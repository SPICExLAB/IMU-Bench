"""MAE + DINOv1 CLS combined SSL objective.

Student encoder processes a masked view (BERT-style [MASK] tokens):
  - Patch tokens at masked positions → linear reconstruction head → MSE loss (MAE)
  - CLS token → DINO head → prototype distribution → DINO CLS loss

Teacher encoder (EMA, no grad) processes a differently-augmented full view:
  - CLS token → DINO head (EMA copy) → teacher prototype distribution

Total loss = lambda_mae * MSE + lambda_cls * DINO_CLS + lambda_koleo * KoLeo

Simpler than DINOv2/iBOT:
  - No iBOT patch-level distillation
  - No patch-level DINO head
  - Reconstruction target is the raw signal (interpretable)
  - One centering buffer (CLS only)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.objectives.masked_1d import generate_patch_mask
from imu_lm.objectives.dino_1d import koleo_loss
from imu_lm.utils.helpers import cfg_get


# ---------------------------------------------------------------------------
# CLS-only DINO loss
# ---------------------------------------------------------------------------

class DINOCLSLoss(nn.Module):
    """DINOv1-style CLS self-distillation loss with EMA centering.

    Only tracks a single centering buffer (CLS), unlike DINOv2 which also
    centers patch tokens for iBOT.
    """

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.10,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.teacher_temp    = teacher_temp
        self.student_temp    = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def _update_center(self, t_cls: torch.Tensor) -> None:
        self.center = (
            self.center * self.center_momentum
            + t_cls.mean(dim=0, keepdim=True) * (1.0 - self.center_momentum)
        )

    def forward(self, s_cls: torch.Tensor, t_cls: torch.Tensor) -> torch.Tensor:
        """Cross-entropy between student and sharpened+centered teacher CLS."""
        t_prob = F.softmax(
            (t_cls - self.center) / self.teacher_temp, dim=-1
        ).detach()
        loss = -(t_prob * F.log_softmax(s_cls / self.student_temp, dim=-1)).sum(dim=-1).mean()
        if self.training:
            self._update_center(t_cls.detach())
        return loss


# ---------------------------------------------------------------------------
# Per-patch reconstruction head
# ---------------------------------------------------------------------------

class PatchReconHead(nn.Module):
    """Lightweight head: patch embedding → reconstructed signal values.

    Applied only to masked patch tokens. Output shape matches the raw
    signal at that patch position: patch_size timesteps (one channel).
    """

    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(x))


# ---------------------------------------------------------------------------
# Helper: extract raw signal values at masked patch positions
# ---------------------------------------------------------------------------

def _extract_masked_targets(
    x: torch.Tensor,            # [B, C, T]
    patch_mask: torch.Tensor,   # [B, N]  True = masked, N = C * Pt
    n_patches_per_ch: int,
    patch_size: int,
) -> torch.Tensor:
    """Return signal at masked positions: [N_masked_total, patch_size]."""
    B, C, _ = x.shape
    # Reshape to [B, C, Pt, patch_size]
    x_patches = x.reshape(B, C, n_patches_per_ch, patch_size)
    # patch_mask channel-major: reshape to [B, C, Pt]
    mask_2d = patch_mask.view(B, C, n_patches_per_ch)
    return x_patches[mask_2d]   # [N_masked_total, patch_size]


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_loss(
    batch:               Any,
    student_encoder:     nn.Module,
    teacher_encoder_ema: nn.Module,
    dino_head_s:         nn.Module,
    dino_head_t:         nn.Module,
    recon_head:          nn.Module,
    dino_loss_fn:        DINOCLSLoss,
    cfg:                 Any,
    lambda_koleo:        float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined MAE + DINOv1 CLS forward pass.

    Args:
        batch               : (x_student [B,C,T], x_teacher [B,C,T], labels)
                              produced by OCDetectDataset with multi_view=True
        student_encoder     : ViT1DEncoder with use_cls_token=True (trainable)
        teacher_encoder_ema : EMA copy of student_encoder (no grad)
        dino_head_s         : ViT1DDINOHead for student (trainable)
        dino_head_t         : EMA copy of dino_head_s (no grad)
        recon_head          : PatchReconHead (trainable)
        dino_loss_fn        : DINOCLSLoss module
        cfg                 : merged experiment config dict
        lambda_koleo        : weight on KoLeo regularizer
    """
    two_view = (
        isinstance(batch, (tuple, list))
        and len(batch) >= 3
        and isinstance(batch[1], torch.Tensor)
        and batch[1].shape == batch[0].shape
    )
    x_s = batch[0]                              # [B, C, T]  student view (augmented)
    x_t = batch[1] if two_view else batch[0]    # [B, C, T]  teacher view (different aug or same at val)
    B, C, T = x_s.shape

    obj_cfg    = cfg_get(cfg, ["objective"], {}) or {}
    mask_ratio = float(obj_cfg.get("mask_ratio",   0.50))
    strategy   = str(obj_cfg.get("mask_strategy",  "random,temporal"))
    lambda_mae = float(obj_cfg.get("lambda_mae",   1.0))
    lambda_cls = float(obj_cfg.get("lambda_cls",   1.0))

    Pt = T // student_encoder.patch_size
    N  = C * Pt

    # Always mask — MAE loss is meaningful at both train and val time.
    # At val, x_t == x_s (single view); teacher still sees the full signal
    # while student sees the masked version, so mae_loss and cls_loss are valid.
    patch_mask = generate_patch_mask(B, Pt, C, mask_ratio, strategy, x_s.device)

    # ── Student forward (masked view, BERT-style [MASK] tokens) ───────────
    # forward_dino replaces masked positions with learned mask_token and
    # processes all N tokens through the transformer.
    s_cls, s_patches = student_encoder.forward_dino(x_s, patch_mask=patch_mask)
    # s_cls:     [B, D]
    # s_patches: [B, N, D]

    # ── MAE loss: reconstruct signal at masked positions ───────────────────
    if patch_mask.any():
        masked_tokens = s_patches[patch_mask]                              # [N_msk, D]
        pred_signal   = recon_head(masked_tokens)                          # [N_msk, patch_size]
        tgt_signal    = _extract_masked_targets(x_s, patch_mask, Pt,
                                                student_encoder.patch_size)  # [N_msk, patch_size]
        mae_loss = F.mse_loss(pred_signal, tgt_signal)
    else:
        mae_loss = x_s.new_zeros(())

    # ── DINO CLS loss ──────────────────────────────────────────────────────
    s_cls_proj = dino_head_s(s_cls)   # [B, K]

    with torch.no_grad():
        teacher_encoder_ema.eval()
        dino_head_t.eval()
        t_cls, _ = teacher_encoder_ema.forward_dino(x_t, patch_mask=None)
        t_cls_proj = dino_head_t(t_cls)                                    # [B, K]

    cls_loss = dino_loss_fn(s_cls_proj, t_cls_proj)

    # ── KoLeo regularizer on student CLS ──────────────────────────────────
    if lambda_koleo > 0.0:
        kl = koleo_loss(s_cls)
        koleo_val = float(kl.detach())
    else:
        kl = x_s.new_zeros(())
        koleo_val = 0.0

    total = lambda_mae * mae_loss + lambda_cls * cls_loss + lambda_koleo * kl

    # ── Diagnostics ────────────────────────────────────────────────────────
    with torch.no_grad():
        cos_sim = float(
            F.cosine_similarity(s_cls_proj.detach(), t_cls_proj.detach(), dim=-1).mean()
        )
        t_centered  = t_cls_proj.detach() - dino_loss_fn.center
        t_probs     = F.softmax(t_centered / dino_loss_fn.teacher_temp, dim=-1)
        avg_probs   = t_probs.mean(dim=0)
        proto_H     = float(-(avg_probs * (avg_probs + 1e-8).log()).sum())
        proto_eff_k = float(math.exp(proto_H))

    return total, {
        "loss":         float(total.detach()),
        "mae_loss":     float(mae_loss.detach()),
        "cls_loss":     float(cls_loss.detach()),
        "koleo_loss":   koleo_val,
        "mask_ratio":   float(patch_mask.float().mean()),
        "cosine_sim":   cos_sim,
        "proto_entropy": proto_H,
        "proto_eff_k":  proto_eff_k,
    }


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """θ_teacher ← τ · θ_teacher + (1 − τ) · θ_student."""
    for p_s, p_t in zip(student.parameters(), teacher.parameters()):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
