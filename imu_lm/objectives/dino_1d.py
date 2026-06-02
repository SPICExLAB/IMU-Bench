"""DINOv2-style Self-Supervised Objective for 1D IMU signals.

Architecture
------------
Student encoder (trainable) + Teacher encoder (EMA copy, no grad) share a
single ViT1DDINOHead projection head.  Each training step uses two independently-
augmented views of the same window.

  View 1 → student encoder  (with iBOT patch masking) → head → student logits
  View 2 → teacher encoder  (no masking)              → head → teacher logits

Two losses are computed and summed:

  CLS loss  (DINO)  — cross-entropy on global CLS tokens.
                      Teacher output is centred + sharpened; student is just sharpened.
                      Drives the encoder to produce view-consistent global representations.

  Patch loss (iBOT) — cross-entropy on MASKED patch tokens only.
                      Student sees [MASK] at those positions; teacher sees the full signal.
                      Drives local, spatially-aware representations — analogous to
                      masked-patch prediction but in representation space, not signal space.

Collapse prevention
-------------------
  Centering   : running-mean buffer subtracted from teacher logits before softmax.
  Sharpening  : teacher temperature τ_t < student temperature τ_s.
  Weight-norm : ViT1DDINOHead last layer has fixed ||w||=1; only direction is trained.

EMA update
----------
  θ_teacher ← τ · θ_teacher + (1 − τ) · θ_student
  τ is typically annealed from 0.996 → 1.0 during training.

References
----------
  Caron et al. 2021  — DINO
  Zhou et al. 2021   — iBOT
  Oquab et al. 2023  — DINOv2 (KoLeo regularizer)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.objectives.masked_1d import generate_patch_mask
from imu_lm.utils.helpers import cfg_get


# ---------------------------------------------------------------------------
# Loss module
# ---------------------------------------------------------------------------

class DINOLoss(nn.Module):
    """Cross-entropy DINO/iBOT loss with EMA centering for the teacher.

    Registers two buffers (``center_cls``, ``center_patch``) that track the
    running mean of the teacher's CLS and patch projections respectively.
    Centering is updated every forward call during training.

    Args:
        out_dim          : prototype vocabulary size K (must match ViT1DDINOHead)
        teacher_temp     : teacher softmax temperature (low → sharp, default 0.04)
        student_temp     : student softmax temperature (default 0.1)
        center_momentum  : EMA momentum for centering buffers (default 0.9)
        lambda_patch     : weight on the iBOT patch loss term (default 1.0)
    """

    def __init__(
        self,
        out_dim: int,
        teacher_temp: float = 0.04,
        student_temp: float = 0.10,
        center_momentum: float = 0.90,
        lambda_patch: float = 1.0,
    ):
        super().__init__()
        self.teacher_temp    = teacher_temp
        self.student_temp    = student_temp
        self.center_momentum = center_momentum
        self.lambda_patch    = lambda_patch
        self.register_buffer("center_cls",   torch.zeros(1, out_dim))
        self.register_buffer("center_patch", torch.zeros(1, out_dim))

    @torch.no_grad()
    def _update_center(self, t_cls: torch.Tensor, t_patch_flat: torch.Tensor) -> None:
        """EMA update centering buffers from the current batch's teacher outputs."""
        self.center_cls = (
            self.center_cls   * self.center_momentum
            + t_cls.mean(dim=0, keepdim=True)        * (1.0 - self.center_momentum)
        )
        self.center_patch = (
            self.center_patch * self.center_momentum
            + t_patch_flat.mean(dim=0, keepdim=True) * (1.0 - self.center_momentum)
        )

    def forward(
        self,
        s_cls:     torch.Tensor,   # [B, K]   student CLS logits
        t_cls:     torch.Tensor,   # [B, K]   teacher CLS logits
        s_patches: torch.Tensor,   # [B, N, K] student patch logits
        t_patches: torch.Tensor,   # [B, N, K] teacher patch logits
        patch_mask: torch.Tensor,  # [B, N]   True = masked (iBOT target)
    ) -> Tuple[torch.Tensor, float, float]:
        """Returns (total_loss, cls_loss_scalar, patch_loss_scalar)."""

        # ── CLS DINO loss ──────────────────────────────────────────────────
        t_cls_prob = F.softmax(
            (t_cls - self.center_cls) / self.teacher_temp, dim=-1
        ).detach()
        cls_loss = -(t_cls_prob * F.log_softmax(s_cls / self.student_temp, dim=-1)).sum(dim=-1).mean()

        # ── iBOT patch loss (masked positions only) ────────────────────────
        if patch_mask.any():
            t_pat_sel = t_patches[patch_mask]   # [N_masked, K]
            s_pat_sel = s_patches[patch_mask]   # [N_masked, K]
            t_pat_prob = F.softmax(
                (t_pat_sel - self.center_patch) / self.teacher_temp, dim=-1
            ).detach()
            patch_loss = -(t_pat_prob * F.log_softmax(s_pat_sel / self.student_temp, dim=-1)).sum(dim=-1).mean()
        else:
            patch_loss = cls_loss.new_zeros(())

        # ── Update centering (teacher statistics) ─────────────────────────
        if self.training:
            self._update_center(t_cls.detach(), t_patches.detach().reshape(-1, t_patches.shape[-1]))

        total = cls_loss + self.lambda_patch * patch_loss
        return total, float(cls_loss.detach()), float(patch_loss.detach())


# ---------------------------------------------------------------------------
# KoLeo regularizer
# ---------------------------------------------------------------------------

def koleo_loss(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Kolmogorov-Leonenko entropy estimator (Oquab et al. 2023, DINOv2).

    Maximises the minimum pairwise distance between L2-normalised representations
    within the batch, encouraging uniform coverage of the unit hypersphere.

    Loss = -mean_i log( min_{j≠i} ||z_i - z_j|| )

    For unit vectors: ||z_i - z_j||^2 = 2 - 2·(z_i · z_j), so distances are
    computed efficiently via a single matrix multiply.

    Args:
        z   : [B, D]  representations (need not be pre-normalised)
        eps : floor added inside log for numerical stability

    Returns:
        scalar KoLeo loss (minimise to spread representations)
    """
    z = F.normalize(z, dim=-1, p=2)          # unit hypersphere
    # Pairwise squared L2 distances via dot products: 2 - 2·(z·zᵀ)
    dot  = z @ z.T                            # [B, B]
    sq_d = (2.0 - 2.0 * dot).clamp(min=0.0)  # [B, B], numerical safety
    sq_d.fill_diagonal_(float("inf"))         # exclude self-distance
    nn_sq_d = sq_d.min(dim=1).values          # [B]  nearest-neighbour sq dist
    # ||·|| = sqrt(sq_d); log(sqrt(x)) = 0.5·log(x)
    return -0.5 * (nn_sq_d + eps).log().mean()


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def forward_loss(
    batch:                Any,
    student_encoder:      nn.Module,
    teacher_encoder_ema:  nn.Module,
    student_dino_head:    nn.Module,
    teacher_dino_head:    nn.Module,
    dino_loss_fn:         DINOLoss,
    cfg:                  Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """DINOv2-style forward pass for 1D IMU.

    Args:
        batch               : (x_student [B,C,T], x_teacher [B,C,T], _labels)
                              produced by OCDetectDataset with multi_view=True
        student_encoder     : ViT1DEncoder with use_cls_token=True (trainable)
        teacher_encoder_ema : EMA copy of student encoder (no grad, EMA-updated)
        student_dino_head   : ViT1DDINOHead for student (trainable)
        teacher_dino_head   : EMA copy of student_dino_head (no grad, EMA-updated)
        dino_loss_fn        : DINOLoss module
        cfg                 : merged experiment config dict

    Returns:
        (loss, log_dict)
    """
    # Two-view (train): batch = (x_student, x_teacher, labels)
    # Single-view (val): batch = (x, labels) — use same tensor for both; skip iBOT mask
    two_view = isinstance(batch, (tuple, list)) and len(batch) >= 3 and isinstance(batch[1], torch.Tensor) and batch[1].shape == batch[0].shape
    x_student = batch[0]                              # [B, C, T]
    x_teacher = batch[1] if two_view else batch[0]    # [B, C, T]
    B, C, T = x_student.shape

    obj_cfg      = cfg_get(cfg, ["objective"], {}) or {}
    mask_ratio   = float(obj_cfg.get("mask_ratio",    0.40))
    strategy     = str(obj_cfg.get("mask_strategy",   "temporal"))
    lambda_koleo = float(
        (cfg_get(cfg, ["objective", "dino_ocdetect_anyrot"], {}) or {})
        .get("loss", {}).get("lambda_koleo", 0.1)
    )

    Pt = T // student_encoder.patch_size
    N  = C * Pt

    # Patch mask for iBOT (applied to student only; skipped in single-view / val mode)
    if two_view:
        patch_mask = generate_patch_mask(B, Pt, C, mask_ratio, strategy, x_student.device)
    else:
        patch_mask = torch.zeros(B, N, dtype=torch.bool, device=x_student.device)

    # ── Student: masked patches → BERT-style [MASK] token inside encoder ──
    s_cls, s_patches = student_encoder.forward_dino(x_student, patch_mask=patch_mask)
    # [B, D] and [B, N, D]

    s_cls_proj   = student_dino_head(s_cls)                                          # [B, K]
    s_patch_proj = student_dino_head(s_patches.reshape(B * N, -1)).reshape(B, N, -1) # [B, N, K]

    # ── Teacher: full (unmasked) view → EMA encoder + EMA head ───────────
    with torch.no_grad():
        teacher_encoder_ema.eval()
        teacher_dino_head.eval()
        t_cls, t_patches = teacher_encoder_ema.forward_dino(x_teacher, patch_mask=None)
        t_cls_proj   = teacher_dino_head(t_cls)                                          # [B, K]
        t_patch_proj = teacher_dino_head(t_patches.reshape(B * N, -1)).reshape(B, N, -1)

    # ── Loss ───────────────────────────────────────────────────────────────
    loss, cls_loss, patch_loss = dino_loss_fn(
        s_cls_proj, t_cls_proj,
        s_patch_proj, t_patch_proj,
        patch_mask,
    )

    # KoLeo: applied to student CLS encoder output (before head), training only
    # Encourages uniform spread on the unit hypersphere → combats proto collapse
    if two_view and lambda_koleo > 0.0:
        kl = koleo_loss(s_cls)
        loss = loss + lambda_koleo * kl
        koleo_loss_val = float(kl.detach())
    else:
        koleo_loss_val = 0.0

    # Collapse diagnostics (all no_grad)
    with torch.no_grad():
        # Cosine similarity between student and teacher CLS
        cos_sim = float(
            F.cosine_similarity(
                s_cls_proj.detach(), t_cls_proj.detach(), dim=-1
            ).mean().item()
        )

        # Prototype entropy: how spread are teacher assignments across K prototypes?
        # Use centered logits (same as DINOLoss) so we measure what the loss actually sees.
        # H = -sum p(k) log p(k);  max = log(K) ≈ 8.32 for K=4096
        # Low entropy → few prototypes used → possible collapse
        t_centered = t_cls_proj.detach() - dino_loss_fn.center_cls                    # [B, K]
        t_probs = F.softmax(t_centered / dino_loss_fn.teacher_temp, dim=-1)           # [B, K]
        avg_probs = t_probs.mean(dim=0)                                                # [K]
        proto_entropy = float(-(avg_probs * (avg_probs + 1e-8).log()).sum().item())

        # Effective number of prototypes: exp(H), intuitive count in [1, K]
        proto_eff_k = float(math.exp(proto_entropy))

    return loss, {
        "loss":          float(loss.detach()),
        "cls_loss":      cls_loss,
        "patch_loss":    patch_loss,
        "koleo_loss":    koleo_loss_val,  # 0 at val time (single-view)
        "mask_ratio":    float(patch_mask.float().mean()),
        "cosine_sim":    cos_sim,
        "proto_entropy": proto_entropy,   # H in nats; max=log(4096)≈8.32
        "proto_eff_k":   proto_eff_k,     # exp(H); max=4096, collapse→1
    }


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(
    student_encoder:     nn.Module,
    teacher_encoder_ema: nn.Module,
    momentum:            float,
) -> None:
    """θ_teacher ← τ · θ_teacher + (1 − τ) · θ_student."""
    for p_s, p_t in zip(
        student_encoder.parameters(), teacher_encoder_ema.parameters()
    ):
        p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)
