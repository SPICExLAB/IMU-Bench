"""BEiT v2 style IMU pretraining: predict discrete RVQ token indices on masked patches.

Architecture
------------
  Encoder : ViT1D with CLS token  (``forward_dino`` API)
            - All positions processed: CLS + visible patches + [MASK] tokens
            - CLS token learns a global representation (used for downstream tasks)
  Tokenizer: Frozen RVQ model loaded from a pre-trained run directory
             - Produces K discrete indices per temporal patch position
  Heads   : K linear heads  (one per RVQ level):  D → num_codes
            - Applied to the masked patch token outputs

BEiT v2 conventions used
-------------------------
  - Masked patches replaced with a learnable [MASK] token before the encoder
  - CLS attends to ALL positions (masked + visible) — same as BERT
  - Loss applied ONLY to masked patches (cross-entropy on predicted token index)
  - CLS token output is not directly supervised during pretraining

Token index alignment
---------------------
  The RVQ tokenizer uses channel pooling (pool_channels=True), producing one
  index per *temporal* patch position:
      token index shape per level: [B, Pt]   (Pt = T // patch_size)

  The encoder has C*Pt patch tokens in channel-major layout:
      position n → temporal position  t = n % Pt

  For each masked patch n, the target is ``token_indices[:, n % Pt, k]`` for level k.
  Channels share the same temporal token index.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.objectives.masked_1d import generate_patch_mask
from imu_lm.utils.helpers import cfg_get


# ---------------------------------------------------------------------------
# Frozen RVQ tokenizer wrapper
# ---------------------------------------------------------------------------

class FrozenRVQTokenizer(nn.Module):
    """Wraps a pre-trained RVQ encoder+quantizer as a frozen target tokenizer.

    All parameters are frozen (no gradients, no EMA updates during BEiT training).
    """

    def __init__(
        self,
        tok_encoder:   nn.Module,
        rvq:           nn.Module,
        projection:    Optional[nn.Module],
        pool_channels: bool = True,
        num_channels:  int  = 6,
    ):
        super().__init__()
        self.tok_encoder   = tok_encoder
        self.rvq           = rvq
        self.projection    = projection
        self.pool_channels = pool_channels
        self.num_channels  = num_channels

        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def get_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize input and return discrete indices.

        Args:
            x : [B, C, T]

        Returns:
            indices : [B, Pt, K]  int64  (K = num RVQ levels, Pt = temporal patches)
        """
        # Disable autocast: frozen tokenizer must run in fp32.
        # Also lazily move tokenizer to the input device — the trainer moves the
        # main encoder to GPU after init, but the frozen tokenizer is not part of
        # the main model and won't be moved automatically.
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()
            dev = x_fp32.device
            self.tok_encoder.to(dev)
            self.rvq.to(dev)
            if self.projection is not None:
                self.projection.to(dev)
            tokens, Pt = self.tok_encoder.forward_tokens(x_fp32)  # [B, C*Pt, D]
            B, C = x.shape[0], x.shape[1]
            D = tokens.shape[-1]

            if self.pool_channels:
                tokens_q = tokens.view(B, C, Pt, D).mean(dim=1)  # [B, Pt, D]
            else:
                tokens_q = tokens  # [B, C*Pt, D]

            z = self.projection(tokens_q) if self.projection is not None else tokens_q
            z = F.normalize(z, dim=-1)  # must match RVQ training

            _, indices_list, _ = self.rvq(z)  # list of K tensors [B, N_q]
            return torch.stack(indices_list, dim=-1)  # [B, Pt, K]


def load_frozen_tokenizer(
    run_dir:   str,
    device:    torch.device,
    cfg_key:   str = "rvq_tokenizer_ocdetect_anyrot",
) -> FrozenRVQTokenizer:
    """Load a frozen RVQ tokenizer from a checkpoint directory.

    Reads the checkpoint's ``cfg`` to reconstruct the exact tokenizer architecture,
    then loads weights and freezes all parameters.

    Args:
        run_dir : training run directory (must contain checkpoints/latest.pt)
        device  : target device
        cfg_key : objective config key used when the tokenizer was trained

    Returns:
        FrozenRVQTokenizer (all params frozen, eval mode)
    """
    import os
    from imu_lm.models.ViT1D.model import ViT1DEncoder
    from imu_lm.objectives.rvq_tokenizer import (
        ResidualVectorQuantizer,
    )

    ckpt_path = os.path.join(run_dir, "checkpoints", "latest.pt")
    if not os.path.exists(ckpt_path):
        ckpts = sorted(f for f in os.listdir(os.path.join(run_dir, "checkpoints"))
                       if f.endswith(".pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found under {run_dir}/checkpoints/")
        ckpt_path = os.path.join(run_dir, "checkpoints", ckpts[-1])

    state = torch.load(ckpt_path, map_location=device)
    tok_cfg = state["cfg"]

    # Build tokenizer encoder
    tok_encoder = ViT1DEncoder(tok_cfg).to(device)
    tok_encoder.load_state_dict(state["model"], strict=True)
    tok_encoder.eval()

    # Read RVQ hyper-params from saved cfg
    rvq_cfg        = cfg_get(tok_cfg, ["objective", cfg_key], {}) or {}
    num_levels     = int(rvq_cfg.get("num_levels",    4))
    num_codes      = int(rvq_cfg.get("num_codes",     512))
    code_dim       = int(rvq_cfg.get("code_dim",      256))
    ema_decay      = float(rvq_cfg.get("ema_decay",   0.99))
    cmt_w          = float(rvq_cfg.get("commitment_weight", 0.25))
    pool_channels  = bool(rvq_cfg.get("pool_channels", True))
    num_channels   = int(rvq_cfg.get("num_channels",   6))

    rvq = ResidualVectorQuantizer(
        num_levels=num_levels,
        num_codes=num_codes,
        code_dim=code_dim,
        ema_decay=ema_decay,
        commitment_weight=cmt_w,
    ).to(device)
    rvq.load_state_dict(state["rvq"], strict=True)
    rvq.eval()

    # Optional encoder→code_dim projection
    proj = None
    if tok_encoder.embed_dim != code_dim and "rvq_proj" in state:
        proj = nn.Linear(tok_encoder.embed_dim, code_dim).to(device)
        proj.load_state_dict(state["rvq_proj"], strict=True)
        proj.eval()

    tokenizer = FrozenRVQTokenizer(
        tok_encoder=tok_encoder,
        rvq=rvq,
        projection=proj,
        pool_channels=pool_channels,
        num_channels=num_channels,
    ).to(device)

    print(
        f"[beit_1d] Loaded frozen tokenizer from step {state.get('step', '?')}  "
        f"| {num_levels} levels × {num_codes} codes × {code_dim}D  "
        f"| pool_channels={pool_channels}"
    )
    return tokenizer


# ---------------------------------------------------------------------------
# Forward loss
# ---------------------------------------------------------------------------

def forward_loss(
    batch:       Any,
    encoder:     nn.Module,
    token_heads: nn.ModuleList,     # K Linear(D, num_codes) heads
    cfg:         Any,
    tokenizer:   FrozenRVQTokenizer,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BEiT v2 forward pass.

    Steps
    -----
    1. Tokenize input with frozen RVQ → target indices [B, Pt, K]
    2. Generate patch mask (65 %, composite strategy)
    3. BEiT encoder: replace masked patches with [MASK] token, prepend CLS
    4. Linear heads predict RVQ index for each masked patch, per level
    5. Cross-entropy loss on masked patches, averaged over K levels

    Args:
        batch       : (x [B,C,T], *labels)
        encoder     : ViT1DEncoder with use_cls_token=True
        token_heads : nn.ModuleList of K Linear(embed_dim, num_codes)
        cfg         : merged experiment config dict
        tokenizer   : FrozenRVQTokenizer (already on correct device)

    Returns:
        loss : scalar
        logs : dict with per-level CE loss, accuracy, mask ratio
    """
    x = batch[0] if isinstance(batch, (list, tuple)) else batch  # [B, C, T]
    B, C, T = x.shape
    Pt = T // encoder.patch_size
    N  = C * Pt  # total patch tokens (channel-major layout)

    # Read config from whichever beit_* variant is active
    obj_key     = cfg_get(cfg, ["objective", "type"], "beit_ocdetect_anyrot")
    beit_cfg    = cfg_get(cfg, ["objective", obj_key], {}) or {}
    mask_ratio  = float(beit_cfg.get("mask_ratio",  0.65))
    strategy    = str(beit_cfg.get("mask_strategy", "composite"))
    num_levels  = len(token_heads)

    # ── 1. Frozen tokenizer → target indices ─────────────────────────────────
    #    pool_channels=True  → [B, Pt, K]   (one token per temporal position)
    #    pool_channels=False → [B, C*Pt, K] (one token per channel-patch pair)
    target_indices = tokenizer.get_indices(x)   # [B, Pt or N, K]  int64

    # ── 2. Patch mask [B, C*Pt] bool, True = masked ──────────────────────────
    patch_mask = generate_patch_mask(
        B, Pt, C, mask_ratio, strategy, x.device
    )   # [B, N]

    # ── 3. BEiT encoder: CLS + mask-token replacement ────────────────────────
    cls_out, patch_out = encoder.forward_dino(x, patch_mask)
    # patch_out: [B, N, D]

    # ── 4. Map tokenizer indices to encoder patch positions ───────────────────
    #    pool_channels=True : tokenizer has Pt tokens → map position n → n % Pt
    #    pool_channels=False: tokenizer has C*Pt tokens → direct correspondence
    N_tok = target_indices.shape[1]
    if N_tok == Pt:
        time_positions = torch.arange(N, device=x.device) % Pt  # [N]
        targets_all = target_indices[:, time_positions, :]       # [B, N, K]
    else:
        targets_all = target_indices                              # [B, N, K]

    total_loss = x.new_zeros(())
    logs: Dict[str, float] = {}

    for k, head in enumerate(token_heads):
        logits  = head(patch_out)              # [B, N, num_codes]
        targets = targets_all[:, :, k]         # [B, N]

        # Loss only on masked patches
        logits_m  = logits[patch_mask]         # [M, num_codes]
        targets_m = targets[patch_mask]        # [M]

        loss_k = F.cross_entropy(logits_m, targets_m)
        total_loss = total_loss + loss_k

        with torch.no_grad():
            acc = (logits_m.argmax(-1) == targets_m).float().mean()

        logs[f"ce_loss_l{k+1}"] = float(loss_k.detach())
        logs[f"acc_l{k+1}"]     = float(acc)

    total_loss = total_loss / num_levels

    logs.update({
        "loss":       float(total_loss.detach()),
        "mask_ratio": float(patch_mask.float().mean()),
    })

    return total_loss, logs
