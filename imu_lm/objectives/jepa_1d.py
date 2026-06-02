"""I-JEPA style Self-Supervised Objective for 1D IMU signals (Assran et al. 2023).

Joint Embedding Predictive Architecture for sensor data:

  context_encoder  (trainable)   : [B, C, T] → context token representations
                                   receives only the UNmasked (context) patches
  target_encoder   (EMA, no-grad): [B, C, T] → full token representations
                                   receives the FULL unmasked input
  predictor        (trainable)   : context tokens → predicted embeddings
                                   at masked (target) positions

Loss
----
  Smooth-L1 between predictor output and (layer-normed) target encoder output
  at every target (masked) position.  The loss is zero at context positions.

EMA update
----------
  θ_target ← τ · θ_target + (1 - τ) · θ_context
  where τ is the momentum (default 0.996).  Called at the end of objective_fn
  while the encoder is in training mode; the predictor's optimizer step has
  not yet occurred (one-step delay), which is acceptable in practice.

Masking
-------
  Reuses generate_patch_mask from masked_1d.py:
    - patch_mask[b, i] = True  → position i is a target (removed from context)
    - mask_ratio controls the fraction of positions that are targets
    - strategy controls the spatial pattern (random | temporal | signal | composite)

Key differences from MAE
-------------------------
  - Prediction is in *latent space* (D_enc-dimensional), not in raw signal space
  - Target is produced by an EMA (momentum) copy of the encoder
  - No reconstruction of the original signal → more abstract representations
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.objectives.masked_1d import generate_patch_mask
from imu_lm.utils.helpers import cfg_get


def forward_loss(
    batch: Any,
    context_encoder: nn.Module,
    target_encoder_ema: nn.Module,
    predictor: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """JEPA forward pass: masked context → predict target encoder embeddings.

    Args:
        batch            : (x, _) where x is [B, C, T] raw IMU signal
        context_encoder  : ViT1DEncoder (trainable), processes visible patches only
        target_encoder_ema: ViT1DEncoder EMA copy (no grad), processes full input
        predictor        : JEPA1DPredictor, maps context → target embeddings
        cfg              : experiment config dict

    Returns:
        (loss, logs)
    """
    padding_mask = None
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        x, _, padding_mask = batch[:3]
    else:
        x, _ = batch
    B, C, T = x.shape

    obj_cfg    = cfg_get(cfg, ["objective"], {}) or {}
    mask_ratio = float(obj_cfg.get("mask_ratio", 0.75))
    strategy   = str(obj_cfg.get("mask_strategy", "composite"))

    patch_size              = context_encoder.patch_size
    num_patches_per_channel = T // patch_size
    N                       = C * num_patches_per_channel

    # Generate patch mask: True = target (removed from context encoder)
    patch_mask = generate_patch_mask(
        B, num_patches_per_channel, C,
        mask_ratio, strategy,
        x.device, padding_mask=padding_mask,
    )

    # -------------------------------------------------------------------
    # Context encoder: sees only context (unmasked) patches
    # -------------------------------------------------------------------
    ctx_tokens, ctx_padding_mask = context_encoder(
        x,
        mask=patch_mask,
        padding_mask=padding_mask,
        return_keep_padding_mask=True,
    )  # ctx_tokens: [B, N_ctx_max, D_enc]

    # -------------------------------------------------------------------
    # Target encoder (EMA, no gradient): sees the full input
    # -------------------------------------------------------------------
    with torch.no_grad():
        target_encoder_ema.eval()
        full_tokens, _ = target_encoder_ema.forward_tokens(x)  # [B, N, D_enc]
        # Layer-norm targets (I-JEPA / data2vec: prevents representational collapse)
        target_tokens = F.layer_norm(full_tokens, [full_tokens.shape[-1]])

    # -------------------------------------------------------------------
    # Predictor: context → target embedding predictions
    # -------------------------------------------------------------------
    pred_tokens = predictor(
        ctx_tokens,
        patch_mask,
        num_patches_per_channel,
        C,
        ctx_padding_mask=ctx_padding_mask,
    )  # [B, N, D_enc]

    # -------------------------------------------------------------------
    # Loss: smooth-L1 at target positions only
    # -------------------------------------------------------------------
    target_mask = patch_mask.clone()
    if padding_mask is not None:
        target_mask = target_mask & ~padding_mask

    pred_flat   = pred_tokens[target_mask]    # [N_tgt_total, D_enc]
    target_flat = target_tokens[target_mask]  # [N_tgt_total, D_enc]

    loss = F.smooth_l1_loss(pred_flat, target_flat)

    actual_ratio = float(patch_mask.float().mean().item())
    cosine_sim   = float(
        F.cosine_similarity(pred_flat.detach(), target_flat.detach(), dim=-1).mean().item()
    )

    return loss, {
        "loss":         float(loss.detach().item()),
        "mask_ratio":   actual_ratio,
        "cosine_sim":   cosine_sim,
    }


@torch.no_grad()
def update_ema(
    context_encoder: nn.Module,
    target_encoder_ema: nn.Module,
    momentum: float,
) -> None:
    """Exponential moving average update: θ_target ← τ·θ_target + (1-τ)·θ_context."""
    for p_ctx, p_tgt in zip(
        context_encoder.parameters(), target_encoder_ema.parameters()
    ):
        p_tgt.data.mul_(momentum).add_(p_ctx.data, alpha=1.0 - momentum)
