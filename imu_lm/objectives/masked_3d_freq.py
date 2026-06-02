"""MAE objective for the 3D frequency-token architecture (v6).

The encoder represents each token as (channel c, time-patch t, frequency band f).
Band 0 is the original signal; bands 1..F-1 are FFT-filtered sub-bands.

Masking is applied at the (channel, time) level and tiled across the frequency
axis — a masked (c, t) position hides ALL its frequency bands, so the model
must infer the missing content from neighbouring visible (c, t) positions
(otherwise reconstruction would be trivial: band 0 ≈ sum of sub-bands).

Reconstruction loss = weighted sum of per-band MSE on masked patches:

    loss = Σ_f  (w_f / Σw) * MSE(recon[:, f][mask], target[:, f][mask])

where target[:, f] is the f-th frequency decomposition of the original signal.

Config keys (under ``objective``):
    mask_ratio       float  default 0.75
    mask_strategy    str    default "composite"
    freq_3d.band_weights  list of floats  default uniform
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.objectives.masked_1d import (
    generate_patch_mask,
    _patch_mask_to_signal_mask,
)
from imu_lm.utils.helpers import cfg_get


def forward_loss(
    batch: Tuple[torch.Tensor, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MAE forward pass with 3D frequency-token reconstruction.

    Args:
        batch   : (x, _) where x is [B, C, T]
        encoder : ViT1DEncoder3D
        decoder : ViT1DDecoder3D
    Returns:
        loss, logs
    """
    if isinstance(batch, (tuple, list)):
        x = batch[0]
    else:
        x = batch
    B, C, T = x.shape

    obj_cfg    = cfg_get(cfg, ["objective"],             {}) or {}
    freq_cfg   = cfg_get(cfg, ["objective", "freq_3d"],  {}) or {}
    mask_ratio = float(obj_cfg.get("mask_ratio",  0.75))
    strategy   = str(obj_cfg.get("mask_strategy", "composite"))

    ps  = encoder.patch_size
    Pt  = T // ps
    Fn  = encoder.num_freq_bands

    raw_weights = freq_cfg.get("band_weights", [1.0] * Fn)
    weights     = [float(w) for w in raw_weights]
    assert len(weights) == Fn, f"band_weights must have {Fn} entries (one per band)"
    weight_sum  = sum(weights)

    # ── Mask at (channel, time) level, then tile across frequency ─────────────
    patch_mask_ct = generate_patch_mask(B, Pt, C, mask_ratio, strategy, x.device)  # [B, C*Pt]
    patch_mask_3d = patch_mask_ct.repeat(1, Fn)                                     # [B, F*C*Pt]

    # ── Encode visible tokens, decode all bands ───────────────────────────────
    visible_tokens, vis_pad = encoder(
        x, mask=patch_mask_3d, return_keep_padding_mask=True,
    )
    reconstructed = decoder(
        visible_tokens, patch_mask_3d, C, visible_padding_mask=vis_pad,
    )  # [B, F, C, T]

    # ── Targets: frequency decomposition of the original signal ───────────────
    target = encoder.decompose(x)                      # [B, F, C, T]

    T_recon = reconstructed.shape[-1]
    if T_recon < target.shape[-1]:
        target = target[..., :T_recon]

    # Signal-level (c, t) mask [B, C, T] — same for every band
    signal_mask_ct = _patch_mask_to_signal_mask(patch_mask_ct, Pt, C, ps)  # [B, C, T]
    if T_recon < signal_mask_ct.shape[-1]:
        signal_mask_ct = signal_mask_ct[:, :, :T_recon]

    # ── Per-band weighted MSE on masked patches ───────────────────────────────
    total_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
    logs: Dict[str, float] = {}
    for f in range(Fn):
        recon_f = reconstructed[:, f]                  # [B, C, T]
        tgt_f   = target[:, f]                          # [B, C, T]
        band_loss = F.mse_loss(recon_f[signal_mask_ct], tgt_f[signal_mask_ct])
        w_norm = weights[f] / weight_sum
        total_loss = total_loss + w_norm * band_loss
        band_label = "orig" if encoder.bands[f] is None else \
                     f"{encoder.bands[f][0]:.0f}_{encoder.bands[f][1]:.0f}hz"
        logs[f"loss_band_{band_label}"] = float(band_loss.detach())

    logs["mae_3d_loss"]    = float(total_loss.detach())
    logs["mask_ratio_act"] = float(patch_mask_ct.float().mean().item())

    return total_loss, logs
