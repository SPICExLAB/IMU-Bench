"""MAE objective with frequency-band-weighted reconstruction loss + spectral loss.

Time-domain loss (on masked patches):
    loss_time = Σ_b  (w_b / Σw) * MSE(recon_b[mask], x_b[mask])

Spectral loss (optional, on masked patches):
    loss_spec = L1( |FFT(recon * mask)|, |FFT(x * mask)| )

Total:
    loss = loss_time + spectral_weight * loss_spec

Config keys (all optional, under ``objective.freq_mae``):
    bands            list of [lo, hi] pairs  default see BANDS_DEFAULT
    band_weights     list of floats          default [1.0, 1.0, 1.0, 1.0]
    fs               float                   default 50.0
    spectral_weight  float                   default 0.0  (disabled)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.objectives.masked_1d import (
    generate_patch_mask,
    _patch_mask_to_signal_mask,
)
from imu_lm.utils.helpers import cfg_get

BANDS_DEFAULT   = [(0, 1), (1, 3), (3, 8), (8, 25)]
WEIGHTS_DEFAULT = [1.0, 1.0, 1.0, 1.0]


def _bandpass(x: torch.Tensor, lo: float, hi: float, fs: float) -> torch.Tensor:
    """Band-pass filter x [B, C, T] via FFT zeroing. Returns same shape."""
    T     = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=x.device)
    X_f   = torch.fft.rfft(x.float(), dim=-1)
    keep  = (freqs >= lo) & (freqs < hi)
    X_out = torch.zeros_like(X_f)
    X_out[..., keep] = X_f[..., keep]
    out = torch.fft.irfft(X_out, n=T, dim=-1)
    return out.to(x.dtype)


def forward_loss(
    batch: Tuple[torch.Tensor, Any],
    encoder: nn.Module,
    decoder: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MAE forward pass with frequency-band-weighted loss on masked patches.

    Args:
        batch    : (x, _) where x is [B, C, T]
        encoder  : ViT1DEncoder
        decoder  : ViT1DDecoder
        cfg      : config dict

    Returns:
        loss : weighted sum of per-band MSE on masked positions
        logs : dict with per-band losses and total loss
    """
    padding_mask = None
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        x, _, padding_mask = batch[:3]
    else:
        x, _ = batch
    B, C, T = x.shape

    obj_cfg    = cfg_get(cfg, ["objective"],          {}) or {}
    freq_cfg   = cfg_get(cfg, ["objective", "freq_mae"], {}) or {}
    mask_ratio = float(obj_cfg.get("mask_ratio",    0.75))
    strategy   = str(obj_cfg.get("mask_strategy",   "composite"))
    fs         = float(freq_cfg.get("fs",            50.0))

    raw_bands   = freq_cfg.get("bands",        BANDS_DEFAULT)
    raw_weights = freq_cfg.get("band_weights", WEIGHTS_DEFAULT)
    bands   = [(float(b[0]), float(b[1])) for b in raw_bands]
    weights = [float(w) for w in raw_weights]
    assert len(bands) == len(weights), "bands and band_weights must have same length"

    patch_size              = encoder.patch_size
    num_patches_per_channel = T // patch_size

    # Generate mask
    patch_mask = generate_patch_mask(
        B, num_patches_per_channel, C,
        mask_ratio, strategy, x.device,
        padding_mask=padding_mask,
    )

    # Encode: BERT mode keeps all N tokens; standard MAE drops masked tokens
    if getattr(encoder, '_use_bert_mask', False):
        # Encoder sees all N tokens with learnable mask tokens at masked positions
        all_tokens = encoder(
            x, mask=patch_mask,
            padding_mask=padding_mask,
            return_keep_padding_mask=False,
        )  # [B, N, D]
        # Decoder sees all tokens — no additional mask token insertion needed
        empty_mask = torch.zeros_like(patch_mask)
        reconstructed = decoder(
            all_tokens, empty_mask,
            num_patches_per_channel, C,
            padding_mask=padding_mask,
        )
    else:
        # Standard MAE: encoder sees only visible patches
        visible_tokens, visible_padding_mask = encoder(
            x, mask=patch_mask,
            padding_mask=padding_mask,
            return_keep_padding_mask=True,
        )
        reconstructed = decoder(
            visible_tokens, patch_mask,
            num_patches_per_channel, C,
            padding_mask=padding_mask,
            visible_padding_mask=visible_padding_mask,
        )

    T_recon = reconstructed.shape[-1]
    if T_recon < x.shape[-1]:
        x = x[:, :, :T_recon]

    # ── Full-signal plain MSE (loss_type: full_mse) ───────────────────────────
    loss_type = str(obj_cfg.get("loss_type", "freq_weighted"))
    if loss_type == "full_mse":
        loss = F.mse_loss(reconstructed, x)
        logs: Dict[str, float] = {
            "loss":            float(loss.detach()),
            "mse_loss":        float(loss.detach()),
            "mask_ratio_act":  float(patch_mask.float().mean().item()),
        }
        # During validation: also log per-band and mae_freq_loss (same as v3)
        # so v0 and v3 are directly comparable in WandB without training on them.
        if not encoder.training:
            sm = _patch_mask_to_signal_mask(patch_mask, num_patches_per_channel, C, patch_size)
            if T_recon < sm.shape[-1]:
                sm = sm[:, :, :T_recon]
            ws = sum(weights)
            freq_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
            for (lo, hi), w in zip(bands, weights):
                x_b = _bandpass(x,             lo, hi, fs)
                r_b = _bandpass(reconstructed, lo, hi, fs)
                bl  = F.mse_loss(r_b[sm], x_b[sm])
                logs[f"loss_band_{lo:.0f}_{hi:.0f}hz"] = float(bl.detach())
                freq_loss = freq_loss + (w / ws) * bl
            logs["mae_freq_loss"] = float(freq_loss.detach())
        return loss, logs

    # ── Frequency-band-weighted time-domain loss (default) ────────────────────
    signal_mask = _patch_mask_to_signal_mask(
        patch_mask, num_patches_per_channel, C, patch_size
    )
    if T_recon < signal_mask.shape[-1]:
        signal_mask = signal_mask[:, :, :T_recon]

    # Frequency-band-weighted time-domain loss
    total_loss = torch.zeros(1, device=x.device, dtype=x.dtype)
    logs: Dict[str, float] = {}
    weight_sum = sum(weights)

    for (lo, hi), w in zip(bands, weights):
        x_band     = _bandpass(x,             lo, hi, fs)
        recon_band = _bandpass(reconstructed, lo, hi, fs)
        band_loss  = F.mse_loss(recon_band[signal_mask], x_band[signal_mask])
        w_norm     = w / weight_sum
        total_loss = total_loss + w_norm * band_loss
        logs[f"loss_band_{lo:.0f}_{hi:.0f}hz"] = float(band_loss.detach())

    logs["mae_freq_loss"] = float(total_loss.detach())

    # Spectral loss: L1 on FFT magnitude of full reconstructed vs original signal
    spectral_weight = float(freq_cfg.get("spectral_weight", 0.0))
    if spectral_weight > 0.0:
        x_mag     = torch.fft.rfft(x.float(),             dim=-1).abs()
        recon_mag = torch.fft.rfft(reconstructed.float(), dim=-1).abs()
        spec_loss  = F.l1_loss(recon_mag, x_mag)
        total_loss = total_loss + spectral_weight * spec_loss
        logs["spectral_loss"] = float(spec_loss.detach())

    logs["loss"]           = float(total_loss.detach())
    logs["mask_ratio_act"] = float(patch_mask.float().mean().item())

    return total_loss, logs
