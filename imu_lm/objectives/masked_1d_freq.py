"""MAE objective with frequency-band-weighted reconstruction loss + spectral loss.

Time-domain loss (on masked patches):
    loss_time = Σ_b  (w_b / Σw) * MSE(recon_b[mask], x_b[mask])

Spectral loss (optional, full signal): length-invariant log-magnitude L1
    loss_spec = L1( log1p|rfft(recon, ortho)|, log1p|rfft(x, ortho)| )
    (orthonormal FFT → scale invariant to window length; log-mag → band-balanced)

Total:
    loss = loss_time + spectral_weight * loss_spec
    (with loss_type: full_mse, loss_time is the plain full-signal MSE instead)

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


def _spectral_loss(recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Length-invariant log-magnitude spectral L1 between recon and x [B, C, T].

    - Orthonormal FFT (``norm="ortho"``) is Parseval-normalized, so the scale is
      invariant to the window length T — important with variable-length windows.
    - ``log1p`` on the magnitudes compresses the gravity-dominated low-frequency
      content so every band contributes, instead of low freq drowning out high.
    - Magnitude only (phase ignored): the time-domain MSE already supervises
      phase/alignment, so this term just matches the energy distribution.
    """
    recon_mag = torch.fft.rfft(recon.float(), dim=-1, norm="ortho").abs()
    x_mag     = torch.fft.rfft(x.float(),     dim=-1, norm="ortho").abs()
    return F.l1_loss(torch.log1p(recon_mag), torch.log1p(x_mag))


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
        if getattr(encoder, 'use_cls_token', False) and padding_mask is None:
            # CLS-token encoder: forward_dino prepends a CLS token that attends over
            # all patches (incl. masked, replaced in-place with the mask token) and
            # returns it separately, mirroring the alt-attn encoder. The CLS is not
            # reconstructed — only the patch tokens go to the decoder.
            _cls_out, all_tokens = encoder.forward_dino(x, patch_mask)  # [B, D], [B, N, D]
        else:
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

    # ── Full-signal plain MSE (loss_type: full_mse), optionally + spectral ─────
    loss_type = str(obj_cfg.get("loss_type", "freq_weighted"))
    if loss_type == "full_mse":
        mse = F.mse_loss(reconstructed, x)
        loss = mse
        logs: Dict[str, float] = {
            "mse_loss":        float(mse.detach()),
            "mask_ratio_act":  float(patch_mask.float().mean().item()),
        }
        # Optional explicit frequency-band-weighted reconstruction term (full signal):
        #   band_loss_weight * Σ_b (w_b/Σw)·MSE(recon_b, x_b)
        # band_loss_weight == 0 (default) keeps plain full_mse unchanged. With uniform
        # band_weights the bands partition the spectrum, so this ≈ a scaled full MSE;
        # set non-uniform band_weights to actually emphasize a band.
        band_loss_weight = float(freq_cfg.get("band_loss_weight", 0.0))
        if band_loss_weight > 0.0:
            ws_b = sum(weights)
            band_term = torch.zeros((), device=x.device, dtype=x.dtype)
            for (lo, hi), w in zip(bands, weights):
                x_b = _bandpass(x,             lo, hi, fs)
                r_b = _bandpass(reconstructed, lo, hi, fs)
                bl  = F.mse_loss(r_b, x_b)
                band_term = band_term + (w / ws_b) * bl
                logs[f"loss_band_{lo:.0f}_{hi:.0f}hz"] = float(bl.detach())
            loss = loss + band_loss_weight * band_term
            logs["mae_freq_loss"] = float(band_term.detach())
        # Optional spectral term: L1 on the full-signal FFT magnitude, added to
        # the full-signal MSE. spectral_weight == 0 (default) leaves plain
        # full_mse unchanged, so existing full_mse configs are unaffected.
        spectral_weight = float(freq_cfg.get("spectral_weight", 0.0))
        if spectral_weight > 0.0:
            spec_loss = _spectral_loss(reconstructed, x)
            loss = loss + spectral_weight * spec_loss
            logs["spectral_loss"] = float(spec_loss.detach())
        logs["loss"] = float(loss.detach())
        # During validation: also log per-band and mae_freq_loss (on masked patches)
        # so plain full_mse runs stay comparable in WandB. Skip when band_loss_weight>0,
        # since the band term above already logs full-signal per-band values.
        if not encoder.training and band_loss_weight == 0.0:
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

    # Spectral loss: length-invariant log-magnitude L1 (full reconstructed vs original)
    spectral_weight = float(freq_cfg.get("spectral_weight", 0.0))
    if spectral_weight > 0.0:
        spec_loss  = _spectral_loss(reconstructed, x)
        total_loss = total_loss + spectral_weight * spec_loss
        logs["spectral_loss"] = float(spec_loss.detach())

    logs["loss"]           = float(total_loss.detach())
    logs["mask_ratio_act"] = float(patch_mask.float().mean().item())

    return total_loss, logs
