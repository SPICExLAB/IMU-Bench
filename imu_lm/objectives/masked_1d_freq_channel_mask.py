"""MAE objective for mixed-modality combined datasets (e.g. Capture24, which
is accelerometer-only: its 3 "gyro" channels are zero-filled placeholders,
not real data — see ``imu_lm.data.combined_mae_multidata_loader``).

Identical to ``imu_lm.objectives.masked_1d_freq`` except the batch carries a
third element, ``channel_valid_mask [B, C]`` (bool, True = real channel,
all-True for every source except capture24). It is used two ways:

1. Blocked channels (``~channel_valid_mask``) are unioned into the random SSL
   patch mask, so they are *always* hidden from the encoder — the model never
   sees a fake zero as if it were a real accelerometer/gyroscope reading.
2. The reconstruction loss is computed only where a position is BOTH masked
   AND backed by a real channel — "masked acc" for capture24, "every masked
   channel" for the other, full 6-channel sources (whose channel_valid_mask
   is all-True, so this reduces to the ordinary masked-patch loss for them).

This file is only used by the ``mae_multidata_varlen_altattn_flu`` objective;
``masked_1d_freq.py`` (used by many other objective types) is untouched.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.objectives.masked_1d import generate_patch_mask, _patch_mask_to_signal_mask
from imu_lm.objectives.masked_1d_freq import (
    BANDS_DEFAULT, WEIGHTS_DEFAULT, _bandpass, _spectral_loss,
)
from imu_lm.utils.helpers import cfg_get


def forward_loss(
    batch: Tuple[torch.Tensor, Any, torch.Tensor],
    encoder: nn.Module,
    decoder: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MAE forward pass with a per-sample channel-validity mask.

    Args:
        batch    : (x [B, C, T], _, channel_valid_mask [B, C] bool or None)
        encoder  : ViT1DEncoder (or ViT1DEncoderAltAttn)
        decoder  : ViT1DDecoder
        cfg      : config dict

    Returns:
        loss : masked-and-valid reconstruction loss (full_mse or freq-weighted,
               per objective.loss_type)
        logs : dict with loss breakdown, mask_ratio_act, blocked_channel_frac
    """
    # batch is (x, _, channel_valid_mask[, temporal_valid_mask]). The 4th element appears only
    # when a source contributes SHORT clips zero-padded to the window length (mocap sources under
    # the 1.28-10.24 s variable-length regime); padded samples must never be reconstructed.
    temporal_valid_mask = batch[3] if len(batch) > 3 else None
    x, _, channel_valid_mask = batch[0], batch[1], batch[2]
    B, C, T = x.shape

    if channel_valid_mask is None:
        channel_valid_mask = torch.ones(B, C, dtype=torch.bool, device=x.device)
    channel_valid_mask = channel_valid_mask.to(device=x.device, dtype=torch.bool)

    obj_cfg    = cfg_get(cfg, ["objective"],             {}) or {}
    freq_cfg   = cfg_get(cfg, ["objective", "freq_mae"], {}) or {}
    mask_ratio = float(obj_cfg.get("mask_ratio",  0.75))
    strategy   = str(obj_cfg.get("mask_strategy", "composite"))
    fs         = float(freq_cfg.get("fs",          50.0))

    raw_bands   = freq_cfg.get("bands",        BANDS_DEFAULT)
    raw_weights = freq_cfg.get("band_weights", WEIGHTS_DEFAULT)
    bands   = [(float(b[0]), float(b[1])) for b in raw_bands]
    weights = [float(w) for w in raw_weights]
    assert len(bands) == len(weights), "bands and band_weights must have same length"

    patch_size              = encoder.patch_size
    num_patches_per_channel = T // patch_size

    # Random SSL mask, then force blocked channels (e.g. capture24 gyro) to
    # always be masked/hidden from the encoder — they carry no real signal,
    # regardless of the random mask_ratio draw.
    logs: Dict[str, float] = {}
    patch_mask = generate_patch_mask(
        B, num_patches_per_channel, C, mask_ratio, strategy, x.device,
    )
    # objective.modality_mask_prob: with this probability a sample masks ALL of one modality
    # (acc <-> gyro) instead of drawing an independent patch mask, forcing genuine cross-modal
    # prediction rather than within-timestep interpolation between correlated channels.
    mmp = float(obj_cfg.get("modality_mask_prob", 0.0))
    if mmp > 0.0 and C > int(obj_cfg.get("n_acc", 3)):
        na = int(obj_cfg.get("n_acc", 3))
        draw = torch.rand(B, device=x.device)
        hide_gyro = draw < mmp / 2.0
        hide_acc = (draw >= mmp / 2.0) & (draw < mmp)
        mod = torch.zeros(B, C, dtype=torch.bool, device=x.device)
        mod[hide_gyro, na:] = True
        mod[hide_acc, :na] = True
        patch_mask = patch_mask | mod.repeat_interleave(num_patches_per_channel, dim=1)
        logs["modality_masked_frac"] = float((hide_gyro | hide_acc).float().mean().item())

    blocked_channels = ~channel_valid_mask                                          # [B, C]
    blocked_patches  = blocked_channels.repeat_interleave(num_patches_per_channel, dim=1)  # [B, C*num_patches]
    patch_mask = patch_mask | blocked_patches

    pad_patches = None
    if temporal_valid_mask is not None:
        # a patch is padding if ANY of its samples is padding -> hide it from the encoder and drop
        # it from the loss, exactly as for the fake gyro channels of acc-only sources
        tv = temporal_valid_mask.to(device=x.device, dtype=torch.bool)[:, :T]
        pv = tv.view(B, num_patches_per_channel, patch_size).all(dim=-1)              # [B, Pt]
        pad_patches = (~pv).repeat(1, C)                                              # [B, C*Pt]
        patch_mask = patch_mask | pad_patches
        logs_pad = float(pad_patches.float().mean().item())
    else:
        logs_pad = 0.0

    if getattr(encoder, "_use_bert_mask", False):
        if getattr(encoder, "use_cls_token", False):
            _cls_out, all_tokens = encoder.forward_dino(x, patch_mask)
        else:
            all_tokens = encoder(x, mask=patch_mask, padding_mask=None, return_keep_padding_mask=False)
        empty_mask = torch.zeros_like(patch_mask)
        reconstructed = decoder(all_tokens, empty_mask, num_patches_per_channel, C, padding_mask=None)
    else:
        visible_tokens, visible_padding_mask = encoder(
            x, mask=patch_mask, padding_mask=None, return_keep_padding_mask=True,
        )
        reconstructed = decoder(
            visible_tokens, patch_mask, num_patches_per_channel, C,
            padding_mask=None, visible_padding_mask=visible_padding_mask,
        )

    T_recon = reconstructed.shape[-1]
    if T_recon < x.shape[-1]:
        x = x[:, :, :T_recon]

    # Loss mask: positions that are masked AND backed by a real channel.
    signal_mask  = _patch_mask_to_signal_mask(patch_mask, num_patches_per_channel, C, patch_size)
    signal_valid = channel_valid_mask.unsqueeze(-1).expand(-1, -1, signal_mask.shape[-1]).clone()
    if temporal_valid_mask is not None:
        tvs = temporal_valid_mask.to(device=x.device, dtype=torch.bool)[:, :signal_valid.shape[-1]]
        signal_valid &= tvs.unsqueeze(1)                     # padded timesteps never enter the loss
    if T_recon < signal_valid.shape[-1]:
        signal_valid = signal_valid[:, :, :T_recon]
    loss_mask = signal_mask & signal_valid

    # ---- patch-normalised reconstruction targets (norm-pix, MAE paper §3) -------------
    # objective.norm_pix_loss: true -> standardise every (channel, patch) of BOTH target and
    # prediction by the TARGET patch's own mean/std before the MSE. Without it, raw/DC-preserving
    # input makes the loss overwhelmingly about predicting the ~9.8 m/s^2 gravity offset, and the
    # encoder collapses onto it (measured 2026-08-25: effective rank 1.4/384, 76 % of frozen-feature
    # variance explained by the window gravity vector). Statistics come from the target only, so no
    # information leaks from the prediction; padded/invalid samples are excluded via loss_mask.
    if bool(obj_cfg.get("norm_pix_loss", False)):
        Bx, Cx, Tx = x.shape
        npc = Tx // patch_size
        def _to_patches(v):
            return v[:, :, :npc * patch_size].reshape(Bx, Cx, npc, patch_size)
        xp, rp = _to_patches(x), _to_patches(reconstructed)
        mu = xp.mean(dim=-1, keepdim=True)
        # norm_pix_mode (2026-08-25):
        #   "mean"    (default) subtract the patch mean only -- kills the trivial "predict gravity"
        #             solution that collapsed the raw-input encoders (effective rank 1.4/384, 76 % of
        #             frozen-feature variance = window gravity) while PRESERVING amplitude, which the
        #             from-scratch sweep showed is real signal for HAR (z-score was worst on 10/12 tasks).
        #   "meanstd" also divide by sqrt(var + norm_pix_var_floor) -- the image-MAE recipe. On wrist
        #             IMU it misbehaves: patch std spans 3 orders of magnitude (EatingSpeed acc median
        #             0.0115 vs OCDetect 0.0594; p1 ~0.0009, some exactly 0), so quiet patches are
        #             amplified 20-1000x, dominate the gradient and the loss rises (measured 51->118).
        #   Signal-to-trivial ratio on real data: mean 1855x, meanstd(1e-2) 13.5x, meanstd(2.5e-3) 5.5x.
        if str(obj_cfg.get("norm_pix_mode", "mean")).lower() == "meanstd":
            var_floor = float(obj_cfg.get("norm_pix_var_floor", 1e-2))
            sd = (xp.var(dim=-1, keepdim=True, unbiased=False) + var_floor).sqrt()
        else:
            sd = torch.ones_like(mu)
        x = ((xp - mu) / sd).reshape(Bx, Cx, npc * patch_size)
        reconstructed = ((rp - mu) / sd).reshape(Bx, Cx, npc * patch_size)
        logs["norm_pix_target_std_mean"] = float(((xp - mu) / sd).std().detach())

    loss_type = str(obj_cfg.get("loss_type", "freq_weighted"))

    if loss_type == "full_mse":
        loss = F.mse_loss(reconstructed[loss_mask], x[loss_mask])
        logs["mse_loss"] = float(loss.detach())
    else:
        weight_sum = sum(weights)
        loss = torch.zeros((), device=x.device, dtype=x.dtype)
        for (lo, hi), w in zip(bands, weights):
            x_band     = _bandpass(x,             lo, hi, fs)
            recon_band = _bandpass(reconstructed, lo, hi, fs)
            band_loss  = F.mse_loss(recon_band[loss_mask], x_band[loss_mask])
            loss = loss + (w / weight_sum) * band_loss
            logs[f"loss_band_{lo:.0f}_{hi:.0f}hz"] = float(band_loss.detach())
        logs["mae_freq_loss"] = float(loss.detach())

    spectral_weight = float(freq_cfg.get("spectral_weight", 0.0))
    if spectral_weight > 0.0:
        spec_loss = _spectral_loss(reconstructed, x)
        loss = loss + spectral_weight * spec_loss
        logs["spectral_loss"] = float(spec_loss.detach())

    # ---- per-modality weighting + reporting (2026-08-18) ----------------------
    # Acc-only sources (NHANES, capture24) can dominate the mixture, so most gradient
    # would flow through the accelerometer pathway. `objective.modality_loss_weights:
    # {acc: 1.0, gyro: 2.0}` re-weights the two halves of the reconstruction loss, and
    # the acc/gyro losses are always logged separately so the balance is visible.
    n_acc = int(obj_cfg.get("n_acc", 3))
    mw = obj_cfg.get("modality_loss_weights", None)
    with torch.no_grad():
        for name, sl in (("acc", slice(0, min(n_acc, C))), ("gyro", slice(min(n_acc, C), C))):
            m = loss_mask[:, sl]
            logs[f"{name}_mse"] = float(F.mse_loss(reconstructed[:, sl][m], x[:, sl][m]).detach()) if m.any() else float("nan")
            logs[f"{name}_frac"] = float(m.float().mean().item())
    if mw and C > n_acc:
        wa, wg = float(mw.get("acc", 1.0)), float(mw.get("gyro", 1.0))
        ma, mg = loss_mask[:, :n_acc], loss_mask[:, n_acc:]
        la = F.mse_loss(reconstructed[:, :n_acc][ma], x[:, :n_acc][ma]) if ma.any() else loss.new_zeros(())
        lg = F.mse_loss(reconstructed[:, n_acc:][mg], x[:, n_acc:][mg]) if mg.any() else loss.new_zeros(())
        # keep the band-weighted term as the backbone and add the modality-balanced MSE on top
        extra = (wa * la + wg * lg) / max(wa + wg, 1e-8)
        lam = float(obj_cfg.get("modality_loss_lambda", 1.0))
        if str(obj_cfg.get("modality_loss_mode", "add")).lower() == "replace":
            loss = extra                      # the modality-balanced MSE IS the objective
        else:
            loss = loss + lam * extra         # legacy: an extra term on top (doubles the MSE)
        logs["modality_extra"] = float(extra.detach())

    logs["loss"]                 = float(loss.detach())
    logs["mask_ratio_act"]       = float(patch_mask.float().mean().item())
    logs["blocked_channel_frac"] = float(blocked_channels.float().mean().item())
    logs["padded_patch_frac"] = logs_pad
    return loss, logs
