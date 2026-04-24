"""1D Masked Autoencoder objective (MAE / LSM-2 style).

Patch-level masking with dropout removal:
- Encoder sees ONLY visible (unmasked) patches — no zero-filling
- Decoder inserts learnable mask tokens and reconstructs per-patch
- MSE loss computed only on masked patches

Masking strategies (randomly selected per sample when strategy=composite):
- random:   mask random patches uniformly
- temporal: mask all channels at random time positions
- signal:   mask all time positions for random channels

Token layout: [C0_P0, C0_P1, ..., C0_Pt, C1_P0, ..., C1_Pt, ..., Cc_Pt]
  where t = num_patches_per_channel - 1, c = num_channels - 1
"""

from __future__ import annotations

import random as pyrandom
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from imu_lm.utils.helpers import cfg_get


def generate_patch_mask(
    B: int,
    num_patches_per_channel: int,
    num_channels: int,
    mask_ratio: float,
    strategy: str,
    device: torch.device,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a patch-level boolean mask [B, N] where True = masked.
    
    All samples in the batch have the same number of masked tokens (num_mask)
    so visible token counts are uniform — required for batched dropout removal.
    
    Token ordering: channel-major [C0_P0, C0_P1, ..., C1_P0, ...].
    
    Args:
        B: batch size
        num_patches_per_channel: time patches per channel
        num_channels: number of input channels (C)
        mask_ratio: fraction of tokens to mask (0-1)
        strategy: "random" | "temporal" | "signal" | "composite"
        device: torch device
        
    Returns:
        [B, N] boolean mask, True = masked, with exactly num_mask True per row
    """
    N = num_channels * num_patches_per_channel
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    
    for b in range(B):
        valid_mask = torch.ones(N, dtype=torch.bool, device=device)
        if padding_mask is not None:
            valid_mask = ~padding_mask[b]
        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        valid_N = int(valid_idx.numel())
        if valid_N <= 1:
            continue
        num_mask = int(mask_ratio * valid_N)
        num_mask = max(1, min(num_mask, valid_N - 1))

        # Pick strategy for this sample
        if strategy == "composite":
            strat = pyrandom.choice(["random", "temporal", "signal"])
        else:
            strat = strategy
        
        if strat == "random":
            perm = valid_idx[torch.randperm(valid_N, device=device)]
            mask[b, perm[:num_mask]] = True
            
        elif strat == "temporal":
            valid_grid = valid_mask.view(num_channels, num_patches_per_channel)
            valid_times = torch.nonzero(valid_grid.all(dim=0), as_tuple=False).squeeze(-1)
            num_time_to_mask = min(int(max(1, num_mask // num_channels)), int(valid_times.numel()))
            if num_time_to_mask > 0:
                time_perm = valid_times[torch.randperm(len(valid_times), device=device)]
                times_to_mask = time_perm[:num_time_to_mask]
                for c in range(num_channels):
                    mask[b, c * num_patches_per_channel + times_to_mask] = True
            deficit = num_mask - int(mask[b].sum().item())
            if deficit > 0:
                unmasked = torch.nonzero(valid_mask & ~mask[b], as_tuple=False).squeeze(-1)
                extras = unmasked[torch.randperm(len(unmasked), device=device)[:deficit]]
                mask[b, extras] = True
                
        elif strat == "signal":
            valid_grid = valid_mask.view(num_channels, num_patches_per_channel)
            valid_channels = torch.nonzero(valid_grid.any(dim=1), as_tuple=False).squeeze(-1)
            num_ch_to_mask = min(max(1, num_mask // num_patches_per_channel), int(valid_channels.numel()))
            if num_ch_to_mask > 0:
                ch_perm = valid_channels[torch.randperm(len(valid_channels), device=device)]
                for c in ch_perm[:num_ch_to_mask]:
                    start = int(c.item()) * num_patches_per_channel
                    channel_valid = valid_grid[int(c.item())]
                    mask[b, start:start + num_patches_per_channel] = channel_valid
            deficit = num_mask - int(mask[b].sum().item())
            if deficit > 0:
                unmasked = torch.nonzero(valid_mask & ~mask[b], as_tuple=False).squeeze(-1)
                extras = unmasked[torch.randperm(len(unmasked), device=device)[:deficit]]
                mask[b, extras] = True
            elif deficit < 0:
                surplus = -deficit
                masked_idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
                to_unmask = masked_idx[torch.randperm(len(masked_idx), device=device)[:surplus]]
                mask[b, to_unmask] = False
        else:
            raise ValueError(f"Unknown mask strategy: {strat}")
    
    return mask


def _patch_mask_to_signal_mask(
    patch_mask: torch.Tensor,
    num_patches_per_channel: int,
    num_channels: int,
    patch_size: int,
) -> torch.Tensor:
    """Convert patch-level mask [B, N] to signal-level mask [B, C, T] for loss computation.
    
    Token ordering: channel-major [C0_P0, C0_P1, ..., C1_P0, ...].
    """
    B = patch_mask.shape[0]
    # Reshape to [B, C, num_patches_per_channel]
    patch_mask_2d = patch_mask.view(B, num_channels, num_patches_per_channel)
    # Expand each patch to patch_size timesteps: [B, C, num_patches_per_channel, patch_size]
    signal_mask = patch_mask_2d.unsqueeze(-1).expand(-1, -1, -1, patch_size)
    # Flatten to [B, C, T]
    signal_mask = signal_mask.reshape(B, num_channels, num_patches_per_channel * patch_size)
    return signal_mask


def forward_loss(
    batch: Tuple[torch.Tensor, torch.Tensor],
    encoder: nn.Module,
    decoder: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """MAE forward: patch masking → encoder (visible only) → decoder → MSE on masked patches.
    
    Args:
        batch: (x, y) where x is [B, C, T] raw 1D signal
        encoder: ViT1DEncoder with forward(x, mask=None) or forward(x, mask=[B,N])
        decoder: ViT1DDecoder with forward(visible_tokens, mask, npc, nc)
        cfg: Config dict with objective.* params
        
    Returns:
        loss: Scalar MSE loss on masked patch positions
        logs: Dict with loss, mask_ratio, strategy stats
    """
    padding_mask = None
    if isinstance(batch, (tuple, list)) and len(batch) >= 3:
        x, _, padding_mask = batch[:3]
    else:
        x, _ = batch  # x: [B, C, T]
    B, C, T = x.shape
    
    obj_cfg = cfg_get(cfg, ["objective"], {}) or {}
    mask_ratio = float(obj_cfg.get("mask_ratio", 0.75))
    strategy = str(obj_cfg.get("mask_strategy", "composite"))
    
    patch_size = encoder.patch_size
    num_patches_per_channel = T // patch_size
    
    # Generate patch-level mask [B, N], True = masked (removed from encoder)
    patch_mask = generate_patch_mask(
        B,
        num_patches_per_channel,
        C,
        mask_ratio,
        strategy,
        x.device,
        padding_mask=padding_mask,
    )
    
    # Encoder: only visible patches go through transformer
    visible_tokens, visible_padding_mask = encoder(
        x,
        mask=patch_mask,
        padding_mask=padding_mask,
        return_keep_padding_mask=True,
    )
    
    # Decoder: insert mask tokens, reconstruct full signal
    reconstructed = decoder(
        visible_tokens,
        patch_mask,
        num_patches_per_channel,
        C,
        padding_mask=padding_mask,
        visible_padding_mask=visible_padding_mask,
    )
    
    # MSE loss on masked patches only
    signal_mask = _patch_mask_to_signal_mask(patch_mask, num_patches_per_channel, C, patch_size)
    # Trim signal_mask if reconstructed is shorter (shouldn't happen normally)
    T_recon = reconstructed.shape[-1]
    if T_recon < signal_mask.shape[-1]:
        signal_mask = signal_mask[:, :, :T_recon]
        x = x[:, :, :T_recon]
    
    loss = F.mse_loss(reconstructed[signal_mask], x[signal_mask])
    
    actual_ratio = float(patch_mask.float().mean().item())
    valid_ratio = 1.0 if padding_mask is None else float((~padding_mask).float().mean().item())
    
    return loss, {
        "loss": float(loss.detach().item()),
        "mask_ratio": actual_ratio,
        "valid_ratio": valid_ratio,
        "mask_strategy": strategy,
    }
