"""Pose regression objective for ViT1D with ViT1DPoseDecoder.

Predicts per-patch 6D joint rotations from raw IMU windows.

Contract
--------
    forward_loss(batch, encoder, pose_decoder, cfg) -> (loss, logs)

    batch : (x, y_pose)
        x       : [B, C, T]          raw IMU windows
        y_pose  : [B, P_t, J*6]      ground-truth 6D rotations per time patch
                                     (from NymeriaWristDataset)

    loss : MSE between predicted and ground-truth 6D rotations.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


def forward_loss(
    batch: Tuple[torch.Tensor, torch.Tensor],
    encoder: nn.Module,
    pose_decoder: nn.Module,
    cfg: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute per-patch 6D rotation regression loss.

    Args:
        batch:        (x [B, C, T], y_pose [B, P_t, J*6])
        encoder:      ViT1DEncoder  (called via forward_tokens)
        pose_decoder: ViT1DPoseDecoder
        cfg:          Config dict

    Returns:
        loss : scalar MSE loss
        logs : dict with loss value
    """
    x, y_pose = batch
    x = x.float()
    y_pose = y_pose.float()  # [B, P_t, J*6]

    # Encode: full per-token output [B, C * P_t, D]
    tokens, _ = encoder.forward_tokens(x)

    # Decode: [B, P_t, J*6]
    pred = pose_decoder(tokens)

    # MSE over all patches, joints, and rotation dims
    loss = torch.nn.functional.mse_loss(pred, y_pose)

    logs: Dict[str, float] = {
        "loss": float(loss.detach().item()),
        "pose_mse": float(loss.detach().item()),
    }
    return loss, logs
