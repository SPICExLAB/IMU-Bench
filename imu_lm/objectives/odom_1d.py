"""Odometry regression objective for ViT-1D.

Target: per-patch root (pelvis) XYZ displacement in the global/world frame.
Loss:   mean squared error on [B, P_t, 3] predictions.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn.functional as F


def forward_loss(
    batch: Any,
    encoder: "ViT1DEncoder",  # noqa: F821
    odom_decoder: "ViT1DOdomDecoder",  # noqa: F821
    cfg: Any,
) -> Tuple[torch.Tensor, dict]:
    """Compute MSE loss for per-patch root displacement regression.

    Args:
        batch:        (x, y) where
                        x : [B, 6, T]          global-frame wrist IMU
                        y : [B, N_patches, 3]  root displacement labels (metres)
        encoder:      ViT1DEncoder
        odom_decoder: ViT1DOdomDecoder
        cfg:          experiment config dict

    Returns:
        (loss_tensor, metrics_dict)
    """
    x, y = batch
    x = x.float()
    y = y.float()

    # Encoder → full token grid [B, C*Pt, D]
    tokens, _num_patches_per_channel = encoder.forward_tokens(x)

    # Decoder → [B, Pt, 3]
    pred = odom_decoder(tokens)

    loss = F.mse_loss(pred, y)

    with torch.no_grad():
        mae = (pred.detach() - y).abs().mean()

    metrics = {
        "loss":      float(loss.detach()),
        "odom_mse":  float(loss.detach()),
        "odom_mae":  float(mae),
    }
    return loss, metrics
