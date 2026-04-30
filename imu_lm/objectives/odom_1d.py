"""Odometry regression objective for ViT-1D.

Target: per-patch root displacement labels.
  - out_dim=3 : global-frame XYZ displacement  [B, P_t, 3]
  - out_dim=4 : local-frame displacement + delta yaw  [B, P_t, 4]
               [..., 0:3] = body-frame [dx, dy, dz]  (metres)
               [...,   3] = delta yaw Δψ              (radians, Z-up world)
Loss:   mean squared error over all output dimensions.
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
    """Compute MSE loss for per-patch displacement regression.

    Args:
        batch:        (x, y) where
                        x : [B, 6, T]          wrist FLU IMU
                        y : [B, N_patches, 3]  global XYZ displacement  OR
                            [B, N_patches, 4]  [dx,dy,dz,dψ] local disp + delta yaw
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

    # Decoder → [B, Pt, out_dim]
    pred = odom_decoder(tokens)

    loss = F.mse_loss(pred, y)

    with torch.no_grad():
        mae = (pred.detach() - y).abs().mean()

    metrics = {
        "loss":      float(loss.detach()),
        "odom_mse":  float(loss.detach()),
        "odom_mae":  float(mae),
    }

    # Split metrics when out_dim=4 (local displacement + delta yaw)
    if y.shape[-1] == 4:
        with torch.no_grad():
            disp_mse = F.mse_loss(pred[..., :3], y[..., :3])
            yaw_mse  = F.mse_loss(pred[..., 3:], y[..., 3:])
            metrics["disp_mse"] = float(disp_mse.detach())
            metrics["yaw_mse"]  = float(yaw_mse.detach())

    return loss, metrics
