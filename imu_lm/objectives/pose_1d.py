"""Pose regression objective for ViT1D with ViT1DPoseDecoder.

Predicts per-patch 6D joint rotations from raw IMU windows.

Contract
--------
    forward_loss(batch, encoder, pose_decoder, cfg) -> (loss, logs)

    batch : (x, y_rot) or longer (extra elements ignored)
        x     : [B, C, T]       raw IMU windows
        y_rot : [B, P_t, K*J*6] ground-truth 6D rotations per time patch

    loss : rotation MSE + optional jerk smoothness penalty.
    During validation (encoder.training == False), MPJPE (mm) is also logged
    when body_model, y_full_rot, and gt_pos are provided.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """[..., 6] → [..., 3, 3] via Gram-Schmidt."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _compute_val_mpjpe(
    pred_rot: torch.Tensor,
    y_full_rot: torch.Tensor,
    gt_pos: torch.Tensor,
    sub_patches: int,
    num_joints: int,
    body_model,
) -> float:
    """MPJPE in mm for joints 1..num_joints using FK with GT root spliced in.

    pred_rot   : [B, Pt, K * num_joints * 6]  predicted 6D rotations (no root)
    y_full_rot : [B, Pt, K * 24 * 6]          GT full-body 6D rotations (joint 0 = root)
    gt_pos     : [B, Pt, K * 24 * 3]          GT FK joint positions, root-relative
    """
    B, Pt, _ = pred_rot.shape
    K = sub_patches
    J = num_joints
    J_loader = y_full_rot.shape[-1] // (K * 6)  # typically 24

    # GT root rotation matrix  [B, Pt, K, 1, 3, 3]
    root_6d = y_full_rot.reshape(B, Pt, K, J_loader, 6)[:, :, :, 0:1, :]
    root_R  = _rot6d_to_matrix(root_6d)

    # Predicted rotation matrices  [B, Pt, K, J, 3, 3]
    pred_R = _rot6d_to_matrix(pred_rot.reshape(B, Pt, K, J, 6))

    # Full 24-joint rotation: [B, Pt, K, 24, 3, 3]
    full_R = torch.cat([root_R, pred_R], dim=3)

    # FK on body model's device
    bm_device = body_model._J_regressor.device
    full_R_flat = full_R.reshape(B * Pt * K, -1).to(bm_device)
    with torch.no_grad():
        _, pos_pred = body_model.forward_kinematics(full_R_flat)  # [N, 24, 3]

    pos_pred = pos_pred.to(pred_rot.device)
    pos_pred_rel = pos_pred - pos_pred[:, 0:1]        # root-relative
    pos_pred_rel = pos_pred_rel.reshape(B, Pt, K, J_loader, 3)

    # GT positions are already root-relative: [B, Pt, K, 24, 3]
    gt_pos_4d = gt_pos.reshape(B, Pt, K, J_loader, 3)

    diff = pos_pred_rel[:, :, :, 1:J + 1] - gt_pos_4d[:, :, :, 1:J + 1]
    return diff.norm(dim=-1).mean().item() * 1000.0   # mm


def _compute_fk_pos_loss(
    pred_rot: torch.Tensor,
    y_full_rot: torch.Tensor,
    gt_pos: torch.Tensor,
    sub_patches: int,
    num_joints: int,
    body_model,
) -> torch.Tensor:
    """MSE joint position loss via differentiable FK (gradients flow through pred_rot).

    Same layout as _compute_val_mpjpe but no torch.no_grad() and returns a tensor.
    """
    B, Pt, _ = pred_rot.shape
    K = sub_patches
    J = num_joints
    J_loader = y_full_rot.shape[-1] // (K * 6)

    root_6d = y_full_rot.reshape(B, Pt, K, J_loader, 6)[:, :, :, 0:1, :]
    root_R  = _rot6d_to_matrix(root_6d)

    pred_R = _rot6d_to_matrix(pred_rot.reshape(B, Pt, K, J, 6))
    full_R = torch.cat([root_R, pred_R], dim=3)

    bm_device = body_model._J_regressor.device
    full_R_flat = full_R.reshape(B * Pt * K, -1).to(bm_device)

    _, pos_pred = body_model.forward_kinematics(full_R_flat)   # [N, J_loader, 3]

    pos_pred = pos_pred.to(pred_rot.device)
    pos_pred_rel = pos_pred - pos_pred[:, 0:1]
    pos_pred_rel = pos_pred_rel.reshape(B, Pt, K, J_loader, 3)

    gt_pos_4d = gt_pos.reshape(B, Pt, K, J_loader, 3).to(pred_rot.device)

    diff = pos_pred_rel[:, :, :, 1:J + 1] - gt_pos_4d[:, :, :, 1:J + 1]
    return (diff ** 2).mean()


def forward_loss(
    batch: Tuple,
    encoder: nn.Module,
    pose_decoder: nn.Module,
    cfg: Any,
    jerk_weight: float = 0.0,
    joint_pos_weight: float = 0.0,
    body_model=None,
    y_full_rot: Optional[torch.Tensor] = None,
    gt_pos: Optional[torch.Tensor] = None,
    **_kwargs,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute per-patch 6D rotation regression loss.

    Args:
        batch:            (x, y_rot, ...)  — extra elements ignored
        encoder:          ViT1DEncoder (called via forward_tokens)
        pose_decoder:     ViT1DPoseDecoder
        cfg:              Config dict
        jerk_weight:      Weight for 3rd-order jerk smoothness penalty
        joint_pos_weight: Weight for FK joint position MSE loss (train only)

    Returns:
        loss : scalar total loss
        logs : dict with individual loss components
    """
    x, y_rot = batch[0].float(), batch[1].float()   # [B, C, T], [B, P_t, K*J*6]

    tokens, _ = encoder.forward_tokens(x)
    pred_rot  = pose_decoder(tokens)                 # [B, P_t, K*J*6]

    rot_mse = nn.functional.mse_loss(pred_rot, y_rot)

    jerk_loss = pred_rot.new_tensor(0.0)
    if jerk_weight > 0.0:
        K = pose_decoder.sub_patches
        B, Pt, dim_total = pred_rot.shape
        J_dim    = dim_total // K
        pred_seq = pred_rot.reshape(B, Pt * K, J_dim)
        diff1    = pred_seq[:, 1:] - pred_seq[:, :-1]
        diff2    = diff1[:, 1:] - diff1[:, :-1]
        diff3    = diff2[:, 1:] - diff2[:, :-1]
        jerk_loss = (diff3 ** 2).mean()

    # FK joint position loss — added to loss during train, logged in both splits
    fk_pos_loss = pred_rot.new_tensor(0.0)
    if (joint_pos_weight > 0.0
            and body_model is not None
            and y_full_rot is not None
            and gt_pos is not None):
        try:
            if encoder.training:
                fk_pos_loss = _compute_fk_pos_loss(
                    pred_rot, y_full_rot, gt_pos,
                    pose_decoder.sub_patches, pose_decoder.num_joints, body_model,
                )
            else:
                with torch.no_grad():
                    fk_pos_loss = _compute_fk_pos_loss(
                        pred_rot, y_full_rot, gt_pos,
                        pose_decoder.sub_patches, pose_decoder.num_joints, body_model,
                    )
        except Exception:
            pass

    loss = rot_mse + jerk_weight * jerk_loss + (joint_pos_weight * fk_pos_loss if encoder.training else pred_rot.new_tensor(0.0))

    logs: Dict[str, float] = {
        "loss":     float(loss.detach().item()),
        "pose_mse": float(rot_mse.detach().item()),
    }
    if jerk_weight > 0.0:
        logs["pose_jerk"] = float(jerk_loss.detach().item())
    if joint_pos_weight > 0.0:
        logs["pose_fk_pos"] = float(fk_pos_loss.detach().item())

    if (not encoder.training
            and body_model is not None
            and y_full_rot is not None
            and gt_pos is not None):
        try:
            logs["mpjpe_mm"] = _compute_val_mpjpe(
                pred_rot, y_full_rot, gt_pos,
                pose_decoder.sub_patches, pose_decoder.num_joints, body_model,
            )
        except Exception:
            pass

    return loss, logs
