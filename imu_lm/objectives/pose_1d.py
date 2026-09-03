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


def _assemble_full_R(pred_rot, y_full_rot, B, Pt, K, J, J_loader):
    """Build the full J_loader-joint rotation tensor FK needs.

    Two conventions coexist in this codebase:
      - root EXCLUDED from predictions (Nymeria: num_joints=23, J_loader=24)
        -> splice the GT root (joint 0) onto the front of the predicted joints.
      - root INCLUDED in predictions (TotalCapture: num_joints=24, J_loader=24)
        -> pred_rot already covers every joint including root; nothing to splice.
    Distinguished by J == J_loader (nothing missing) vs. J == J_loader - 1
    (root missing, the only other case this objective is used with).
    """
    pred_R = _rot6d_to_matrix(pred_rot.reshape(B, Pt, K, J, 6))
    if J == J_loader:
        return pred_R
    root_6d = y_full_rot.reshape(B, Pt, K, J_loader, 6)[:, :, :, 0:1, :]
    root_R  = _rot6d_to_matrix(root_6d)
    return torch.cat([root_R, pred_R], dim=3)


def _compute_val_mpjpe(
    pred_rot: torch.Tensor,
    y_full_rot: torch.Tensor,
    gt_pos: torch.Tensor,
    sub_patches: int,
    num_joints: int,
    body_model,
) -> float:
    """MPJPE in mm over the NON-ROOT joints (1..J_loader-1) using FK (GT root
    spliced in when predictions exclude it -- see _assemble_full_R).

    Bug fixed 2026-07-28: this used to slice ``j_start:j_start+J`` with
    ``j_start = 0 if J == J_loader else 1``, so ``num_joints: 24`` configs
    (e.g. the TotalCapture 24-joint probe/finetune family) averaged over
    joints 0..23 -- *including joint 0*. Both tensors here are root-relative
    (``pos_pred_rel`` subtracts ``pos_pred[:, 0:1]`` two lines below; the GT
    is root-relative on arrival, see core/articulate/model.py:89 "Root joint
    is aligned at zero"), so joint 0's term is identically zero and simply
    diluted the mean: reported MPJPE for 24-joint configs came out exactly
    23/24 (~4.2%) LOWER than the same error measured the 23-joint way, i.e.
    not comparable to the Nymeria 23-joint numbers it sat next to in tables.
    The metric now always excludes the root, for every J.

    pred_rot   : [B, Pt, K * num_joints * 6]  predicted 6D rotations
    y_full_rot : [B, Pt, K * 24 * 6]          GT full-body 6D rotations (joint 0 = root)
    gt_pos     : [B, Pt, K * 24 * 3]          GT FK joint positions, root-relative
    """
    B, Pt, _ = pred_rot.shape
    K = sub_patches
    J = num_joints
    J_loader = y_full_rot.shape[-1] // (K * 6)  # typically 24

    full_R = _assemble_full_R(pred_rot, y_full_rot, B, Pt, K, J, J_loader)

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

    # Always skip joint 0 (the root): it is exactly zero in both tensors under
    # the root-relative convention, so including it would only dilute the mean
    # and make 24-joint configs incomparable to 23-joint ones. Both tensors are
    # in the same J_loader-joint FK space here, so this slice is aligned for
    # J == J_loader (model predicts the root too) and J == J_loader - 1
    # (GT root spliced in by _assemble_full_R) alike.
    diff = pos_pred_rel[:, :, :, 1:J_loader] - gt_pos_4d[:, :, :, 1:J_loader]
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

    NOTE (2026-07-28): unlike ``_compute_val_mpjpe``, this deliberately KEEPS
    the ``j_start`` slicing, i.e. ``num_joints: 24`` configs still include the
    always-zero root term. That term contributes exactly 0 to the mean and has
    zero gradient, so its only effect is scaling this loss by 23/24 -- which is
    indistinguishable from setting ``joint_pos_weight`` 4.3% lower, and those
    weights were tuned with this scaling in place. Changing it here would
    silently retune every 24-joint config's effective loss balance, so it is
    left as-is on purpose; the reported *metric* is what needed to be
    comparable, and that is fixed above.
    """
    B, Pt, _ = pred_rot.shape
    K = sub_patches
    J = num_joints
    J_loader = y_full_rot.shape[-1] // (K * 6)
    j_start  = 0 if J == J_loader else 1

    full_R = _assemble_full_R(pred_rot, y_full_rot, B, Pt, K, J, J_loader)

    bm_device = body_model._J_regressor.device
    full_R_flat = full_R.reshape(B * Pt * K, -1).to(bm_device)

    _, pos_pred = body_model.forward_kinematics(full_R_flat)   # [N, J_loader, 3]

    pos_pred = pos_pred.to(pred_rot.device)
    pos_pred_rel = pos_pred - pos_pred[:, 0:1]
    pos_pred_rel = pos_pred_rel.reshape(B, Pt, K, J_loader, 3)

    gt_pos_4d = gt_pos.reshape(B, Pt, K, J_loader, 3).to(pred_rot.device)

    diff = pos_pred_rel[:, :, :, j_start:j_start + J] - gt_pos_4d[:, :, :, j_start:j_start + J]
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
