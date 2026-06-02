"""Dataset-level pose evaluation for the ViT1D right-arm model.

Runs inference over every pkl in a test directory, aggregates per-joint
rotation error and MPJPE (Mean Per-Joint Position Error) across the whole
test set, and saves summary PNGs.

Metrics
-------
  Geodesic error (°) — angular distance between predicted and GT rotation matrices
  MPJPE (mm)         — L2 distance between predicted and GT 3-D joint positions
                       derived via SMPL forward kinematics (requires core.articulate)
  MPJVE (cm)         — Mean Per Joint Vertex Error: mean Euclidean distance across all
                       SMPL mesh vertices between predicted and GT poses, with the root
                       joint (pelvis) aligned. Reported in centimetres.

Output PNGs (--out-dir)
-----------------------
  per_joint_geodesic.png   — mean ± std geodesic error per right-arm joint
  per_joint_mpjpe.png      — mean ± std MPJPE  per right-arm joint
  geodesic_cdf.png         — CDF of per-frame geodesic error for each joint
  error_vs_sequence.png    — mean geodesic error per test sequence (sorted)

Usage
-----
    python scripts/run_pose_eval_dataset.py \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_pose.yaml \\
        --run          vit1d_nymeria_pose \\
        --test-dir     /mnt/nas2/processed_data_test_body_Ty \\
        [--ckpt        best]          # "best" | "latest" | abs path
        [--max-frames  0]             # per-sequence cap (0 = full)
        [--fps         50]
        [--out-dir     data_vis/pose_eval_dataset]
        [--num-joints  5]             # 5 (right-arm chain) or 24 (full body)
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRot
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.imuposerdataset import (
    _resample_linear,
    _resample_nearest,
    _resample_rotmat_nlerp,
    _resample_rotmat_nearest,
)
from imu_lm.data.nymeria_wrist_loader import (
    RIGHT_WRIST_CHAIN_JOINTS,
    _XSENS_RIGHT_WRIST_JOINT,
    _xsens_to_smpl_rotmat,
)
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DPoseDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

# Right-arm chain joint names (5-joint mode)
JOINT_NAMES_5  = ["R-Collar", "R-Shoulder", "R-Elbow", "R-Wrist", "R-Hand"]
SMPL_JOINT_NAMES = [
    "Pelvis", "L-Hip", "R-Hip", "Spine1", "L-Knee", "R-Knee",
    "Spine2", "L-Ankle", "R-Ankle", "Spine3", "L-Foot", "R-Foot",
    "Neck", "L-Collar", "R-Collar", "Head", "L-Shoulder", "R-Shoulder",
    "L-Elbow", "R-Elbow", "L-Wrist", "R-Wrist", "L-Hand", "R-Hand",
]


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def geodesic_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    R_diff = torch.matmul(R_pred, R_gt.transpose(-1, -2))
    trace  = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos    = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


# ---------------------------------------------------------------------------
# Per-sequence data loading
# ---------------------------------------------------------------------------

def load_sequence(pkl_path: str, fps: int, max_frames: int, num_joints: int,
                  add_gravity: bool = False, loader_type: str = ""):
    """Load IMU, GT 6D rotations, and full SMPL local rotation matrices.

    Supports both Nymeria pkls (``gt_data["xsens_pose"]``) and IMUPoser pkls
    (``gt_data["pose"]`` + ``real_imu_data``).

    Returns (X, gt_6d, smpl_R_local, actual_fps) or raises on failure.
      X             : [N, 6]             — right-wrist FLU-frame IMU
      gt_6d         : [N, num_joints, 6] — GT 6D rotations for queried joints
      smpl_R_local  : [N, 24, 3, 3]     — full SMPL local rotmats (for FK)
      actual_fps    : float
    """
    import core.articulate as art
    from core.paths import Paths

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    gt_data = raw.get("gt_data", {})

    # ------------------------------------------------------------------
    # IMUPoser format: gt_data["pose"] = [N, 24, 3, 3] SMPL local rotmats
    # ------------------------------------------------------------------
    if "pose" in gt_data and "xsens_pose" not in gt_data:
        # R_WRIST_FROM_FLU (Z-up world, same as imuposer_wrist_loader_FLU.py)
        R_WRIST_FROM_FLU_ZUP = np.array(
            [[-1., 0., 0.],
             [ 0., 0., 1.],
             [ 0., 1., 0.]], dtype=np.float64,
        )

        actual_fps = float(raw.get("sampling_rate_hz", fps))

        pose_local = np.asarray(gt_data["pose"], dtype=np.float32)   # [N, 24, 3, 3]
        N_raw = pose_local.shape[0]

        ri         = raw.get("real_imu_data", {})
        real_acc   = np.asarray(ri["real_acc"],  dtype=np.float64)   # [N, 5, 3]  Z-up world
        real_ori   = np.asarray(ri["real_ori"],  dtype=np.float64)   # [N, 5, 3, 3]  R_world_from_sensor
        joint_pos  = np.asarray(gt_data["joint_pos"], dtype=np.float32)   # [N, 24, 3]

        N = N_raw
        if max_frames > 0:
            N = min(N, max_frames)
        pose_local = pose_local[:N]
        real_acc   = real_acc[:N]
        real_ori   = real_ori[:N]
        joint_pos  = joint_pos[:N]

        # Resample to target fps — must match the training loader
        if abs(actual_fps - fps) > 1e-3:
            new_len     = int(round((N - 1) * fps / actual_fps)) + 1
            up          = fps > actual_fps
            _pose_t     = torch.from_numpy(pose_local)
            _jpos_t     = torch.from_numpy(joint_pos)
            _acc_t      = torch.from_numpy(real_acc.astype(np.float32))
            _ori_t      = torch.from_numpy(real_ori.astype(np.float32))
            pose_local  = (_resample_rotmat_nearest if up else _resample_rotmat_nlerp)(_pose_t, new_len).numpy().astype(np.float32)
            joint_pos   = (_resample_nearest        if up else _resample_linear       )(_jpos_t, new_len).numpy().astype(np.float32)
            real_acc    = (_resample_nearest        if up else _resample_linear       )(_acc_t,  new_len).numpy().astype(np.float64)
            real_ori    = (_resample_rotmat_nearest if up else _resample_rotmat_nlerp)(_ori_t,  new_len).numpy().astype(np.float64)
            N           = new_len
            actual_fps  = float(fps)

        smpl_R_local = pose_local   # already local rotations

        # FK to get world-frame wrist rotation (needed for FLU frame)
        body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
        root_trans = torch.from_numpy(joint_pos[:, 0])   # [N, 3]
        pose_global_t, _ = body_model.forward_kinematics_masked(
            torch.from_numpy(pose_local), tran=root_trans, calc_mesh=False,
        )   # [N, 24, 3, 3]
        R_world_from_wrist = pose_global_t[:, 21].cpu().numpy().astype(np.float64)   # [N,3,3]

        # FLU frame transform
        R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU_ZUP)
        R_X_from_world = R_world_from_X.transpose(0, 2, 1)

        # Accelerometer: right-wrist sensor is index 1 in real_imu_data
        acc_world = real_acc[:, 1]   # [N, 3]  Z-up world, gravity-subtracted
        if add_gravity:
            acc_world = acc_world + np.array([0., 0., 9.81])   # Z-up specific force
        acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world).astype(np.float32)

        # Gyroscope from consecutive sensor orientations
        R_world_from_sensor = real_ori[:, 1]   # [N, 3, 3]
        dR = np.einsum("nij,njk->nik",
                       R_world_from_sensor[:-1].transpose(0, 2, 1),
                       R_world_from_sensor[1:])   # [N-1, 3, 3]
        omega_sensor = np.zeros((N, 3), dtype=np.float64)
        omega_sensor[:-1] = (
            ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec() * actual_fps
        )
        omega_sensor[-1] = omega_sensor[-2]
        omega_world = np.einsum("nij,nj->ni", R_world_from_sensor, omega_sensor)
        gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, omega_world).astype(np.float32)

        X = np.concatenate([acc_flu, gyro_flu], axis=1)   # [N, 6]
        # Measured wrist orientation ≈ R_world_from_smpl_wrist (sensor calibrated to SMPL frame)
        R_wrist_world_meas = real_ori[:, 1].astype(np.float32)  # [N, 3, 3]

    # ------------------------------------------------------------------
    # Nymeria format: gt_data["xsens_pose"] = [N, 23, 4, 4]
    # ------------------------------------------------------------------
    else:
        xsens_pose = gt_data["xsens_pose"]
        if hasattr(xsens_pose, "numpy"):
            xsens_pose = xsens_pose.detach().cpu().numpy()
        xsens_pose = np.asarray(xsens_pose, dtype=np.float32)   # [N, 23, 4, 4]

        body_fps_raw = float(
            raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or fps
        )
        actual_fps = float(fps)

        if loader_type in ("nymeria_real_wrist_flu", "nymeria_real_wrist_full_body_flu"):
            # --- Real wrist IMU in FLU frame (matches training loader) ---
            _R_WRIST_FROM_FLU_EVAL = np.array(
                [[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], dtype=np.float64
            )
            wrist_fps_raw = float(raw.get("wrist_imu_sampling_rate_hz", body_fps_raw))
            body_ds  = max(1, round(body_fps_raw  / fps))
            wrist_ds = max(1, round(wrist_fps_raw / fps))

            xsens_pose = xsens_pose[::body_ds]

            wrist_imu_d = raw.get("wrist_imu_data", {})
            wrist_gt_d  = raw.get("wrist_gt_data",  {})

            def _to_np64(x):
                if hasattr(x, "detach"):
                    return x.detach().cpu().numpy().astype(np.float64)
                return np.asarray(x, dtype=np.float64)

            acc_local  = _to_np64(wrist_imu_d["accel"])[::wrist_ds]
            gyro_local = _to_np64(wrist_imu_d["gyro"])[::wrist_ds]

            orient_raw = wrist_gt_d["orientation"]
            try:
                import pypose as pp
                _is_lt = isinstance(orient_raw, pp.LieTensor)
            except ImportError:
                _is_lt = False
            if _is_lt:
                R_world_from_wrist = (
                    orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
                )
            else:
                orient_np = _to_np64(orient_raw)[::wrist_ds]
                R_world_from_wrist = (
                    ScipyRot.from_quat(orient_np).as_matrix().astype(np.float64)
                )

            N = min(xsens_pose.shape[0], acc_local.shape[0])
            if max_frames > 0:
                N = min(N, max_frames)
            xsens_pose         = xsens_pose[:N]
            acc_local          = acc_local[:N]
            gyro_local         = gyro_local[:N]
            R_world_from_wrist = R_world_from_wrist[:N]

            acc_world  = np.einsum("nij,nj->ni", R_world_from_wrist, acc_local)
            gyro_world = np.einsum("nij,nj->ni", R_world_from_wrist, gyro_local)
            if not add_gravity:
                acc_world = acc_world + np.array([0., -9.81, 0.])

            R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, _R_WRIST_FROM_FLU_EVAL)
            R_X_from_world = R_world_from_X.transpose(0, 2, 1)
            acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, acc_world)
            gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, gyro_world)
            X = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)
            # wrist_gt_data orientation = R_world_from_smpl_wrist in Nymeria Y-up world
            R_wrist_world_meas = R_world_from_wrist.astype(np.float32)  # [N, 3, 3]

        else:
            # --- Synthetic IMU from Xsens pose (global frame) ---
            actual_fps = body_fps_raw
            N = xsens_pose.shape[0]
            if max_frames > 0:
                N = min(N, max_frames)
                xsens_pose = xsens_pose[:N]

            j   = _XSENS_RIGHT_WRIST_JOINT
            pos = xsens_pose[:, j, :3, 3]
            rot = xsens_pose[:, j, :3, :3]

            acc       = np.zeros((N, 3), dtype=np.float32)
            acc[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (actual_fps ** 2)
            acc[0]    = acc[1];  acc[-1] = acc[-2]

            dR       = np.einsum("nij,njk->nik", rot[:-1].transpose(0, 2, 1), rot[1:])
            gyro_raw = (ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec() * actual_fps)
            gyro     = np.zeros((N, 3), dtype=np.float32)
            gyro[:-1] = gyro_raw.astype(np.float32);  gyro[-1] = gyro[-2]
            X = np.concatenate([acc, gyro], axis=1)   # [N, 6]
            R_wrist_world_meas = None   # synthetic IMU: no real wrist sensor

        smpl_R_global = torch.from_numpy(
            _xsens_to_smpl_rotmat(xsens_pose[:, :, :3, :3])
        )  # [N, 24, 3, 3]
        body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
        smpl_R_local = (
            body_model
            .inverse_kinematics_R(smpl_R_global.reshape(N, -1))
            .view(N, 24, 3, 3)
            .contiguous().clone().cpu().numpy()
        )  # [N, 24, 3, 3]

    # ---- GT 6D for selected joints ---- #
    if num_joints == 24:
        sel_R = smpl_R_local
    elif num_joints == 23:
        sel_R = smpl_R_local[:, 1:]   # drop root (joint 0)
    else:
        sel_R = smpl_R_local[:, RIGHT_WRIST_CHAIN_JOINTS]   # [N, 5, 3, 3]

    gt_6d = np.concatenate([sel_R[..., 0], sel_R[..., 1]], axis=-1)  # [N, J, 6]
    return X, gt_6d, smpl_R_local, actual_fps, R_wrist_world_meas


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(X: np.ndarray, encoder, pose_decoder, cfg, device, fps: int) -> np.ndarray:
    """Return per-frame 6D predictions [N, num_joints, 6]."""
    T, _       = compute_T_and_hop(cfg)
    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    num_joints = int(cfg_get(cfg, ["objective", "pose", "num_joints"], 5))
    stats      = PreprocessStats()
    N          = len(X)

    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X

    pred_full = np.zeros((len(X_pad), num_joints, 6), dtype=np.float32)
    for w in range(len(X_pad) // T):
        start = w * T
        Xw    = X_pad[start:start + T]
        Xproc = preprocess_window(Xw, cfg, stats)
        if Xproc is None:
            continue
        x = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)
        tokens, _ = encoder.forward_tokens(x)
        pred_np   = pose_decoder(tokens).squeeze(0).cpu().numpy()   # [Pt, J*6]
        for p in range(pred_np.shape[0]):
            f0 = start + p * patch_size
            f1 = f0 + patch_size
            pred_full[f0:f1] = pred_np[p].reshape(num_joints, 6)

    return pred_full[:N]


# ---------------------------------------------------------------------------
# MPJPE via forward kinematics
# ---------------------------------------------------------------------------

def compute_mpjpe_mm(
    pred_6d: np.ndarray,          # [N, num_joints, 6]
    smpl_R_gt_local: np.ndarray,  # [N, 24, 3, 3]
    num_joints: int,
    device: torch.device,
) -> np.ndarray:
    """Return MPJPE in millimetres per frame per joint  [N, num_joints].

    Uses SMPL forward kinematics (core.articulate) to convert local rotations
    → 3-D joint world positions, then computes per-joint L2 distance.

    Requires core.articulate. Returns None if unavailable.
    """
    try:
        import core.articulate as art
        from core.paths import Paths
    except ImportError:
        return None

    N = pred_6d.shape[0]

    # ---- Convert predicted 6D → local rotation matrices ---- #
    pred_R_sel = rotation_6d_to_matrix(
        torch.from_numpy(pred_6d)
    ).numpy()   # [N, num_joints, 3, 3]

    # Build full predicted local rotation matrices:
    # replace queried joints in GT, keep everything else from GT
    pred_R_full = smpl_R_gt_local.copy()   # [N, 24, 3, 3]
    if num_joints == 24:
        pred_R_full = pred_R_sel
    elif num_joints == 23:
        pred_R_full[:, 1:] = pred_R_sel   # joints 1-23, skip root
    else:
        for ci, ji in enumerate(RIGHT_WRIST_CHAIN_JOINTS):
            pred_R_full[:, ji] = pred_R_sel[:, ci]

    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)

    def _fk_positions(R_local_np):
        # forward_kinematics returns (pose_global [N,24,3,3], joint_positions [N,24,3])
        # We need the joint POSITIONS (index 1), not the rotation matrices (index 0).
        R_t    = torch.from_numpy(R_local_np).float().to(device)
        R_flat = R_t.reshape(N, -1)
        try:
            result = body_model.forward_kinematics(R_flat)
            if isinstance(result, (tuple, list)):
                pos = result[1]   # index 1 = joint positions [N, 24, 3] in metres
            else:
                pos = result
        except AttributeError:
            return None
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        return pos  # [N, 24, 3]  in metres

    pos_gt   = _fk_positions(smpl_R_gt_local)
    pos_pred = _fk_positions(pred_R_full)
    if pos_gt is None or pos_pred is None:
        logger.warning("FK unavailable — skipping MPJPE")
        return None

    if num_joints == 24:
        sel_idx = list(range(24))
    elif num_joints == 23:
        sel_idx = list(range(1, 24))   # SMPL joints 1-23, skip root
    else:
        sel_idx = RIGHT_WRIST_CHAIN_JOINTS

    # L2 distance in mm
    mpjpe = (
        np.linalg.norm(pos_pred[:, sel_idx] - pos_gt[:, sel_idx], axis=-1) * 1000.0
    )  # [N, num_joints]
    return mpjpe


def compute_mpjve_cm(
    pred_6d: np.ndarray,          # [N, num_joints, 6]
    smpl_R_gt_local: np.ndarray,  # [N, 24, 3, 3]
    num_joints: int,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return MPJVE in centimetres per frame [N].

    MPJVE = Mean Per Joint Vertex Error: mean Euclidean distance across all SMPL
    mesh vertices between predicted and GT poses, with the root joint (pelvis)
    aligned (both meshes translated so their pelvis joint sits at the origin).

    Requires core.articulate (SMPL model). Returns None if unavailable.
    """
    try:
        import core.articulate as art
        from core.paths import Paths
    except ImportError:
        return None

    N = pred_6d.shape[0]

    # Build full predicted local rotation matrices (same logic as compute_mpjpe_mm)
    pred_R_sel = rotation_6d_to_matrix(
        torch.from_numpy(pred_6d)
    ).numpy()   # [N, num_joints, 3, 3]

    pred_R_full = smpl_R_gt_local.copy()   # [N, 24, 3, 3]
    if num_joints == 24:
        pred_R_full = pred_R_sel
    elif num_joints == 23:
        pred_R_full[:, 1:] = pred_R_sel
    else:
        for ci, ji in enumerate(RIGHT_WRIST_CHAIN_JOINTS):
            pred_R_full[:, ji] = pred_R_sel[:, ci]

    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)

    mpjve_frames = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Nb = end - start
        R_gt_b   = torch.from_numpy(smpl_R_gt_local[start:end]).float().to(device)
        R_pred_b = torch.from_numpy(pred_R_full[start:end]).float().to(device)
        try:
            _, jpos_gt,   verts_gt   = body_model.forward_kinematics(
                R_gt_b.reshape(Nb, -1), calc_mesh=True)
            _, jpos_pred, verts_pred = body_model.forward_kinematics(
                R_pred_b.reshape(Nb, -1), calc_mesh=True)
        except Exception:
            return None

        # Pelvis alignment: subtract root joint (joint 0) from both meshes
        verts_gt_aligned   = verts_gt   - jpos_gt[:, :1]    # [Nb, V, 3]
        verts_pred_aligned = verts_pred - jpos_pred[:, :1]  # [Nb, V, 3]

        # Mean vertex L2 error per frame, convert metres → cm
        err_per_frame = (
            (verts_pred_aligned - verts_gt_aligned)
            .norm(dim=-1)       # [Nb, V]
            .mean(dim=-1)       # [Nb]
            * 100.0             # cm
        )
        mpjve_frames.append(err_per_frame.detach().cpu().numpy())

    return np.concatenate(mpjve_frames)   # [N]


# ---------------------------------------------------------------------------
# Wrist-pelvis helpers — derive global pelvis from measured wrist orientation
# ---------------------------------------------------------------------------

def _build_pred_R_full(
    pred_6d: np.ndarray,          # [N, num_joints, 6]
    smpl_R_gt_local: np.ndarray,  # [N, 24, 3, 3]
    num_joints: int,
) -> np.ndarray:                  # [N, 24, 3, 3]
    """Convert predicted 6D rotations → full 24-joint local rotation matrices.

    Non-predicted joints (e.g. the pelvis root for num_joints==23) are filled
    from the GT local rotations.
    """
    pred_R_sel = rotation_6d_to_matrix(torch.from_numpy(pred_6d)).numpy()
    pred_R_full = smpl_R_gt_local.copy()
    if num_joints == 24:
        pred_R_full = pred_R_sel
    elif num_joints == 23:
        pred_R_full[:, 1:] = pred_R_sel   # joints 1-23; leave root as GT placeholder
    else:
        for ci, ji in enumerate(RIGHT_WRIST_CHAIN_JOINTS):
            pred_R_full[:, ji] = pred_R_sel[:, ci]
    return pred_R_full


def _derive_pelvis_from_wrist(
    pred_R_full: np.ndarray,     # [N, 24, 3, 3] local rotations (root is placeholder)
    R_wrist_world: np.ndarray,   # [N, 3, 3] R_world_from_smpl_wrist (measured)
    body_model,
    device: torch.device,
) -> np.ndarray:                  # [N, 3, 3] estimated global pelvis rotation
    """Estimate global pelvis rotation from measured wrist orientation + predicted pose.

    SMPL FK chain from root to right wrist (joint 21):
        R_global[21] = R_global_pelvis @ C_wrist

    Setting root = I to isolate the chain product C_wrist via FK, then:
        R_global_pelvis = R_wrist_world_meas @ C_wrist.T

    This gives the pelvis orientation such that, combined with the predicted
    body pose, the right-wrist FK output matches the measured wrist orientation.
    """
    N = pred_R_full.shape[0]
    R_no_root = pred_R_full.copy()
    R_no_root[:, 0] = np.eye(3, dtype=np.float32)   # identity root
    R_t = torch.from_numpy(R_no_root).float().to(device)
    # FK with identity root → R_global[21] = C_wrist (chain product root→wrist)
    R_global = body_model.forward_kinematics_R(R_t.reshape(N, -1))   # [N, 24, 3, 3]
    C_wrist = R_global[:, 21].detach().cpu().numpy()                   # [N, 3, 3]
    # R_global_pelvis = R_wrist_world @ C_wrist.T
    R_global_pelvis = np.matmul(
        R_wrist_world.astype(np.float32),
        C_wrist.transpose(0, 2, 1),
    )
    return R_global_pelvis   # [N, 3, 3]


def compute_mpjpe_wrist_pelvis_mm(
    pred_6d: np.ndarray,          # [N, num_joints, 6]
    smpl_R_gt_local: np.ndarray,  # [N, 24, 3, 3]
    num_joints: int,
    R_wrist_world: np.ndarray,    # [N, 3, 3] measured R_world_from_smpl_wrist
    device: torch.device,
) -> np.ndarray:
    """MPJPE (mm) using the global pelvis rotation derived from measured wrist orientation.

    Unlike compute_mpjpe_mm (which uses the GT pelvis rotation for all predicted
    non-root joints), this function back-computes the global pelvis rotation
    from the predicted body pose + the measured wrist IMU orientation.
    """
    try:
        import core.articulate as art
        from core.paths import Paths
    except ImportError:
        return None

    N = pred_6d.shape[0]
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)
    pred_R_full = _build_pred_R_full(pred_6d, smpl_R_gt_local, num_joints)
    R_pelvis = _derive_pelvis_from_wrist(pred_R_full, R_wrist_world, body_model, device)
    pred_R_full_wp = pred_R_full.copy()
    pred_R_full_wp[:, 0] = R_pelvis

    def _fk_pos(R_local_np):
        R_t = torch.from_numpy(R_local_np).float().to(device)
        try:
            result = body_model.forward_kinematics(R_t.reshape(N, -1))
            pos = result[1] if isinstance(result, (tuple, list)) else result
        except AttributeError:
            return None
        return pos.detach().cpu().numpy() if hasattr(pos, "detach") else pos

    pos_gt   = _fk_pos(smpl_R_gt_local)
    pos_pred = _fk_pos(pred_R_full_wp)
    if pos_gt is None or pos_pred is None:
        return None

    if num_joints == 24:
        sel_idx = list(range(24))
    elif num_joints == 23:
        sel_idx = list(range(1, 24))
    else:
        sel_idx = RIGHT_WRIST_CHAIN_JOINTS

    return np.linalg.norm(pos_pred[:, sel_idx] - pos_gt[:, sel_idx], axis=-1) * 1000.0


def compute_mpjve_wrist_pelvis_cm(
    pred_6d: np.ndarray,          # [N, num_joints, 6]
    smpl_R_gt_local: np.ndarray,  # [N, 24, 3, 3]
    num_joints: int,
    R_wrist_world: np.ndarray,    # [N, 3, 3] measured R_world_from_smpl_wrist
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """MPJVE (cm, pelvis-aligned) using pelvis derived from measured wrist orientation."""
    try:
        import core.articulate as art
        from core.paths import Paths
    except ImportError:
        return None

    N = pred_6d.shape[0]
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=device)
    pred_R_full = _build_pred_R_full(pred_6d, smpl_R_gt_local, num_joints)
    R_pelvis = _derive_pelvis_from_wrist(pred_R_full, R_wrist_world, body_model, device)
    pred_R_full_wp = pred_R_full.copy()
    pred_R_full_wp[:, 0] = R_pelvis

    mpjve_frames = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        Nb = end - start
        R_gt_b   = torch.from_numpy(smpl_R_gt_local[start:end]).float().to(device)
        R_pred_b = torch.from_numpy(pred_R_full_wp[start:end]).float().to(device)
        try:
            _, jpos_gt,   verts_gt   = body_model.forward_kinematics(
                R_gt_b.reshape(Nb, -1), calc_mesh=True)
            _, jpos_pred, verts_pred = body_model.forward_kinematics(
                R_pred_b.reshape(Nb, -1), calc_mesh=True)
        except Exception:
            return None
        verts_gt_aligned   = verts_gt   - jpos_gt[:, :1]
        verts_pred_aligned = verts_pred - jpos_pred[:, :1]
        err = (verts_pred_aligned - verts_gt_aligned).norm(dim=-1).mean(dim=-1) * 100.0
        mpjve_frames.append(err.detach().cpu().numpy())

    return np.concatenate(mpjve_frames)   # [N]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved  %s", path)


def plot_per_joint_bar(means, stds, ylabel, title, out_path, joint_names, color):
    fig, ax = plt.subplots(figsize=(max(8, len(joint_names) * 0.7 + 2), 5))
    x = np.arange(len(joint_names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=color, width=0.55, alpha=0.85,
                  error_kw=dict(ecolor="dimgray", lw=1.2))
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(means) * 0.015,
                f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(joint_names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    # Overall mean annotation
    overall = float(np.mean(means))
    ax.axhline(overall, color="r", lw=1.2, ls="--",
               label=f"Overall mean: {overall:.1f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_path)


def plot_cdf(all_errors_per_joint, joint_names, out_path, xlabel):
    """CDF of per-frame errors for each joint on one axes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.tab10
    for ji, (jname, errs) in enumerate(zip(joint_names, all_errors_per_joint)):
        errs_sorted = np.sort(errs)
        cdf = np.arange(1, len(errs_sorted) + 1) / len(errs_sorted)
        ax.plot(errs_sorted, cdf, label=jname, color=cmap(ji % 10), linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"CDF — {xlabel}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_error_per_sequence(seq_names, seq_geo_means, out_path):
    """Bar chart: mean geodesic error for each test sequence (sorted)."""
    order = np.argsort(seq_geo_means)
    names_sorted = [seq_names[i] for i in order]
    vals_sorted  = [seq_geo_means[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(10, len(seq_names) * 0.35 + 2), 5))
    x = np.arange(len(names_sorted))
    ax.bar(x, vals_sorted, color="steelblue", width=0.8, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names_sorted, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Mean geodesic error (°)")
    ax.set_title("Per-sequence mean geodesic error (all right-arm joints)")
    ax.axhline(np.mean(vals_sorted), color="r", lw=1.2, ls="--",
               label=f"Dataset mean: {np.mean(vals_sorted):.1f}°")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser(
        description="Dataset-level pose eval — geodesic + MPJPE across all test pkls"
    )
    ap.add_argument("--config",       required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run",          required=True, help="Run name under runs/<run>")
    ap.add_argument("--test-dir",     required=True, help="Directory of Nymeria test pkl files")
    ap.add_argument("--ckpt",         default="best")
    ap.add_argument("--max-frames",   type=int, default=0, help="Per-sequence frame cap (0=all)")
    ap.add_argument("--fps",          type=int, default=50)
    ap.add_argument("--out-dir",      default="data_vis/pose_eval_dataset")
    ap.add_argument("--num-joints",   type=int, default=None,
                    help="Override num_joints (default: read from model config)")
    args = ap.parse_args()

    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg = deep_update(base_cfg, model_cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Load checkpoint -------------------------------------------------- #
    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_dir   = os.path.join(runs_root, args.run)
    if args.ckpt in ("best", "latest"):
        ckpt_path = os.path.join(run_dir, "checkpoints", f"{args.ckpt}.pt")
    else:
        ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    state    = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = state["cfg"]

    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    T_win, _   = compute_T_and_hop(ckpt_cfg)
    vit_cfg    = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))

    # Resolve num_joints from the saved checkpoint config, checking both
    # the end-to-end *pose* key and the probe/finetune variants.
    def _resolve_num_joints(c):
        for path in (
            ["objective", "pose",            "num_joints"],
            ["objective", "pose_probe",       "num_joints"],
            ["objective", "pose_finetune",    "num_joints"],
        ):
            v = cfg_get(c, path, None)
            if v is not None:
                return int(v)
        return None

    num_joints = args.num_joints or _resolve_num_joints(ckpt_cfg) or _resolve_num_joints(cfg) or 5
    if num_joints == 24:
        joint_names = SMPL_JOINT_NAMES
    elif num_joints == 23:
        joint_names = SMPL_JOINT_NAMES[1:]   # drop Pelvis (root)
    else:
        joint_names = JOINT_NAMES_5
    logger.info("num_joints=%d  joint_names=%s", num_joints, joint_names)

    # Resolve add_gravity from checkpoint config
    def _resolve_add_gravity(c):
        for section in ("pose", "pose_probe", "pose_finetune"):
            v = cfg_get(c, ["objective", section, "add_gravity"], None)
            if v is not None:
                return bool(v)
        return False

    add_gravity = _resolve_add_gravity(ckpt_cfg) or _resolve_add_gravity(cfg)
    logger.info("add_gravity=%s", add_gravity)

    def _resolve_loader_type(c):
        for section in ("pose", "pose_probe", "pose_finetune"):
            v = cfg_get(c, ["objective", section, "loader"], None)
            if v is not None:
                return str(v).lower()
        return ""

    loader_type = _resolve_loader_type(ckpt_cfg) or _resolve_loader_type(cfg)
    logger.info("loader_type=%s", loader_type)

    pose_decoder = ViT1DPoseDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        num_joints=num_joints,
        cfg=ckpt_cfg,
    ).to(device)
    pose_decoder.load_state_dict(state["pose_decoder"])
    pose_decoder.eval()

    # Override cfg so run_inference reads the right num_joints
    cfg.setdefault("objective", {}).setdefault("pose", {})["num_joints"] = num_joints

    # ---- Gather pkl files ------------------------------------------------- #
    pkl_files = sorted(glob.glob(os.path.join(args.test_dir, "*.pkl")))
    if not pkl_files:
        raise SystemExit(f"No pkl files found in {args.test_dir}")
    logger.info("Found %d pkl files in %s", len(pkl_files), args.test_dir)

    # ---- Per-sequence accumulators ---------------------------------------- #
    # geo_all[j] = flat list of per-frame geodesic errors for joint j
    geo_all       = [[] for _ in range(num_joints)]
    mpjpe_all     = [[] for _ in range(num_joints)]
    mpjve_all     = []
    mpjpe_wp_all  = [[] for _ in range(num_joints)]   # wrist-pelvis variant
    mpjve_wp_all  = []                                 # wrist-pelvis variant
    seq_names   = []
    seq_geo_means = []
    seq_failed  = []
    compute_mpjpe    = True   # set False if FK unavailable
    compute_mpjve    = True   # set False if FK unavailable
    compute_mpjpe_wp = True   # set False if FK or wrist data unavailable
    compute_mpjve_wp = True   # set False if FK or wrist data unavailable

    for pkl_path in tqdm(pkl_files, desc="Evaluating sequences", unit="seq"):
        seq_name = os.path.basename(pkl_path).replace(".pkl", "")
        try:
            X, gt_6d, smpl_R_local, actual_fps, R_wrist_world = load_sequence(
                pkl_path, args.fps, args.max_frames, num_joints,
                add_gravity=add_gravity, loader_type=loader_type,
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", seq_name, exc)
            seq_failed.append(seq_name)
            continue

        pred_6d = run_inference(X, encoder, pose_decoder, cfg, device, int(actual_fps))
        N = len(pred_6d)

        # ---- Geodesic ---- #
        pred_t = torch.from_numpy(pred_6d)
        gt_t   = torch.from_numpy(gt_6d)
        pred_R = rotation_6d_to_matrix(pred_t)
        gt_R   = rotation_6d_to_matrix(gt_t)
        geo_seq = geodesic_error_deg(
            pred_R.reshape(-1, 3, 3), gt_R.reshape(-1, 3, 3)
        ).reshape(N, num_joints).numpy()   # [N, J]

        for j in range(num_joints):
            geo_all[j].extend(geo_seq[:, j].tolist())

        seq_geo_means.append(float(geo_seq.mean()))
        seq_names.append(seq_name)

        # ---- MPJPE ---- #
        if compute_mpjpe:
            mpjpe_seq = compute_mpjpe_mm(pred_6d, smpl_R_local, num_joints, device)
            if mpjpe_seq is None:
                compute_mpjpe = False
                logger.warning("FK unavailable — MPJPE will not be computed.")
            else:
                for j in range(num_joints):
                    mpjpe_all[j].extend(mpjpe_seq[:, j].tolist())

        # ---- MPJVE ---- #
        if compute_mpjve:
            mpjve_seq = compute_mpjve_cm(pred_6d, smpl_R_local, num_joints, device)
            if mpjve_seq is None:
                compute_mpjve = False
                logger.warning("FK unavailable — MPJVE will not be computed.")
            else:
                mpjve_all.extend(mpjve_seq.tolist())

        # ---- MPJPE (wrist-pelvis) ---- #
        if compute_mpjpe_wp:
            if R_wrist_world is None:
                compute_mpjpe_wp = False
                logger.info("No measured wrist orientation — skipping wrist-pelvis MPJPE.")
            else:
                mpjpe_wp_seq = compute_mpjpe_wrist_pelvis_mm(
                    pred_6d, smpl_R_local, num_joints, R_wrist_world, device)
                if mpjpe_wp_seq is None:
                    compute_mpjpe_wp = False
                    logger.warning("FK unavailable — wrist-pelvis MPJPE will not be computed.")
                else:
                    for j in range(num_joints):
                        mpjpe_wp_all[j].extend(mpjpe_wp_seq[:, j].tolist())

        # ---- MPJVE (wrist-pelvis) ---- #
        if compute_mpjve_wp:
            if R_wrist_world is None:
                compute_mpjve_wp = False
                logger.info("No measured wrist orientation — skipping wrist-pelvis MPJVE.")
            else:
                mpjve_wp_seq = compute_mpjve_wrist_pelvis_cm(
                    pred_6d, smpl_R_local, num_joints, R_wrist_world, device)
                if mpjve_wp_seq is None:
                    compute_mpjve_wp = False
                    logger.warning("FK unavailable — wrist-pelvis MPJVE will not be computed.")
                else:
                    mpjve_wp_all.extend(mpjve_wp_seq.tolist())

    if not seq_names:
        raise SystemExit("No sequences evaluated successfully.")

    # ---- Aggregate statistics --------------------------------------------- #
    geo_all_arr   = [np.array(g) for g in geo_all]   # per-joint flat arrays
    geo_means     = np.array([g.mean() for g in geo_all_arr])
    geo_stds      = np.array([g.std()  for g in geo_all_arr])

    # Print table
    total_frames = sum(len(g) for g in geo_all_arr)
    print(f"\n{'='*72}")
    print(f"  Dataset-level evaluation — {len(seq_names)} sequences  {total_frames} frames")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'='*72}")
    print(f"\n{'Joint':<16} {'Geodesic mean':>15} {'Geodesic std':>13}", end="")
    if compute_mpjpe and any(len(m) > 0 for m in mpjpe_all):
        print(f" {'MPJPE mean (mm)':>16} {'MPJPE std':>11}", end="")
    print()
    print("-" * (72 if not compute_mpjpe else 88))

    for ji, jname in enumerate(joint_names):
        row = f"{jname:<16} {geo_means[ji]:>14.2f}°  {geo_stds[ji]:>12.2f}°"
        if compute_mpjpe and len(mpjpe_all[ji]) > 0:
            m = np.array(mpjpe_all[ji])
            row += f"  {m.mean():>15.2f}   {m.std():>10.2f}"
        print(row)

    print("-" * (72 if not compute_mpjpe else 88))
    print(f"{'Mean':<16} {geo_means.mean():>14.2f}°  {geo_stds.mean():>12.2f}°", end="")
    if compute_mpjpe and any(len(m) > 0 for m in mpjpe_all):
        all_mpjpe_flat = np.concatenate([np.array(m) for m in mpjpe_all if len(m) > 0])
        print(f"  {all_mpjpe_flat.mean():>15.2f}   {all_mpjpe_flat.std():>10.2f}", end="")
    print()
    if compute_mpjve and len(mpjve_all) > 0:
        mpjve_arr_full = np.array(mpjve_all)
        print(f"\n  MPJVE (pelvis-aligned, GT pelvis): {mpjve_arr_full.mean():.2f} ± {mpjve_arr_full.std():.2f} cm")
    if compute_mpjpe_wp and any(len(m) > 0 for m in mpjpe_wp_all):
        all_mpjpe_wp_flat = np.concatenate([np.array(m) for m in mpjpe_wp_all if len(m) > 0])
        print(f"  MPJPE  (wrist-derived pelvis)    : {all_mpjpe_wp_flat.mean():.2f} ± {all_mpjpe_wp_flat.std():.2f} mm")
    if compute_mpjve_wp and len(mpjve_wp_all) > 0:
        mpjve_wp_arr = np.array(mpjve_wp_all)
        print(f"  MPJVE  (wrist-derived pelvis)    : {mpjve_wp_arr.mean():.2f} ± {mpjve_wp_arr.std():.2f} cm")
    if seq_failed:
        print(f"\n  ⚠ Skipped {len(seq_failed)} sequences: {seq_failed[:5]}{'...' if len(seq_failed) > 5 else ''}")
    print()

    # ---- Save plots ------------------------------------------------------- #
    plot_per_joint_bar(
        geo_means, geo_stds,
        ylabel="Mean geodesic error (°)",
        title=f"Per-Joint Geodesic Error  ({len(seq_names)} test sequences)",
        out_path=os.path.join(args.out_dir, "per_joint_geodesic.png"),
        joint_names=joint_names,
        color="steelblue",
    )
    plot_cdf(
        geo_all_arr, joint_names,
        out_path=os.path.join(args.out_dir, "geodesic_cdf.png"),
        xlabel="Geodesic error (°)",
    )
    plot_error_per_sequence(
        seq_names, seq_geo_means,
        out_path=os.path.join(args.out_dir, "error_vs_sequence.png"),
    )

    if compute_mpjpe and any(len(m) > 0 for m in mpjpe_all):
        mpjpe_arr   = [np.array(m) for m in mpjpe_all]
        mpjpe_means = np.array([m.mean() for m in mpjpe_arr])
        mpjpe_stds  = np.array([m.std()  for m in mpjpe_arr])
        plot_per_joint_bar(
            mpjpe_means, mpjpe_stds,
            ylabel="MPJPE (mm)",
            title=f"Per-Joint MPJPE  ({len(seq_names)} test sequences)",
            out_path=os.path.join(args.out_dir, "per_joint_mpjpe.png"),
            joint_names=joint_names,
            color="tomato",
        )
        plot_cdf(
            mpjpe_arr, joint_names,
            out_path=os.path.join(args.out_dir, "mpjpe_cdf.png"),
            xlabel="MPJPE (mm)",
        )

    if compute_mpjpe_wp and any(len(m) > 0 for m in mpjpe_wp_all):
        mpjpe_wp_arr   = [np.array(m) for m in mpjpe_wp_all]
        mpjpe_wp_means = np.array([m.mean() for m in mpjpe_wp_arr])
        mpjpe_wp_stds  = np.array([m.std()  for m in mpjpe_wp_arr])
        plot_per_joint_bar(
            mpjpe_wp_means, mpjpe_wp_stds,
            ylabel="MPJPE (mm)",
            title=f"Per-Joint MPJPE — wrist-derived pelvis  ({len(seq_names)} test sequences)",
            out_path=os.path.join(args.out_dir, "per_joint_mpjpe_wrist_pelvis.png"),
            joint_names=joint_names,
            color="darkorange",
        )
        plot_cdf(
            mpjpe_wp_arr, joint_names,
            out_path=os.path.join(args.out_dir, "mpjpe_wrist_pelvis_cdf.png"),
            xlabel="MPJPE (mm)",
        )

    if compute_mpjve and len(mpjve_all) > 0:
        mpjve_arr_full = np.array(mpjve_all)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(mpjve_arr_full, bins=60, color="mediumpurple", alpha=0.75, edgecolor="none")
        ax.axvline(mpjve_arr_full.mean(), color="r", lw=1.4, ls="--",
                   label=f"Mean: {mpjve_arr_full.mean():.2f} cm")
        ax.set_xlabel("MPJVE (cm)")
        ax.set_ylabel("Frame count")
        ax.set_title(f"MPJVE — GT pelvis  ({len(seq_names)} test sequences, pelvis-aligned)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, os.path.join(args.out_dir, "mpjve_hist.png"))

    if compute_mpjve_wp and len(mpjve_wp_all) > 0:
        mpjve_wp_arr_full = np.array(mpjve_wp_all)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(mpjve_wp_arr_full, bins=60, color="teal", alpha=0.75, edgecolor="none")
        ax.axvline(mpjve_wp_arr_full.mean(), color="r", lw=1.4, ls="--",
                   label=f"Mean: {mpjve_wp_arr_full.mean():.2f} cm")
        ax.set_xlabel("MPJVE (cm)")
        ax.set_ylabel("Frame count")
        ax.set_title(f"MPJVE — wrist-derived pelvis  ({len(seq_names)} test sequences, pelvis-aligned)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, os.path.join(args.out_dir, "mpjve_wrist_pelvis_hist.png"))

    logger.info("All plots written to %s", args.out_dir)


if __name__ == "__main__":
    main()
