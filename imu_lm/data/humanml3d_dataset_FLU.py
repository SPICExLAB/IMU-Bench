"""
HumanML3D dataloaders — FLU wrist-IMU variant.

Identical to humanml3d_dataset.py **except** the synthetic right-wrist IMU is
expressed in the FLU (Forward-Left-Up) body frame, matching the convention of
``imu_lm/data/nymeria_wrist_loader_FLU.py``.

FLU frame definition (T-pose, right wrist, geometric arm-chain frame)
----------------------------------------------------------------------
_build_wrist_rotation builds a frame from shoulder→elbow→wrist positions:

    col 0 (x_geom) = normalize(z × upper_arm)  — lateral (≈ thumb side)
    col 1 (y_geom) = normalize(z × x)           — dorsal  (out of watch face)
    col 2 (z_geom) = normalize(wrist − elbow)   — toward fingertips

The FLU mapping from this geometric frame:

    Forward (FLU x)  =  +z_geom  (toward fingertips, away from arm)
    Left    (FLU y)  =  +x_geom  (toward thumb side of right wrist)
    Up      (FLU z)  =  +y_geom  (out of watch face, dorsal, away from skin)

Note: this is a DIFFERENT R_WRIST_FROM_FLU than nymeria_wrist_loader_FLU.py,
which uses the SMPL FK joint frame (x=proximal, y=dorsal, z=thumb).

IMU derivation pipeline (per frame, world frame → FLU)
-------------------------------------------------------
1.  Resample HumanML3D joint positions from 20 Hz → 50 Hz.
2.  Build wrist rotation matrix R_world_from_wrist(t) from arm-chain geometry
    at every frame (shoulder → elbow → wrist joint positions).
3.  Propagate FLU frame to world:
        R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
4.  Compute world-frame linear acceleration of the wrist joint:
        a_world(t) = d²p_wrist / dt² × fps²
5.  Compute angular velocity in the wrist local frame via SO3 log-map, then
    rotate to world:
        ω_world(t) = R_world_from_wrist(t) @ ω_wrist_local(t)
6.  Project both into FLU:
        acc_flu(t)  = R_X_from_world(t) @ a_world(t)
        gyro_flu(t) = R_X_from_world(t) @ ω_world(t)

This is the **exact same pipeline** as ``nymeria_wrist_loader_FLU.py``; the
only difference is that R_world_from_wrist is estimated from joint positions
instead of SMPL FK rotation matrices.

Exported symbols
----------------
- ``_joints_to_imu_flu``            core IMU derivation (FLU frame)
- ``HumanML3DIMUDatasetFLU``        map-style Dataset
- ``HumanML3DIMUVarLenDatasetFLU``  iterable (variable T) Dataset
- ``make_humanml3d_imu_loaders_flu`` DataLoader factory
- ``imu_text_collate_fn``           (re-exported from base module)
"""

from __future__ import annotations

import logging
import os
import random as pyrandom
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from os.path import join as pjoin
from tqdm import tqdm

from imu_lm.utils.helpers import cfg_get
from imu_lm.data.augmentations.preprocess import filter_window

# Re-export helpers that are shared with the base module
from imu_lm.data.humanml3d_dataset import (
    _resample_joints,
    _snap_T,
    imu_text_collate_fn,
    _HML3D_FPS,
    _TARGET_FPS,
    _R_SHOULDER,
    _R_ELBOW,
    _R_WRIST,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FLU frame definition — SMPL FK joint frame (same as nymeria_wrist_loader_FLU.py)
# ---------------------------------------------------------------------------
# R_world_from_wrist(t) is taken from SMPL FK pose_global[:, 21].
# In the SMPL T-pose the right-wrist local axes are:
#   x_smpl  = proximal direction (toward elbow)
#   y_smpl  = dorsal  (out of watch face)
#   z_smpl  = thumb side (lateral)
#
# FLU mapping (identical constants to nymeria_wrist_loader_FLU.py):
#   F (forward / fingertips) = −x_smpl
#   L (left    / thumb side) = +z_smpl
#   U (up      / watch face) = +y_smpl
#
# Columns of R_WRIST_FROM_FLU are the F, L, U axes in SMPL wrist-local coords:
R_WRIST_FROM_FLU: np.ndarray = np.array(
    [[-1.0,  0.0,  0.0],   # F
     [ 0.0,  0.0,  1.0],   # L
     [ 0.0,  1.0,  0.0]],  # U
    dtype=np.float64,
)

assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)

# SMPL right wrist joint index (same as nymeria_wrist_loader_FLU.py)
_SMPL_R_WRIST = 21


# ---------------------------------------------------------------------------
# Geometric arm-chain helper (kept for external / visualisation use)
# ---------------------------------------------------------------------------

def _build_wrist_rotation(
    shoulder: np.ndarray,   # [N, 3]
    elbow:    np.ndarray,   # [N, 3]
    wrist:    np.ndarray,   # [N, 3]
) -> np.ndarray:
    """Geometric wrist frame from arm-chain joint positions [N, 3, 3].

    Not used in the FLU IMU derivation pipeline (which uses SMPL FK rotation
    matrices from _load_wrist_rotmat_20hz). Kept for external / visualisation use.
    """
    eps = 1e-8
    z = wrist - elbow
    z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + eps)
    upper = elbow - shoulder
    upper = upper / (np.linalg.norm(upper, axis=-1, keepdims=True) + eps)
    x = np.cross(z, upper)
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y, axis=-1, keepdims=True) + eps)
    return np.stack([x, y, z], axis=-1).astype(np.float64)


# ---------------------------------------------------------------------------
# SMPL FK helpers — load wrist rotation matrices from AMASS data
# ---------------------------------------------------------------------------

def _rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping unit vector a → b (Rodrigues formula)."""
    a = a / np.linalg.norm(a);  b = b / np.linalg.norm(b)
    v = np.cross(a, b);  s = np.linalg.norm(v);  c = float(a @ b)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1., -1., -1.])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)


# Module-level AMASS clip index cache  {root: {"clip.npy": (npz_path, s, e)}}
_amass_index_cache: Dict[str, Dict[str, Tuple[str, int, int]]] = {}


def _get_amass_index(root: str) -> Dict[str, Tuple[str, int, int]]:
    """Build (once per process) and return the AMASS clip index for this root."""
    if root in _amass_index_cache:
        return _amass_index_cache[root]
    index: Dict[str, Tuple[str, int, int]] = {}
    index_csv = pjoin(root, "index.csv")
    if os.path.exists(index_csv):
        with open(index_csv) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) != 4:
                    continue
                src, s, e, name = parts
                try:
                    s, e = int(s), int(e)
                except ValueError:
                    continue
                pose_npy  = pjoin(root, src.lstrip("./ "))
                amass_npz = (pose_npy
                             .replace("/pose_data/", "/amass_data/")
                             .replace(".npy", ".npz"))
                index[name] = (amass_npz, s, e)
    _amass_index_cache[root] = index
    return index


# Lazy SMPL body model singleton (loaded once per process / DataLoader worker)
_smpl_body_model = None


def _get_smpl_body_model():
    global _smpl_body_model
    if _smpl_body_model is None:
        import torch
        import core.articulate as art
        from core.paths import Paths
        _smpl_body_model = art.model.ParametricModel(
            str(Paths.SMPL_FILE), device=torch.device("cpu")
        )
    return _smpl_body_model


def _resample_rotmat(R: np.ndarray, N_out: int) -> np.ndarray:
    """Resample [N_in, 3, 3] rotation matrices to [N_out, 3, 3] via SLERP."""
    from scipy.spatial.transform import Rotation as ScipyRot, Slerp
    N_in = R.shape[0]
    if N_in == N_out:
        return R
    t_in  = np.linspace(0.0, 1.0, N_in)
    t_out = np.linspace(0.0, 1.0, N_out)
    return Slerp(t_in, ScipyRot.from_matrix(R))(t_out).as_matrix().astype(np.float64)


def _load_wrist_rotmat_20hz(
    amass_path: str,
    start_f:    int,
    end_f:      int,
    target_N:   int,
) -> Optional[np.ndarray]:
    """Run SMPL FK on an AMASS clip; return canonical wrist rotation [N, 3, 3] at 20 Hz.

    Pipeline (mirrors nymeria_wrist_loader_FLU.py steps 2-4):
      1. Load AMASS poses, resample to 20 Hz, slice to [start_f : end_f].
      2. SMPL FK (no root translation) → pose_global[:, 21].
      3. Apply canonical Y-rotation (face +Z) matching HumanML3D.
      4. Return [min(N_amass, target_N), 3, 3].
    """
    if not os.path.exists(amass_path):
        return None
    try:
        d = np.load(amass_path)
    except Exception:
        return None

    import torch
    from core.articulate.math import axis_angle_to_rotation_matrix

    a_fps   = float(d["mocap_framerate"])
    stride  = max(1, int(round(a_fps / _HML3D_FPS)))
    poses20 = d["poses"][::stride]
    end     = end_f if end_f >= 0 else len(poses20)
    poses_c = poses20[start_f:end].astype(np.float32)
    N       = min(poses_c.shape[0], target_N)
    if N < 3:
        return None
    poses_c = poses_c[:N]

    body    = _get_smpl_body_model()
    pose_aa = torch.from_numpy(poses_c[:, :72].reshape(N, 24, 3))
    R_local = axis_angle_to_rotation_matrix(pose_aa.reshape(-1, 3)).reshape(N, 24, 3, 3)

    # Global rotations from FK
    R_global = body.forward_kinematics_R(R_local).detach().cpu().numpy()   # [N, 24, 3, 3]

    # Canonical Y-rotation: make person face +Z at frame 0 (same as HumanML3D)
    _, jnt_t = body.forward_kinematics_masked(R_local, tran=None, calc_mesh=False)
    jnt = jnt_t.detach().cpu().numpy()   # [N, 24, 3]
    r_hip = jnt[0, 2];  l_hip = jnt[0, 1]
    r_sdr = jnt[0, 17]; l_sdr = jnt[0, 16]
    across  = (r_hip - l_hip) + (r_sdr - l_sdr);  across[1] = 0.
    forward = np.cross(np.array([0., 1., 0.]), across)
    R_y     = _rot_a_to_b(forward, np.array([0., 0., 1.]))

    # R_canon = R_y @ R_world_from_wrist
    R_wrist = R_global[:, _SMPL_R_WRIST]                   # [N, 3, 3]
    return np.einsum("ij,njk->nik", R_y, R_wrist)          # [N, 3, 3]


def _load_all_joint_rotmat_20hz(
    amass_path: str,
    start_f:    int,
    end_f:      int,
    target_N:   int,
    num_joints: int = 22,
) -> Optional[np.ndarray]:
    """Same pipeline as ``_load_wrist_rotmat_20hz`` but returns canonical
    rotations for ALL ``num_joints`` SMPL joints (raw SMPL/AMASS joint order:
    0=pelvis, 1=L_hip, 2=R_hip, 3=spine1, ..., 21=R_wrist) instead of just
    joint 21. ``body.forward_kinematics_R`` already computes all 24 joints'
    global rotations internally -- this function differs from
    ``_load_wrist_rotmat_20hz`` only in *not* discarding the other 21 before
    returning.

    Returns [N, num_joints, 3, 3] or None if the clip is invalid / too short.
    """
    if not os.path.exists(amass_path):
        return None
    try:
        d = np.load(amass_path)
    except Exception:
        return None

    import torch
    from core.articulate.math import axis_angle_to_rotation_matrix

    a_fps   = float(d["mocap_framerate"])
    stride  = max(1, int(round(a_fps / _HML3D_FPS)))
    poses20 = d["poses"][::stride]
    end     = end_f if end_f >= 0 else len(poses20)
    poses_c = poses20[start_f:end].astype(np.float32)
    N       = min(poses_c.shape[0], target_N)
    if N < 3:
        return None
    poses_c = poses_c[:N]

    body    = _get_smpl_body_model()
    pose_aa = torch.from_numpy(poses_c[:, :72].reshape(N, 24, 3))
    R_local = axis_angle_to_rotation_matrix(pose_aa.reshape(-1, 3)).reshape(N, 24, 3, 3)

    # Global rotations from FK
    R_global = body.forward_kinematics_R(R_local).detach().cpu().numpy()   # [N, 24, 3, 3]

    # Canonical Y-rotation: make person face +Z at frame 0 (same as HumanML3D)
    _, jnt_t = body.forward_kinematics_masked(R_local, tran=None, calc_mesh=False)
    jnt = jnt_t.detach().cpu().numpy()   # [N, 24, 3]
    r_hip = jnt[0, 2];  l_hip = jnt[0, 1]
    r_sdr = jnt[0, 17]; l_sdr = jnt[0, 16]
    across  = (r_hip - l_hip) + (r_sdr - l_sdr);  across[1] = 0.
    forward = np.cross(np.array([0., 1., 0.]), across)
    R_y     = _rot_a_to_b(forward, np.array([0., 0., 1.]))

    # R_canon[:, j] = R_y @ R_world_from_joint[:, j] for every joint j
    R_joints = R_global[:, :num_joints]                        # [N, J, 3, 3]
    return np.einsum("ij,npjk->npik", R_y, R_joints)           # [N, J, 3, 3]


# ---------------------------------------------------------------------------
# All-joint rotation from HumanML3D's own new_joint_vecs (no AMASS needed)
# ---------------------------------------------------------------------------
#
# HumanML3D's official post-processing (motion_representation.ipynb) derives
# a 263-dim per-frame feature vector (new_joint_vecs/<id>.npy) via Inverse
# Kinematics run directly on joint POSITIONS (not on AMASS pose parameters).
# It covers ALL 29,228 mirrored clips -- vs. only ~8,246/23,384 (35%) reached
# by the AMASS-archive-index path above, since this cluster's local AMASS
# mirror is incomplete (see EXPERIMENTS.md). Joint numbering matches
# standard SMPL exactly (verified against paramUtil.py's t2m_kinematic_chain
# -- e.g. chain [9,14,17,19,21] = spine3->R_collar->R_shoulder->R_elbow->
# R_wrist, matching _SMPL_R_WRIST=21).
#
# Fidelity caveat (real, not hidden): each non-root joint's local rotation
# comes from qbetween(rest_offset, current_offset) -- the minimal rotation
# aligning two vectors. That recovers 2 of 3 rotational DOF (bending
# direction) but NOT twist/roll about the bone's own long axis, since a
# vector-to-vector alignment is fundamentally underdetermined there. The
# AMASS path above uses genuine SMPL pose parameters (captures true twist);
# this path trades that fidelity for ~3.5x more training clips.

_T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]


def _cont6d_to_matrix(cont6d: np.ndarray) -> np.ndarray:
    """[..., 6] continuous 6D rotation (Zhou et al.) -> [..., 3, 3] matrix."""
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw, axis=-1)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.stack([x, y, z], axis=-1)  # columns (x, y, z)


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """[..., 4] quaternion (w, x, y, z) -> [..., 3, 3] rotation matrix."""
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / (q * q).sum(-1)
    o = np.stack(
        [
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def _load_all_joint_rotmat_from_vecs(
    new_joint_vec_path: str,
    target_N:           int,
    num_joints:          int = 22,
    root_rotmat_path:    Optional[str] = None,
) -> Optional[np.ndarray]:
    """Reconstruct canonical per-joint global rotations from HumanML3D's own
    new_joint_vecs/<id>.npy (263-dim feature vector), no AMASS required.

    Pipeline (mirrors motion_representation.ipynb's recover_root_rot_pos /
    recover_from_rot exactly):
      1. Root Y-axis rotation: cumulative sum of stored rotation velocity
         (data[:,0]) -> quaternion [cos, 0, sin, 0]. If root_rotmat_path is
         given, the root rotation is instead read directly from that file
         (root_rotations_3d/<id>.npy, [T,3,3]) rather than re-derived here --
         verified numerically identical (max abs diff 2.1e-7) on spot
         checks, so this is a data-provenance choice, not a behavior change.
      2. Non-root local rotations: rot_data block (data[:,67:193],
         21 joints x 6D, natural joint-index order 1..21) -> matrices, via
         the SMPL/T2M kinematic chain (same joint topology as AMASS/SMPL).
      3. Compose local -> global along _T2M_KINEMATIC_CHAIN.

    Both new_joints/<id>.npy (positions) and this function's rotation output
    share the same per-frame canonical alignment (HumanML3D's own "face +Z
    at frame 0" convention baked into new_joint_vecs), so they're frame-
    consistent with each other by construction.

    Returns [N, num_joints, 3, 3] or None if the file is invalid / too short.
    """
    if not os.path.exists(new_joint_vec_path):
        return None
    try:
        data = np.load(new_joint_vec_path).astype(np.float64)  # [T, 263]
    except Exception:
        return None

    N = min(data.shape[0], target_N)
    if N < 3:
        return None
    data = data[:N]

    if root_rotmat_path is not None:
        if not os.path.exists(root_rotmat_path):
            return None
        try:
            R_root = np.load(root_rotmat_path).astype(np.float64)[:N]  # [N, 3, 3]
        except Exception:
            return None
        if R_root.shape[0] < N:
            return None
    else:
        # Root Y-axis rotation, reconstructed by cumulative integration.
        rot_vel          = data[:, 0]
        r_rot_ang         = np.zeros(N, dtype=np.float64)
        r_rot_ang[1:]     = rot_vel[:-1]
        r_rot_ang         = np.cumsum(r_rot_ang)
        r_rot_quat        = np.zeros((N, 4), dtype=np.float64)
        r_rot_quat[:, 0]  = np.cos(r_rot_ang)
        r_rot_quat[:, 2]  = np.sin(r_rot_ang)
        R_root = _quat_wxyz_to_matrix(r_rot_quat)  # [N, 3, 3]

    # Non-root local rotations (21 joints x 6D, natural joint-index order).
    start = 1 + 2 + 1 + (num_joints - 1) * 3
    end   = start + (num_joints - 1) * 6
    cont6d = data[:, start:end].reshape(N, num_joints - 1, 6)
    R_local_nonroot = _cont6d_to_matrix(cont6d)  # [N, 21, 3, 3]

    # Compose local -> global along the kinematic chain.
    R_global = np.zeros((N, num_joints, 3, 3), dtype=np.float64)
    R_global[:, 0] = R_root
    for chain in _T2M_KINEMATIC_CHAIN:
        R = R_root.copy()
        for i in range(1, len(chain)):
            j = chain[i]
            R_local_j = R_local_nonroot[:, j - 1]
            R = np.einsum("nij,njk->nik", R, R_local_j)
            R_global[:, j] = R

    return R_global[:, :num_joints]


# ---------------------------------------------------------------------------
# Core IMU derivation — FLU frame via SMPL FK (mirrors nymeria_wrist_loader_FLU.py)
# ---------------------------------------------------------------------------

def _joints_to_imu_flu(
    joints:        np.ndarray,   # [N, 22, 3]  canonical joint positions at target fps
    R_wrist_world: np.ndarray,   # [N, 3, 3]   SMPL FK wrist rotation at target fps
    fps:           float = _TARGET_FPS,
    gravity_mps2:  float = 0.0,
) -> np.ndarray:
    """Derive right-wrist IMU in the FLU body frame.

    Follows nymeria_wrist_loader_FLU.py exactly:

    1.  R_world_from_X(t) = R_wrist_world(t) @ R_WRIST_FROM_FLU
    2.  Linear acc from second derivative of canonical wrist position (from joints.npy).
    3.  Angular velocity: SO3 log of consecutive R_wrist_world, rotated to world.
    4.  Project both into FLU via R_X_from_world(t).

    Args:
        joints:        [N, 22, 3]  canonical joint positions (target fps)
        R_wrist_world: [N, 3, 3]   SMPL FK global wrist rotation (canonical, target fps)
        fps:           frame rate (default _TARGET_FPS = 50 Hz)

    gravity_mps2: added to the WORLD-frame +Y (up) axis of a_world before
        rotating into FLU, so the returned acc represents PROPER acceleration
        (what a real accelerometer measures: reaction force to gravity plus
        linear acceleration) instead of pure kinematic acceleration. Default
        0.0 preserves the original (gravity-free) behavior exactly. Debugging
        note (2026-07-25): HumanML3D's world frame is Y-up (confirmed
        empirically: joint 0 / pelvis sits at Y≈0.95 with low variance,
        X/Z show unconstrained horizontal translation). A stationary real
        accelerometer reads ~+9.81 m/s^2 (reaction force opposing gravity,
        which points -Y), so gravity_mps2=9.81 is the physically-correct
        value to test whether HumanML3D's synthetic acc -- which has never
        included this term -- was missing a genuine signal-STRUCTURE
        component (gravity's changing projection onto the wrist's own axes
        as it rotates) that per-window z-score normalization cannot restore
        (z-scoring only removes each window's own mean/DC offset, not this
        kind of correlated, rotation-dependent structure).

    Returns:
        [N, 6]  float32 — (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    N     = joints.shape[0]
    wrist = joints[:, _R_WRIST].astype(np.float64)   # [N, 3]

    # ── Step 1: FLU frame in world ──────────────────────────────────────────
    R_world_from_X = np.einsum("nij,jk->nik", R_wrist_world, R_WRIST_FROM_FLU)  # [N, 3, 3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                            # [N, 3, 3]

    # ── Step 2: world-frame linear acceleration of wrist joint ──────────────
    a_world = np.zeros((N, 3), dtype=np.float64)
    a_world[1:-1] = (wrist[2:] - 2.0 * wrist[1:-1] + wrist[:-2]) * (fps ** 2)
    a_world[0]    = a_world[1]
    a_world[-1]   = a_world[-2]

    if gravity_mps2:
        # Proper acceleration = kinematic acceleration - gravity_vector.
        # World is Y-up, so gravity points -Y -> subtracting it adds +Y.
        a_world[:, 1] += gravity_mps2

    # ── Step 3: angular velocity in wrist local frame → world ───────────────
    dR = np.einsum(
        "nij,njk->nik",
        R_wrist_world[:-1].transpose(0, 2, 1),
        R_wrist_world[1:],
    )  # [N-1, 3, 3]
    omega_wl       = np.zeros((N, 3), dtype=np.float64)
    omega_wl[:-1]  = (ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
                      .as_rotvec().astype(np.float64) * fps)
    omega_wl[-1]   = omega_wl[-2]
    omega_world    = np.einsum("nij,nj->ni", R_wrist_world, omega_wl)  # [N, 3]

    # ── Step 4: project into FLU ────────────────────────────────────────────
    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, a_world).astype(np.float32)  # a_world may include gravity_mps2
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, omega_world).astype(np.float32)

    return np.concatenate([acc_flu, gyro_flu], axis=1)   # [N, 6]


# ---------------------------------------------------------------------------
# Clip loader (FLU version)
# ---------------------------------------------------------------------------

def _load_clip_imu_unimts_sim(
    root:             str,
    clip_id:          str,
    patch_size:       int,
    T_min:            int,
    T_max:            int,
    resample_method:  str = 'cubic',
    captions_subpath: str = 'aug_texts',
    wrist_node:       int = _SMPL_R_WRIST,
) -> Optional[Dict]:
    """Load one clip's pre-synthesized wrist IMU directly from the user-
    provided ``UniMTS_data/data/sim/<id>.npy`` export (2026-07-28), instead
    of deriving acc/gyro ourselves from joint positions + reconstructed
    rotation (the ``_load_clip_imu_flu`` default path).

    ``UniMTS_data/data/sim/<id>.npy`` is ``[T_20hz, 22, 6]`` float32, already
    in UniMTS's own node ordering (node 21 = right wrist, confirmed via
    ``UNIMTS_TO_SMPL22[21] == 21`` in ``prepare_humanml3d_alljoint.py`` and
    via this exact data already being consumed as-is by UniMTS's own
    unmodified ``pretrain.py``/``evaluate_custom.py --joint_list 21``
    elsewhere in this repo). Spot-checked numerically (clip 000000): T
    matches ``HumanML3D/new_joints/000000.npy`` exactly (116 frames, same
    20 Hz clock, same clip-id convention) and the wrist channel's
    acceleration norm averages ~12.7 m/s^2 (vs. ~10.0 m/s^2 at the
    near-stationary pelvis root) -- i.e. gravity is ALREADY baked into this
    signal, unlike our own ``_joints_to_imu_flu`` derivation which adds it
    separately via ``gravity_mps2``. This function does NOT add gravity_mps2
    -- doing so here would double-count it.

    Resampled 20 Hz -> 50 Hz directly on the 6-channel acc+gyro signal
    itself (reusing ``_resample_joints``, which interpolates generically
    along axis 0 regardless of trailing shape) -- a more principled
    upsampling than our own position-then-finite-difference pipeline, since
    acc/gyro are already physical quantities here, not positions that still
    need differentiating.

    Returns the same ``{'imu': [T,6] float32, 'T': int, 'caps': list[str]}``
    shape as ``_load_clip_imu_flu``, or None if the clip is invalid/missing.
    """
    unimts_root = pjoin(root, 'UniMTS_data', 'data')
    sim_path    = pjoin(unimts_root, 'sim', f'{clip_id}.npy')
    # Captions come EXCLUSIVELY from UniMTS_data/data/<captions_subpath>
    # (aug_texts), and clips without a caption file there are DROPPED --
    # matching UniMTS's own CLIPDataset exactly (data.py takes the
    # intersection of sim/ and aug_texts/ listings; no fallback source).
    # BUG FIX history (2026-07-28, found in logic review): the first version
    # of this function "fell back" to a path inside UniMTS_data/data that
    # does not exist, so the ~15.5% of train clips (3,629/23,380) and ~18%
    # of val clips missing an aug_texts file silently trained/validated
    # against an EMPTY caption '' -- in the InfoNCE loss every such sample
    # shares one identical text embedding, so the multi-positive logic pulled
    # 15% of unrelated motions toward one arbitrary anchor. Job 8129177 was
    # affected. A brief intermediate fix fell back to the HumanML3D root's
    # texts/ (full coverage) but was superseded before any training ran by
    # the user's explicit direction to use UniMTS_data/data/aug_texts --
    # hence the drop-if-missing semantics (UniMTS's own convention).
    text_path = pjoin(unimts_root, captions_subpath, f'{clip_id}.txt')
    if not os.path.exists(text_path):
        return None

    if not os.path.exists(sim_path):
        return None

    try:
        sim = np.load(sim_path).astype(np.float32)   # [T_20hz, 22, 6]
    except Exception:
        return None
    if sim.ndim != 3 or sim.shape[0] < 2 or sim.shape[1] <= wrist_node:
        return None

    wrist_20hz = sim[:, wrist_node, :]                # [T_20hz, 6]
    wrist_50hz = _resample_joints(wrist_20hz, method=resample_method)  # [N, 6]
    N_raw      = wrist_50hz.shape[0]

    T = _snap_T(N_raw, patch_size, T_min, T_max)
    if T is None:
        return None

    imu = wrist_50hz[:T]                              # [T, 6]  already acc+gyro, gravity-inclusive

    captions: List[str] = []
    with open(text_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('#')
            if parts and parts[0]:
                captions.append(parts[0])
    if not captions:
        # A caption file that exists but parses to nothing is dropped, same
        # as a missing one -- never train against an empty '' caption (see
        # the bug-fix note above).
        return None

    return {'imu': imu, 'T': T, 'caps': captions}


def _load_clip_imu_flu(
    root:       str,
    clip_id:    str,
    joints_num: int,
    patch_size: int,
    T_min:      int,
    T_max:      int,
    gravity_mps2: float = 0.0,
    resample_method: str = 'linear',
    joints_subpath: str = 'joints',
    rotation_source: str = 'amass',
    captions_subpath: str = 'texts',
    data_source: str = 'derived',
) -> Optional[Dict]:
    """Load one HumanML3D clip and derive 50 Hz FLU IMU.

    Returns a dict::

        {'imu':  np.ndarray [N, 6]  float32   — FLU frame
         'T':    int                            actual length (multiple of patch_size)
         'caps': list[str]}

    or None if the clip is invalid / too short.

    joints_subpath: relative path (under root) to the joint-position .npy
        files, default 'joints' (the original per-clip [T,52,3] arrays).
        Pass 'HumanML3D/new_joints' to use HumanML3D's own official
        post-processed positions instead (2026-07-26: this cluster's local
        'joints/' mirror is incomplete/shrinking -- new_joints/ has full
        ~29,228-clip coverage and is confirmed high-fidelity, 0.9998
        correlation against independently-computed AMASS-FK positions on
        spot checks -- see EXPERIMENTS.md).

    rotation_source: 'amass' (default) reads true SMPL pose parameters from
        the local AMASS mocap mirror (_load_wrist_rotmat_20hz) -- physically
        complete 3-DOF rotation but only ~35% clip coverage on this cluster
        (incomplete local AMASS archive). 'new_joint_vecs' instead
        reconstructs rotation from HumanML3D's own new_joint_vecs/<id>.npy
        (position-only inverse kinematics, via
        _load_all_joint_rotmat_from_vecs) -- full ~29,228-clip coverage, no
        AMASS dependency at all, but a median 56deg wrist-rotation fidelity
        gap vs. AMASS ground truth (position-only IK cannot recover twist
        about the bone's own axis, see EXPERIMENTS.md 2026-07-26). Explicit
        user choice (2026-07-26): accept this fidelity cost for full
        coverage.

    captions_subpath: relative path (under root) to the caption .txt files,
        default 'texts' (HumanML3D's own original captions, 4 per clip
        typically). Pass 'aug_texts' to additionally draw from GPT-
        paraphrased caption variants (2026-07-26, user-provided directory,
        24,660/29,232 clip coverage) -- each aug_texts/<id>.txt contains the
        SAME original '#'-delimited lines as texts/<id>.txt PLUS extra
        plain-sentence paraphrases appended (no '#' suffix; the existing
        `line.split('#')[0]` parsing below already handles both formats
        identically, no parsing change needed). Falls back to 'texts' for
        any clip missing an aug_texts file, so caption coverage never drops
        below the 'texts'-only baseline.

    data_source: 'derived' (default) uses the position+rotation pipeline
        above (joints_subpath/rotation_source). 'unimts_sim' instead reads
        pre-synthesized acc+gyro directly from the user-provided
        ``UniMTS_data/data/sim/<id>.npy`` export -- see
        ``_load_clip_imu_unimts_sim`` for details (2026-07-28).
    """
    if data_source == 'unimts_sim':
        return _load_clip_imu_unimts_sim(
            root, clip_id, patch_size, T_min, T_max,
            resample_method=resample_method,
            captions_subpath=captions_subpath,
        )

    joints_path = pjoin(root, joints_subpath, f'{clip_id}.npy')
    text_path   = pjoin(root, captions_subpath, f'{clip_id}.txt')
    if not os.path.exists(text_path):
        text_path = pjoin(root, 'texts', f'{clip_id}.txt')

    if not os.path.exists(joints_path):
        return None

    joints_raw  = np.load(joints_path, allow_pickle=True).astype(np.float32)  # [T_20hz, 52, 3]
    if joints_raw.ndim == 2:
        # Degenerate single-frame clip -- some new_joints/<id>.npy files drop
        # the leading time axis when T==1 (e.g. 000990.npy, shape [22,3] not
        # [1,22,3]). Restore it so downstream shape handling is uniform; the
        # existing "too short" validity checks below reject it naturally
        # (finite-difference acc/gyro need >=3 frames regardless).
        joints_raw = joints_raw[None]
    if joints_raw.ndim != 3:
        return None
    joints_raw  = joints_raw[:, :joints_num, :]          # [T_20hz, 22, 3]

    joints_50hz = _resample_joints(joints_raw, method=resample_method)  # [N, 22, 3]
    N_raw       = joints_50hz.shape[0]

    T = _snap_T(N_raw, patch_size, T_min, T_max)
    if T is None:
        return None

    joints_50hz = joints_50hz[:T]                        # [T, 22, 3]

    # ── wrist rotation at 20 Hz, source selectable ──────────────────────────
    if rotation_source == "new_joint_vecs":
        vecs_path = pjoin(root, "HumanML3D", "new_joint_vecs", f"{clip_id}.npy")
        root_rotmat_path = pjoin(root, "HumanML3D", "root_rotations_3d", f"{clip_id}.npy")
        R_all_20hz = _load_all_joint_rotmat_from_vecs(
            vecs_path, target_N=joints_raw.shape[0], num_joints=joints_num,
            root_rotmat_path=root_rotmat_path if os.path.exists(root_rotmat_path) else None,
        )
        R_wrist_20hz = R_all_20hz[:, _SMPL_R_WRIST] if R_all_20hz is not None else None
    else:
        amass_index = _get_amass_index(root)
        amass_entry = amass_index.get(f"{clip_id}.npy")
        if amass_entry is None:
            return None
        amass_path_clip, start_f, end_f = amass_entry
        R_wrist_20hz = _load_wrist_rotmat_20hz(
            amass_path_clip, start_f, end_f, target_N=joints_raw.shape[0]
        )
    if R_wrist_20hz is None:
        return None

    # Resample wrist rotation matrices 20 Hz → 50 Hz (SLERP), truncate to T
    R_wrist_50hz = _resample_rotmat(R_wrist_20hz, N_raw)[:T]   # [T, 3, 3]

    # ── FLU IMU derivation via SMPL FK (matches nymeria_wrist_loader_FLU.py) ─
    imu = _joints_to_imu_flu(joints_50hz, R_wrist_50hz, fps=_TARGET_FPS, gravity_mps2=gravity_mps2)  # [T, 6]

    # ── captions ─────────────────────────────────────────────────────────
    captions: List[str] = []
    if os.path.exists(text_path):
        with open(text_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('#')
                if parts and parts[0]:
                    captions.append(parts[0])
    if not captions:
        captions = ['']

    return {'imu': imu, 'T': T, 'caps': captions}


# ---------------------------------------------------------------------------
# All-joint IMU derivation (UniMTS-architecture from-scratch training)
# ---------------------------------------------------------------------------
#
# Unlike _joints_to_imu_flu (single right-wrist joint, output rotated into
# a fixed wrist-specific FLU convention via R_WRIST_FROM_FLU), this derives
# per-joint IMU for ALL 22 SMPL joints, each expressed in that joint's OWN
# canonical (Y-rotated) SMPL FK frame -- there is no single shared "FLU"
# definition across joints, so each sensor simply reports in its own local
# frame, matching how a real IMU reports regardless of mounting joint.

def _all_joints_to_imu(
    joints:       np.ndarray,   # [N, J, 3]  canonical joint positions at target fps
    R_all_world:  np.ndarray,   # [N, J, 3, 3]  canonical per-joint SMPL FK rotation
    fps:          float,
    gravity_mps2: float = 0.0,
) -> np.ndarray:
    """Derive per-joint local-frame IMU (acc+gyro) for all J joints.

    For joint j: acc_j(t) is the world-frame linear acceleration of that
    joint's position (+ optional gravity reaction term) expressed in the
    joint's own local frame; gyro_j(t) is the local-frame angular velocity
    from the SO3 log of consecutive R_all_world[:, j] rotations (no extra
    world round-trip needed since R[t]^T @ R[t+1] is already the local-frame
    delta rotation).

    Returns [N, J, 6] float32 -- (acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z) per
    joint, each in that joint's own canonical SMPL FK frame.
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    N, J = joints.shape[0], joints.shape[1]
    pos  = joints.astype(np.float64)         # [N, J, 3]
    R    = R_all_world.astype(np.float64)    # [N, J, 3, 3]

    # ── world-frame linear acceleration, central difference ─────────────────
    a_world = np.zeros_like(pos)
    a_world[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    a_world[0]    = a_world[1]
    a_world[-1]   = a_world[-2]
    if gravity_mps2:
        # World is Y-up; a stationary sensor reads +gravity_mps2 (reaction
        # force), same convention as _joints_to_imu_flu.
        a_world[:, :, 1] += gravity_mps2

    # ── local-frame angular velocity: SO3 log of R[t]^T @ R[t+1] ────────────
    dR = np.einsum(
        "njik,njkl->njil",
        R[:-1].transpose(0, 1, 3, 2),
        R[1:],
    )  # [N-1, J, 3, 3]
    omega_local      = np.zeros((N, J, 3), dtype=np.float64)
    omega_local[:-1] = (ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
                         .as_rotvec().reshape(N - 1, J, 3) * fps)
    omega_local[-1]  = omega_local[-2]

    # ── project world-frame linear acceleration into each joint's own frame ─
    R_local_from_world = R.transpose(0, 1, 3, 2)                              # [N, J, 3, 3]
    acc_local = np.einsum("njik,njk->nji", R_local_from_world, a_world)       # [N, J, 3]

    return np.concatenate([acc_local, omega_local], axis=-1).astype(np.float32)  # [N, J, 6]


def _load_clip_imu_alljoint(
    root:         str,
    clip_id:      str,
    num_joints:   int = 22,
    gravity_mps2: float = 0.0,
    joints_subpath: str = 'joints',
    rotation_source: str = 'amass',
    captions_subpath: str = 'texts',
) -> Optional[Dict]:
    """Load one HumanML3D clip and derive all-joint IMU at native 20 Hz.

    Unlike _load_clip_imu_flu (which resamples to 50 Hz to match this
    benchmark's other data sources), this stays at HumanML3D/AMASS's native
    20 Hz throughout -- UniMTS's own architecture and pretraining recipe
    work natively at 20 Hz, so no resampling round-trip is needed.

    joints_subpath / rotation_source / captions_subpath: same meaning and
    same 2026-07-26 options as _load_clip_imu_flu -- 'HumanML3D/new_joints'
    for positions, 'new_joint_vecs' for AMASS-free rotation (root read from
    HumanML3D/root_rotations_3d when present, numerically identical to the
    internal re-derivation), 'aug_texts' for GPT-paraphrased captions. When
    joints_subpath is NOT the top-level 'joints' default, the array is
    assumed already [T, 22, 3] (HumanML3D/new_joints' own shape) rather
    than the original [T, 52, 3] -- no extra joints beyond num_joints to
    slice off.

    Returns a dict::

        {'imu':  np.ndarray [N, num_joints, 6]  float32 -- raw SMPL joint
                 order (0=pelvis, ..., 21=R_wrist), each channel in that
                 joint's own local frame
         'N':    int  actual length
         'caps': list[str]}

    or None if the clip is invalid / too short.
    """
    joints_path = pjoin(root, joints_subpath, f'{clip_id}.npy')
    text_path   = pjoin(root, captions_subpath, f'{clip_id}.txt')
    if not os.path.exists(text_path):
        text_path = pjoin(root, 'texts', f'{clip_id}.txt')

    if not os.path.exists(joints_path):
        return None

    joints_raw = np.load(joints_path, allow_pickle=True).astype(np.float32)  # [N_20hz, 52-or-22, 3]
    if joints_raw.ndim == 2:
        joints_raw = joints_raw[None]
    if joints_raw.ndim != 3:
        return None
    joints_raw = joints_raw[:, :num_joints, :]                                # [N_20hz, J, 3]
    if joints_raw.shape[0] < 3:
        return None

    if rotation_source == "new_joint_vecs":
        vecs_path = pjoin(root, "HumanML3D", "new_joint_vecs", f"{clip_id}.npy")
        root_rotmat_path = pjoin(root, "HumanML3D", "root_rotations_3d", f"{clip_id}.npy")
        R_all_20hz = _load_all_joint_rotmat_from_vecs(
            vecs_path, target_N=joints_raw.shape[0], num_joints=num_joints,
            root_rotmat_path=root_rotmat_path if os.path.exists(root_rotmat_path) else None,
        )
    else:
        amass_index = _get_amass_index(root)
        amass_entry = amass_index.get(f"{clip_id}.npy")
        if amass_entry is None:
            return None
        amass_path_clip, start_f, end_f = amass_entry
        R_all_20hz = _load_all_joint_rotmat_20hz(
            amass_path_clip, start_f, end_f,
            target_N=joints_raw.shape[0], num_joints=num_joints,
        )
    if R_all_20hz is None:
        return None

    N = min(joints_raw.shape[0], R_all_20hz.shape[0])
    if N < 3:
        return None
    joints_raw = joints_raw[:N]
    R_all_20hz = R_all_20hz[:N]

    imu = _all_joints_to_imu(joints_raw, R_all_20hz, fps=_HML3D_FPS, gravity_mps2=gravity_mps2)  # [N, J, 6]

    captions: List[str] = []
    if os.path.exists(text_path):
        with open(text_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('#')
                if parts and parts[0]:
                    captions.append(parts[0])
    if not captions:
        captions = ['']

    return {'imu': imu, 'N': N, 'caps': captions}


# ---------------------------------------------------------------------------
# Map-style Dataset
# ---------------------------------------------------------------------------

class HumanML3DIMUDatasetFLU(Dataset):
    """Map-style Dataset: one item per HumanML3D clip, FLU wrist IMU.

    Drop-in replacement for ``HumanML3DIMUDataset`` with FLU-frame IMU.

    Returns ``(x, caption)`` where::

        x:       torch.Tensor [6, T]   float32  — FLU acc (3) + FLU gyro (3), 50 Hz
        caption: str                             — one randomly selected annotation

    Args:
        root:        HumanML3D workspace root.
        split:       ``'train'``, ``'val'``, ``'test'``, etc.
        patch_size:  ViT1D patch size (T must be a multiple; default 16).
        T_min:       Minimum T at 50 Hz (default 64 = 1.28 s).
        T_max:       Maximum T at 50 Hz (default 512 = 10.24 s).
        joints_num:  Leading SMPL joints to keep (default 22).
        seed:        RNG seed for caption sampling (default 42).
    """

    def __init__(
        self,
        root:       str,
        split:      str = 'train',
        patch_size: int = 16,
        T_min:      int = 64,
        T_max:      int = 512,
        joints_num: int = 22,
        seed:       int = 42,
        cfg:        Optional[Any] = None,
        split_name: str = "train",
    ):
        self.patch_size = patch_size
        self.T_min      = T_min
        self.T_max      = T_max
        self.rng        = pyrandom.Random(seed)
        # Optional: apply the shared random_rotation/gain_jitter/etc. augmentation
        # catalogue (imu_lm.data.augmentations.transform.apply_augment) so this
        # source is treated consistently with every other loader when combined
        # into one dataset (see imu_lm.data.combined_mae_multidata_loader). None
        # (default) preserves the exact prior behavior for existing callers
        # (e.g. imu_text_clip_flu) that don't pass cfg.
        self.cfg = cfg
        # Canonical split ("train"/"val"), used only to gate augmentation.
        # Passed separately from `split` because `split` is the backing .txt
        # FILE name, which may be a derived/capped variant like
        # "train_capped_4000" (see _capped_split_name) -- string-matching on it
        # would be fragile. Bug fixed 2026-07-28: augmentation used to be
        # applied whenever cfg was not None, i.e. to the val split too.
        self.split_name = split_name

        split_file = pjoin(root, 'HumanML3D', f'{split}.txt')
        with open(split_file) as f:
            all_ids = [l.strip() for l in f if l.strip()]

        self._clips: List[Dict] = []
        skipped = 0
        # Bug fix (2026-07-28): this call used to omit joints_subpath/
        # rotation_source/captions_subpath/data_source entirely, silently
        # falling back to _load_clip_imu_flu's defaults (joints_subpath=
        # 'joints', now removed from the cluster -- see the matching fix in
        # HumanML3DIMUVarLenDatasetFLU.__init__ above). Threading these
        # through from cfg here too so MAE Stage-1 pretraining (which uses
        # this map-style class via combined_mae_multidata_loader.py) picks
        # up the same humanml3d.* config block as Stage-2 CLIP-align.
        for clip_id in tqdm(all_ids, desc=f"Loading {split} clips (HumanML3D wrist FLU)",
                             unit="clip", dynamic_ncols=True):
            clip = _load_clip_imu_flu(
                root, clip_id, joints_num, patch_size, T_min, T_max,
                gravity_mps2=cfg_get(cfg, ["humanml3d", "gravity_mps2"], 0.0),
                resample_method=cfg_get(cfg, ["humanml3d", "resample_method"], "linear"),
                joints_subpath=cfg_get(cfg, ["humanml3d", "joints_subpath"], "joints"),
                rotation_source=cfg_get(cfg, ["humanml3d", "rotation_source"], "amass"),
                captions_subpath=cfg_get(cfg, ["humanml3d", "captions_subpath"], "texts"),
                data_source=cfg_get(cfg, ["humanml3d", "data_source"], "derived"),
            )
            if clip is None:
                skipped += 1
                continue
            clip['id'] = clip_id
            self._clips.append(clip)

        logger.info(
            '[HumanML3DIMUDatasetFLU] split=%s  loaded=%d  skipped=%d',
            split, len(self._clips), skipped,
        )

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        clip    = self._clips[idx]
        imu     = clip['imu']   # [T, 6]  FLU frame
        caption = self.rng.choice(clip['caps'])

        # Bug fix (2026-07-25): every real-dataset loader (WindowDataset/
        # RealdataCLIPDataset via preprocess_window) applies preprocess.filter
        # (lowpass, enabled by base.yaml's own default) before z-scoring --
        # HumanML3D's own loader never did, a pure omission. Confirmed
        # empirically this matters: HumanML3D's synthetic acc (2nd-order
        # finite difference of joint positions) has a MUCH higher high-
        # frequency "roughness" than real accelerometers (2nd-diff std ~13-21
        # vs ~1-2 on WISDM/PAMAP2) -- numerical differentiation amplifies any
        # jitter in the underlying position signal. Real target-dataset
        # windows always get this smoothed away at 24Hz before the encoder
        # ever sees them; HumanML3D windows never did.
        imu = filter_window(imu, self.cfg)

        # Bug fix (2026-07-28, user-caught): augment (random rotation) BEFORE
        # the z-score, in raw physical units -- see the detailed note in
        # HumanML3DIMUVarLenDatasetFLU.__iter__. Rotating after the per-
        # channel z-score is not a rotation of the physical signal (z-score
        # is anisotropic scaling) and leaves non-unit channel variances.
        if self.cfg is not None and self.split_name == "train":
            from imu_lm.data.augmentations.transform import apply_augment
            imu = apply_augment(imu, self.cfg)

        # Per-window z-score (identical to nymeria_wrist_loader_FLU.py)
        mean = imu.mean(axis=0, keepdims=True)
        std  = imu.std(axis=0,  keepdims=True) + 1e-8
        imu  = (imu - mean) / std

        x = torch.from_numpy(imu.T).float()   # [6, T]
        return x, caption


# ---------------------------------------------------------------------------
# Iterable variable-length Dataset
# ---------------------------------------------------------------------------

class HumanML3DIMUVarLenDatasetFLU(IterableDataset):
    """Infinite IterableDataset with variable T per batch — FLU wrist IMU.

    Drop-in replacement for ``HumanML3DIMUVarLenDataset`` with FLU-frame IMU.
    API is identical to ``NymeriaVarLenMAEDataset`` and
    ``HumanML3DIMUVarLenDataset``.

    Each batch yields::

        (x [B, 6, T],  texts List[str])           # without text_emb_cache
        (x [B, 6, T],  embs  FloatTensor [B, D])  # with    text_emb_cache

    T is sampled uniformly from multiples of ``patch_size`` in [T_min, T_max]
    at each batch boundary.

    Args:
        root:           HumanML3D workspace root.
        split:          ``'train'``, ``'val'``, etc.
        batch_size:     Samples per batch.
        patch_size:     ViT1D patch size; T is always a multiple (default 16).
        T_min:          Minimum T at 50 Hz (default 64 = 1.28 s).
        T_max:          Maximum T at 50 Hz (default 512 = 10.24 s).
        joints_num:     Leading SMPL joints (default 22).
        seed:           Base RNG seed.
        text_emb_cache: Optional pre-computed caption → embedding dict.
        pad_mode:       How to fill clips shorter than the sampled T:
                         ``'zero'`` (default, matches every existing run in
                         this benchmark) zero-fills the tail; ``'wrap'``
                         cyclically repeats the clip's own signal instead, so
                         every timestep carries real motion content (matches
                         UniMTS's ``load_custom_data`` wrap-pad convention,
                         see ``baselines/language_alignment/UniMTS/data.py``).
        cfg:            Optional full training cfg. When given, each sampled
                         window is passed through the shared
                         ``imu_lm.data.augmentations.transform.apply_augment``
                         catalogue (random_rotation/gaussian_noise/etc., same
                         top-level config blocks as every other loader in this
                         repo, e.g. ``vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml``).
                         ``None`` (default) preserves prior no-augmentation
                         behavior.
    """

    def __init__(
        self,
        root:           str,
        split:          str  = 'train',
        batch_size:     int  = 32,
        patch_size:     int  = 16,
        T_min:          int  = 64,
        T_max:          int  = 512,
        joints_num:     int  = 22,
        seed:           int  = 42,
        text_emb_cache: Optional[Dict[str, np.ndarray]] = None,
        pad_mode:       str  = 'zero',
        cfg:            Optional[Any] = None,
        augment:        bool = True,
    ):
        super().__init__()
        self.root           = root
        self.batch_size     = batch_size
        self.pad_mode       = pad_mode
        self.patch_size     = patch_size
        self.joints_num     = joints_num
        # Gate apply_augment (random_rotation/gaussian_noise/etc.) to the
        # train split only. Bug fix (2026-07-26): this class previously
        # applied it whenever cfg is not None regardless of split -- the val
        # loader (make_humanml3d_imu_loaders_flu) is built with the SAME cfg,
        # so every enabled augmentation (e.g. random_rotation in every
        # "anyrot" config used this whole session) was ALSO being applied to
        # val windows, injecting fresh per-call randomness into the val_loss
        # early stopping/best-checkpoint selection relies on being a stable,
        # comparable measurement. RealdataCLIPDataset already gated this
        # correctly (split_name == "train"); this class never did.
        self._augment       = augment
        self.T_min          = T_min
        self.T_max          = T_max
        self.seed           = seed
        self.cfg             = cfg
        self._text_emb_cache = text_emb_cache

        # Clip-validity floor passed to _load_clip_imu_flu / _snap_T (drops
        # clips with native T below this). Must NOT reuse T_min directly when
        # pad_mode='wrap': T_min also defines the per-batch sampled window's
        # lower bound, and if T_min==T_max (e.g. a fixed-window config), every
        # clip shorter than T_max gets dropped as "invalid" before it ever
        # reaches __iter__'s wrap-pad branch -- since HumanML3D's own clips
        # cap out at 10.0s (500 samples @ 50Hz), a T_min=T_max=512 (10.24s)
        # config drops 100% of clips this way, leaving zero valid batches and
        # an infinite retry spin in __iter__ (observed: job alive, zero
        # progress for 22h+). wrap-mode's whole point is to pad short clips
        # up to T rather than skip them, so its validity floor only needs to
        # exclude degenerate clips (< 1 patch), not enforce the sampled
        # window's own minimum.
        self._clip_min_T = patch_size if pad_mode == 'wrap' else T_min

        if text_emb_cache is not None:
            _sample_emb    = next(iter(text_emb_cache.values()))
            self._emb_dim  = int(_sample_emb.shape[0])
            self._zero_emb = np.zeros(self._emb_dim, dtype=np.float32)
            logger.info(
                '[HumanML3DIMUVarLenDatasetFLU] text_emb_cache: %d captions, dim=%d',
                len(text_emb_cache), self._emb_dim,
            )
        else:
            self._emb_dim  = None
            self._zero_emb = None

        self.allowed_Ts: List[int] = list(range(T_min, T_max + 1, patch_size))
        if not self.allowed_Ts:
            raise ValueError(f'No valid T in [{T_min}, {T_max}] with patch_size={patch_size}')

        split_file = pjoin(root, 'HumanML3D', f'{split}.txt')
        with open(split_file) as f:
            all_ids = [l.strip() for l in f if l.strip()]

        # Bug fix (2026-07-28): this used to hardcode 'joints' here regardless
        # of the configurable humanml3d.joints_subpath used by __iter__ below
        # -- silently restricting every joints_subpath='HumanML3D/new_joints'
        # config (the AMASS-free path, ~29,228-clip coverage) to whatever
        # subset of clips happened to still have a file under the legacy,
        # separately-degrading 'joints/' mirror (~35% coverage at best, now
        # 0 -- that directory has since been removed from the cluster
        # entirely, which is what surfaced this: every HumanML3DIMUVarLenDatasetFLU
        # config started hard-failing with "no joints/*.npy found", including
        # ones that never intended to touch 'joints/' at all).
        # data_source='unimts_sim' (2026-07-28) bypasses joints_subpath
        # entirely -- its clip validity gate is the INTERSECTION of
        # UniMTS_data/data/sim/<id>.npy and UniMTS_data/data/<captions_subpath>/
        # <id>.txt, exactly mirroring UniMTS's own CLIPDataset enumeration
        # (data.py: common_files = sim listing ∩ aug_texts listing; clips
        # without a caption file are dropped, no fallback). Keeping the
        # filter in sync with _load_clip_imu_unimts_sim's own drop-if-missing
        # logic keeps the logged clip count honest and avoids wasted
        # resampling retries in __iter__.
        _data_source = cfg_get(cfg, ["humanml3d", "data_source"], "derived")
        if _data_source == "unimts_sim":
            _caps_subpath = cfg_get(cfg, ["humanml3d", "captions_subpath"], "aug_texts")
            _filter_subpath = pjoin("UniMTS_data", "data", "sim")
            self._clip_ids: List[str] = [
                cid for cid in all_ids
                if os.path.exists(pjoin(root, _filter_subpath, f'{cid}.npy'))
                and os.path.exists(pjoin(root, "UniMTS_data", "data", _caps_subpath, f'{cid}.txt'))
            ]
        else:
            _filter_subpath = cfg_get(cfg, ["humanml3d", "joints_subpath"], "joints")
            self._clip_ids = [
                cid for cid in all_ids
                if os.path.exists(pjoin(root, _filter_subpath, f'{cid}.npy'))
            ]

        if not self._clip_ids:
            raise RuntimeError(
                f'HumanML3DIMUVarLenDatasetFLU: no {_filter_subpath}/*.npy found under {root}'
            )

        logger.info(
            '[HumanML3DIMUVarLenDatasetFLU] split=%s  clips=%d  '
            'T=[%d,%d]  bs=%d  (lazy, FLU frame)',
            split, len(self._clip_ids), T_min, T_max, batch_size,
        )

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (worker_info.id if worker_info is not None else 0)
        rng         = np.random.default_rng(worker_seed)
        py_rng      = pyrandom.Random(int(worker_seed))

        # Worker-local clip cache (loaded once, reused for life of worker)
        cache: Dict[str, Optional[Dict]] = {}

        gravity_mps2     = cfg_get(self.cfg, ["humanml3d", "gravity_mps2"], 0.0)
        resample_method  = cfg_get(self.cfg, ["humanml3d", "resample_method"], "linear")
        crop_mode        = cfg_get(self.cfg, ["humanml3d", "crop_mode"], "random")
        zscore           = bool(cfg_get(self.cfg, ["humanml3d", "zscore"], True))
        joints_subpath   = cfg_get(self.cfg, ["humanml3d", "joints_subpath"], "joints")
        rotation_source  = cfg_get(self.cfg, ["humanml3d", "rotation_source"], "amass")
        captions_subpath = cfg_get(self.cfg, ["humanml3d", "captions_subpath"], "texts")
        data_source      = cfg_get(self.cfg, ["humanml3d", "data_source"], "derived")

        def get_clip(cid: str) -> Optional[Dict]:
            if cid not in cache:
                cache[cid] = _load_clip_imu_flu(
                    self.root, cid, self.joints_num,
                    self.patch_size, self._clip_min_T, self.T_max,
                    gravity_mps2=gravity_mps2,
                    resample_method=resample_method,
                    joints_subpath=joints_subpath,
                    rotation_source=rotation_source,
                    captions_subpath=captions_subpath,
                    data_source=data_source,
                )
            return cache[cid]

        clip_ids = self._clip_ids

        apply_augment = None
        if self.cfg is not None and self._augment:
            from imu_lm.data.augmentations.transform import apply_augment as _apply_augment
            apply_augment = _apply_augment

        while True:
            T = int(rng.choice(self.allowed_Ts))

            x_list:    List[np.ndarray] = []
            text_list: List[str]        = []
            attempts = 0

            while len(x_list) < self.batch_size:
                attempts += 1
                if attempts > self.batch_size * 50:
                    break

                cid  = clip_ids[int(rng.integers(len(clip_ids)))]
                clip = get_clip(cid)
                if clip is None:
                    continue

                imu    = clip['imu']       # [clip_T, 6]  FLU
                clip_T = imu.shape[0]
                cap    = py_rng.choice(clip['caps'])

                if clip_T >= T:
                    max_start = clip_T - T
                    if crop_mode == "first":
                        # Matches UniMTS's own CLIPDataset.__getitem__
                        # (baselines/language_alignment/UniMTS/data.py):
                        # imu = imu[:padding_size] -- always the first T
                        # samples, deterministic, no per-epoch crop diversity.
                        start = 0
                    else:
                        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
                    window    = imu[start : start + T]    # [T, 6]

                    # Bug fix (2026-07-25): match every real-dataset loader's
                    # preprocess.filter step (lowpass, enabled by base.yaml's
                    # own default) -- see HumanML3DIMUDatasetFLU.__getitem__'s
                    # own fix for the full rationale (HumanML3D's synthetic
                    # acc has much higher high-frequency roughness than real
                    # accelerometers, and was never smoothed the way real
                    # target-dataset windows always are).
                    window = filter_window(window, self.cfg)

                    # Bug fix (2026-07-28, user-caught): augmentation (random
                    # rotation) must run BEFORE the per-window z-score, in raw
                    # physical units. A reoriented sensor produces R@a_raw(t)
                    # which the real pipeline then normalizes -- whereas
                    # rotating AFTER the per-channel z-score (the previous
                    # order, used by every prior "anyrot" checkpoint) is not
                    # a rotation of the physical signal at all: z-score is an
                    # anisotropic per-channel scaling, and a rotation
                    # conjugated by anisotropic scaling is no longer a
                    # rotation. It also left training inputs with non-unit
                    # channel variances (rotation of unit-variance correlated
                    # channels gives diag(R @ Sigma @ R^T) != 1) while eval
                    # windows are always exactly zero-mean/unit-variance --
                    # a train/eval moment mismatch. UniMTS's own augment_data()
                    # also rotates raw signals before its BatchNorm. Filter
                    # commutes with rotation (same linear filter per channel),
                    # so filter -> rotate -> zscore is the faithful order.
                    if apply_augment is not None:
                        window = apply_augment(window, self.cfg)

                    if zscore:
                        # Per-window z-score (same as nymeria_wrist_loader_FLU.py)
                        mean   = window.mean(axis=0, keepdims=True)
                        std    = window.std(axis=0,  keepdims=True) + 1e-8
                        window = (window - mean) / std
                else:
                    # Shorter than T: filter, [augment,] [normalise,] then pad
                    # to T (augment-before-zscore: see bug-fix note above;
                    # rotation also commutes with wrap/zero padding, so
                    # rotating the unpadded clip is equivalent and simpler).
                    imu = filter_window(imu, self.cfg)
                    if apply_augment is not None:
                        imu = apply_augment(imu, self.cfg)
                    if zscore:
                        mean   = imu.mean(axis=0, keepdims=True)
                        std    = imu.std(axis=0,  keepdims=True) + 1e-8
                        normed = (imu - mean) / std
                    else:
                        normed = imu
                    if self.pad_mode == 'wrap':
                        # Cyclically repeat the clip's own signal (matches
                        # UniMTS's load_custom_data) -- every timestep stays
                        # informative instead of diluting the mean-pooled
                        # representation with dead zero regions.
                        window = np.pad(normed, ((0, T - clip_T), (0, 0)), mode='wrap').astype(np.float32)
                    else:
                        window = np.zeros((T, 6), dtype=np.float32)
                        window[:clip_T] = normed

                # (augmentation now happens BEFORE the z-score above -- see
                # the 2026-07-28 bug-fix note; no augment call here anymore)

                x_list.append(window.T)   # [6, T]
                text_list.append(cap)

            if len(x_list) < self.batch_size:
                continue   # retry

            x_tensor = torch.from_numpy(np.stack(x_list, axis=0)).float()  # [B, 6, T]

            if self._text_emb_cache is not None:
                emb_list = [
                    self._text_emb_cache.get(cap, self._zero_emb)
                    for cap in text_list
                ]
                yield x_tensor, torch.from_numpy(np.stack(emb_list, axis=0)).float()
            else:
                yield x_tensor, text_list


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_humanml3d_imu_loaders_flu(
    root:             str,
    cfg:              Any                              = None,
    patch_size:       int                              = 16,
    T_min:            int                              = 64,
    T_max:            int                              = 512,
    batch_size:       int                              = 32,
    num_workers:      int                              = 4,
    seed:             int                              = 42,
    text_emb_cache:   Optional[Dict[str, np.ndarray]] = None,
    train_split_name: str                              = 'train',
) -> Dict[str, Optional[DataLoader]]:
    """Build train and val DataLoaders for HumanML3D with FLU wrist IMU.

    Drop-in replacement for ``make_humanml3d_imu_loaders`` — API is identical.
    ``cfg`` key overrides are the same::

        vit1d.patch_size              (int)
        windowing.min_window_seconds  (float)
        windowing.max_window_seconds  (float)
        windowing.pad_mode            (str, 'zero' or 'wrap'; default 'zero')
        data.batch_size               (int)
        data.num_workers              (int)
        trainer.seed                  (int)

    ``cfg`` (the full merged config, not just the keys above) is also passed
    straight through to ``HumanML3DIMUVarLenDatasetFLU`` so its top-level
    augmentation blocks (``random_rotation``, ``gaussian_noise``, etc. — same
    convention as every other loader in this repo) apply to sampled windows
    when enabled; a config with no augmentation blocks enabled behaves
    exactly as before.

    ``train_split_name``: which ``<root>/HumanML3D/<name>.txt`` file backs
    the *train* loader (default ``'train'``, the official split — every
    existing config in this benchmark keeps the default). The val loader
    always reads the official ``'val'`` split regardless of this override.
    Exists specifically to build a train/eval-overlap leakage control (e.g.
    ``train_split_name='train_val'``, HumanML3D's own official train+val
    union — every val clip becomes part of training, so scoring against
    ``val.txt`` afterward measures the effect of training directly on the
    eval clips, not genuine held-out generalization). See EXPERIMENTS.md
    Section 6b.

    Returns a dict with keys ``'train_loader'`` and ``'val_loader'``.
    DataLoaders use ``batch_size=None`` (the dataset pre-batches).
    """
    pad_mode = 'zero'
    if cfg is not None:
        from imu_lm.utils.helpers import cfg_get
        vit_cfg = cfg_get(cfg, ['vit1d'],    {}) or {}
        win_cfg = cfg_get(cfg, ['windowing'], {}) or {}
        dat_cfg = cfg_get(cfg, ['data'],     {}) or {}
        patch_size  = int(vit_cfg.get('patch_size', patch_size))
        T_min       = int(round(float(win_cfg.get('min_window_seconds', T_min / _TARGET_FPS)) * _TARGET_FPS / patch_size)) * patch_size
        T_max       = int(round(float(win_cfg.get('max_window_seconds', T_max / _TARGET_FPS)) * _TARGET_FPS / patch_size)) * patch_size
        batch_size  = int(dat_cfg.get('batch_size', batch_size))
        num_workers = int(dat_cfg.get('num_workers', num_workers))
        seed        = int(cfg_get(cfg, ['trainer', 'seed'], seed))
        pad_mode    = str(win_cfg.get('pad_mode', pad_mode))

    loaders: Dict[str, Optional[DataLoader]] = {}
    for split, split_seed, split_file_name in [
        ('train', seed, train_split_name),
        ('val', seed + 1000, 'val'),
    ]:
        split_file = pjoin(root, 'HumanML3D', f'{split_file_name}.txt')
        if not os.path.exists(split_file):
            logger.warning(
                '[make_humanml3d_imu_loaders_flu] split file not found: %s', split_file
            )
            loaders[f'{split}_loader'] = None
            continue

        dataset = HumanML3DIMUVarLenDatasetFLU(
            root           = root,
            split          = split_file_name,
            batch_size     = batch_size,
            patch_size     = patch_size,
            T_min          = T_min,
            T_max          = T_max,
            seed           = split_seed,
            text_emb_cache = text_emb_cache,
            pad_mode       = pad_mode,
            cfg            = cfg,
            augment        = (split == 'train'),
        )
        loaders[f'{split}_loader'] = DataLoader(
            dataset,
            batch_size         = None,
            num_workers        = num_workers if split == 'train' else min(num_workers, 2),
            pin_memory         = True,
            persistent_workers = num_workers > 0,
        )

    return loaders


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    root = sys.argv[1] if len(sys.argv) > 1 else '.'

    ds = HumanML3DIMUDatasetFLU(root=root, split='test', patch_size=16,
                                 T_min=64, T_max=512)
    print(f'Clips loaded : {len(ds)}')
    x, cap = ds[0]
    print(f'x shape      : {x.shape}   (expect [6, T])')
    print(f'caption      : {cap!r}')
    print(f'acc_F  mean/std : {x[0].mean():.4f} / {x[0].std():.4f}')
    print(f'gyro_F mean/std : {x[3].mean():.4f} / {x[3].std():.4f}')
