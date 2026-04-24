from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WRIST_SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

RIGHT_WRIST_CHAIN_JOINTS = [14, 17, 19, 21, 23]
NUM_CHAIN_JOINTS = len(RIGHT_WRIST_CHAIN_JOINTS)
ROT6D_DIM = 6

# SMPL right wrist joint index
_SMPL_RIGHT_WRIST = 21

# ---------------------------------------------------------------------------
# Watch-mounted FLU coordinate system on the right wrist (T-pose)
# ---------------------------------------------------------------------------
# Forward (FLU x) = -x_smpl  →  toward fingertips (away from arm)
# Left    (FLU y) = +z_smpl  →  toward thumb side of right wrist
# Up      (FLU z) = +y_smpl  →  out of watch face (dorsal, away from skin)
#
# In SMPL T-pose right-wrist local frame:
#   +x points toward the elbow (proximal), +y points dorsal (out of watch face),
#   +z points toward the thumb (lateral on right wrist).
# Columns are FLU axes expressed in SMPL WRIST-LOCAL coordinates.
R_WRIST_FROM_FLU = np.array([
    [-1.0, 0.0, 0.0],   # F = -x_smpl  (fingertips)
    [ 0.0, 0.0, 1.0],   # L = +z_smpl  (thumb side)
    [ 0.0, 1.0, 0.0],   # U = +y_smpl  (watch face / dorsal)
], dtype=np.float64)

# sanity: should be a proper rotation
assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Xsens → SMPL joint mapping (inlined from nymeriadataset.py)
_XSENS_TO_SMPL = [0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18,
                   5, 11,  7, 6, 12,  8, 13,  9, 13,  9, 13,  9]


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def xsens_to_SMPL(xsens_rotmat: np.ndarray) -> np.ndarray:
    """Convert Xsens *global* rotation matrices [N,23,3,3] → SMPL *global* [N,24,3,3].

    Applies the Nymeria coordinate-system fix (Rx90 · Rz90) and remaps joints.
    """
    perm = [1, 2, 0]
    xsens_rotmat = xsens_rotmat[:, :, perm, :]
    xsens_rotmat = xsens_rotmat[:, :, :, perm]

    xsens_t = torch.from_numpy(xsens_rotmat).float()

    Rx90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    Rz90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    xsens_t = Rz90 @ (Rx90 @ xsens_t)  # broadcast over batch/joint dims

    smpl_rotmat = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(
        xsens_t.shape[0], 24, -1, -1
    ).clone()
    for smpl_idx, xsens_idx in enumerate(_XSENS_TO_SMPL):
        smpl_rotmat[:, smpl_idx] = xsens_t[:, xsens_idx]

    return smpl_rotmat.numpy()  # [N,24,3,3]


def _add_mems_gyro_noise(
    gyro: np.ndarray,
    fps: float,
    white_noise_std: float = 0.02,
    bias_drift_std: float  = 0.002,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Add two-component MEMS gyroscope noise to a clean synthetic gyro signal.

    Standard IMU error model (IEEE Std 952):
        omega_measured(t) = omega_true(t) + b(t) + n_w(t)

    Where:
      n_w  ~ N(0, white_noise_std²) per sample    [Angle Random Walk]
             Typical wrist IMU: 0.01–0.05 rad/s RMS
      b(t) = integrated random walk               [Rate Random Walk / Bias Instability]
             b(t+dt) = b(t) + N(0, bias_drift_std² * dt),  dt = 1/fps
             Typical wrist IMU: 0.001–0.005 rad/s/√s

    Args:
        gyro:            [N, 3] clean angular velocity in rad/s
        fps:             sampling rate in Hz
        white_noise_std: std of additive white noise (rad/s).  Default 0.02 rad/s.
        bias_drift_std:  std of bias random-walk per √second (rad/s/√s).  Default 0.002.
        rng:             optional RandomState for reproducibility

    Returns:
        [N, 3] noisy gyro signal, same dtype as input
    """
    if rng is None:
        rng = np.random.RandomState()

    N = gyro.shape[0]
    dt = 1.0 / fps
    out = gyro.copy()

    # --- 1) White noise (Angle Random Walk) ---
    out = out + rng.normal(0.0, white_noise_std, size=(N, 3))

    # --- 2) Bias drift (Rate Random Walk) ---
    # Each sample the bias steps by N(0, bias_drift_std² * dt)
    bias_steps = rng.normal(0.0, bias_drift_std * np.sqrt(dt), size=(N, 3))
    bias = np.cumsum(bias_steps, axis=0)          # integrated walk
    out = out + bias

    return out.astype(gyro.dtype)


def _compute_acc_world(pos_world: np.ndarray, fps: float) -> np.ndarray:
    """Second finite difference."""
    N = pos_world.shape[0]
    acc = np.zeros((N, 3), dtype=np.float64)
    acc[1:-1] = (pos_world[2:] - 2.0 * pos_world[1:-1] + pos_world[:-2]) * (fps ** 2)
    acc[0] = acc[1]
    acc[-1] = acc[-2]
    return acc


def _compute_omega_local_from_global_rot(R_world_from_joint: np.ndarray, fps: float) -> np.ndarray:
    """
    Angular velocity in the joint LOCAL frame.

    dR_t = R_t^T R_{t+1}
    rotvec(dR_t) * fps
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    dR = np.einsum(
        "nij,njk->nik",
        R_world_from_joint[:-1].transpose(0, 2, 1),
        R_world_from_joint[1:],
    )  # [N-1,3,3]

    omega_raw = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3))
        .as_rotvec()
        .astype(np.float64) * fps
    )  # [N-1,3]

    N = R_world_from_joint.shape[0]
    omega = np.zeros((N, 3), dtype=np.float64)
    omega[:-1] = omega_raw
    omega[-1] = omega_raw[-1]
    return omega


# ---------------------------------------------------------------------------
# Core session loader
# ---------------------------------------------------------------------------

def _load_nymeria_session_flu(
    pkl_path: str,
    target_hz: int,
    add_gravity: bool = True,    gyro_white_noise_std: float = 0.3,
    gyro_bias_drift_std:  float = 0.1,
    rng: Optional[np.random.RandomState] = None,) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Pipeline (follows the user's FK → world → FLU idea):

    1.  Load Nymeria ``xsens_pose`` [N,23,4,4] (global Xsens transforms).
    2.  Convert Xsens global rotations → SMPL global rotations via
        ``xsens_to_SMPL``, then IK → SMPL *local* rotations.
    3.  Define FLU coordinate frame X at the right-wrist joint in T-pose:
            Forward (FLU x) = −x_smpl,  Left (FLU y) = +z_smpl,  Up (FLU z) = +y_smpl
        so R_wrist_from_flu = R_WRIST_FROM_FLU  (constant).
    4.  Apply SMPL FK (body_model.forward_kinematics_masked) to obtain the
        wrist's world-frame rotation R_world_from_wrist(t) and position
        p_world_wrist(t).
    5.  Propagate X to world frame:
            R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
        The inverse is:
            R_X_from_world(t) = R_world_from_X(t)^T
    6.  Get acc and gyro in the *world* frame:
            a_world  = d²p_world_wrist / dt²
            ω_world  = R_world_from_wrist(t) @ ω_wrist_local(t)
    7.  Express both in X (FLU) via R_X_from_world(t):
            acc_flu  = R_X_from_world(t) @ a_world(t)
            gyro_flu = R_X_from_world(t) @ ω_world(t)

    Returns
    -------
    X_imu : np.ndarray  [N, 6]    float32  — [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    y     : np.ndarray  [N, 5, 6] float32  — 6D local rotations for right-arm chain
    """
    import core.articulate as art
    from core.paths import Paths

    logger.info("[nymeria_wrist_flu] loading: %s", os.path.basename(pkl_path))
    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        logger.warning("[nymeria_wrist_flu] cannot open %s: %s", pkl_path, exc)
        return None

    gt_data = raw.get("gt_data", {})
    xsens_pose = gt_data.get("xsens_pose")
    if xsens_pose is None:
        logger.warning("[nymeria_wrist_flu] %s: 'xsens_pose' missing.", pkl_path)
        return None

    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)   # [N, 23, 4, 4]

    N = xsens_pose.shape[0]
    if N < 3:
        logger.warning("[nymeria_wrist_flu] %s: too short (%d frames).", pkl_path, N)
        return None

    fps = float(
        raw.get("imu_sampling_rate_hz")
        or raw.get("pose_sampling_rate_hz")
        or target_hz
    )

    # ------------------------------------------------------------------
    # Step 2 — Xsens global → SMPL local rotations + root translation
    # ------------------------------------------------------------------
    body_model = art.model.ParametricModel(
        Paths.SMPL_FILE, device=torch.device("cpu")
    )

    xsens_R = xsens_pose[:, :, :3, :3].astype(np.float32)    # [N, 23, 3, 3]
    smpl_global_R = xsens_to_SMPL(xsens_R)                    # [N, 24, 3, 3]

    smpl_global_t = torch.from_numpy(smpl_global_R).float()
    smpl_local_t = body_model.inverse_kinematics_R(smpl_global_t)  # [N, 24, 3, 3]
    smpl_local_R = smpl_local_t.cpu().numpy().astype(np.float64)   # [N, 24, 3, 3]

    # Root (pelvis) world translation from Xsens joint 0
    root_trans = xsens_pose[:, 0, :3, 3].astype(np.float32)   # [N, 3]

    # ------------------------------------------------------------------
    # Step 4 — SMPL FK: world-frame wrist rotation + position
    # ------------------------------------------------------------------
    pose_global_t, joint_world_t = body_model.forward_kinematics_masked(
        smpl_local_t,
        tran=torch.from_numpy(root_trans).float(),
        calc_mesh=False,
    )  # [N,24,3,3], [N,24,3]

    R_world_from_wrist = pose_global_t[:, _SMPL_RIGHT_WRIST].cpu().numpy().astype(np.float64)  # [N,3,3]
    p_world_wrist      = joint_world_t[:, _SMPL_RIGHT_WRIST].cpu().numpy().astype(np.float64)  # [N,3]

    # ------------------------------------------------------------------
    # Step 5 — propagate X (= FLU) to world frame per timestep
    # R_world_from_X(t) = R_world_from_wrist(t) @ R_WRIST_FROM_FLU
    # ------------------------------------------------------------------
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)  # [N,3,3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                                # [N,3,3]

    # ------------------------------------------------------------------
    # Step 6 — compute acc and gyro in world frame
    # ------------------------------------------------------------------
    # 6a) linear acceleration of wrist origin in world
    a_world = _compute_acc_world(p_world_wrist, fps)            # [N,3]

    if add_gravity:
        # subtract gravity so the signal matches what a real accelerometer reads
        # (Nymeria is Y-up: gravity = [0, -9.81, 0])
        g_world = np.array([0.0, -9.81, 0.0], dtype=np.float64)
        a_world = a_world - g_world

    # 6b) angular velocity in world frame
    #     ω_wrist_local is in the wrist joint's local frame;
    #     rotate it to world: ω_world = R_world_from_wrist @ ω_wrist_local
    omega_wrist_local = _compute_omega_local_from_global_rot(R_world_from_wrist, fps)   # [N,3]
    omega_world = np.einsum("nij,nj->ni", R_world_from_wrist, omega_wrist_local)        # [N,3]

    # ------------------------------------------------------------------
    # Step 7 — express acc and gyro in X (FLU) frame
    #     acc_flu  = R_X_from_world(t) @ a_world(t)
    #     gyro_flu = R_X_from_world(t) @ ω_world(t)
    # ------------------------------------------------------------------
    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, a_world)    # [N,3]
    gyro_flu = np.einsum("nij,nj->ni", R_X_from_world, omega_world) # [N,3]

    # -- Optional: add MEMS-style gyro noise to close the sim-to-real gap --
    if gyro_white_noise_std > 0.0 or gyro_bias_drift_std > 0.0:
        gyro_flu = _add_mems_gyro_noise(
            gyro_flu, fps,
            white_noise_std = gyro_white_noise_std,
            bias_drift_std  = gyro_bias_drift_std,
            rng             = rng,
        )

    X_imu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)  # [N,6]

    # ------------------------------------------------------------------
    # Labels — SMPL local rotations for right-arm chain
    # ------------------------------------------------------------------
    chain_R = smpl_local_R[:, RIGHT_WRIST_CHAIN_JOINTS]                   # [N,5,3,3]
    y = matrix_to_rotation_6d(torch.from_numpy(chain_R.astype(np.float32))).numpy()  # [N,5,6]

    logger.info(
        "[nymeria_wrist_flu] loaded %s: N=%d fps=%.0f smpl-fk-flu",
        os.path.basename(pkl_path), N, fps,
    )
    return X_imu, y


# ---------------------------------------------------------------------------
# Window index builder
# ---------------------------------------------------------------------------

def _build_window_specs(
    session_lengths: List[int],
    T: int,
    hop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total = sum(max(0, (n - T) // hop + 1) for n in session_lengths)
    sess_idx = np.empty(total, dtype=np.int32)
    starts   = np.empty(total, dtype=np.int32)
    offset = 0
    for ki, n in enumerate(session_lengths):
        if n < T:
            continue
        n_win = (n - T) // hop + 1
        sess_idx[offset:offset + n_win] = ki
        starts[offset:offset + n_win]   = np.arange(0, n - T + 1, hop, dtype=np.int32)
        offset += n_win
    return sess_idx, starts


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NymeriaWristDatasetFLU(Dataset):
    def __init__(
        self,
        pkl_paths: List[str],
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 50,
        add_gravity: bool = True,
        gyro_white_noise_std: float = 0.0,
        gyro_bias_drift_std:  float = 0.0,
        noise_seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.split_name = split_name
        self.target_hz = target_hz
        self.add_gravity = add_gravity
        self.gyro_white_noise_std = gyro_white_noise_std
        self.gyro_bias_drift_std  = gyro_bias_drift_std
        self.stats = PreprocessStats()
        rng = np.random.RandomState(noise_seed) if (gyro_white_noise_std > 0.0 or gyro_bias_drift_std > 0.0) else None

        self._T, self._hop = compute_T_and_hop(cfg)

        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size = int(vit_cfg.get("patch_size", 16))
        self._num_patches = self._T // self._patch_size

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []
        session_lengths: List[int] = []

        for path in tqdm(
            pkl_paths,
            desc=f"Loading {split_name} sessions (FLU)",
            unit="session",
            dynamic_ncols=True,
        ):
            result = _load_nymeria_session_flu(
                path,
                target_hz=target_hz,
                add_gravity=add_gravity,
                gyro_white_noise_std=gyro_white_noise_std,
                gyro_bias_drift_std =gyro_bias_drift_std,
                rng=rng,
            )
            if result is None:
                logger.warning("[nymeria_wrist_flu] skipping: %s", path)
                continue
            X, y = result
            self._sessions.append((X, y))
            session_lengths.append(len(X))

        if not self._sessions:
            raise RuntimeError(
                "NymeriaWristDatasetFLU: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        si = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T = self._T

        X_raw, y_arr = self._sessions[si]
        Xw = X_raw[start : start + T]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc = apply_augment(Xproc.T, self.cfg).T
        x_tensor = torch.from_numpy(Xproc).float()

        P = self._patch_size
        patch_labels = np.empty(
            (self._num_patches, NUM_CHAIN_JOINTS * ROT6D_DIM), dtype=np.float32
        )
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y_arr[frame].reshape(-1)

        y_tensor = torch.from_numpy(patch_labels).float()
        return x_tensor, y_tensor


def collate_skip_none(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


def make_nymeria_loaders_flu(
    cfg: Any,
    pkl_root: str,
    train_files: List[str],
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 50,
    add_gravity: bool = True,
    gyro_white_noise_std: float = 0.0,
    gyro_bias_drift_std:  float = 0.0,
    noise_seed: int = 0,
) -> Dict[str, DataLoader]:
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))
    def _abs(files: List[str]) -> List[str]:
        return [f if os.path.isabs(f) else os.path.join(pkl_root, f)
                for f in files]

    def _make(name: str, files: List[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = NymeriaWristDatasetFLU(
            pkl_paths=_abs(files),
            cfg=cfg,
            split_name=name,
            target_hz=target_hz,
            add_gravity=add_gravity,
            gyro_white_noise_std=gyro_white_noise_std,
            gyro_bias_drift_std =gyro_bias_drift_std,
            noise_seed=noise_seed,
        )
        persistent = num_workers > 0
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_skip_none,
            persistent_workers=persistent,
        )

    loaders: Dict[str, DataLoader] = {}
    for name, files, bs, shuf in [
        ("train", train_files,         batch_size,      True),
        ("val",   val_files  or [],    eval_batch_size, False),
        ("test",  test_files or [],    eval_batch_size, False),
    ]:
        loader = _make(name, files, bs, shuf)
        if loader is not None:
            loaders[f"{name}_loader"] = loader

    return loaders