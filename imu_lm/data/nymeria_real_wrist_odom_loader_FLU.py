"""nymeria_real_wrist_odom_loader_FLU.py
Nymeria real right-wrist IMU (FLU body frame) → per-patch pelvis-root velocity
in the **wrist FLU sensor frame**.

Mirrors ``imuposer_wrist_local_odom_loader_FLU.py`` exactly in label
convention, making the two datasets interchangeable for local-velocity
odometry pretraining / fine-tuning.

Motivation
----------
Expressing the pelvis velocity in the wrist FLU frame keeps inputs and labels
in a consistent local coordinate system.  The network receives acc/gyro already
in FLU, and the velocity label is also in FLU, so the body-frame direction of
motion is directly observable without a separate global-yaw estimate.

IMU derivation (Nymeria)
------------------------
  acc  : wrist_imu_data["accel"]          – CPF-local; rotated to world via orientation
  gyro : wrist_imu_data["gyro"]           – CPF-local; rotated to world via orientation
  ori  : wrist_gt_data["orientation"]     – R_world_from_wrist_cpf  (quat xyzw or LieTensor)

Data flow
---------
1.  R_world_from_wrist_cpf  from wrist_gt_data["orientation"]
      → R_world_from_X(t) = R_world_from_wrist_cpf(t) @ R_WRIST_FROM_FLU
      → R_X_from_world(t) = R_world_from_X(t)^T
2.  Project real acc and gyro into FLU.
3.  Root (pelvis) world position from gt_data["xsens_pose"][:, 0, :3, 3].
4.  Store pos_root and R_X_from_world for label computation.

Per-patch label
---------------
Each patch p (P frames) is divided into 4 sub-patches of Q = P // 4 frames.
For sub-patch k (k = 0 ... 3):

    t_start = start + p*P + k*Q
    t_end   = t_start + Q
    t_mid   = t_start + Q // 2              (midpoint -- used for FLU frame)

    v_world[k] = (pos_root[t_end] - pos_root[t_start]) / (Q / fps)
    v_flu[k]   = R_X_from_world[t_mid] @ v_world[k]   (pelvis vel in wrist FLU, m/s)

    shape  : [N_patches, 4, 3]  float32

FLU frame convention (right wrist, T-pose)
------------------------------------------
  F (x) = -x_smpl  (finger / forward direction)
  L (y) = +z_smpl  (thumb / left side)
  U (z) = +y_smpl  (dorsal / watch-face / up)

  R_WRIST_FROM_FLU = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]

Inputs / outputs per window
---------------------------
  x : [6, T]              float32  -- (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
  y : [N_patches, 4, 3]   float32  -- pelvis velocity in wrist FLU frame (m/s)
                                       y[p, k] = velocity at sub-patch k of patch p

Usage
-----
    from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import (
        make_nymeria_real_wrist_odom_loaders_flu
    )
    loaders = make_nymeria_real_wrist_odom_loaders_flu(
        cfg=cfg,
        pkl_root="/mnt/nas2/processed_data_body_Ty_rwrist",
        train_files=train_files,
        val_files=val_files,
        target_hz=50,
        add_gravity=True,
    )
"""

from __future__ import annotations

import logging
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Xsens joint 0 = pelvis/root (used for root position labels)
_XSENS_ROOT_JOINT = 0

# FLU frame rotation matrix (right wrist T-pose, same as all other *_FLU loaders)
R_WRIST_FROM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0],   # F = -x_smpl
        [ 0.0, 0.0, 1.0],   # L = +z_smpl
        [ 0.0, 1.0, 0.0],   # U = +y_smpl
    ],
    dtype=np.float64,
)

assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, w) quaternions to rotation matrices. [N, 4] -> [N, 3, 3]."""
    from scipy.spatial.transform import Rotation as ScipyRot
    return ScipyRot.from_quat(quat).as_matrix().astype(np.float64)


# ---------------------------------------------------------------------------
# Window index builder
# ---------------------------------------------------------------------------

def _build_window_specs(
    session_lengths: List[int],
    T: int,
    hop: int,
) -> Tuple[np.ndarray, np.ndarray]:
    total    = sum(max(0, (n - T) // hop + 1) for n in session_lengths)
    sess_idx = np.empty(total, dtype=np.int32)
    starts   = np.empty(total, dtype=np.int32)
    offset   = 0
    for ki, n in enumerate(session_lengths):
        if n < T:
            continue
        n_win = (n - T) // hop + 1
        sess_idx[offset : offset + n_win] = ki
        starts[offset : offset + n_win]   = np.arange(0, n - T + 1, hop, dtype=np.int32)
        offset += n_win
    return sess_idx, starts


# ---------------------------------------------------------------------------
# Session loader
# ---------------------------------------------------------------------------

def load_session_local_vel_flu(
    pkl_path: str,
    target_hz: int = 50,
    add_gravity: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a Nymeria session -> (X_flu, pos_root, R_X_from_world).

    Parameters
    ----------
    pkl_path    : absolute path to a Nymeria ``.pkl`` file (has_wrist_imu=True).
    target_hz   : target sampling rate (data downsampled if raw > target).
    add_gravity : if False, subtracts gravity from the rotated acc (acc_world -
                  [0, -9.81, 0]) so the signal matches kinematic acceleration.
                  Default True keeps the raw specific-force (gravity included).

    Returns
    -------
    X_flu          : [N, 6]    float32  -- FLU-frame acc + gyro
    pos_root       : [N, 3]    float32  -- pelvis world position (Y-up, metres)
    R_X_from_world : [N, 3, 3] float32  -- R_FLU_from_world at every frame
    None on any failure.
    """
    logger.info("[nymeria_real_wrist_odom_flu] loading: %s", os.path.basename(pkl_path))

    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        logger.warning("[nymeria_real_wrist_odom_flu] cannot open %s: %s", pkl_path, exc)
        return None

    if not raw.get("has_wrist_imu", False):
        logger.warning(
            "[nymeria_real_wrist_odom_flu] %s: has_wrist_imu=False, skipping", pkl_path
        )
        return None

    wrist_imu  = raw.get("wrist_imu_data", {})
    wrist_gt   = raw.get("wrist_gt_data",  {})
    xsens_pose = raw.get("gt_data", {}).get("xsens_pose")

    for key in ("accel", "gyro"):
        if key not in wrist_imu:
            logger.warning(
                "[nymeria_real_wrist_odom_flu] %s: '%s' missing in wrist_imu_data",
                pkl_path, key,
            )
            return None

    if "orientation" not in wrist_gt:
        logger.warning(
            "[nymeria_real_wrist_odom_flu] %s: 'orientation' missing in wrist_gt_data",
            pkl_path,
        )
        return None

    if xsens_pose is None:
        logger.warning(
            "[nymeria_real_wrist_odom_flu] %s: 'xsens_pose' missing (needed for root position)",
            pkl_path,
        )
        return None

    # ------------------------------------------------------------------
    # Sampling rates and downsample
    # ------------------------------------------------------------------
    body_fps_raw  = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz
    )
    wrist_fps_raw = float(raw.get("wrist_imu_sampling_rate_hz", body_fps_raw))

    body_ds  = max(1, round(body_fps_raw  / target_hz))
    wrist_ds = max(1, round(wrist_fps_raw / target_hz))

    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)[::body_ds]   # [N_b, 23, 4, 4]

    def _to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    acc_local  = _to_np(wrist_imu["accel"]).astype(np.float64)[::wrist_ds]   # [N_w, 3]
    gyro_local = _to_np(wrist_imu["gyro"]).astype(np.float64)[::wrist_ds]    # [N_w, 3]
    orient_raw = wrist_gt["orientation"]

    try:
        import pypose as pp
        _is_lietensor = isinstance(orient_raw, pp.LieTensor)
    except ImportError:
        _is_lietensor = False

    if _is_lietensor:
        R_world_from_wrist = (
            orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
        )   # [N_w, 3, 3]
    else:
        orient_np          = _to_np(orient_raw).astype(np.float64)[::wrist_ds]
        R_world_from_wrist = _quat_xyzw_to_rotmat(orient_np)   # [N_w, 3, 3]

    # ------------------------------------------------------------------
    # Align lengths
    # ------------------------------------------------------------------
    N = min(xsens_pose.shape[0], acc_local.shape[0])
    if N < 3:
        logger.warning(
            "[nymeria_real_wrist_odom_flu] %s: too short after alignment (%d frames)",
            pkl_path, N,
        )
        return None

    xsens_pose         = xsens_pose[:N]
    acc_local          = acc_local[:N]
    gyro_local         = gyro_local[:N]
    R_world_from_wrist = R_world_from_wrist[:N]

    fps = float(target_hz)

    # ------------------------------------------------------------------
    # Build FLU frame from wrist CPF orientation
    # ------------------------------------------------------------------
    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, R_WRIST_FROM_FLU)  # [N,3,3]
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)                                # [N,3,3]

    # ------------------------------------------------------------------
    # CPF-local acc -> world -> FLU
    # ------------------------------------------------------------------
    acc_world = np.einsum("nij,nj->ni", R_world_from_wrist, acc_local)   # [N, 3]
    if not add_gravity:
        # Subtract gravity (Y-up: g = [0, -9.81, 0]) to get kinematic acceleration
        g_world = np.array([0.0, -9.81, 0.0], dtype=np.float64)
        acc_world = acc_world + g_world

    acc_flu = np.einsum("nij,nj->ni", R_X_from_world, acc_world)         # [N, 3]

    # ------------------------------------------------------------------
    # CPF-local gyro -> world -> FLU
    # ------------------------------------------------------------------
    gyro_world = np.einsum("nij,nj->ni", R_world_from_wrist, gyro_local)  # [N, 3]
    gyro_flu   = np.einsum("nij,nj->ni", R_X_from_world, gyro_world)      # [N, 3]

    X_flu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)  # [N, 6]

    # ------------------------------------------------------------------
    # Root world position + per-frame FLU-from-world rotation for labels
    # ------------------------------------------------------------------
    pos_root    = xsens_pose[:, _XSENS_ROOT_JOINT, :3, 3].astype(np.float32)  # [N, 3]
    R_X_fw_f32  = R_X_from_world.astype(np.float32)                            # [N, 3, 3]

    logger.info(
        "[nymeria_real_wrist_odom_flu] %s: N=%d fps=%.0f",
        os.path.basename(pkl_path), N, fps,
    )
    return X_flu, pos_root, R_X_fw_f32


# ---------------------------------------------------------------------------
# Sliding-window Dataset
# ---------------------------------------------------------------------------

class NymeriaRealWristOdomDatasetFLU(Dataset):
    """Sliding-window dataset: real wrist FLU IMU -> per-patch pelvis velocity in FLU frame.

    Each item::

        x : [6, T]              float32 -- FLU-frame wrist acc + gyro (real sensor)
        y : [N_patches, 4, 3]   float32 -- pelvis velocity (m/s) in wrist FLU frame

    Each patch p is split into 4 sub-patches of Q = patch_size // 4 frames::

        t_start = start + p*P + k*Q
        t_end   = t_start + Q
        t_mid   = t_start + Q // 2
        v_world = (pos_root[t_end] - pos_root[t_start]) / (Q / fps)
        y[p, k] = R_X_from_world[t_mid] @ v_world   (3-D velocity in FLU, m/s)

    patch_size must be divisible by 4.
    """

    def __init__(
        self,
        pkl_paths:   List[str],
        cfg:         Any,
        split_name:  str  = "train",
        target_hz:   int  = 50,
        add_gravity: bool = True,
    ):
        super().__init__()
        self.cfg        = cfg
        self.split_name = split_name
        self.stats      = PreprocessStats()

        vit_cfg            = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size   = int(vit_cfg.get("patch_size", 16))
        self._T, self._hop = compute_T_and_hop(cfg)
        self._num_patches  = self._T // self._patch_size
        self._target_hz    = float(target_hz)

        assert self._T % self._patch_size == 0, (
            f"T={self._T} must be divisible by patch_size={self._patch_size}"
        )
        assert self._patch_size % 4 == 0, (
            f"patch_size={self._patch_size} must be divisible by 4 "
            "(4 velocity sub-patches per patch)"
        )

        # Each session: (X_flu [N,6], pos_root [N,3], R_X_from_world [N,3,3])
        self._sessions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        session_lengths: List[int] = []

        _n_workers = min(32, (os.cpu_count() or 4) * 2)
        _load      = lambda p: load_session_local_vel_flu(
            p, target_hz=target_hz, add_gravity=add_gravity
        )
        _ordered: List = [None] * len(pkl_paths)
        with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
            _fut_to_idx = {_pool.submit(_load, p): i for i, p in enumerate(pkl_paths)}
            with tqdm(
                total=len(pkl_paths),
                desc=f"Loading {split_name} sessions (nymeria real wrist FLU odom)",
                unit="session",
                dynamic_ncols=True,
            ) as _pbar:
                for _fut in as_completed(_fut_to_idx):
                    _ordered[_fut_to_idx[_fut]] = _fut.result()
                    _pbar.update(1)

        for path, result in zip(pkl_paths, _ordered):
            if result is None:
                logger.warning("[nymeria_real_wrist_odom_flu] skipping: %s", path)
                continue
            X_flu, pos_root, R_X_fw = result
            # Need at least T + patch_size frames so last patch has a valid end-frame
            if X_flu.shape[0] < self._T + self._patch_size:
                logger.warning(
                    "[nymeria_real_wrist_odom_flu] %s too short (%d), skipping",
                    os.path.basename(path), X_flu.shape[0],
                )
                continue
            self._sessions.append((X_flu, pos_root, R_X_fw))
            session_lengths.append(X_flu.shape[0] - self._patch_size)

        if not self._sessions:
            raise RuntimeError(
                "NymeriaRealWristOdomDatasetFLU: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )
        logger.info(
            "[nymeria_real_wrist_odom_flu] split=%s  sessions=%d  windows=%d  T=%d  hop=%d",
            split_name, len(self._sessions), len(self._starts), self._T, self._hop,
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T
        P     = self._patch_size

        X_flu, pos_root, R_X_fw = self._sessions[si]   # [N,6], [N,3], [N,3,3]

        Xw    = X_flu[start : start + T]              # [T, 6]
        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        if self.split_name == "train":
            Xproc = apply_augment(Xproc.T, self.cfg).T

        x_tensor = torch.from_numpy(Xproc).float()    # [6, T]

        # Per-patch label: 4 x 3-D velocity in wrist FLU frame (m/s)
        # Each patch is divided into 4 sub-patches of Q frames.
        # At 50 Hz with patch_size=16: Q=4 frames = 0.08 s per sub-patch,
        # covering 0.32 s per patch in total.
        Q    = P // 4                          # frames per sub-patch
        dt_Q = Q / self._target_hz             # duration of one sub-patch (seconds)
        vel  = np.empty((self._num_patches, 4, 3), dtype=np.float32)
        for p in range(self._num_patches):
            t_p = start + p * P
            for k in range(4):
                t_s   = t_p + k * Q
                t_e   = t_s + Q
                t_mid = t_s + Q // 2
                v_w   = (pos_root[t_e].astype(np.float64)
                         - pos_root[t_s].astype(np.float64)) / dt_Q   # world velocity [3]
                vel[p, k] = (R_X_fw[t_mid].astype(np.float64) @ v_w).astype(np.float32)

        y_tensor = torch.from_numpy(vel).float()       # [N_patches, 4, 3]
        return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_skip_none(
    batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_nymeria_real_wrist_odom_loaders_flu(
    cfg:         Any,
    pkl_root:    str,
    train_files: List[str],
    val_files:   Optional[List[str]] = None,
    test_files:  Optional[List[str]] = None,
    target_hz:   int  = 50,
    add_gravity: bool = True,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for Nymeria FLU-frame local velocity.

    Each batch::

        x : [B, 6, T]              float32 -- real right-wrist FLU IMU
        y : [B, N_patches, 4, 3]   float32 -- pelvis velocity in wrist FLU frame (m/s)
                                              y[b, p, k] = velocity at sub-patch k of patch p

    Parameters
    ----------
    cfg          : experiment config (OmegaConf or dict).
    pkl_root     : root directory; relative paths in *_files are joined here.
    train_files  : list of pkl paths for training.
    val_files    : list of pkl paths for validation (or None).
    test_files   : list of pkl paths for test (or None).
    target_hz    : target sampling rate (default 50 Hz).
    add_gravity  : see ``load_session_local_vel_flu``.
    """
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))
    persistent      = num_workers > 0

    def _abs(files: List[str]) -> List[str]:
        return [
            f if os.path.isabs(f) else os.path.join(pkl_root, f)
            for f in files
        ]

    def _make(name: str, files: List[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = NymeriaRealWristOdomDatasetFLU(
            pkl_paths=_abs(files),
            cfg=cfg,
            split_name=name,
            target_hz=target_hz,
            add_gravity=add_gravity,
        )
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
