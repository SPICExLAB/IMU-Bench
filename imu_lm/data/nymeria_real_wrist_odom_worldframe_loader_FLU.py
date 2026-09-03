"""nymeria_real_wrist_odom_worldframe_loader_FLU.py
Nymeria real right-wrist IMU (FLU body frame) → per-patch pelvis-root velocity
in a **per-window-canonicalized world frame**.

Complementary alternative to ``nymeria_real_wrist_odom_loader_FLU.py``. That
module expresses the pelvis velocity in the *wrist's own, continuously
rotating* FLU frame -- reconstructing a trajectory from its predictions
requires continuously tracking wrist orientation via real gyro integration
(``exp_so3`` at every timestep). This module instead predicts **world-frame**
velocity (Y-up, metres), but to remove the arbitrary global compass heading
(which varies session to session and gives the network nothing learnable),
each *window* is canonicalized once: rotated about the vertical (Y) axis so
that its own net horizontal displacement over the window points along +X.

Motivation
----------
Pelvis *orientation* isn't available in this dataset (only position), so
"facing +X" is defined via net movement direction rather than a body-frame
axis -- the standard canonicalization used in trajectory forecasting when only
position trajectories are available. One useful side effect: because the
canonicalizing rotation is *fixed per window* (computed once from start/end
position, not integrated over time), reconstructing a trajectory from
predicted velocities needs no gyro integration at all -- just a straight
cumulative sum, since the frame doesn't rotate within a window.

Data flow
---------
1.  Reuse ``load_session_local_vel_flu`` for X_flu / pos_root (only pos_root
    is used here; X_flu is used as-is for the model input, matching the other
    odometry loaders exactly).
2.  For window [s, s+T):
      disp        = pos_root[s+T] - pos_root[s]            (net displacement)
      theta       = atan2(disp_z, disp_x)                  (heading angle, X-Z floor plane)
      R_canon     = R_y(theta)  (rotation about the Y/up axis only)
                  = [[cosθ, 0, sinθ], [0, 1, 0], [-sinθ, 0, cosθ]]
      -> R_canon @ [disp_x, disp_y, disp_z] = [|disp_xz|, disp_y, 0]  (now facing +X)
    Near-stationary windows (|disp_xz| below a threshold) get an identity
    R_canon instead -- heading is ill-defined when there's no real motion.
3.  Per sub-patch k of patch p (same 4-sub-patches-per-patch convention as the
    wrist-FLU loader):
      v_world[k] = (pos_root[t_end] - pos_root[t_start]) / dt
      y[p, k]    = R_canon @ v_world[k]   (world velocity, this window's canonical frame)

Inputs / outputs per window
---------------------------
  x       : [6, T]              float32  -- (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U), same as the wrist-FLU loader
  y       : [N_patches, 4, 3]   float32  -- pelvis velocity (m/s), canonical world frame
  R_canon : [3, 3]              float32  -- this window's canonicalizing rotation (world -> canonical),
                                             needed to reconstruct/compare trajectories at eval time
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import (
    load_session_local_vel_flu, _build_window_specs,
)
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)

# Below this net horizontal displacement (metres) over the whole window, the
# heading direction is considered too noisy/undefined to canonicalize against
# -- use identity rotation instead of dividing by a near-zero norm.
_MIN_DISP_FOR_HEADING = 0.05


def _heading_rotation(disp: np.ndarray) -> np.ndarray:
    """[3] world displacement (Y-up) -> [3,3] rotation about Y so it points +X."""
    dx, dz = float(disp[0]), float(disp[2])
    norm_xz = (dx * dx + dz * dz) ** 0.5
    if norm_xz < _MIN_DISP_FOR_HEADING:
        return np.eye(3, dtype=np.float64)
    theta = np.arctan2(dz, dx)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Sliding-window Dataset
# ---------------------------------------------------------------------------

class NymeriaRealWristOdomWorldFrameDatasetFLU(Dataset):
    """Sliding-window dataset: real wrist FLU IMU -> per-patch pelvis velocity
    in a per-window heading-canonicalized world frame (see module docstring).

    Each item::

        x       : [6, T]              float32
        y       : [N_patches, 4, 3]   float32  -- world-frame velocity (m/s), this window's canonical frame
        R_canon : [3, 3]              float32  -- world -> canonical rotation for this window
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

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []  # (X_flu, pos_root) per session
        session_lengths: List[int] = []

        _n_workers = min(32, (os.cpu_count() or 4) * 2)
        _load = lambda p: load_session_local_vel_flu(p, target_hz=target_hz, add_gravity=add_gravity)
        _ordered: List = [None] * len(pkl_paths)
        with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
            _fut_to_idx = {_pool.submit(_load, p): i for i, p in enumerate(pkl_paths)}
            with tqdm(
                total=len(pkl_paths),
                desc=f"Loading {split_name} sessions (nymeria real wrist FLU odom worldframe)",
                unit="session", dynamic_ncols=True,
            ) as _pbar:
                from concurrent.futures import as_completed
                for _fut in as_completed(_fut_to_idx):
                    _ordered[_fut_to_idx[_fut]] = _fut.result()
                    _pbar.update(1)

        for path, result in zip(pkl_paths, _ordered):
            if result is None:
                logger.warning("[nymeria_real_wrist_odom_worldframe_flu] skipping: %s", path)
                continue
            X_flu, pos_root, _R_X_fw = result   # rotation not needed for this frame convention
            if X_flu.shape[0] < self._T + self._patch_size:
                logger.warning(
                    "[nymeria_real_wrist_odom_worldframe_flu] %s too short (%d), skipping",
                    os.path.basename(path), X_flu.shape[0],
                )
                continue
            self._sessions.append((X_flu, pos_root))
            session_lengths.append(X_flu.shape[0] - self._patch_size)

        if not self._sessions:
            raise RuntimeError(
                "NymeriaRealWristOdomWorldFrameDatasetFLU: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )
        logger.info(
            "[nymeria_real_wrist_odom_worldframe_flu] split=%s  sessions=%d  windows=%d  T=%d  hop=%d",
            split_name, len(self._sessions), len(self._starts), self._T, self._hop,
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T
        P     = self._patch_size

        X_flu, pos_root = self._sessions[si]

        Xw    = X_flu[start : start + T]
        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        if self.split_name == "train":
            Xproc = apply_augment(Xproc.T, self.cfg).T

        x_tensor = torch.from_numpy(Xproc).float()

        R_canon = _heading_rotation(
            pos_root[start + T].astype(np.float64) - pos_root[start].astype(np.float64)
        )

        Q    = P // 4
        dt_Q = Q / self._target_hz
        vel  = np.empty((self._num_patches, 4, 3), dtype=np.float32)
        for p in range(self._num_patches):
            t_p = start + p * P
            for k in range(4):
                t_s = t_p + k * Q
                t_e = t_s + Q
                v_w = (pos_root[t_e].astype(np.float64)
                       - pos_root[t_s].astype(np.float64)) / dt_Q
                vel[p, k] = (R_canon @ v_w).astype(np.float32)

        y_tensor = torch.from_numpy(vel).float()
        r_tensor = torch.from_numpy(R_canon.astype(np.float32))
        return x_tensor, y_tensor, r_tensor


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys, rs = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), torch.stack(rs, 0)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_nymeria_real_wrist_odom_worldframe_loaders_flu(
    cfg:         Any,
    pkl_root:    str,
    train_files: List[str],
    val_files:   Optional[List[str]] = None,
    test_files:  Optional[List[str]] = None,
    target_hz:   int  = 50,
    add_gravity: bool = True,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for Nymeria canonical-world-frame
    local velocity.

    Each batch::

        x       : [B, 6, T]              float32
        y       : [B, N_patches, 4, 3]   float32  -- world-frame velocity, this window's canonical frame
        R_canon : [B, 3, 3]              float32  -- world -> canonical rotation per window
    """
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))
    persistent      = num_workers > 0

    def _abs(files: List[str]) -> List[str]:
        return [f if os.path.isabs(f) else os.path.join(pkl_root, f) for f in files]

    def _make(name: str, files: List[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = NymeriaRealWristOdomWorldFrameDatasetFLU(
            pkl_paths=_abs(files), cfg=cfg, split_name=name,
            target_hz=target_hz, add_gravity=add_gravity,
        )
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
            pin_memory=pin_memory, drop_last=False,
            collate_fn=collate_skip_none, persistent_workers=persistent,
        )

    loaders: Dict[str, DataLoader] = {}
    for name, files, bs, shuf in [
        ("train", train_files,      batch_size,      True),
        ("val",   val_files or [],  eval_batch_size, False),
        ("test",  test_files or [], eval_batch_size, False),
    ]:
        loader = _make(name, files, bs, shuf)
        if loader is not None:
            loaders[f"{name}_loader"] = loader

    return loaders
