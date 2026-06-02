"""NymeriaSynWristDatasetFLU: synthesized right-wrist IMU in FLU local frame.

Synthesizes IMU from Nymeria body SMPL (xsens_pose joint 9 = right wrist) via
finite-difference kinematics, then rotates from the world frame into the FLU
(Forward-Left-Up) sensor body frame using the wrist joint orientation.

This is the synthetic counterpart of NymeriaRealWristDatasetFLU:
  • Real loader:  reads wrist_imu_data; rotates local CPF → world → FLU
  • This loader:  synthesizes acc+gyro in world from xsens_pose; rotates world → FLU

Input / output
--------------
    x : [6, T]  float32 — [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    y : [6, T]  float32 — identical copy of x (dummy; ignored by MAE objective)

Coordinate conventions
-----------------------
World frame: Y-up (Nymeria / SMPL convention) — X right, Y up, Z backward.
FLU frame:   attached to the wrist, defined by R_WRIST_FROM_FLU (same constant
             as nymeria_real_wrist_full_body_loader_FLU.py).

IMU synthesis from xsens_pose[:, 9, :, :] (joint 9 = right wrist)
--------------------------------------------------------------------
    pos [N,3]     = xsens_pose[:, 9, :3, 3]   world-frame positions
    rot [N,3,3]   = xsens_pose[:, 9, :3, :3]  R_world_from_wrist

    acc_world  = (pos[i+1] - 2*pos[i] + pos[i-1]) * fps²   (central diff)
    if add_gravity: acc_world[:, 1] += 9.81                 (Y-up bias)
    dR_world_i = rot[i+1] @ rot[i].T
    gyro_world = rotvec(dR_world_i) * fps

    R_world_from_FLU[n] = rot[n] @ R_WRIST_FROM_FLU
    R_FLU_from_world[n] = R_world_from_FLU[n].T
    acc_FLU  = R_FLU_from_world @ acc_world
    gyro_FLU = R_FLU_from_world @ gyro_world

Usage
-----
    from imu_lm.data.nymeria_syn_wrist_loader_FLU import (
        make_nymeria_syn_wrist_loaders_flu
    )

    loaders = make_nymeria_syn_wrist_loaders_flu(
        cfg=cfg,
        pkl_root="/mnt/nas2/processed_data_body_Ty_rwrist",
        train_files=train_files,
        val_files=val_files,
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

_XSENS_RIGHT_WRIST_JOINT = 9

# FLU frame: same convention as nymeria_real_wrist_full_body_loader_FLU.py
R_WRIST_FROM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0],   # F (forward)
        [ 0.0, 0.0, 1.0],   # L (left)
        [ 0.0, 1.0, 0.0],   # U (up)
    ],
    dtype=np.float64,
)

assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Core session loader
# ---------------------------------------------------------------------------

def _load_nymeria_syn_wrist_session_flu(
    pkl_path: str,
    target_hz: int,
    add_gravity: bool = True,
) -> Optional[np.ndarray]:
    """Synthesize right wrist IMU in FLU frame from xsens_pose.

    Returns
    -------
    X_imu : [N, 6]  float32  — [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    or None on failure.
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    logger.info("[nymeria_syn_wrist_flu] loading: %s", os.path.basename(pkl_path))
    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        logger.warning("[nymeria_syn_wrist_flu] cannot open %s: %s", pkl_path, exc)
        return None

    xsens_pose = raw.get("gt_data", {}).get("xsens_pose")
    if xsens_pose is None:
        logger.warning("[nymeria_syn_wrist_flu] %s: 'xsens_pose' missing; skipping", pkl_path)
        return None

    if hasattr(xsens_pose, "detach"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)   # [N, 23, 4, 4]

    if xsens_pose.shape[0] < 3:
        logger.warning("[nymeria_syn_wrist_flu] %s: too short; skipping", pkl_path)
        return None

    fps_raw = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz
    )
    ds = max(1, round(fps_raw / target_hz))
    xsens_pose = xsens_pose[::ds]
    N = xsens_pose.shape[0]
    if N < 3:
        return None
    fps = float(target_hz)

    j   = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, j, :3, 3].copy()    # [N, 3] world-frame positions
    rot = xsens_pose[:, j, :3, :3].copy()   # [N, 3, 3] R_world_from_wrist

    # Linear acceleration via central finite difference (world frame)
    acc_world = np.zeros((N, 3), dtype=np.float64)
    acc_world[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc_world[0]    = acc_world[1]
    acc_world[-1]   = acc_world[-2]

    # Gravity bias: real static IMU reads +g in the up axis (Y-up world)
    if add_gravity:
        acc_world[:, 1] += 9.81

    # Angular velocity in world frame: dR_world = R_{i+1} @ R_i^T
    dR_world = np.einsum("nij,nkj->nik", rot[1:], rot[:-1])   # [N-1, 3, 3]
    gyro_raw = ScipyRot.from_matrix(dR_world).as_rotvec().astype(np.float64) * fps
    gyro_world = np.zeros((N, 3), dtype=np.float64)
    gyro_world[:-1] = gyro_raw
    gyro_world[-1]  = gyro_raw[-1]

    # World → FLU: R_world_from_FLU = rot @ R_WRIST_FROM_FLU
    R_world_from_FLU = np.einsum("nij,jk->nik", rot, R_WRIST_FROM_FLU)  # [N, 3, 3]
    R_FLU_from_world = R_world_from_FLU.transpose(0, 2, 1)               # [N, 3, 3]

    acc_flu  = np.einsum("nij,nj->ni", R_FLU_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_FLU_from_world, gyro_world)

    X_imu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)  # [N, 6]

    logger.info(
        "[nymeria_syn_wrist_flu] loaded %s: N=%d fps=%d add_gravity=%s",
        os.path.basename(pkl_path), N, target_hz, add_gravity,
    )
    return X_imu


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

class NymeriaSynWristDatasetFLU(Dataset):
    """Sliding-window dataset: synthetic right-wrist IMU in FLU frame.

    Input X: [6, T]  float32  — [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    Label y: [6, T]  float32  — copy of x (dummy; MAE objective ignores it)
    """

    def __init__(
        self,
        pkl_paths: List[str],
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 50,
        add_gravity: bool = True,
    ):
        super().__init__()
        self.cfg        = cfg
        self.split_name = split_name
        self.stats      = PreprocessStats()

        self._T, self._hop = compute_T_and_hop(cfg)

        self._sessions: List[np.ndarray] = []
        session_lengths: List[int] = []

        _n_workers = min(32, (os.cpu_count() or 4) * 2)
        _load = lambda p: _load_nymeria_syn_wrist_session_flu(
            p, target_hz=target_hz, add_gravity=add_gravity
        )
        _ordered: List = [None] * len(pkl_paths)
        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            fut_to_idx = {pool.submit(_load, p): i for i, p in enumerate(pkl_paths)}
            with tqdm(
                total=len(pkl_paths),
                desc=f"Loading {split_name} sessions (syn wrist FLU)",
                unit="session",
                dynamic_ncols=True,
            ) as pbar:
                for fut in as_completed(fut_to_idx):
                    _ordered[fut_to_idx[fut]] = fut.result()
                    pbar.update(1)

        for path, result in zip(pkl_paths, _ordered):
            if result is None:
                logger.warning("[nymeria_syn_wrist_flu] skipping: %s", path)
                continue
            self._sessions.append(result)
            session_lengths.append(len(result))

        if not self._sessions:
            raise RuntimeError(
                "NymeriaSynWristDatasetFLU: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )

        logger.info(
            "[nymeria_syn_wrist_flu] split=%s sessions=%d windows=%d",
            split_name, len(self._sessions), len(self._starts),
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T

        Xw = self._sessions[si][start : start + T]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        if self.split_name == "train":
            Xproc = apply_augment(Xproc.T, self.cfg).T
        x_tensor = torch.from_numpy(Xproc).float()   # [6, T]
        return x_tensor, x_tensor   # y = x dummy for MAE objective


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
# Loader factory
# ---------------------------------------------------------------------------

def make_nymeria_syn_wrist_loaders_flu(
    cfg: Any,
    pkl_root: str,
    train_files: List[str],
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 50,
    add_gravity: bool = True,
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders for MAE on Nymeria synthetic wrist FLU."""
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))

    def _abs(files: List[str]) -> List[str]:
        return [
            f if os.path.isabs(f) else os.path.join(pkl_root, f)
            for f in files
        ]

    def _make(name: str, files: List[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = NymeriaSynWristDatasetFLU(
            pkl_paths=_abs(files),
            cfg=cfg,
            split_name=name,
            target_hz=target_hz,
            add_gravity=add_gravity,
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
