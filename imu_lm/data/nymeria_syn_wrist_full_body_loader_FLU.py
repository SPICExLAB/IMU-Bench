"""NymeriaSynWristFullBodyDatasetFLU: synthesized wrist IMU in FLU frame → 24-joint SMPL labels.

Identical to ``nymeria_syn_wrist_full_body_loader_global.py`` for the label side
(24-joint SMPL 6D rotations from xsens_pose IK), but the IMU input is expressed
in the **FLU (Forward-Left-Up) body frame** rather than the world frame.

This makes the synthetic data directly compatible with the real-wrist FLU loader
(NymeriaRealWristFullBodyDatasetFLU) for joint training / transfer experiments.

Input / labels
--------------
    x : [6, T]            float32  — [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    y : [N_patches, 144]  float32  — per-patch 6D rotation for all 24 SMPL joints

Coordinate convention
---------------------
FLU frame (right wrist, T-pose):
    F (+x) = finger direction (-x_smpl)
    L (+y) = thumb side  (+z_smpl)
    U (+z) = dorsal/up   (+y_smpl)

  R_WRIST_FROM_FLU = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]

IMU synthesis (same as global loader but rotated)
-------------------------------------------------
    pos  = xsens_pose[:, 9, :3, 3]   world-frame wrist positions
    rot  = xsens_pose[:, 9, :3, :3]  R_world_from_wrist

    acc_world  = (pos[i+1] - 2*pos[i] + pos[i-1]) * fps²
    if add_gravity: acc_world[:, 1] += 9.81
    gyro_world = rotvec(rot[i+1] @ rot[i].T) * fps

    R_world_from_FLU = rot @ R_WRIST_FROM_FLU
    R_FLU_from_world = R_world_from_FLU.T
    acc_flu  = R_FLU_from_world @ acc_world
    gyro_flu = R_FLU_from_world @ gyro_world

Usage
-----
    from imu_lm.data.nymeria_syn_wrist_full_body_loader_FLU import (
        make_nymeria_syn_wrist_full_body_loaders_flu
    )

    loaders = make_nymeria_syn_wrist_full_body_loaders_flu(
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

NUM_BODY_JOINTS = 24
ROT6D_DIM = 6
BODY_ROT6D_DIM = NUM_BODY_JOINTS * ROT6D_DIM   # 144

_XSENS_RIGHT_WRIST_JOINT = 9

R_WRIST_FROM_FLU = np.array(
    [
        [-1.0, 0.0, 0.0],
        [ 0.0, 0.0, 1.0],
        [ 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

assert np.allclose(R_WRIST_FROM_FLU.T @ R_WRIST_FROM_FLU, np.eye(3), atol=1e-6)
assert np.isclose(np.linalg.det(R_WRIST_FROM_FLU), 1.0, atol=1e-6)

_XSENS_TO_SMPL = [
    0, 19, 15, 1, 20, 16, 3, 21, 17, 4, 22, 18,
    5, 11,  7, 6, 12,  8, 13,  9, 13,  9, 13,  9,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _rot6d_to_mat(r6d: np.ndarray) -> np.ndarray:
    a1 = r6d[:3].astype(np.float64)
    a2 = r6d[3:6].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0).astype(np.float32)


def _mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    return mat[:2, :].flatten().astype(np.float32)


def xsens_to_SMPL(xsens_rotmat: np.ndarray) -> np.ndarray:
    """[N,23,3,3] Xsens global rotations → [N,24,3,3] SMPL global rotations."""
    perm = [1, 2, 0]
    xsens_rotmat = xsens_rotmat[:, :, perm, :]
    xsens_rotmat = xsens_rotmat[:, :, :, perm]
    xsens_t = torch.from_numpy(xsens_rotmat).float()
    Rx90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    Rz90 = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    xsens_t = Rz90 @ (Rx90 @ xsens_t)
    smpl_rotmat = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(
        xsens_t.shape[0], 24, -1, -1
    ).clone()
    for smpl_idx, xsens_idx in enumerate(_XSENS_TO_SMPL):
        smpl_rotmat[:, smpl_idx] = xsens_t[:, xsens_idx]
    return smpl_rotmat.numpy()


# ---------------------------------------------------------------------------
# Core session loader
# ---------------------------------------------------------------------------

def _load_nymeria_syn_wrist_full_body_session_flu(
    pkl_path: str,
    target_hz: int,
    add_gravity: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Synthesize wrist IMU in FLU frame + compute 24-joint SMPL labels.

    Returns
    -------
    X_imu : [N, 6]     float32  — [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    y     : [N, 24, 6] float32  — 6D local rotations for all 24 SMPL joints
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    import core.articulate as art
    from core.paths import Paths

    logger.info("[nymeria_syn_full_body_flu] loading: %s", os.path.basename(pkl_path))
    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        logger.warning("[nymeria_syn_full_body_flu] cannot open %s: %s", pkl_path, exc)
        return None

    xsens_pose = raw.get("gt_data", {}).get("xsens_pose")
    if xsens_pose is None:
        logger.warning("[nymeria_syn_full_body_flu] %s: 'xsens_pose' missing; skipping", pkl_path)
        return None

    if hasattr(xsens_pose, "detach"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)   # [N, 23, 4, 4]

    N = xsens_pose.shape[0]
    if N < 3:
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
    pos = xsens_pose[:, j, :3, 3].copy()    # [N, 3] world positions
    rot = xsens_pose[:, j, :3, :3].copy()   # [N, 3, 3] R_world_from_wrist

    # Linear acceleration (world frame, central difference)
    acc_world = np.zeros((N, 3), dtype=np.float64)
    acc_world[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc_world[0]    = acc_world[1]
    acc_world[-1]   = acc_world[-2]

    if add_gravity:
        acc_world[:, 1] += 9.81   # Y-up gravity bias

    # Angular velocity (world frame)
    dR_world = np.einsum("nij,nkj->nik", rot[1:], rot[:-1])
    gyro_raw = ScipyRot.from_matrix(dR_world).as_rotvec().astype(np.float64) * fps
    gyro_world = np.zeros((N, 3), dtype=np.float64)
    gyro_world[:-1] = gyro_raw
    gyro_world[-1]  = gyro_raw[-1]

    # World → FLU
    R_world_from_FLU = np.einsum("nij,jk->nik", rot, R_WRIST_FROM_FLU)   # [N, 3, 3]
    R_FLU_from_world = R_world_from_FLU.transpose(0, 2, 1)                # [N, 3, 3]

    acc_flu  = np.einsum("nij,nj->ni", R_FLU_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_FLU_from_world, gyro_world)

    X_imu = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)  # [N, 6]

    # SMPL labels via Xsens → SMPL → IK
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))
    xsens_R       = xsens_pose[:, :, :3, :3].astype(np.float32)
    smpl_global_R = xsens_to_SMPL(xsens_R)                               # [N, 24, 3, 3]
    smpl_local_t  = body_model.inverse_kinematics_R(
        torch.from_numpy(smpl_global_R).float()
    )
    smpl_local_R = smpl_local_t.cpu().numpy().astype(np.float64)
    y = matrix_to_rotation_6d(
        torch.from_numpy(smpl_local_R.astype(np.float32))
    ).numpy()   # [N, 24, 6]

    logger.info(
        "[nymeria_syn_full_body_flu] loaded %s: N=%d fps=%d add_gravity=%s",
        os.path.basename(pkl_path), N, target_hz, add_gravity,
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

class NymeriaSynWristFullBodyDatasetFLU(Dataset):
    """Sliding-window dataset: synthetic wrist FLU IMU → 24-joint SMPL labels.

    Input  X: [6, T]            float32  — [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
    Label  y: [N_patches, 144]  float32  — per-patch 6D rotation for all 24 SMPL joints
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

        vit_cfg  = cfg_get(cfg, ["vit1d"], {}) or {}
        dec_cfg  = (vit_cfg.get("pose_decoder") or {})
        pose_cfg = cfg_get(cfg, ["objective", "pose"], {}) or \
                   cfg_get(cfg, ["objective", "pose_probe"], {}) or {}

        self._patch_size  = int(vit_cfg.get("patch_size", 16))
        self._num_patches = self._T // self._patch_size
        self._sub_patches = int(dec_cfg.get("sub_patches", 1))
        self._reset_root  = bool(pose_cfg.get("reset_root_orientation", False))

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []
        session_lengths: List[int] = []

        _n_workers = min(32, (os.cpu_count() or 4) * 2)
        _load = lambda p: _load_nymeria_syn_wrist_full_body_session_flu(
            p, target_hz=target_hz, add_gravity=add_gravity
        )
        _ordered: List = [None] * len(pkl_paths)
        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            fut_to_idx = {pool.submit(_load, p): i for i, p in enumerate(pkl_paths)}
            with tqdm(
                total=len(pkl_paths),
                desc=f"Loading {split_name} sessions (syn wrist full-body FLU)",
                unit="session",
                dynamic_ncols=True,
            ) as pbar:
                for fut in as_completed(fut_to_idx):
                    _ordered[fut_to_idx[fut]] = fut.result()
                    pbar.update(1)

        for path, result in zip(pkl_paths, _ordered):
            if result is None:
                logger.warning("[nymeria_syn_full_body_flu] skipping: %s", path)
                continue
            X, y = result
            self._sessions.append((X, y))
            session_lengths.append(len(X))

        if not self._sessions:
            raise RuntimeError(
                "NymeriaSynWristFullBodyDatasetFLU: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )
        logger.info(
            "[nymeria_syn_full_body_flu] split=%s sessions=%d windows=%d",
            split_name, len(self._sessions), len(self._starts),
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T

        X_raw, y_arr = self._sessions[si]
        Xw = X_raw[start : start + T]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        if self.split_name == "train":
            Xproc = apply_augment(Xproc.T, self.cfg).T
        x_tensor = torch.from_numpy(Xproc).float()

        P  = self._patch_size
        K  = self._sub_patches
        Q  = P // K
        out_dim = K * BODY_ROT6D_DIM
        patch_labels = np.empty((self._num_patches, out_dim), dtype=np.float32)

        for p in range(self._num_patches):
            for k in range(K):
                frame = start + p * P + k * Q + Q // 2
                patch_labels[p, k * BODY_ROT6D_DIM : (k + 1) * BODY_ROT6D_DIM] = (
                    y_arr[frame].reshape(-1)
                )

        if self._reset_root:
            R0     = _rot6d_to_mat(patch_labels[0, :6])
            R0_inv = R0.T
            for p in range(self._num_patches):
                for k in range(K):
                    base = k * BODY_ROT6D_DIM
                    Ri = _rot6d_to_mat(patch_labels[p, base : base + 6])
                    patch_labels[p, base : base + 6] = _mat_to_rot6d(R0_inv @ Ri)

        y_tensor = torch.from_numpy(patch_labels).float()
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
# Loader factory
# ---------------------------------------------------------------------------

def make_nymeria_syn_wrist_full_body_loaders_flu(
    cfg: Any,
    pkl_root: str,
    train_files: List[str],
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 50,
    add_gravity: bool = True,
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders for full-body pose on Nymeria synthetic wrist FLU."""
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
        ds = NymeriaSynWristFullBodyDatasetFLU(
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
