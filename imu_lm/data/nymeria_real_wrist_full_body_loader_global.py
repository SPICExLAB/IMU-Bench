"""NymeriaRealWristFullBodyDatasetGlobal: real wrist IMU in global (world) frame → 24-joint SMPL labels.

Identical to ``nymeria_real_wrist_full_body_loader_FLU.py`` except that the
input signal stays in the **world (global) coordinate frame** rather than being
rotated into the wrist FLU body frame.

Input / labels
--------------
    x : [6, T]            float32  — [acc_x_w, acc_y_w, acc_z_w, gyro_x_w, gyro_y_w, gyro_z_w]
                                      real wrist IMU expressed in the world (Y-up) frame
    y : [N_patches, 144]  float32  — per-patch 6D rotation for all 24 SMPL joints
                                      (24 × 6 = 144 dims), one label per patch centre

Coordinate convention
---------------------
World frame is Y-up (Nymeria / SMPL convention):
    X — right,  Y — up,  Z — backward (out of scene)
Gravity points in the -Y direction, so with ``add_gravity=True`` (default) the
accelerometer output is biased by ~9.81 m/s² in the +Y direction.

Data source
-----------
pkl files with ``has_wrist_imu=True`` (from ``processed_data_body_Ty_rwrist``).
Keys used:
    - ``wrist_imu_data['accel']``, ``wrist_imu_data['gyro']``  — real IMU signal (local CPF)
    - ``wrist_gt_data['orientation']``                          — CPF→world rotation
    - ``gt_data['xsens_pose']``                                 — SMPL label source

Usage
-----
    from imu_lm.data.nymeria_real_wrist_full_body_loader_global import (
        make_nymeria_real_wrist_full_body_loaders_global
    )

    loaders = make_nymeria_real_wrist_full_body_loaders_global(
        cfg=cfg,
        pkl_root="/mnt/nas2/processed_data_body_Ty_rwrist",
        train_files=train_files,
        val_files=val_files,
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
BODY_ROT6D_DIM = NUM_BODY_JOINTS * ROT6D_DIM  # 144

# Xsens → SMPL joint mapping (same as FLU variant)
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


def xsens_to_SMPL(xsens_rotmat: np.ndarray) -> np.ndarray:
    """Convert Xsens global rotation matrices [N,23,3,3] → SMPL global [N,24,3,3]."""
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
    return smpl_rotmat.numpy()  # [N, 24, 3, 3]


def _quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, w) quaternions to rotation matrices. [N, 4] → [N, 3, 3]."""
    from scipy.spatial.transform import Rotation as ScipyRot
    return ScipyRot.from_quat(quat).as_matrix().astype(np.float64)


# ---------------------------------------------------------------------------
# Core session loader
# ---------------------------------------------------------------------------

def _load_nymeria_real_wrist_full_body_session_global(
    pkl_path: str,
    target_hz: int,
    add_gravity: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load a Nymeria session: real wrist IMU in world frame → all 24 SMPL joint labels.

    Pipeline
    --------
    1. Read ``wrist_imu_data['accel']`` / ``['gyro']`` (local wrist CPF frame).
    2. Read ``wrist_gt_data['orientation']`` → R_world_from_wrist_cpf [N, 3, 3].
    3. Rotate acc and gyro into the world frame:
           acc_world  = R_world_from_wrist_cpf @ acc_local
           gyro_world = R_world_from_wrist_cpf @ gyro_local
    4. Gravity handling (Y-up world, gravity = -Y):
           add_gravity=True  → keep as-is (real IMU, biased by +9.81 in Y).
           add_gravity=False → subtract gravity: acc_world += [0, -9.81, 0]
               (converts from IMU-reading convention to linear-acceleration).
       NOTE: unlike the FLU variant the gravity direction in world coords is
             [0, -9.81, 0] (Y-up), no additional frame rotation needed.
    5. Stack into X_imu = [acc_world | gyro_world] — stays in world frame.
    6. Build SMPL labels via xsens → SMPL IK (same as FLU variant).

    Returns
    -------
    X_imu : [N, 6]     float32  — world-frame acc + gyro
    y     : [N, 24, 6] float32  — 6D local rotations for all 24 SMPL joints
    """
    import core.articulate as art
    from core.paths import Paths

    logger.info("[nymeria_real_wrist_global] loading: %s", os.path.basename(pkl_path))
    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        logger.warning("[nymeria_real_wrist_global] cannot open %s: %s", pkl_path, exc)
        return None

    # ------------------------------------------------------------------
    # Check required keys
    # ------------------------------------------------------------------
    if not raw.get("has_wrist_imu", False):
        logger.warning(
            "[nymeria_real_wrist_global] %s: has_wrist_imu=False, skipping", pkl_path
        )
        return None

    wrist_imu = raw.get("wrist_imu_data", {})
    wrist_gt  = raw.get("wrist_gt_data",  {})

    for key in ("accel", "gyro"):
        if key not in wrist_imu:
            logger.warning(
                "[nymeria_real_wrist_global] %s: '%s' missing in wrist_imu_data", pkl_path, key
            )
            return None

    if "orientation" not in wrist_gt:
        logger.warning(
            "[nymeria_real_wrist_global] %s: 'orientation' missing in wrist_gt_data", pkl_path
        )
        return None

    xsens_pose = raw.get("gt_data", {}).get("xsens_pose")
    if xsens_pose is None:
        logger.warning(
            "[nymeria_real_wrist_global] %s: 'xsens_pose' missing (needed for labels)", pkl_path
        )
        return None

    # ------------------------------------------------------------------
    # Determine sampling rates and downsample to target_hz
    # ------------------------------------------------------------------
    body_fps_raw  = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz
    )
    wrist_fps_raw = float(raw.get("wrist_imu_sampling_rate_hz", body_fps_raw))

    body_ds  = max(1, round(body_fps_raw  / target_hz))
    wrist_ds = max(1, round(wrist_fps_raw / target_hz))

    # Body pose (for labels)
    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)[::body_ds]   # [N_b, 23, 4, 4]

    # Wrist IMU (local CPF frame)
    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    acc_local  = _to_numpy(wrist_imu["accel"]).astype(np.float64)[::wrist_ds]   # [N_w, 3]
    gyro_local = _to_numpy(wrist_imu["gyro"]).astype(np.float64)[::wrist_ds]    # [N_w, 3]

    # Wrist GT orientation → R_world_from_wrist_cpf  [N_w, 3, 3]
    orient_raw = wrist_gt["orientation"]

    try:
        import pypose as pp
        _is_lietensor = isinstance(orient_raw, pp.LieTensor)
    except ImportError:
        _is_lietensor = False

    if _is_lietensor:
        R_world_from_wrist_cpf = (
            orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
        )
    else:
        orient_np = _to_numpy(orient_raw).astype(np.float64)[::wrist_ds]
        R_world_from_wrist_cpf = _quat_xyzw_to_rotmat(orient_np)

    # ------------------------------------------------------------------
    # Align lengths
    # ------------------------------------------------------------------
    N = min(xsens_pose.shape[0], acc_local.shape[0])
    if N < 3:
        logger.warning(
            "[nymeria_real_wrist_global] %s: too short after alignment (%d frames)",
            pkl_path, N
        )
        return None

    xsens_pose             = xsens_pose[:N]
    acc_local              = acc_local[:N]
    gyro_local             = gyro_local[:N]
    R_world_from_wrist_cpf = R_world_from_wrist_cpf[:N]

    # ------------------------------------------------------------------
    # Step 3: local CPF → world frame
    #   acc_world  = R_world_from_wrist_cpf @ acc_local
    #   gyro_world = R_world_from_wrist_cpf @ gyro_local
    # ------------------------------------------------------------------
    acc_world  = np.einsum("nij,nj->ni", R_world_from_wrist_cpf, acc_local)   # [N, 3]
    gyro_world = np.einsum("nij,nj->ni", R_world_from_wrist_cpf, gyro_local)  # [N, 3]

    # ------------------------------------------------------------------
    # Step 4: gravity handling (Y-up: gravity = [0, -9.81, 0] in world)
    #   Real IMU:  acc_world = a_linear_world - g_world  (g_world = [0,-9.81,0])
    #   add_gravity=True  → keep as-is (includes gravity bias)
    #   add_gravity=False → a_linear = acc_world - g_world = acc_world + [0, 9.81, 0]
    # ------------------------------------------------------------------
    if not add_gravity:
        g_world = np.array([0.0, -9.81, 0.0], dtype=np.float64)
        acc_world = acc_world - g_world   # subtract gravity to get linear accel

    # ------------------------------------------------------------------
    # Step 5: build input — stays in world frame (no body-frame rotation)
    # ------------------------------------------------------------------
    X_imu = np.concatenate([acc_world, gyro_world], axis=1).astype(np.float32)  # [N, 6]

    # ------------------------------------------------------------------
    # Step 6: SMPL IK/FK → all 24 joint local rotations (6D)
    # ------------------------------------------------------------------
    body_model = art.model.ParametricModel(Paths.SMPL_FILE, device=torch.device("cpu"))

    xsens_R       = xsens_pose[:, :, :3, :3].astype(np.float32)   # [N, 23, 3, 3]
    smpl_global_R = xsens_to_SMPL(xsens_R)                         # [N, 24, 3, 3]

    smpl_global_t = torch.from_numpy(smpl_global_R).float()
    smpl_local_t  = body_model.inverse_kinematics_R(smpl_global_t)  # [N, 24, 3, 3]
    smpl_local_R  = smpl_local_t.cpu().numpy().astype(np.float64)

    y = matrix_to_rotation_6d(
        torch.from_numpy(smpl_local_R.astype(np.float32))
    ).numpy()   # [N, 24, 6]

    logger.info(
        "[nymeria_real_wrist_global] loaded %s: N=%d fps=%.0f",
        os.path.basename(pkl_path), N, target_hz,
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

class NymeriaRealWristFullBodyDatasetGlobal(Dataset):
    """Sliding-window dataset: real wrist IMU in world frame → 24-joint SMPL labels.

    Input  X: [6, T]            float32  — world-frame [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    Label  y: [N_patches, 144]  float32  — per-patch 6D rotation for all 24 SMPL joints
               where N_patches = T // patch_size
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
        self.cfg         = cfg
        self.split_name  = split_name
        self.target_hz   = target_hz
        self.add_gravity = add_gravity
        self.stats       = PreprocessStats()

        self._T, self._hop = compute_T_and_hop(cfg)

        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size  = int(vit_cfg.get("patch_size", 16))
        self._num_patches = self._T // self._patch_size

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []
        session_lengths: List[int] = []

        _n_workers = min(32, (os.cpu_count() or 4) * 2)
        _load = lambda p: _load_nymeria_real_wrist_full_body_session_global(
            p, target_hz=target_hz, add_gravity=add_gravity
        )
        _ordered: List = [None] * len(pkl_paths)
        with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
            _fut_to_idx = {_pool.submit(_load, p): i for i, p in enumerate(pkl_paths)}
            with tqdm(
                total=len(pkl_paths),
                desc=f"Loading {split_name} sessions (real wrist global full-body)",
                unit="session",
                dynamic_ncols=True,
            ) as _pbar:
                for _fut in as_completed(_fut_to_idx):
                    _ordered[_fut_to_idx[_fut]] = _fut.result()
                    _pbar.update(1)

        for path, result in zip(pkl_paths, _ordered):
            if result is None:
                logger.warning("[nymeria_real_wrist_global] skipping: %s", path)
                continue
            X, y = result
            self._sessions.append((X, y))
            session_lengths.append(len(X))

        if not self._sessions:
            raise RuntimeError(
                "NymeriaRealWristFullBodyDatasetGlobal: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )

        logger.info(
            "[nymeria_real_wrist_global] split=%s sessions=%d windows=%d",
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

        P = self._patch_size
        patch_labels = np.empty((self._num_patches, BODY_ROT6D_DIM), dtype=np.float32)
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y_arr[frame].reshape(-1)   # [24, 6] → [144]

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

def make_nymeria_real_wrist_full_body_loaders_global(
    cfg: Any,
    pkl_root: str,
    train_files: List[str],
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 50,
    add_gravity: bool = True,
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders for full-body pose on Nymeria real wrist (world frame)."""
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
        ds = NymeriaRealWristFullBodyDatasetGlobal(
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
