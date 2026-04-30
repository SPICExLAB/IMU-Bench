"""NymeriaSynWristFullBodyDatasetGlobal: synthesized wrist IMU in global frame → 24-joint SMPL labels.

Identical to ``nymeria_wrist_full_body_loader.py`` except:
1. The synthetic accelerometer is left in the **world (global) coordinate frame**
   (the acceleration from position finite-differences is already in world frame).
2. The synthetic gyroscope is expressed in the **world frame** via
   ``dR_world = R_{i+1} @ R_i^T`` rather than the body-frame
   ``dR_body = R_i^T @ R_{i+1}`` used in the original loader.
3. An optional gravity bias ``[0, +9.81, 0]`` (Y-up Nymeria/SMPL world) is added
   to the accelerometer to mimic a real IMU (``add_gravity=True``, the default).

Input / labels
--------------
    x : [6, T]            float32  — [acc_x_w, acc_y_w, acc_z_w, gyro_x_w, gyro_y_w, gyro_z_w]
                                      synthesized right-wrist IMU in world (Y-up) frame
    y : [N_patches, 144]  float32  — per-patch 6D rotation for all 24 SMPL joints
                                      (24 × 6 = 144 dims), one label per patch centre

Coordinate convention
---------------------
World frame is Y-up (Nymeria / SMPL convention):
    X — right,  Y — up,  Z — backward
Gravity is ``-Y``. With ``add_gravity=True`` the accelerometer output is biased
by ``+9.81 m/s²`` in the Y direction (mimics a real static IMU on a flat surface).

IMU synthesis
-------------
From ``gt_data['xsens_pose']`` [N, 23, 4, 4], joint index 9 (right wrist):
    pos [N, 3]    = xsens_pose[:, 9, :3, 3]  — world-frame positions
    rot [N, 3, 3] = xsens_pose[:, 9, :3, :3] — rotation matrices (body→world)

    acc_world  = (pos[i+1] - 2*pos[i] + pos[i-1]) * fps²     (central finite difference)
    dR_world_i = rot[i+1] @ rot[i].T                          (world-frame relative rotation)
    gyro_world = rotvec(dR_world_i) * fps                      (world-frame angular velocity)

Usage
-----
    from imu_lm.data.nymeria_syn_wrist_full_body_loader_global import (
        make_nymeria_syn_wrist_full_body_loaders_global
    )

    loaders = make_nymeria_syn_wrist_full_body_loaders_global(
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
BODY_ROT6D_DIM = NUM_BODY_JOINTS * ROT6D_DIM  # 144

# Xsens joint index used as right-wrist proxy for IMU synthesis (joint 9).
_XSENS_RIGHT_WRIST_JOINT = 9

# Xsens → SMPL joint mapping
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


# ---------------------------------------------------------------------------
# Core session loader
# ---------------------------------------------------------------------------

def _load_nymeria_syn_wrist_full_body_session_global(
    pkl_path: str,
    target_hz: int,
    add_gravity: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Synthesize wrist IMU in world frame from Xsens GT and compute SMPL labels.

    Pipeline
    --------
    1. Read ``gt_data['xsens_pose']`` [N, 23, 4, 4].
       Joint 9 (right wrist): positions in world frame, rotation matrices body→world.

    2. Synthesize accelerometer in world frame:
           acc_world = (pos[i+1] - 2*pos[i] + pos[i-1]) * fps²
       This is the true linear acceleration in world frame.

    3. Optionally add gravity bias (Y-up: g_world = [0, -9.81, 0]):
           add_gravity=True  → acc_world += [0, +9.81, 0]
               (mimics real IMU: static sensor on flat surface reads +9.81 in Y)
           add_gravity=False → keep linear acceleration as-is

    4. Synthesize gyroscope in world frame:
           dR_world_i = rot[i+1] @ rot[i].T    (world-frame relative rotation)
           gyro_world = rotvec(dR_world_i) * fps
       Contrast with the body-frame gyro (used in nymeria_wrist_full_body_loader.py):
           dR_body_i  = rot[i].T @ rot[i+1]
           gyro_body  = rotvec(dR_body_i) * fps

    5. Stack X_imu = [acc_world | gyro_world].

    6. Build SMPL labels via xsens → SMPL → IK (same as other full-body loaders).

    Returns
    -------
    X_imu : [N, 6]     float32  — world-frame acc + gyro
    y     : [N, 24, 6] float32  — 6D local rotations for all 24 SMPL joints
    """
    from scipy.spatial.transform import Rotation as ScipyRot
    import core.articulate as art
    from core.paths import Paths

    logger.info("[nymeria_syn_global] loading: %s", os.path.basename(pkl_path))
    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        logger.warning("[nymeria_syn_global] cannot open %s: %s", pkl_path, exc)
        return None

    gt_data    = raw.get("gt_data", {})
    xsens_pose = gt_data.get("xsens_pose")
    if xsens_pose is None:
        logger.warning("[nymeria_syn_global] %s: 'xsens_pose' missing; skipping", pkl_path)
        return None

    if hasattr(xsens_pose, "detach"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)   # [N, 23, 4, 4]

    N = xsens_pose.shape[0]
    if N < 3:
        logger.warning("[nymeria_syn_global] %s: too short (%d frames); skipping", pkl_path, N)
        return None

    fps_raw  = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz
    )
    ds = max(1, round(fps_raw / target_hz))
    xsens_pose = xsens_pose[::ds]
    N = xsens_pose.shape[0]
    if N < 3:
        return None
    fps = target_hz

    # ------------------------------------------------------------------
    # Xsens joint 9 (right wrist) in world frame
    # ------------------------------------------------------------------
    j   = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, j, :3, 3].astype(np.float64)    # [N, 3] world-frame positions
    rot = xsens_pose[:, j, :3, :3].astype(np.float64)   # [N, 3, 3] body→world

    # ------------------------------------------------------------------
    # Step 2: world-frame linear acceleration (central difference)
    # ------------------------------------------------------------------
    acc_world = np.zeros((N, 3), dtype=np.float64)
    acc_world[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc_world[0]    = acc_world[1]
    acc_world[-1]   = acc_world[-2]

    # ------------------------------------------------------------------
    # Step 3: optional gravity bias (Y-up: gravity vector = [0, -9.81, 0])
    # Real IMU in static position: a_sensor = -g = [0, +9.81, 0]
    # ------------------------------------------------------------------
    if add_gravity:
        acc_world[:, 1] += 9.81   # bias +Y to mimic real accelerometer

    # ------------------------------------------------------------------
    # Step 4: world-frame angular velocity
    # dR_world_i = rot[i+1] @ rot[i].T  →  gyro = rotvec(dR_world) * fps
    # ------------------------------------------------------------------
    dR_world = np.einsum("nij,nkj->nik", rot[1:], rot[:-1])   # [N-1, 3, 3]  (R_{i+1} @ R_i^T)
    gyro_raw = ScipyRot.from_matrix(dR_world).as_rotvec().astype(np.float64) * fps  # [N-1, 3]
    gyro_world = np.zeros((N, 3), dtype=np.float64)
    gyro_world[:-1] = gyro_raw
    gyro_world[-1]  = gyro_raw[-1]

    # ------------------------------------------------------------------
    # Step 5: stack input
    # ------------------------------------------------------------------
    X_imu = np.concatenate([acc_world, gyro_world], axis=1).astype(np.float32)  # [N, 6]

    # ------------------------------------------------------------------
    # Step 6: SMPL labels — identical to other Nymeria full-body loaders
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
        "[nymeria_syn_global] loaded %s: N=%d fps=%d add_gravity=%s",
        os.path.basename(pkl_path), N, fps, add_gravity,
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

class NymeriaSynWristFullBodyDatasetGlobal(Dataset):
    """Sliding-window dataset: synthetic wrist IMU in world frame → 24-joint SMPL labels.

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
        _load = lambda p: _load_nymeria_syn_wrist_full_body_session_global(
            p, target_hz=target_hz, add_gravity=add_gravity
        )
        _ordered: List = [None] * len(pkl_paths)
        with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
            _fut_to_idx = {_pool.submit(_load, p): i for i, p in enumerate(pkl_paths)}
            with tqdm(
                total=len(pkl_paths),
                desc=f"Loading {split_name} sessions (syn wrist global full-body)",
                unit="session",
                dynamic_ncols=True,
            ) as _pbar:
                for _fut in as_completed(_fut_to_idx):
                    _ordered[_fut_to_idx[_fut]] = _fut.result()
                    _pbar.update(1)

        for path, result in zip(pkl_paths, _ordered):
            if result is None:
                logger.warning("[nymeria_syn_global] skipping: %s", path)
                continue
            X, y = result
            self._sessions.append((X, y))
            session_lengths.append(len(X))

        if not self._sessions:
            raise RuntimeError(
                "NymeriaSynWristFullBodyDatasetGlobal: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )

        logger.info(
            "[nymeria_syn_global] split=%s sessions=%d windows=%d",
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

def make_nymeria_syn_wrist_full_body_loaders_global(
    cfg: Any,
    pkl_root: str,
    train_files: List[str],
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 50,
    add_gravity: bool = True,
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders for full-body pose on Nymeria synthetic wrist (world frame)."""
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
        ds = NymeriaSynWristFullBodyDatasetGlobal(
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
