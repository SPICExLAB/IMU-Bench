"""NymeriaWristOdomDataset: wrist IMU (sensor/body frame) → root displacement labels.

Inputs / labels
---------------
    x : [6, T]          float32  —  right-wrist acc + gyro **in the sensor/body frame**
                                     (identical derivation to NymeriaWristDataset / MAE)
    y : [N_patches, 3]  float32  —  root (pelvis) XYZ displacement over each patch interval
                                     y[p] = pos_root[p*P + P] − pos_root[p*P]  (metres)
                                     N_patches = T // patch_size  (default 8 = 128 // 16)

IMU derivation (sensor frame, same as MAE loader)
-------------------------------------------------
Let R[t] = xsens_pose[t, wrist, :3, :3] be the world←sensor rotation matrix.

  acc[t]        = second-order finite difference of world-frame wrist position × fps²

  gyro[t]       = log(R[t]ᵀ R[t+1]) / Δt   (angular velocity in sensor/body frame)

Root displacement label
-----------------------
  pos_root[t] = xsens_pose[t, 0, :3, 3]   (Xsens joint 0 = root/pelvis, world frame)
  y[p]        = pos_root[start + (p+1)*P] − pos_root[start + p*P]    p = 0 … P_t−1
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
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

# Xsens joint indices
_XSENS_RIGHT_WRIST_JOINT = 9   # right wrist (used for IMU)
_XSENS_ROOT_JOINT        = 0   # pelvis / root (used for displacement labels)

WRIST_SENSOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------

@dataclass
class _OdomSessionCache:
    path: Optional[str] = None
    X:    Optional[np.ndarray] = None   # [N, 6]   float32  global-frame acc + gyro
    pos:  Optional[np.ndarray] = None   # [N, 3]   float32  root world position


# ---------------------------------------------------------------------------
# Window index helpers
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
# Session loader
# ---------------------------------------------------------------------------

def _load_odom_session(
    pkl_path: str,
    target_hz: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load one Nymeria pkl and return (X_sensor, pos_root).

    X_sensor : [N, 6]  float32
        acc   — second-order finite diff of wrist world-frame position × fps²
        gyro  — log(R[t]ᵀ R[t+1]) * fps  (angular velocity in sensor/body frame)

    pos_root : [N, 3]  float32
        World-frame position of Xsens joint 0 (pelvis/root) at each frame.
        Used to compute per-patch displacement labels inside __getitem__.

    Returns None on load failure.
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    try:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
    except Exception as exc:
        logger.warning("[odom] failed to open %s: %s", pkl_path, exc)
        return None

    gt_data    = raw.get("gt_data", {})
    xsens_pose = gt_data.get("xsens_pose")
    if xsens_pose is None:
        logger.warning("[odom] %s: no 'xsens_pose'; skipping.", pkl_path)
        return None

    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float32)   # [N, 23, 4, 4]

    N = xsens_pose.shape[0]
    if N < 3:
        logger.warning("[odom] %s: too short (%d frames); skipping.", pkl_path, N)
        return None

    fps = float(
        raw.get("imu_sampling_rate_hz")
        or raw.get("pose_sampling_rate_hz")
        or target_hz
    )

    # ------------------------------------------------------------------
    # RIGHT WRIST: global-frame acc + gyro
    # ------------------------------------------------------------------
    jw  = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, jw, :3, 3]    # [N, 3]  world-frame position
    R   = xsens_pose[:, jw, :3, :3]   # [N, 3, 3]  world←sensor rotation

    # --- acceleration in world frame (pure kinematic, no gravity term) ---
    acc_global          = np.zeros((N, 3), dtype=np.float32)
    acc_global[1:-1]    = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc_global[0]       = acc_global[1]
    acc_global[-1]      = acc_global[-2]

    # --- angular velocity in sensor/body frame (matches MAE loader) ---
    # dR[t] = R[t].T @ R[t+1]  →  rotvec in sensor frame
    dR           = np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])
    omega_sensor = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec()
        .reshape(N - 1, 3).astype(np.float32)
        * fps
    )
    gyro         = np.zeros((N, 3), dtype=np.float32)
    gyro[:-1]    = omega_sensor
    gyro[-1]     = omega_sensor[-1]

    X = np.concatenate([acc_global, gyro], axis=1)  # [N, 6]

    # ------------------------------------------------------------------
    # ROOT: world-frame position for displacement labels
    # ------------------------------------------------------------------
    jr       = _XSENS_ROOT_JOINT
    pos_root = xsens_pose[:, jr, :3, 3].astype(np.float32)  # [N, 3]

    logger.info(
        "[odom] loaded %s: N=%d  fps=%.0f",
        os.path.basename(pkl_path), N, fps,
    )
    return X, pos_root


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NymeriaWristOdomDataset(Dataset):
    """Sliding-window dataset: sensor-frame wrist IMU → root displacement.

    Returns ``(x, y)`` pairs where::

        x : torch.Tensor  [6, T]          float32
            Right-wrist acc + gyro in the **sensor/body frame** (same as MAE loader).

        y : torch.Tensor  [N_patches, 3]  float32
            Root (pelvis) XYZ displacement per patch interval (metres).
            y[p] = pos_root[start + (p+1)*P] − pos_root[start + p*P]
            N_patches = T // patch_size  (default 8 for T=128, P=16)

    Args:
        pkl_paths:   Absolute paths to Nymeria ``.pkl`` session files.
        cfg:         Config dict (base.yaml structure).
        split_name:  Logging label.
        target_hz:   Sampling rate (default 50 Hz).
    """

    def __init__(
        self,
        pkl_paths: List[str],
        cfg: Any,
        split_name: str = "train",
        target_hz: int = 50,
    ):
        super().__init__()
        self.cfg        = cfg
        self.split_name = split_name
        self.target_hz  = target_hz
        self.stats      = PreprocessStats()

        self._T, self._hop = compute_T_and_hop(cfg)

        vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size  = int(vit_cfg.get("patch_size", 16))
        self._num_patches = self._T // self._patch_size

        self._sessions: List[Tuple[np.ndarray, np.ndarray]] = []  # [(X, pos_root)]
        session_lengths: List[int] = []

        for path in tqdm(
            pkl_paths,
            desc=f"Loading odom sessions [{split_name}]",
            unit="session",
            dynamic_ncols=True,
        ):
            # Need extra P frames beyond T for the last patch's end-point displacement
            result = _load_odom_session(path, target_hz)
            if result is None:
                logger.warning("[odom] skipping: %s", path)
                continue
            X, pos_root = result
            self._sessions.append((X, pos_root))
            # Usable length: N − patch_size so last patch has a valid end-frame
            session_lengths.append(max(0, len(X) - self._patch_size))

        if not self._sessions:
            raise RuntimeError(
                f"NymeriaWristOdomDataset: no valid sessions loaded from "
                f"{len(pkl_paths)} pkl paths."
            )

        self._sess_idx, self._starts = _build_window_specs(
            session_lengths, self._T, self._hop
        )

        logger.info(
            "[odom] split=%s  sessions=%d  windows=%d  T=%d  hop=%d  "
            "patch_size=%d  num_patches=%d",
            split_name, len(self._sessions), len(self._starts),
            self._T, self._hop, self._patch_size, self._num_patches,
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])

        X, pos_root = self._sessions[si]
        P = self._patch_size

        # ---- IMU window ---------------------------------------------------
        Xw    = X[start:start + self._T]                       # [T, 6]
        Xproc = preprocess_window(Xw, self.cfg, self.stats)    # → [C, T]
        if Xproc is None:
            return None
        Xproc = apply_augment(Xproc.T, self.cfg).T             # augment expects [C,T]; result [C, T]
        x_tensor = torch.from_numpy(Xproc).float()             # [6, T]

        # ---- Displacement labels ------------------------------------------
        # y[p] = pos_root[start + (p+1)*P] − pos_root[start + p*P]
        disp = np.empty((self._num_patches, 3), dtype=np.float32)
        for p in range(self._num_patches):
            f0 = start + p * P
            f1 = f0 + P
            disp[p] = pos_root[f1] - pos_root[f0]

        y_tensor = torch.from_numpy(disp).float()              # [N_patches, 3]
        return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def make_nymeria_odom_loaders(
    cfg: Any,
    pkl_root: str,
    train_files: List[str],
    val_files: Optional[List[str]] = None,
    test_files: Optional[List[str]] = None,
    target_hz: int = 50,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for Nymeria odometry training.

    Each batch::

        x : [B, 6, T]           — global-frame wrist acc + gyro
        y : [B, N_patches, 3]   — per-patch root displacement (metres)
    """
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],       0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],    False))

    def _abs(files):
        return [f if os.path.isabs(f) else os.path.join(pkl_root, f) for f in files]

    def _make(name, files, bs, shuffle):
        if not files:
            return None
        ds = NymeriaWristOdomDataset(
            pkl_paths=_abs(files), cfg=cfg, split_name=name, target_hz=target_hz
        )
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_skip_none,
            persistent_workers=(num_workers > 0),
        )

    loaders: Dict[str, DataLoader] = {}
    for name, files, bs, shuf in [
        ("train", train_files,       batch_size,      True),
        ("val",   val_files or [],   eval_batch_size, False),
        ("test",  test_files or [],  eval_batch_size, False),
    ]:
        loader = _make(name, files, bs, shuf)
        if loader is not None:
            loaders[f"{name}_loader"] = loader
            logger.info(
                "[odom] %s_loader: sessions=%d  windows=%d  batch_size=%d",
                name, len(loader.dataset._sessions), len(loader.dataset), bs,
            )
    return loaders
