"""imuposer_wrist_16hz_to_50hz_loader_FLU.py
IMUPoser right-wrist FLU loader with 50→16Hz→50Hz IMU degradation.

Identical to ``imuposer_wrist_loader_FLU.py`` except the raw IMU signal (X)
is degraded after loading:

    original 50 Hz  →  downsample to 16 Hz  →  band-limited resample back to 50 Hz

This simulates a lower-quality 16 Hz sensor recording.  The pose labels (y) are
**not** degraded — they still come from the 50 Hz SMPL FK ground truth.

Degradation uses ``scipy.signal.resample_poly`` (anti-aliased polyphase filter),
matching the implementation in ``realdata_16hz_to_50hz_loader.py``.

Outputs per window (same shape as the non-degraded loader):
    x : [6, T]           float32  – degraded (acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U)
    y : [N_patches, 30]  float32  – right-arm 6D rotations per time patch
                                     SMPL joints [14, 17, 19, 21, 23]

Loader key: ``imuposer_flu_16hz_to_50hz``
"""

from __future__ import annotations

import logging
from math import gcd

import numpy as np
import torch

from scipy.signal import resample_poly

from imu_lm.data.imuposer_wrist_loader_FLU import (
    load_session_flu,
    collate_skip_none,
    NUM_CHAIN_JOINTS,
    ROT6D_DIM,
)

_log = logging.getLogger(__name__)

_SOURCE_HZ   = 50
_DEGRADED_HZ = 16


# ---------------------------------------------------------------------------
# Degradation helper
# ---------------------------------------------------------------------------

def _degrade_imu(X: np.ndarray,
                 source_hz:   int = _SOURCE_HZ,
                 degraded_hz: int = _DEGRADED_HZ) -> np.ndarray:
    """50 Hz → 16 Hz → 50 Hz via anti-aliased polyphase resampling.

    Parameters
    ----------
    X : [N, 6]  float32  – original IMU signal at source_hz
    Returns
    -------
    [N, 6]  float32  – degraded signal at source_hz (Nyquist = degraded_hz / 2)
    """
    g_down = gcd(source_hz, degraded_hz)
    X_low  = resample_poly(X, degraded_hz // g_down, source_hz // g_down, axis=0).astype(np.float32)

    g_up   = gcd(degraded_hz, source_hz)
    X_out  = resample_poly(X_low, source_hz // g_up, degraded_hz // g_up, axis=0).astype(np.float32)

    return X_out[: len(X)]   # polyphase may add 1–2 extra samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class IMUPoserWrist16HzTo50HzFLUDataset(torch.utils.data.Dataset):
    """Sliding-window dataset over IMUPoser sessions with 50→16→50 Hz degraded FLU IMU.

    Each item:
        x : [6, T]           float32  – degraded FLU IMU
        y : [N_patches, 30]  float32  – right-arm 6D rotations per time patch
    """

    def __init__(
        self,
        pkl_paths:   list[str],
        cfg:         dict,
        split_name:  str  = "train",
        target_hz:   int  = 50,
        add_gravity: bool = False,
    ):
        from imu_lm.data.augmentations.preprocess import PreprocessStats
        from imu_lm.data.windowing import compute_T_and_hop
        from imu_lm.utils.helpers import cfg_get
        from tqdm import tqdm

        self.cfg        = cfg
        self.split_name = split_name
        self.stats      = PreprocessStats()

        vit_cfg              = cfg_get(cfg, ["vit1d"], {}) or {}
        self._patch_size     = int(vit_cfg.get("patch_size", 16))
        self._T, self._hop   = compute_T_and_hop(cfg)
        self._num_patches    = self._T // self._patch_size
        assert self._T % self._patch_size == 0, (
            f"T={self._T} must be divisible by patch_size={self._patch_size}"
        )

        self._sessions: list[tuple[np.ndarray, np.ndarray]] = []
        self._sess_idx: list[int] = []
        self._starts:   list[int] = []

        for pkl in tqdm(pkl_paths, desc=f"[imuposer_flu_16hz_to_50hz] loading {split_name}"):
            result = load_session_flu(pkl, target_hz=target_hz, add_gravity=add_gravity)
            if result is None:
                continue
            X, y = result   # X: [N, 6],  y: [N, 5, 6]

            # Apply 50→16→50 Hz degradation to IMU signal only
            X = _degrade_imu(X, source_hz=target_hz, degraded_hz=_DEGRADED_HZ)

            if X.shape[0] < self._T:
                continue

            si = len(self._sessions)
            self._sessions.append((X, y))

            n_windows = (X.shape[0] - self._T) // self._hop + 1
            for s in range(n_windows):
                self._sess_idx.append(si)
                self._starts.append(s * self._hop)

        self._sess_idx = np.array(self._sess_idx, dtype=np.int32)
        self._starts   = np.array(self._starts,   dtype=np.int32)
        _log.info(
            "[imuposer_flu_16hz_to_50hz] split=%s  sessions=%d  windows=%d",
            split_name, len(self._sessions), len(self._starts),
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        from imu_lm.data.augmentations.preprocess import preprocess_window
        from imu_lm.data.augmentations.transform  import apply_augment

        si    = int(self._sess_idx[idx])
        start = int(self._starts[idx])
        T     = self._T
        X, y  = self._sessions[si]   # [N, 6], [N, 5, 6]

        Xw = X[start : start + T]   # [T, 6]

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc    = apply_augment(Xproc.T, self.cfg).T          # [C, T]
        x_tensor = torch.from_numpy(Xproc).float()             # [6, T]

        P = self._patch_size
        patch_labels = np.empty(
            (self._num_patches, NUM_CHAIN_JOINTS * ROT6D_DIM), dtype=np.float32
        )
        for p in range(self._num_patches):
            frame = start + p * P + P // 2
            patch_labels[p] = y[frame].reshape(-1)   # [5, 6] → [30]

        y_tensor = torch.from_numpy(patch_labels).float()   # [N_patches, 30]
        return x_tensor, y_tensor


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_imuposer_wrist_16hz_to_50hz_flu_loaders(
    cfg:          dict,
    train_files:  list[str],
    val_files:    list[str] | None = None,
    test_files:   list[str] | None = None,
    target_hz:    int  = 50,
    add_gravity:  bool = False,
) -> dict[str, torch.utils.data.DataLoader]:
    """Build train / val / test DataLoaders with 50→16→50 Hz degraded FLU IMU.

    Each batch yields::

        x : [B, 6, T]           float32  – degraded right-wrist IMU in FLU body frame
        y : [B, N_patches, 30]  float32  – right-arm 6D rotations per patch
    """
    from imu_lm.utils.helpers import cfg_get

    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],       256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"],  batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],        0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],     False))
    persistent      = num_workers > 0

    def _make(name: str, files: list[str], bs: int, shuffle: bool):
        if not files:
            return None
        ds = IMUPoserWrist16HzTo50HzFLUDataset(
            pkl_paths=files, cfg=cfg, split_name=name,
            target_hz=target_hz, add_gravity=add_gravity,
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle,
            persistent_workers=persistent,
            collate_fn=collate_skip_none,
        )

    return {
        "train_loader": _make("train", train_files,       batch_size,      True),
        "val_loader":   _make("val",   val_files  or [],  eval_batch_size, False),
        "test_loader":  _make("test",  test_files or [],  eval_batch_size, False),
    }
