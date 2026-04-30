"""surfaceXR_dataloader.py
PyTorch DataLoader for the SurfaceXR wrist-IMU dataset.

Data source
-----------
Pre-processed pickle produced by surfaceXR_dataset.py:
    /mnt/nas2/vasco/SurfaceXR/dataset.pkl

Structure
---------
dataset[user_name] : List[Dict]   (one dict per recording session)
  imu_windows    : [N, 200, 24]  float32   Butterworth-filtered, normalised to [-1, 1]
                                             24 = 6 raw channels × 4 frequency bands
  hand_windows   : [N,  60, 109] float64   hand-tracking feature windows (unused here)
  touch_labels   : [N,  60]      bool      per-frame touch boolean → majority → 0/1
  gesture_labels : [N,  60]      int8      per-frame gesture class  → majority vote
  surface        : [9,]          float64   surface embedding (constant per session)
  surface_type   : str           'TABLE' or 'WALL'

Shapes emitted per item
-----------------------
  x : [24, 200]  float32  — (imu_channels, time_steps); channels first for ViT1D
  y : []         int64    — class index (touch: 0/1, gesture: 0-N)

Coordinate frame
----------------
Android Sensor frame (device frame, never rotates with screen):
  +X → right side of screen  +Y → top of screen  +Z → out of screen

Assumed wearing: right wrist, watch face outward, watch crown / top toward fingertips.
This is the default for SurfaceXR (Google Pixel Watch 3, right wrist).

Wrist FLU frame:
  F = forward (fingertip direction)  → Android +Y
  L = left    (thumb side, right wrist) → Android +X
  U = up      (away from skin / watch face) → Android +Z

So:  [F, L, U] = [Y, X, Z]   (no sign flips, only a permutation)

The pre-processed pkl stores 24 channels as 6 raw channels × 4 Butterworth bands,
laid out as: [acc_x_b0..b3, acc_y_b0..b3, acc_z_b0..b3, gyro_x_b0..b3, gyro_y_b0..b3, gyro_z_b0..b3].
The FLU transform permutes complete band-groups:
  FLU channel order: [acc_F(=Y), acc_L(=X), acc_U(=Z), gyro_F(=Y), gyro_L(=X), gyro_U(=Z)]
  → channel permutation indices: [4,5,6,7, 0,1,2,3, 8,9,10,11, 16,17,18,19, 12,13,14,15, 20,21,22,23]

Two split strategies
--------------------
  - LOPO (Leave-One-Person-Out): test on one user, train on the rest
  - Random: pool all users, random 80/10/10 split

Usage
-----
    from imu_lm.data.surfaceXR_dataloader import make_surfaceXR_loaders_lopo

    loaders = make_surfaceXR_loaders_lopo(
        pkl_path='/mnt/nas2/vasco/SurfaceXR/dataset.pkl',
        test_user='Timmy',
        label_type='touch',   # 'touch' or 'gesture'
        batch_size=256,
    )
    train_loader = loaders['train_loader']
    val_loader   = loaders['val_loader']

# /mnt/nas2/vasco/SurfaceXR/dataset.pkl
# /mnt/nas2/vasco/SurfaceXR/dataset_raw_imu.pkl
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of IMU channels in the pre-processed pkl  (6 raw × 4 Butterworth bands)
IMU_CHANNELS = 24
IMU_TIMESTEPS = 200

# All user names present in the dataset (15 participants)
ALL_USERS = [
    "Mingyuan", "Timmy", "Taeyoung", "Jerry", "Xiaan",
    "Jesse", "Trenton", "Anwesha", "Ge", "Brian",
    "Ziyi", "Vasco", "Jhan", "Pouya", "Tony",
]

# ---------------------------------------------------------------------------
# Android → wrist FLU coordinate transform
# ---------------------------------------------------------------------------
# Raw channel layout in the 24-channel pkl (6 raw × 4 Butterworth bands each):
#   groups:  acc_x[0-3]  acc_y[4-7]  acc_z[8-11]  gyro_x[12-15]  gyro_y[16-19]  gyro_z[20-23]
#
# Right wrist, watch face out, watch top toward fingertips:
#   FLU-F = Android +Y  →  acc_F group ← old acc_y group  (indices 4-7)
#   FLU-L = Android +X  →  acc_L group ← old acc_x group  (indices 0-3)
#   FLU-U = Android +Z  →  acc_U group ← old acc_z group  (indices 8-11, unchanged)
#   gyro analogous
#
# Resulting channel order (FLU): [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]
_ANDROID_TO_FLU_PERM: List[int] = [
    4,  5,  6,  7,   # acc_F  = acc_Y
    0,  1,  2,  3,   # acc_L  = acc_X
    8,  9, 10, 11,   # acc_U  = acc_Z
   16, 17, 18, 19,   # gyro_F = gyro_Y
   12, 13, 14, 15,   # gyro_L = gyro_X
   20, 21, 22, 23,   # gyro_U = gyro_Z
]
assert len(_ANDROID_TO_FLU_PERM) == 24 and len(set(_ANDROID_TO_FLU_PERM)) == 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _majority_vote(per_frame_labels: np.ndarray) -> int:
    """Reduce per-frame labels [T] to a single integer via majority vote."""
    vals, counts = np.unique(per_frame_labels, return_counts=True)
    return int(vals[np.argmax(counts)])


def _load_pkl(pkl_path: str) -> Dict:
    _log.info("[surfaceXR] loading pkl: %s", pkl_path)
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    _log.info("[surfaceXR] loaded %d users", len(data))
    return data


def _collect_windows(
    data: Dict,
    users: List[str],
    label_type: str,
    apply_flu: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate pre-computed windows across ``users``.

    Parameters
    ----------
    data       : full dataset dict (user → list of session dicts)
    users      : which users to include
    label_type : ``'touch'`` or ``'gesture'``
    apply_flu  : if True (default), permute channels from Android device frame
                 to wrist FLU frame assuming right wrist, watch face outward,
                 watch crown toward fingertips.

    Returns
    -------
    X : [M, 24, 200]  float32  — imu windows, channels first
    y : [M]           int64    — per-window label (majority vote of per-frame labels)
    """
    label_key = "touch_labels" if label_type == "touch" else "gesture_labels"
    X_list, y_list = [], []

    for user in users:
        if user not in data:
            _log.warning("[surfaceXR] user '%s' not found in pkl, skipping", user)
            continue
        sessions = data[user]
        for sess in sessions:
            imu = sess["imu_windows"]               # [N, 200, 24]
            labels_frame = sess[label_key]          # [N, 60]

            N = len(imu)
            imu_chfirst = imu.transpose(0, 2, 1)   # [N, 24, 200]

            if apply_flu:
                # Permute channel groups: Android device frame → wrist FLU frame
                # [F,L,U] = [Y,X,Z] for both acc and gyro (right wrist, face out, top→fingertips)
                imu_chfirst = imu_chfirst[:, _ANDROID_TO_FLU_PERM, :]

            per_window_labels = np.array(
                [_majority_vote(labels_frame[i]) for i in range(N)],
                dtype=np.int64,
            )

            X_list.append(imu_chfirst.astype(np.float32))
            y_list.append(per_window_labels)

    if not X_list:
        raise RuntimeError(
            f"[surfaceXR] no windows collected for users={users}, label_type={label_type}"
        )

    X = np.concatenate(X_list, axis=0)   # [M, 24, 200]
    y = np.concatenate(y_list, axis=0)   # [M]
    _log.info(
        "[surfaceXR] users=%s → %d windows, label_type=%s, classes=%s, flu=%s",
        users, len(X), label_type, np.unique(y).tolist(), apply_flu,
    )
    return X, y


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SurfaceXRDataset(Dataset):
    """Pre-windowed SurfaceXR IMU dataset.

    Parameters
    ----------
    X          : [M, 24, 200]  float32  — IMU windows (Butterworth, normalised)
    y          : [M]           int64    — class labels
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self._X = torch.from_numpy(X).float()
        self._y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._X[idx], self._y[idx]


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------

def make_surfaceXR_loaders_lopo(
    pkl_path: str,
    test_user: str,
    label_type: str = "touch",
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    apply_flu: bool = True,
) -> Dict[str, DataLoader]:
    """Leave-One-Person-Out split: train on all users except ``test_user``.

    The held-out user is used for **both** val and test loaders.

    Parameters
    ----------
    pkl_path       : path to ``dataset.pkl``
    test_user      : one of ``ALL_USERS`` — held out for evaluation
    label_type     : ``'touch'`` (binary) or ``'gesture'`` (multi-class)
    batch_size     : training batch size
    eval_batch_size: val/test batch size (defaults to ``batch_size``)
    num_workers    : DataLoader worker count
    pin_memory     : DataLoader pin_memory flag
    seed           : random seed (unused here but kept for API consistency)
    apply_flu      : if True (default), remap channels from Android device frame
                     to wrist FLU frame ([F,L,U]=[Y,X,Z], right wrist, watch top→fingertips)

    Returns
    -------
    dict with keys ``'train_loader'``, ``'val_loader'``, ``'test_loader'``
    """
    if test_user not in ALL_USERS:
        raise ValueError(f"test_user must be one of {ALL_USERS}, got '{test_user}'")

    eval_batch_size = eval_batch_size or batch_size
    data = _load_pkl(pkl_path)

    train_users = [u for u in ALL_USERS if u != test_user]

    X_train, y_train = _collect_windows(data, train_users, label_type, apply_flu=apply_flu)
    X_eval,  y_eval  = _collect_windows(data, [test_user], label_type, apply_flu=apply_flu)

    persistent = num_workers > 0

    def _make(X, y, bs, shuffle):
        ds = SurfaceXRDataset(X, y)
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent,
        )

    return {
        "train_loader": _make(X_train, y_train, batch_size,      shuffle=True),
        "val_loader":   _make(X_eval,  y_eval,  eval_batch_size, shuffle=False),
        "test_loader":  _make(X_eval,  y_eval,  eval_batch_size, shuffle=False),
    }


def make_surfaceXR_loaders_random(
    pkl_path: str,
    label_type: str = "touch",
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    apply_flu: bool = True,
) -> Dict[str, DataLoader]:
    """Pool all users and split randomly into train / val / test.

    Parameters
    ----------
    pkl_path       : path to ``dataset.pkl``
    label_type     : ``'touch'`` or ``'gesture'``
    splits         : (train_frac, val_frac, test_frac); must sum to 1.0
    apply_flu      : if True (default), remap channels from Android device frame
                     to wrist FLU frame ([F,L,U]=[Y,X,Z], right wrist, watch top→fingertips)
    batch_size     : training batch size
    eval_batch_size: val/test batch size (defaults to ``batch_size``)
    num_workers    : DataLoader worker count
    pin_memory     : DataLoader pin_memory flag
    seed           : numpy random seed for reproducible splits

    Returns
    -------
    dict with keys ``'train_loader'``, ``'val_loader'``, ``'test_loader'``
    """
    assert abs(sum(splits) - 1.0) < 1e-6, f"splits must sum to 1.0, got {sum(splits)}"
    eval_batch_size = eval_batch_size or batch_size

    data = _load_pkl(pkl_path)
    X, y = _collect_windows(data, ALL_USERS, label_type, apply_flu=apply_flu)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n = len(X)
    n_train = int(splits[0] * n)
    n_val   = int(splits[1] * n)

    X_train, y_train = X[:n_train],          y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],    y[n_train+n_val:]

    _log.info(
        "[surfaceXR] random split — train=%d val=%d test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    persistent = num_workers > 0

    def _make(Xp, yp, bs, shuffle):
        ds = SurfaceXRDataset(Xp, yp)
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=persistent,
        )

    return {
        "train_loader": _make(X_train, y_train, batch_size,      shuffle=True),
        "val_loader":   _make(X_val,   y_val,   eval_batch_size, shuffle=False),
        "test_loader":  _make(X_test,  y_test,  eval_batch_size, shuffle=False),
    }


# ===========================================================================
# 50 Hz / 6-channel / 128-sample pipeline
# ===========================================================================
# Uses dataset_raw_imu.pkl which stores raw (un-filtered) acc_x/y/z + gyro_x/y/z
# at 200 Hz, pre-windowed to 1-second (200-sample) windows.
#
# Processing:
#   1. Decimate each 1-second window from 200 Hz → 50 Hz (50 samples) using a
#      zero-phase IIR anti-aliasing filter (scipy.signal.decimate, zero_phase=True).
#   2. Flatten the decimated windows into a pseudo-contiguous per-session signal.
#      Adjacent original windows may have been produced with a sliding stride, so
#      some samples near boundaries are repeated — an acceptable approximation.
#   3. Slide 128-sample (2.56 s) windows with 64-sample (50 %) hop.
#   4. Map each IMU window to the corresponding 60-Hz hand-tracking label frames
#      via round(sample_idx × 60/50) and take the majority-vote label.
#
# Output per item: x [6, 128] float32,  y [] int64
# Encoder compatibility: identical format to all other 6-channel 50-Hz datasets.
# ===========================================================================

# Raw 6-channel layout in dataset_raw_imu.pkl:
#   channel 0: acc_x   channel 1: acc_y   channel 2: acc_z
#   channel 3: gyro_x  channel 4: gyro_y  channel 5: gyro_z
#
# FLU permutation (right wrist, watch face out, crown toward fingertips):
#   acc_F  = acc_Y  → idx 1    gyro_F = gyro_Y → idx 4
#   acc_L  = acc_X  → idx 0    gyro_L = gyro_X → idx 3
#   acc_U  = acc_Z  → idx 2    gyro_U = gyro_Z → idx 5
_RAW_ANDROID_TO_FLU_PERM_6CH: List[int] = [1, 0, 2, 4, 3, 5]

RAW_IMU_PKL_DEFAULT = "/mnt/nas2/vasco/SurfaceXR/dataset_raw_imu.pkl"

# Canonical resampling parameters
_ORIG_HZ              = 200
_TARGET_HZ            = 50
_DECIMATE_FACTOR      = _ORIG_HZ // _TARGET_HZ   # 4
_HAND_HZ              = 60
TARGET_WINDOW_SAMPLES = 128   # 2.56 s @ 50 Hz  (matches windowing.window_seconds = 2.56)
TARGET_HOP_SAMPLES    = 64    # 50 % overlap
RAW_WINDOW_SAMPLES    = 512   # 2.56 s @ 200 Hz
RAW_HOP_SAMPLES       = 256   # 50 % overlap


def _decimate_imu_batch(imu_windows: np.ndarray, factor: int = _DECIMATE_FACTOR) -> np.ndarray:
    """Decimate [N, T, C] IMU windows along the time axis (axis=1).

    Uses a zero-phase IIR low-pass filter to avoid aliasing and phase distortion.
    Converts to float64 for numerical stability, returns float32.
    """
    from scipy.signal import decimate as _sp_decimate
    T_out = imu_windows.shape[1] // factor
    dec = _sp_decimate(imu_windows.astype(np.float64), q=factor, axis=1, zero_phase=True)
    return dec[:, :T_out, :].astype(np.float32)


def _collect_windows_raw_hz(
    data: Dict,
    users: List[str],
    label_type: str,
    target_hz: int,
    apply_flu: bool = True,
    window_samples: Optional[int] = None,
    hop_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect 6-channel raw-IMU windows from the raw SurfaceXR pkl.

    Supports the existing 50 Hz path by decimating raw 200 Hz windows, and a
    native 200 Hz path that preserves the original sampling rate.
    """

    if target_hz not in (_TARGET_HZ, _ORIG_HZ):
        raise ValueError(f"Unsupported SurfaceXR target_hz={target_hz}; expected 50 or 200")

    if window_samples is None:
        window_samples = TARGET_WINDOW_SAMPLES if target_hz == _TARGET_HZ else RAW_WINDOW_SAMPLES
    if hop_samples is None:
        hop_samples = TARGET_HOP_SAMPLES if target_hz == _TARGET_HZ else RAW_HOP_SAMPLES

    label_key = "touch_labels" if label_type == "touch" else "gesture_labels"
    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for user in users:
        if user not in data:
            _log.warning("[surfaceXR-%dhz] user '%s' not found in pkl, skipping", target_hz, user)
            continue
        for sess in data[user]:
            imu = np.asarray(sess["imu_windows"])      # [N, 200, 6]
            labels = np.asarray(sess[label_key])       # [N, 60]

            if len(imu) == 0:
                continue

            if target_hz == _TARGET_HZ:
                imu_stream = _decimate_imu_batch(imu, factor=_DECIMATE_FACTOR)
            else:
                imu_stream = imu.astype(np.float32, copy=False)

            imu_cont = imu_stream.reshape(-1, imu_stream.shape[2])
            labels_cont = labels.reshape(-1).astype(np.int64)

            M = len(imu_cont)
            L = len(labels_cont)

            for start in range(0, M - window_samples + 1, hop_samples):
                end = start + window_samples
                win_imu = imu_cont[start:end]

                hf_start = round(start * _HAND_HZ / target_hz)
                hf_end = min(round(end * _HAND_HZ / target_hz), L)

                if hf_start >= L or hf_end <= hf_start:
                    break

                label = int(_majority_vote(labels_cont[hf_start:hf_end]))

                if apply_flu:
                    win_imu = win_imu[:, _RAW_ANDROID_TO_FLU_PERM_6CH]

                X_list.append(win_imu.T.astype(np.float32))
                y_list.append(label)

    if not X_list:
        raise RuntimeError(
            f"[surfaceXR-{target_hz}hz] no windows collected for users={users}, label_type={label_type}"
        )

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)
    _log.info(
        "[surfaceXR-%dhz] users=%s → %d windows [6,%d], label=%s, classes=%s, flu=%s",
        target_hz, users, len(X), window_samples, label_type, np.unique(y).tolist(), apply_flu,
    )
    return X, y


def _collect_windows_50hz(
    data: Dict,
    users: List[str],
    label_type: str,
    apply_flu: bool = True,
    window_samples: int = TARGET_WINDOW_SAMPLES,
    hop_samples: int = TARGET_HOP_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect 50 Hz / 6-channel / 128-sample windows from the raw-IMU pkl.

    Parameters
    ----------
    data          : full dataset dict loaded from ``dataset_raw_imu.pkl``
    users         : which users to include
    label_type    : ``'touch'`` or ``'gesture'``
    apply_flu     : if True, permute raw 6 channels to wrist FLU frame
    window_samples: output window length in 50 Hz samples (default 128 = 2.56 s)
    hop_samples   : sliding stride in 50 Hz samples (default 64 = 50 % overlap)

    Returns
    -------
    X : [M, 6, 128]  float32  — windows, channels first
    y : [M]          int64    — per-window majority-vote label
    """
    return _collect_windows_raw_hz(
        data=data,
        users=users,
        label_type=label_type,
        target_hz=_TARGET_HZ,
        apply_flu=apply_flu,
        window_samples=window_samples,
        hop_samples=hop_samples,
    )


def _collect_windows_200hz(
    data: Dict,
    users: List[str],
    label_type: str,
    apply_flu: bool = True,
    window_samples: int = RAW_WINDOW_SAMPLES,
    hop_samples: int = RAW_HOP_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect 200 Hz / 6-channel / 512-sample windows from the raw-IMU pkl."""

    return _collect_windows_raw_hz(
        data=data,
        users=users,
        label_type=label_type,
        target_hz=_ORIG_HZ,
        apply_flu=apply_flu,
        window_samples=window_samples,
        hop_samples=hop_samples,
    )


def make_surfaceXR_loaders_lopo_50hz(
    pkl_path: str = RAW_IMU_PKL_DEFAULT,
    test_user: str = None,
    label_type: str = "gesture",
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    apply_flu: bool = True,
    window_samples: int = TARGET_WINDOW_SAMPLES,
    hop_samples: int = TARGET_HOP_SAMPLES,
) -> Dict[str, DataLoader]:
    """LOPO split — 50 Hz / 6-channel / 128-sample windows (raw-IMU pkl).

    The held-out ``test_user`` is used for both val and test loaders.
    """
    if test_user not in ALL_USERS:
        raise ValueError(f"test_user must be one of {ALL_USERS}, got '{test_user}'")

    eval_batch_size = eval_batch_size or batch_size
    data = _load_pkl(pkl_path)

    train_users = [u for u in ALL_USERS if u != test_user]
    X_train, y_train = _collect_windows_50hz(
        data, train_users, label_type, apply_flu, window_samples, hop_samples
    )
    X_eval, y_eval = _collect_windows_50hz(
        data, [test_user], label_type, apply_flu, window_samples, hop_samples
    )

    persistent = num_workers > 0

    def _make(X, y, bs, shuffle):
        ds = SurfaceXRDataset(X, y)
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False, persistent_workers=persistent,
        )

    return {
        "train_loader": _make(X_train, y_train, batch_size,      shuffle=True),
        "val_loader":   _make(X_eval,  y_eval,  eval_batch_size, shuffle=False),
        "test_loader":  _make(X_eval,  y_eval,  eval_batch_size, shuffle=False),
    }


def make_surfaceXR_loaders_random_50hz(
    pkl_path: str = RAW_IMU_PKL_DEFAULT,
    label_type: str = "gesture",
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    apply_flu: bool = True,
    window_samples: int = TARGET_WINDOW_SAMPLES,
    hop_samples: int = TARGET_HOP_SAMPLES,
) -> Dict[str, DataLoader]:
    """Random 80/10/10 split — 50 Hz / 6-channel / 128-sample windows (raw-IMU pkl)."""
    assert abs(sum(splits) - 1.0) < 1e-6, f"splits must sum to 1.0, got {sum(splits)}"

    eval_batch_size = eval_batch_size or batch_size
    data = _load_pkl(pkl_path)
    X, y = _collect_windows_50hz(
        data, ALL_USERS, label_type, apply_flu, window_samples, hop_samples
    )

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n       = len(X)
    n_train = int(splits[0] * n)
    n_val   = int(splits[1] * n)

    X_train, y_train = X[:n_train],                y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val],  y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:],          y[n_train+n_val:]

    _log.info(
        "[surfaceXR-50hz] random split — train=%d val=%d test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    persistent = num_workers > 0

    def _make(Xp, yp, bs, shuffle):
        ds = SurfaceXRDataset(Xp, yp)
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False, persistent_workers=persistent,
        )

    return {
        "train_loader": _make(X_train, y_train, batch_size,      shuffle=True),
        "val_loader":   _make(X_val,   y_val,   eval_batch_size, shuffle=False),
        "test_loader":  _make(X_test,  y_test,  eval_batch_size, shuffle=False),
    }


def make_surfaceXR_loaders_lopo_200hz(
    pkl_path: str = RAW_IMU_PKL_DEFAULT,
    test_user: str = None,
    label_type: str = "gesture",
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    apply_flu: bool = True,
    window_samples: int = RAW_WINDOW_SAMPLES,
    hop_samples: int = RAW_HOP_SAMPLES,
) -> Dict[str, DataLoader]:
    """LOPO split — 200 Hz / 6-channel / 512-sample windows (raw-IMU pkl)."""
    if test_user not in ALL_USERS:
        raise ValueError(f"test_user must be one of {ALL_USERS}, got '{test_user}'")

    eval_batch_size = eval_batch_size or batch_size
    data = _load_pkl(pkl_path)

    train_users = [u for u in ALL_USERS if u != test_user]
    X_train, y_train = _collect_windows_200hz(
        data, train_users, label_type, apply_flu, window_samples, hop_samples
    )
    X_eval, y_eval = _collect_windows_200hz(
        data, [test_user], label_type, apply_flu, window_samples, hop_samples
    )

    persistent = num_workers > 0

    def _make(X, y, bs, shuffle):
        ds = SurfaceXRDataset(X, y)
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False, persistent_workers=persistent,
        )

    return {
        "train_loader": _make(X_train, y_train, batch_size,      shuffle=True),
        "val_loader":   _make(X_eval,  y_eval,  eval_batch_size, shuffle=False),
        "test_loader":  _make(X_eval,  y_eval,  eval_batch_size, shuffle=False),
    }


def make_surfaceXR_loaders_random_200hz(
    pkl_path: str = RAW_IMU_PKL_DEFAULT,
    label_type: str = "gesture",
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    apply_flu: bool = True,
    window_samples: int = RAW_WINDOW_SAMPLES,
    hop_samples: int = RAW_HOP_SAMPLES,
) -> Dict[str, DataLoader]:
    """Random 80/10/10 split — 200 Hz / 6-channel / 512-sample windows (raw-IMU pkl)."""
    assert abs(sum(splits) - 1.0) < 1e-6, f"splits must sum to 1.0, got {sum(splits)}"

    eval_batch_size = eval_batch_size or batch_size
    data = _load_pkl(pkl_path)
    X, y = _collect_windows_200hz(
        data, ALL_USERS, label_type, apply_flu, window_samples, hop_samples
    )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n = len(X)
    n_train = int(splits[0] * n)
    n_val = int(splits[1] * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    _log.info(
        "[surfaceXR-200hz] random split — train=%d val=%d test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    persistent = num_workers > 0

    def _make(Xp, yp, bs, shuffle):
        ds = SurfaceXRDataset(Xp, yp)
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            drop_last=False, persistent_workers=persistent,
        )

    return {
        "train_loader": _make(X_train, y_train, batch_size,      shuffle=True),
        "val_loader":   _make(X_val,   y_val,   eval_batch_size, shuffle=False),
        "test_loader":  _make(X_test,  y_test,  eval_batch_size, shuffle=False),
    }
