"""utd_mhad_dataloader.py
PyTorch DataLoader for the UTD Multimodal Human Action Dataset (UTD-MHAD).

Data source
-----------
MATLAB .mat files from the Inertial subfolder:
    <data_root>/a{action}_s{subject}_t{trial}_inertial.mat

Each .mat file contains:
    d_iner : [N, 6]  float64   accelerometer (cols 0-2) + gyroscope (cols 3-5) at ~50 Hz

Dataset statistics
------------------
  - 27 action classes (1-indexed in filenames, remapped to 0-25 here)
  - 8 subjects (IDs 1-8)
  - Up to 4 trials per (action, subject) pair
  - IMU: right wrist, 6 channels (acc_x/y/z, gyro_x/y/z), ~50 Hz

Split strategies
----------------
  - loso  : Leave-One-Subject-Out  (held-out subject is val + test)
  - standard: odd subjects (1,3,5,7) = train; even subjects (2,4,6,8) = test
               (matches the standard benchmark split from the original paper)

Window / label
--------------
  Each clip = one action; a 128-sample sliding window is applied over the clip
  with a configurable hop.  All windows from a clip carry the same action label.

Output per item: x [6, 128] float32,  y [] int64
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES      = 27
ALL_SUBJECTS     = list(range(1, 9))          # 1-8
STANDARD_TRAIN   = [1, 3, 5, 7]
STANDARD_TEST    = [2, 4, 6, 8]

TARGET_HZ        = 50
WINDOW_SAMPLES   = 128   # 2.56 s @ 50 Hz
HOP_SAMPLES      = 64    # 50 % overlap

# UTD-MHAD action labels (0-indexed)
ACTION_NAMES = [
    "swipe_left", "swipe_right", "wave", "clap", "throw",
    "arm_cross", "basketball_shoot", "draw_x", "draw_circle_cw", "draw_circle_ccw",
    "draw_triangle", "bowling", "boxing", "baseball_swing", "tennis_swing",
    "arm_curl", "tennis_serve", "push", "knock", "catch",
    "pickup_and_throw", "jog", "walk", "sit2stand", "stand2sit",
    "lunge", "squat",
]
assert len(ACTION_NAMES) == NUM_CLASSES


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_clip(mat_path: str) -> np.ndarray:
    """Load one inertial .mat file → [N, 6] float32."""
    from scipy.io import loadmat
    mat = loadmat(mat_path)
    data = mat["d_iner"].astype(np.float32)   # [N, 6]
    return data


def _parse_filename(fname: str) -> Optional[Tuple[int, int, int]]:
    """Parse a{action}_s{subject}_t{trial}_inertial.mat → (action, subject, trial) 1-indexed.

    Returns None if the filename does not match the expected pattern.
    """
    name = os.path.splitext(os.path.basename(fname))[0]  # strip .mat
    if not name.endswith("_inertial"):
        return None
    name = name[: -len("_inertial")]                      # strip _inertial
    parts = name.split("_")                               # ['a1', 's1', 't1']
    if len(parts) != 3:
        return None
    try:
        action  = int(parts[0][1:])
        subject = int(parts[1][1:])
        trial   = int(parts[2][1:])
    except (ValueError, IndexError):
        return None
    return action, subject, trial


def _collect_clips(
    data_root: str,
    subjects: List[int],
) -> Tuple[List[np.ndarray], List[int]]:
    """Scan *data_root* for .mat files belonging to *subjects*.

    Returns
    -------
    clips  : list of [N, 6] float32 arrays (variable N per clip)
    labels : list of int (0-indexed action label, one per clip)
    """
    clips, labels = [], []

    all_files = sorted(f for f in os.listdir(data_root) if f.endswith("_inertial.mat"))
    if not all_files:
        raise RuntimeError(
            f"[utd_mhad] no *_inertial.mat files found in {data_root!r}"
        )

    for fname in all_files:
        parsed = _parse_filename(fname)
        if parsed is None:
            _log.warning("[utd_mhad] skipping unrecognised file: %s", fname)
            continue
        action, subject, _ = parsed
        if subject not in subjects:
            continue
        if not (1 <= action <= NUM_CLASSES):
            _log.warning("[utd_mhad] action %d out of range, skipping %s", action, fname)
            continue
        mat_path = os.path.join(data_root, fname)
        clip = _load_clip(mat_path)     # [N, 6]
        clips.append(clip)
        labels.append(action - 1)       # 0-indexed

    if not clips:
        raise RuntimeError(
            f"[utd_mhad] no clips found for subjects={subjects} in {data_root!r}"
        )
    return clips, labels


def _window_clips(
    clips: List[np.ndarray],
    labels: List[int],
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide *window_samples* windows over each clip.

    If a clip is shorter than *window_samples* it is zero-padded to one window.

    Returns
    -------
    X : [M, 6, window_samples]  float32  — channels first
    y : [M]                     int64
    """
    X_list, y_list = [], []

    for clip, label in zip(clips, labels):
        N = len(clip)
        if N < window_samples:
            # zero-pad to exactly one window
            pad = np.zeros((window_samples - N, clip.shape[1]), dtype=np.float32)
            clip = np.concatenate([clip, pad], axis=0)
            N = window_samples

        for start in range(0, N - window_samples + 1, hop_samples):
            win = clip[start : start + window_samples]   # [window_samples, 6]
            X_list.append(win.T.astype(np.float32))      # [6, window_samples]
            y_list.append(label)

    X = np.stack(X_list)                   # [M, 6, window_samples]
    y = np.array(y_list, dtype=np.int64)   # [M]
    return X, y


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UTDMHADDataset(Dataset):
    """Pre-windowed UTD-MHAD inertial dataset.

    Parameters
    ----------
    X : [M, 6, T]  float32  — IMU windows, channels first
    y : [M]        int64    — 0-indexed action labels
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

def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = UTDMHADDataset(X, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def make_utd_mhad_loaders_loso(
    data_root: str,
    test_subject: int,
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> Dict[str, DataLoader]:
    """Leave-One-Subject-Out split.

    The held-out *test_subject* is used for both val and test loaders.

    Parameters
    ----------
    data_root    : directory containing *_inertial.mat files
    test_subject : subject ID (1-8) to hold out
    """
    if test_subject not in ALL_SUBJECTS:
        raise ValueError(f"test_subject must be one of {ALL_SUBJECTS}, got {test_subject!r}")

    eval_batch_size = eval_batch_size or batch_size
    train_subjects  = [s for s in ALL_SUBJECTS if s != test_subject]

    clips_tr, labels_tr = _collect_clips(data_root, train_subjects)
    clips_ev, labels_ev = _collect_clips(data_root, [test_subject])

    X_train, y_train = _window_clips(clips_tr, labels_tr, window_samples, hop_samples)
    X_eval,  y_eval  = _window_clips(clips_ev, labels_ev, window_samples, hop_samples)

    _log.info(
        "[utd_mhad] LOSO test_subject=%d  train=%d windows  eval=%d windows",
        test_subject, len(X_train), len(X_eval),
    )
    return {
        "train_loader": _make_loader(X_train, y_train, batch_size,      True,  num_workers, pin_memory),
        "val_loader":   _make_loader(X_eval,  y_eval,  eval_batch_size, False, num_workers, pin_memory),
        "test_loader":  _make_loader(X_eval,  y_eval,  eval_batch_size, False, num_workers, pin_memory),
    }


def make_utd_mhad_loaders_standard(
    data_root: str,
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> Dict[str, DataLoader]:
    """Standard benchmark split: odd subjects train, even subjects test.

    Matches the evaluation protocol from the UTD-MHAD paper.
    """
    eval_batch_size = eval_batch_size or batch_size

    clips_tr, labels_tr = _collect_clips(data_root, STANDARD_TRAIN)
    clips_ev, labels_ev = _collect_clips(data_root, STANDARD_TEST)

    X_train, y_train = _window_clips(clips_tr, labels_tr, window_samples, hop_samples)
    X_eval,  y_eval  = _window_clips(clips_ev, labels_ev, window_samples, hop_samples)

    _log.info(
        "[utd_mhad] standard split  train_subjects=%s  train=%d windows  eval=%d windows",
        STANDARD_TRAIN, len(X_train), len(X_eval),
    )
    return {
        "train_loader": _make_loader(X_train, y_train, batch_size,      True,  num_workers, pin_memory),
        "val_loader":   _make_loader(X_eval,  y_eval,  eval_batch_size, False, num_workers, pin_memory),
        "test_loader":  _make_loader(X_eval,  y_eval,  eval_batch_size, False, num_workers, pin_memory),
    }
