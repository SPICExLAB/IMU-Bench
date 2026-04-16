"""HGAGDataset: PyTorch Dataset for the Hand Gesture Accelerometer and Gyroscope
Dataset (HGAG-DATA, Mendeley Data, doi:10.17632/mkhn7kxjvy.1).

Dataset facts
-------------
* 43 subjects, 11 gestures, 50 repetitions each → 23,650 samples total.
* 6D IMU (acc_x/y/z + gyro_x/y/z) originally at 200 Hz, 250 time steps per sample
  (≈ 1.25 s per gesture).
* Pre-processed by the original authors with a 4th-order Butterworth band-pass
  filter (30–90 Hz) to remove motion artefacts.
* This loader resamples to 50 Hz / 64 timesteps (1.28 s window) to match the
  Nymeria / parquet pretraining regime: last 6 raw samples are duplicated to
  pad the sequence to 256, then every 4th sample is kept (4× decimation).

Gesture label mapping (sorted alphabetically = folder order in HGAG-DATA1)
---------------------------------------------------------------------------
  0  Clapping
  1  Coin Flipping
  2  Finger Snapping
  3  Fist Making
  4  Horizontal Wrist Extension
  5  Index Finger Flick
  6  Index Thumb Tap
  7  Shooting
  8  Thumb Up
  9  Wrist Extension
 10  Wrist Flexion

Directory layout expected (HGAG-DATA1, organised by gesture)
-------------------------------------------------------------
  <root>/
    Clapping/
      Subject_1/.csv/{accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z}_data.csv
      Subject_2/...
      ...
    Coin Flipping/...
    ...

Usage
-----
    from imu_lm.data.hgag_loader import HGAGDataset, make_hgag_loaders

    train_ds, val_ds, test_ds = HGAGDataset.subject_splits(
        root="/mnt/nas2/yiquan_data/Hand Gesture Accelerometer and Gyroscope "
             "Dataset (HGAG-DATA)/HGAG-DATA/HGAG-DATA1",
        val_subjects=5,
        test_subjects=5,
        seed=0,
        normalize=True,
    )
    # or use the factory:
    loaders = make_hgag_loaders(root=..., batch_size=64)
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GESTURE_NAMES: List[str] = [
    "Clapping",
    "Coin Flipping",
    "Finger Snapping",
    "Fist Making",
    "Horizontal Wrist Extension",
    "Index Finger Flick",
    "Index Thumb Tap",
    "Shooting",
    "Thumb Up",
    "Wrist Extension",
    "Wrist Flexion",
]

NUM_CLASSES = len(GESTURE_NAMES)        # 11
NUM_CHANNELS = 6                        # acc_x/y/z, gyro_x/y/z
RAW_TIMESTEPS = 250                     # original samples per gesture at 200 Hz
RAW_HZ = 200
NUM_TIMESTEPS = 64                      # resampled: 50 Hz × 1.28 s
SAMPLING_HZ = 50

_AXIS_FILES = [
    "accel_x_data.csv",
    "accel_y_data.csv",
    "accel_z_data.csv",
    "gyro_x_data.csv",
    "gyro_y_data.csv",
    "gyro_z_data.csv",
]


# ---------------------------------------------------------------------------
# Resampling helper: 200 Hz / 250 samples  →  50 Hz / 64 samples (1.28 s)
# ---------------------------------------------------------------------------

_DECIMATE_FACTOR = RAW_HZ // SAMPLING_HZ   # 4
_TARGET_LEN = NUM_TIMESTEPS                # 64
_PADDED_LEN = _TARGET_LEN * _DECIMATE_FACTOR  # 256


def _resample_to_50hz(data: np.ndarray) -> np.ndarray:
    """Resample a batch of raw 200-Hz sequences to 50 Hz, 64 timesteps.

    Steps
    -----
    1. Pad the time axis from 250 → 256 by repeating the last 6 samples.
    2. Decimate by 4 (take every 4th sample) → 64 timesteps.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(N_reps, 6, 250)`` — raw 200-Hz data.

    Returns
    -------
    np.ndarray
        Shape ``(N_reps, 6, 64)`` — resampled 50-Hz data.
    """
    n_pad = _PADDED_LEN - data.shape[-1]   # 256 - 250 = 6
    if n_pad > 0:
        # Duplicate the last n_pad samples along the time axis
        tail = data[..., -n_pad:]          # (..., 6)
        data = np.concatenate([data, tail], axis=-1)  # (..., 256)
    return data[..., ::_DECIMATE_FACTOR]   # (..., 64)


# ---------------------------------------------------------------------------
# Low-level loader
# ---------------------------------------------------------------------------

def _load_subject_gesture(
    root: str,
    gesture: str,
    subject: str,
) -> Optional[np.ndarray]:
    """Return float32 array of shape (50, 6, 250) or None if files are missing."""
    csv_dir = os.path.join(root, gesture, subject, ".csv")
    if not os.path.isdir(csv_dir):
        return None

    axes: List[np.ndarray] = []
    for fname in _AXIS_FILES:
        fpath = os.path.join(csv_dir, fname)
        if not os.path.isfile(fpath):
            logger.warning("[HGAG] missing file: %s", fpath)
            return None
        arr = np.genfromtxt(fpath, delimiter=",", dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]   # handle single-row edge case
        axes.append(arr)  # (50, 250)

    data = np.stack(axes, axis=1)  # (50, 6, 250)
    data = _resample_to_50hz(data)  # (50, 6, 64)
    return data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HGAGDataset(Dataset):
    """Flat map-style dataset: one item = one gesture repetition.

    Parameters
    ----------
    root:
        Path to HGAG-DATA1 directory (gesture-organised layout).
    subjects:
        List of subject folder names to include, e.g. ["Subject_1", "Subject_2"].
        If None, all subjects found in the first gesture folder are used.
    gestures:
        List of gesture names to include. Defaults to all 11.
    normalize:
        If True, apply per-channel z-score normalisation across the loaded
        split (mean/std computed from this split only).
    transform:
        Optional callable applied to the (6, 250) float32 numpy array
        *before* converting to tensor. E.g. for augmentation.
    """

    def __init__(
        self,
        root: str,
        subjects: Optional[List[str]] = None,
        gestures: Optional[List[str]] = None,
        normalize: bool = True,
        transform=None,
    ):
        super().__init__()
        self.root = root
        self.gestures = gestures if gestures is not None else GESTURE_NAMES
        self.normalize = normalize
        self.transform = transform

        # Resolve subject list
        if subjects is None:
            first_gesture_dir = os.path.join(root, self.gestures[0])
            subjects = sorted(
                d for d in os.listdir(first_gesture_dir)
                if os.path.isdir(os.path.join(first_gesture_dir, d))
            )
        self.subjects = subjects

        # Build (X, y) arrays
        X_list: List[np.ndarray] = []   # each: (N_reps, 6, 250)
        y_list: List[np.ndarray] = []   # each: (N_reps,)

        for gesture in self.gestures:
            label = GESTURE_NAMES.index(gesture)
            for subject in self.subjects:
                data = _load_subject_gesture(root, gesture, subject)
                if data is None:
                    logger.debug(
                        "[HGAG] skipping %s / %s (no data)", gesture, subject
                    )
                    continue
                X_list.append(data)                                      # (50, 6, 250)
                y_list.append(np.full(data.shape[0], label, dtype=np.int64))

        if not X_list:
            raise RuntimeError(
                f"HGAGDataset: no data loaded from '{root}'. "
                "Check that the path points to HGAG-DATA1."
            )

        self._X = np.concatenate(X_list, axis=0)   # (N, 6, 250)
        self._y = np.concatenate(y_list, axis=0)   # (N,)

        n_samples = self._X.shape[0]
        logger.info(
            "[HGAG] split loaded: subjects=%d gestures=%d samples=%d",
            len(self.subjects), len(self.gestures), n_samples,
        )

        # Per-channel z-score normalisation (computed on this split)
        if normalize:
            # mean/std over (N, T) per channel
            mean = self._X.mean(axis=(0, 2), keepdims=True)  # (1, 6, 1)
            std  = self._X.std(axis=(0, 2), keepdims=True)   # (1, 6, 1)
            std  = np.where(std < 1e-8, 1.0, std)
            self._X = (self._X - mean) / std

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._X[idx]   # (6, 250) float32 numpy
        y = int(self._y[idx])

        if self.transform is not None:
            x = self.transform(x)

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    # ------------------------------------------------------------------
    # Convenience: per-class sample counts
    # ------------------------------------------------------------------
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for label, name in enumerate(self.gestures):
            counts[name] = int((self._y == GESTURE_NAMES.index(name)).sum())
        return counts

    # ------------------------------------------------------------------
    # Factory: subject-stratified splits
    # ------------------------------------------------------------------
    @classmethod
    def subject_splits(
        cls,
        root: str,
        val_subjects: int = 5,
        test_subjects: int = 5,
        seed: int = 0,
        normalize: bool = True,
        gestures: Optional[List[str]] = None,
    ) -> Tuple["HGAGDataset", "HGAGDataset", "HGAGDataset"]:
        """Return (train, val, test) datasets split by subject.

        Args:
            root: Path to HGAG-DATA1.
            val_subjects: Number of subjects for validation.
            test_subjects: Number of subjects for test.
            seed: Random seed for subject shuffling.
            normalize: Apply per-split z-score normalisation.
            gestures: Subset of gestures to use (default: all 11).
        """
        gestures = gestures or GESTURE_NAMES
        first_gesture_dir = os.path.join(root, gestures[0])
        all_subjects = sorted(
            d for d in os.listdir(first_gesture_dir)
            if os.path.isdir(os.path.join(first_gesture_dir, d))
        )

        rng = random.Random(seed)
        shuffled = list(all_subjects)
        rng.shuffle(shuffled)

        test_subjs  = shuffled[:test_subjects]
        val_subjs   = shuffled[test_subjects : test_subjects + val_subjects]
        train_subjs = shuffled[test_subjects + val_subjects:]

        logger.info(
            "[HGAG] total_subjects=%d  train=%d  val=%d  test=%d",
            len(all_subjects), len(train_subjs), len(val_subjs), len(test_subjs),
        )

        train_ds = cls(root, subjects=train_subjs, gestures=gestures, normalize=normalize)
        val_ds   = cls(root, subjects=val_subjs,   gestures=gestures, normalize=normalize)
        test_ds  = cls(root, subjects=test_subjs,  gestures=gestures, normalize=normalize)

        return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_hgag_loaders(
    root: str,
    batch_size: int = 64,
    val_subjects: int = 5,
    test_subjects: int = 5,
    seed: int = 0,
    normalize: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    gestures: Optional[List[str]] = None,
) -> Dict[str, DataLoader]:
    """Return {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.

    Args:
        root: Path to HGAG-DATA1 directory.
        batch_size: Mini-batch size.
        val_subjects: Number of subjects held out for validation.
        test_subjects: Number of subjects held out for test.
        seed: Shuffle seed for subject split.
        normalize: Per-split z-score normalisation.
        num_workers: DataLoader workers (data is already in RAM, so 0–4 is fine).
        pin_memory: Pin CUDA memory for faster transfer.
        gestures: Subset of gestures to include (default: all 11).

    Example::

        loaders = make_hgag_loaders(
            root="/mnt/nas2/yiquan_data/Hand Gesture Accelerometer and "
                 "Gyroscope Dataset (HGAG-DATA)/HGAG-DATA/HGAG-DATA1",
            batch_size=64,
        )
        for x, y in loaders["train"]:
            # x: [B, 6, 64]  float32  (50 Hz, 1.28 s)
            # y: [B]          int64   (0–10)
            ...
    """
    train_ds, val_ds, test_ds = HGAGDataset.subject_splits(
        root=root,
        val_subjects=val_subjects,
        test_subjects=test_subjects,
        seed=seed,
        normalize=normalize,
        gestures=gestures,
    )

    def _loader(ds: HGAGDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    loaders = {
        "train": _loader(train_ds, shuffle=True),
        "val":   _loader(val_ds,   shuffle=False),
        "test":  _loader(test_ds,  shuffle=False),
    }
    logger.info(
        "[HGAG] loaders ready — train=%d  val=%d  test=%d  batch=%d",
        len(train_ds), len(val_ds), len(test_ds), batch_size,
    )
    return loaders
