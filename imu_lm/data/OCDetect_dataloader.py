"""OCDetect_dataloader.py
PyTorch DataLoader for the OCDetect OCD-detection wrist-IMU dataset.

Data source
-----------
678 pre-processed CSV files in:
    /mnt/nas2/yiquan_data/preprocessed/
    OCDetect_{subject:02d}_recording_{rec_no:02d}_{rec_id}.csv

Column layout
-------------
  timestamp    : int64   nanoseconds since start of recording
  datetime     : str     ISO wall-clock timestamp
  acc x/y/z   : float   accelerometer  [m s⁻²]  Android sensor frame (raw)
  gyro x/y/z  : float   gyroscope      [rad s⁻¹] Android sensor frame (raw)
  user yes/no  : float   sparse event marker (-1 / 1); NaN between events
  compulsive   : float   sparse binary label (0 = no, 1 = compulsive episode); NaN between events
  urge         : float   sparse annotation; NaN between events
  tense        : float   sparse annotation; NaN between events
  ignore       : int     data-quality mask
                          0 = usable sample
                          2 = exclude (pocket / charging / sensor malfunction)
                          3 / 4 / 5 = additional exclusion reasons
  relabeled    : int     annotation provenance (0 = original, 1/2 = post-hoc correction)

Sampling rate
-------------
  50 Hz (timestamp step = 20,000,000 ns = 20 ms)

Subjects
--------
  22 participants, IDs: 1–5, 7, 9–13, 15, 18–22, 24, 25, 27, 29, 30
  9–47 recordings per subject; 678 recordings total.

Windowing strategy
------------------
  Window  : 128 samples = 2.56 s @ 50 Hz
  Hop     : 64 samples  = 50 % overlap
  Masking : windows are slid only within contiguous ``ignore == 0`` segments.
            Any gap (sample with ignore ≠ 0) resets the segment boundary; no
            window spans a gap.
  Label   : a window receives label 1 if ANY sample in it has ``compulsive == 1.0``
            (NaN → 0, so unannotated samples are treated as background).

Coordinate frame
----------------
  Raw CSV columns are in the Android sensor frame. This loader applies the
  Android → FLU transform before windowing:

      F = +X_android,  L = -Y_android,  U = +Z_android

  Channel order emitted: [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]

Shapes emitted per item
-----------------------
  x : [6, 128]  float32  — (channels, time); channels-first for ViT1D
  y : []        int64    — 0 (background) or 1 (compulsive episode)

Split strategies
----------------
  LOPO (Leave-One-Subject-Out) : held-out subject → val + test; rest → train
  Random                       : pool all windows, random 80/10/10 split
"""

from __future__ import annotations

import glob
import logging
import multiprocessing as _mp
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR_DEFAULT = "/mnt/nas2/yiquan_data/preprocessed"

SAMPLE_HZ      = 50
WINDOW_SAMPLES = 128   # 2.56 s @ 50 Hz
HOP_SAMPLES    = 64    # 50 % overlap

IMU_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]

# All 22 subject IDs present in the dataset
ALL_SUBJECTS: List[int] = [
    1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13,
    15, 18, 19, 20, 21, 22, 24, 25, 27, 29, 30,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_files(data_dir: str) -> List[str]:
    """Return sorted list of OCDetect data files under *data_dir*.

    Prefers ``.parquet`` files (faster reads) over ``.csv`` when both exist.
    """
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.parquet")))
    csv_files     = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.csv")))

    if not parquet_files and not csv_files:
        raise FileNotFoundError(f"[OCDetect] no CSV or parquet files found under {data_dir}")

    if parquet_files:
        # Use parquet where available; fall back to CSV for any missing ones.
        parquet_stems = {os.path.splitext(f)[0] for f in parquet_files}
        csv_fallbacks = [
            f for f in csv_files
            if os.path.splitext(f)[0] not in parquet_stems
        ]
        files = sorted(parquet_files + csv_fallbacks)
        if csv_fallbacks:
            _log.info(
                "[OCDetect] using %d parquet + %d CSV (conversion in progress?)",
                len(parquet_files), len(csv_fallbacks),
            )
        return files

    return csv_files


def _subject_id_from_path(path: str) -> int:
    """Parse the integer subject ID from an OCDetect filename.

    Example: ``OCDetect_30_recording_35_f5c311a4…csv`` → 30
    """
    name = os.path.basename(path)
    m = re.match(r"OCDetect_(\d+)_", name)
    if m is None:
        raise ValueError(f"[OCDetect] cannot parse subject ID from: {name}")
    return int(m.group(1))


def _load_recording(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one CSV and return ``(imu, compulsive, ignore)``.

    The raw Android-frame IMU channels are converted to FLU on load:
        F = +X_android,  L = -Y_android,  U = +Z_android
    Channel order: [acc_F, acc_L, acc_U, gyro_F, gyro_L, gyro_U]

    Returns
    -------
    imu        : [N, 6]  float32 — acc_F/L/U, gyro_F/L/U  (FLU frame)
    compulsive : [N]     float32 — 0.0 (background/NaN→0) or 1.0 (episode)
    ignore     : [N]     int32   — quality mask (0=usable, else exclude)
    """
    df = pd.read_parquet(path, columns=["ignore", "compulsive"] + IMU_COLS) \
        if path.endswith(".parquet") else \
        pd.read_csv(
            path,
            usecols=["ignore", "compulsive"] + IMU_COLS,
            dtype={col: "float32" for col in IMU_COLS},
            low_memory=False,
        )
    imu        = df[IMU_COLS].values.astype(np.float32)
    # Android → FLU: negate Y channels (acc_y → acc_L, gyro_y → gyro_L)
    imu[:, 1] *= -1  # acc_y  →  acc_L
    imu[:, 4] *= -1  # gyro_y →  gyro_L
    compulsive = df["compulsive"].fillna(0.0).values.astype(np.float32)
    ignore     = df["ignore"].values.astype(np.int32)
    return imu, compulsive, ignore


def _process_one_file(
    args: Tuple[str, int, int],
) -> Tuple[np.ndarray, np.ndarray, Optional[str], Optional[str]]:
    """Multiprocessing worker: load and window one OCDetect CSV file."""
    path, window_samples, hop_samples = args
    try:
        imu, compulsive, ignore = _load_recording(path)
    except Exception as exc:
        return (
            np.empty((0, 6, window_samples), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            str(exc),
            os.path.basename(path),
        )

    usable     = (ignore == 0)
    changes    = np.diff(usable.astype(np.int8), prepend=0, append=0)
    seg_starts = np.where(changes ==  1)[0]
    seg_ends   = np.where(changes == -1)[0]

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for start, end in zip(seg_starts, seg_ends):
        seg_len = int(end - start)
        if seg_len < window_samples:
            continue
        seg_imu  = imu[start:end]
        seg_comp = compulsive[start:end]
        n_wins   = (seg_len - window_samples) // hop_samples + 1
        idxs     = np.arange(n_wins) * hop_samples
        idx_mat  = idxs[:, None] + np.arange(window_samples)
        wins_imu  = seg_imu[idx_mat]
        wins_comp = seg_comp[idx_mat]
        labels    = (wins_comp >= 1.0).any(axis=1).astype(np.int64)
        X_parts.append(wins_imu.transpose(0, 2, 1).astype(np.float32))
        y_parts.append(labels)

    if not X_parts:
        return (
            np.empty((0, 6, window_samples), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            None, None,
        )
    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        None, None,
    )


def _collect_windows(
    files: List[str],
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide windows over *files*, respecting the ``ignore`` quality mask.

    Windows are slid within contiguous ``ignore == 0`` segments only. A
    segment boundary (any sample with ``ignore != 0``) causes the current
    window accumulation to restart, so no window ever spans a gap.

    Parameters
    ----------
    files          : sorted list of CSV paths to process
    window_samples : window length in samples (default 128 = 2.56 s @ 50 Hz)
    hop_samples    : stride between windows in samples (default 64 = 50 %)

    Returns
    -------
    X : [M, 6, window_samples]  float32
    y : [M]                     int64    (0 = background, 1 = compulsive)
    """
    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []

    args_list = [(f, window_samples, hop_samples) for f in files]
    n_workers = min(_mp.cpu_count(), 8, len(files))

    # fork is fastest on Linux and avoids the spawn startup cost.
    ctx = _mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        for X_c, y_c, err, fname in tqdm(
            pool.imap_unordered(_process_one_file, args_list),
            total=len(files),
            desc="[OCDetect] loading",
            unit="file",
            dynamic_ncols=True,
        ):
            if err is not None:
                _log.warning("[OCDetect] skipping %s — %s", fname, err)
            if len(X_c) > 0:
                X_chunks.append(X_c)
                y_chunks.append(y_c)

    if not X_chunks:
        raise RuntimeError(
            "[OCDetect] no windows collected — check data_dir and ignore masks"
        )

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)

    _log.info(
        "[OCDetect] collected %d windows | pos=%d (%.2f %%) neg=%d",
        len(X), int((y == 1).sum()), 100.0 * (y == 1).mean(), int((y == 0).sum()),
    )
    return X, y


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OCDetectDataset(Dataset):
    """Pre-windowed OCDetect IMU dataset.

    Parameters
    ----------
    X   : [M, 6, 128]  float32  channels-first (C, T)
    y   : [M]          int64    0 = background, 1 = compulsive episode
    cfg : merged config dict — used for preprocess (impute / filter / normalize)
          and augmentation, matching the pipeline in loaders.py and
          nymeria_real_wrist_loader_FLU.py.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, cfg: dict | None = None) -> None:
        # Keep as numpy so preprocess_window (numpy) can operate without a copy.
        self._X   = X.astype(np.float32)   # [M, 6, 128]
        self._y   = torch.from_numpy(y).long()
        self._cfg = cfg or {}
        self.stats = PreprocessStats()

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Stored as [C, T]; preprocess_window expects [T, C].
        Xw = self._X[idx].T  # [128, 6]

        Xproc = preprocess_window(Xw, self._cfg, self.stats)  # [C, T] or None
        if Xproc is None:
            # Fallback: plain zscore so the batch is never empty.
            Xproc = (Xw - Xw.mean(axis=0, keepdims=True)) / (Xw.std(axis=0, keepdims=True) + 1e-6)
            Xproc = Xproc.T.astype(np.float32)  # [C, T]

        Xproc = apply_augment(Xproc.T, self._cfg).T  # augment in [T,C] space, back to [C,T]
        return torch.from_numpy(Xproc).float(), self._y[idx]


# ---------------------------------------------------------------------------
# Internal loader builder
# ---------------------------------------------------------------------------

def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    cfg: dict | None = None,
) -> DataLoader:
    ds = OCDetectDataset(X, y, cfg=cfg)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Public loader factories
# ---------------------------------------------------------------------------

def make_ocdetect_loaders_lopo(
    data_dir: str = DATA_DIR_DEFAULT,
    test_subject: int = None,
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
    cfg: dict | None = None,
) -> Dict[str, DataLoader]:
    """Leave-One-Subject-Out (LOPO) split.

    The held-out ``test_subject`` provides both val and test loaders.
    All other subjects are pooled for training.

    Parameters
    ----------
    data_dir       : directory containing ``OCDetect_*.csv`` files
    test_subject   : integer subject ID to hold out (must be in ``ALL_SUBJECTS``)
    batch_size     : training batch size
    eval_batch_size: val/test batch size (defaults to ``batch_size``)
    num_workers    : DataLoader worker processes
    pin_memory     : DataLoader pin-memory flag
    seed           : reserved for API consistency (unused)
    window_samples : window length in 50 Hz samples (default 128 = 2.56 s)
    hop_samples    : sliding stride in samples (default 64 = 50 % overlap)

    Returns
    -------
    dict with keys ``'train_loader'``, ``'val_loader'``, ``'test_loader'``
    """
    if test_subject not in ALL_SUBJECTS:
        raise ValueError(
            f"[OCDetect] test_subject must be one of {ALL_SUBJECTS}, got {test_subject!r}"
        )

    eval_batch_size = eval_batch_size or batch_size
    all_files = _list_files(data_dir)

    train_files = [f for f in all_files if _subject_id_from_path(f) != test_subject]
    test_files  = [f for f in all_files if _subject_id_from_path(f) == test_subject]

    _log.info(
        "[OCDetect] LOPO test_subject=%d | %d train files, %d test files",
        test_subject, len(train_files), len(test_files),
    )

    X_train, y_train = _collect_windows(train_files, window_samples, hop_samples)
    X_eval,  y_eval  = _collect_windows(test_files,  window_samples, hop_samples)

    _log.info(
        "[OCDetect] LOPO — train: %d (pos=%d, neg=%d) | eval: %d (pos=%d, neg=%d)",
        len(X_train), int((y_train == 1).sum()), int((y_train == 0).sum()),
        len(X_eval),  int((y_eval  == 1).sum()), int((y_eval  == 0).sum()),
    )

    return {
        "train_loader": _make_loader(X_train, y_train, batch_size,      True,  num_workers, pin_memory, cfg=cfg),
        "val_loader":   _make_loader(X_eval,  y_eval,  eval_batch_size, False, num_workers, pin_memory, cfg=cfg),
        "test_loader":  _make_loader(X_eval,  y_eval,  eval_batch_size, False, num_workers, pin_memory, cfg=cfg),
    }


def make_ocdetect_loaders_random(
    data_dir: str = DATA_DIR_DEFAULT,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = HOP_SAMPLES,
    cfg: dict | None = None,
) -> Dict[str, DataLoader]:
    """Random window-level split pooling all subjects.

    Useful for quickly checking model capacity but does NOT respect subject
    independence — prefer ``make_ocdetect_loaders_lopo`` for rigorous evaluation.

    Parameters
    ----------
    data_dir       : directory containing ``OCDetect_*.csv`` files
    splits         : (train, val, test) fractions that must sum to 1.0
    batch_size     : training batch size
    eval_batch_size: val/test batch size (defaults to ``batch_size``)
    num_workers    : DataLoader worker processes
    pin_memory     : DataLoader pin-memory flag
    seed           : random seed for reproducible shuffling
    window_samples : window length in 50 Hz samples (default 128 = 2.56 s)
    hop_samples    : sliding stride in samples (default 64 = 50 % overlap)

    Returns
    -------
    dict with keys ``'train_loader'``, ``'val_loader'``, ``'test_loader'``
    """
    assert abs(sum(splits) - 1.0) < 1e-6, f"splits must sum to 1.0, got {sum(splits)}"

    eval_batch_size = eval_batch_size or batch_size
    all_files = _list_files(data_dir)

    X, y = _collect_windows(all_files, window_samples, hop_samples)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n       = len(X)
    n_train = int(splits[0] * n)
    n_val   = int(splits[1] * n)

    X_train, y_train = X[:n_train],               y[:n_train]
    X_val,   y_val   = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test,  y_test  = X[n_train + n_val:],         y[n_train + n_val:]

    _log.info(
        "[OCDetect] random split — train: %d (pos=%d) | val: %d (pos=%d) | test: %d (pos=%d)",
        len(X_train), int((y_train == 1).sum()),
        len(X_val),   int((y_val   == 1).sum()),
        len(X_test),  int((y_test  == 1).sum()),
    )

    return {
        "train_loader": _make_loader(X_train, y_train, batch_size,      True,  num_workers, pin_memory, cfg=cfg),
        "val_loader":   _make_loader(X_val,   y_val,   eval_batch_size, False, num_workers, pin_memory, cfg=cfg),
        "test_loader":  _make_loader(X_test,  y_test,  eval_batch_size, False, num_workers, pin_memory, cfg=cfg),
    }