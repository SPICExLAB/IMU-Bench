"""watchhand_loader.py
PyTorch Dataset for the WatchHand IMU gesture dataset.

Reference: Kim et al., "WatchHand: Enabling Continuous Hand Pose Tracking On
Off-the-Shelf Smartwatches", CHI 2026.
Dataset: https://doi.org/10.7298/qf1v-j805

Data sources with IMU: Study-1, Study-2, Study-4.
Study-3 does NOT include IMU data and is silently skipped.

──────────────────────────── Label mapping (18 classes) ─────────────────────
  0  Thumb-bent       │  5  ASL-0  │  10  ASL-5  │  15  Flexion
  1  Index-bent       │  6  ASL-1  │  11  ASL-6  │  16  Extension
  2  Middle-bent      │  7  ASL-2  │  12  ASL-7  │  17  Ulnar_deviation
  3  Ring-bent        │  8  ASL-3  │  13  ASL-8  │
  4  Little-bent      │  9  ASL-4  │  14  ASL-9  │

──────────────────────── records.txt field format ───────────────────────────
  pose_id, start_sec, end_sec, display_label

──────────────────────────── IMU CSV field format ───────────────────────────
  Acce / Gyro / Magnet columns:  "ax ay az ts_ms  ax ay az ts_ms  …"
  Sampling rate: 100 Hz.
  One CSV row per recording session (blockNum == session index 0–9).

──────────────────────────── Tensor shapes ──────────────────────────────────
  x : [6, window_size]  float32  — channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
  y : []                int64    — pose class index 0–17

Sampling rate: raw IMU is 100 Hz; data is downsampled to target_hz (default 50 Hz)
before windowing.  window_size=200 @ 50 Hz ≈ 4 s per gesture window.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset

_log = logging.getLogger(__name__)

csv.field_size_limit(10_000_000)

# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

POSE_ID_TO_NAME: dict[int, str] = {
    0: "Thumb-bent",
    1: "Index-bent",
    2: "Middle-bent",
    3: "Ring-bent",
    4: "Little-bent",
    5:  "ASL-0",
    6:  "ASL-1",
    7:  "ASL-2",
    8:  "ASL-3",
    9:  "ASL-4",
    10: "ASL-5",
    11: "ASL-6",
    12: "ASL-7",
    13: "ASL-8",
    14: "ASL-9",
    15: "Flexion",
    16: "Extension",
    17: "Ulnar_deviation",
}
NUM_CLASSES = 18

# Pose ID groups — use these with the ``pose_ids`` parameter of WatchHandDataset
SIMPLE_POSE_IDS: List[int] = list(range(0, 5))   # Thumb/Index/Middle/Ring/Little-bent
ASL_POSE_IDS:    List[int] = list(range(5, 15))  # ASL-0 through ASL-9
WRIST_POSE_IDS:  List[int] = list(range(15, 18)) # Flexion, Extension, Ulnar_deviation


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _parse_imu_stream(raw: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a packed IMU field into data and timestamps.

    The format is a flat space-separated string of 4-tuples:
        ``x y z timestamp_ms  x y z timestamp_ms  …``

    Returns
    -------
    data : ndarray [N, 3]  float32
    ts_ms : ndarray [N]    float64  Unix milliseconds
    """
    vals = np.fromstring(raw, dtype=np.float64, sep=" ")
    n_samples = len(vals) // 4
    arr = vals[: n_samples * 4].reshape(n_samples, 4)
    ts = arr[:, 3]
    xyz = arr[:, :3].astype(np.float32)
    # Wear OS sometimes emits duplicate timestamps — keep first occurrence only
    _, keep = np.unique(ts, return_index=True)
    return xyz[keep], ts[keep]


def _resample_window(window: np.ndarray, target_len: int) -> np.ndarray:
    """Resample an [N, C] window to [target_len, C] using Fourier resampling."""
    if len(window) == target_len:
        return window
    return resample(window, target_len, axis=0).astype(np.float32)


def _load_session_samples(
    row: dict,
    records_path: Path,
    window_size: int,
    min_samples: int,
    target_hz: int = 50,
    src_hz: int = 100,
) -> List[Tuple[np.ndarray, int]]:
    """Extract all (imu_window, label) pairs from one recording session.

    Parameters
    ----------
    row : CSV row dict with keys ``Acce``, ``Gyro``
    records_path : path to the ``*_records.txt`` file for this session
    window_size : output window length (number of IMU time steps after resampling)
    min_samples : discard gesture windows with fewer raw samples than this
    target_hz : target sampling rate; stream is downsampled from src_hz if different
    src_hz : native sampling rate of the raw IMU stream (100 Hz for WatchHand)

    Returns
    -------
    list of (imu_window [window_size, 6], pose_id) tuples
    """
    acce_data, acce_ts = _parse_imu_stream(row["Acce"])
    gyro_data, gyro_ts = _parse_imu_stream(row["Gyro"])

    # Align accel and gyro on common timestamps (inner join by nearest ms)
    acce_dict = {float(t): v for t, v in zip(acce_ts, acce_data)}
    gyro_dict = {float(t): v for t, v in zip(gyro_ts, gyro_data)}
    common_ts = np.array(sorted(set(acce_ts.tolist()) & set(gyro_ts.tolist())),
                         dtype=np.float64)
    if len(common_ts) == 0:
        return []

    imu_data = np.stack(
        [np.concatenate([acce_dict[t], gyro_dict[t]]) for t in common_ts],
        axis=0,
    ).astype(np.float32)  # [N, 6]

    # Transform from Android coordinates to FLU (right wrist).
    # Android: +X = right of screen, +Y = top of screen, +Z = out of screen face.
    # FLU:     +X = toward fingertips (Forward), +Y = toward thumb (Left), +Z = out of screen (Up).
    # Mapping: FLU_X = +Android_Y,  FLU_Y = -Android_X,  FLU_Z = +Android_Z
    flu = np.empty_like(imu_data)
    flu[:, 0] =  imu_data[:, 1]  # acc  FLU_X = Android_Y
    flu[:, 1] = -imu_data[:, 0]  # acc  FLU_Y = -Android_X
    flu[:, 2] =  imu_data[:, 2]  # acc  FLU_Z = Android_Z  (unchanged)
    flu[:, 3] =  imu_data[:, 4]  # gyro FLU_X = Android_Y
    flu[:, 4] = -imu_data[:, 3]  # gyro FLU_Y = -Android_X
    flu[:, 5] =  imu_data[:, 5]  # gyro FLU_Z = Android_Z  (unchanged)
    imu_data = flu

    # Downsample session stream from src_hz to target_hz (anti-aliased via Fourier resample)
    if target_hz != src_hz and len(imu_data) > 1:
        new_len = int(round(len(imu_data) * target_hz / src_hz))
        imu_data = resample(imu_data, new_len, axis=0).astype(np.float32)
        # Remap timestamps linearly to preserve absolute time range
        common_ts = np.linspace(common_ts[0], common_ts[-1], new_len, dtype=np.float64)

    samples: List[Tuple[np.ndarray, int]] = []
    if not records_path.exists():
        _log.warning("records file not found: %s", records_path)
        return samples

    with open(records_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            pose_id = int(parts[0])
            start_ms = float(parts[1]) * 1000.0
            end_ms = float(parts[2]) * 1000.0

            mask = (common_ts >= start_ms) & (common_ts <= end_ms)
            n_raw = int(mask.sum())
            if n_raw < min_samples:
                continue

            window = _resample_window(imu_data[mask], window_size)
            samples.append((window, pose_id))

    return samples


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WatchHandDataset(Dataset):
    """PyTorch Dataset for WatchHand IMU hand-gesture classification.

    Parameters
    ----------
    watchhand_root : str | Path
        Root of the WatchHand repo (contains ``Study-1/``, ``Study-2/``, etc.).
    window_size : int
        Number of IMU time steps per sample.  Default 200 ≈ 4 s @ 50 Hz.
    subjects : list[int] | None
        Participant IDs to include (e.g. ``[1, 2, 3]``).  ``None`` = all.
    studies : list[str] | None
        Study folder names to search (default: ``["Study-1"]``).
        Study-3 has no IMU and will always be skipped.
    min_samples : int
        Minimum raw IMU samples in a gesture window; shorter windows are
        discarded (catches sync-boundary edge cases).
    target_hz : int
        Target sampling rate after downsampling (default 50 Hz, matching
        imuposer_wrist_loader_FLU.py).  Raw WatchHand IMU is 100 Hz.
    pose_ids : list[int] | None
        Restrict to a subset of pose classes (e.g. ``SIMPLE_POSE_IDS``,
        ``ASL_POSE_IDS``, ``WRIST_POSE_IDS``).  Labels are remapped to a
        contiguous 0-based index within the subset.  ``None`` = all 18.
    transform : callable | None
        Optional transform applied to the IMU tensor ``x`` before returning.
    block_nums : list[int] | None
        Restrict to specific session block indices (0–9 in Study-1).  Useful
        for within-subject splits: e.g. ``block_nums=[0,1,2,3,4,5,6,7]`` for
        train and ``block_nums=[8,9]`` for val.  ``None`` = all blocks.

    Yields
    ------
    x : Tensor [6, window_size]  float32
        IMU channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
    y : Tensor []  int64
        Gesture class index remapped to 0-based within ``pose_ids`` (or 0–17
        when ``pose_ids`` is None).
    """

    def __init__(
        self,
        watchhand_root: str | Path,
        window_size: int = 200,
        subjects: Optional[List[int]] = None,
        studies: Optional[List[str]] = None,
        min_samples: int = 10,
        target_hz: int = 50,
        pose_ids: Optional[List[int]] = None,
        transform=None,
        block_nums: Optional[List[int]] = None,
    ):
        self.window_size = window_size
        self.transform = transform
        self._samples: List[Tuple[np.ndarray, int]] = []
        _block_nums_set = set(block_nums) if block_nums is not None else None

        root = Path(watchhand_root)
        if studies is None:
            studies = ["Study-1"]

        # Recursively collect all directories that contain imu_config.json
        session_dirs: List[Path] = []
        for study_name in studies:
            study_path = root / study_name
            if not study_path.exists():
                continue
            session_dirs.extend(
                p.parent for p in study_path.rglob("imu_config.json")
            )

        _log.info("WatchHand: found %d session folders", len(session_dirs))

        for session_dir in sorted(session_dirs):
            # Skip left-wrist recordings — only use right wrist
            folder_name = session_dir.name
            if "_left_" in folder_name:
                continue

            # Filter by subject
            if subjects is not None:
                # Extract numeric subject ID from the deepest "sub{N}_*" component
                sub_id = _extract_subject_id(session_dir)
                if sub_id is None or sub_id not in subjects:
                    continue

            imu_cfg_path = session_dir / "imu_config.json"
            with open(imu_cfg_path) as f:
                imu_cfg = json.load(f)

            csv_files = imu_cfg.get("imu", {}).get("files", [])
            if not csv_files:
                continue
            csv_path = session_dir / csv_files[0]
            if not csv_path.exists():
                _log.warning("CSV not found: %s", csv_path)
                continue

            gt_files: List[str] = imu_cfg.get("ground_truth", {}).get("files", [])
            if not gt_files:
                continue

            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            n_before = len(self._samples)
            for row in rows:
                session_idx = int(row["blockNum"])
                if session_idx >= len(gt_files):
                    continue
                # Within-subject split: skip blocks not in the requested set
                if _block_nums_set is not None and session_idx not in _block_nums_set:
                    continue
                records_path = session_dir / gt_files[session_idx]
                new_samples = _load_session_samples(
                    row, records_path, window_size, min_samples,
                    target_hz=target_hz, src_hz=100,
                )
                self._samples.extend(new_samples)

            n_added = len(self._samples) - n_before
            _log.debug("%s: +%d samples", session_dir.name, n_added)

        _log.info(
            "WatchHand: loaded %d gesture windows before pose filter",
            len(self._samples),
        )

        # Filter to requested pose classes and remap labels to contiguous 0-based indices
        if pose_ids is not None:
            id_set = set(pose_ids)
            remap = {old: new for new, old in enumerate(sorted(pose_ids))}
            self._samples = [
                (w, remap[lbl]) for w, lbl in self._samples if lbl in id_set
            ]
            self._num_classes = len(pose_ids)
            self._label_names: dict[int, str] = {
                remap[pid]: POSE_ID_TO_NAME[pid] for pid in sorted(pose_ids)
            }
        else:
            self._num_classes = NUM_CLASSES
            self._label_names = POSE_ID_TO_NAME

        _log.info(
            "WatchHand: %d gesture windows, %d classes",
            len(self._samples), self._num_classes,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        window, label = self._samples[idx]
        x = torch.from_numpy(window).T.contiguous()  # [6, window_size]
        y = torch.tensor(label, dtype=torch.long)
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    @property
    def label_names(self) -> dict[int, str]:
        return self._label_names

    @property
    def num_classes(self) -> int:
        return self._num_classes


# ---------------------------------------------------------------------------
# Helper: extract subject ID from a path
# ---------------------------------------------------------------------------

def _extract_subject_id(path: Path) -> Optional[int]:
    """Return the numeric ID from the first ``sub{N}`` component in the path."""
    for part in path.parts:
        if part.startswith("sub"):
            try:
                return int(part.split("_")[0][3:])
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_watchhand_loaders(
    watchhand_root: str | Path,
    window_size: int = 200,
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    studies: Optional[List[str]] = None,
    pose_ids: Optional[List[int]] = None,
    target_hz: int = 50,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train/val DataLoaders for WatchHand IMU gesture data.

    Parameters
    ----------
    watchhand_root : path to WatchHand repo root (contains Study-1/, etc.)
    window_size : IMU window length in samples (default 200 = 4 s @ 50 Hz)
    train_subjects : participant IDs for training (None = all)
    val_subjects : participant IDs for validation (None = no val loader)
    studies : study folders to include (default: Study-1)
    pose_ids : restrict to a pose subset, e.g. SIMPLE_POSE_IDS (None = all 18)
    target_hz : target sampling rate (default 50, matching imuposer loader)
    batch_size, num_workers, pin_memory : DataLoader settings

    Returns
    -------
    (train_loader, val_loader)  — val_loader is None when val_subjects is None
    """
    train_ds = WatchHandDataset(
        watchhand_root=watchhand_root,
        window_size=window_size,
        subjects=train_subjects,
        studies=studies,
        pose_ids=pose_ids,
        target_hz=target_hz,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader: Optional[DataLoader] = None
    if val_subjects is not None:
        val_ds = WatchHandDataset(
            watchhand_root=watchhand_root,
            window_size=window_size,
            subjects=val_subjects,
            studies=studies,
            pose_ids=pose_ids,
            target_hz=target_hz,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Group-specific convenience factories
# ---------------------------------------------------------------------------

def make_watchhand_simple_loaders(
    watchhand_root: str | Path,
    window_size: int = 200,
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    studies: Optional[List[str]] = None,
    target_hz: int = 50,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """DataLoaders for Simple gestures only (5 classes, labels remapped 0–4).

    Classes: Thumb-bent(0), Index-bent(1), Middle-bent(2), Ring-bent(3), Little-bent(4)
    """
    return make_watchhand_loaders(
        watchhand_root=watchhand_root,
        window_size=window_size,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        studies=studies,
        pose_ids=SIMPLE_POSE_IDS,
        target_hz=target_hz,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_watchhand_asl_loaders(
    watchhand_root: str | Path,
    window_size: int = 200,
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    studies: Optional[List[str]] = None,
    target_hz: int = 50,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """DataLoaders for ASL gestures only (10 classes, labels remapped 0–9).

    Classes: ASL-0(0) through ASL-9(9)
    """
    return make_watchhand_loaders(
        watchhand_root=watchhand_root,
        window_size=window_size,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        studies=studies,
        pose_ids=ASL_POSE_IDS,
        target_hz=target_hz,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def make_watchhand_wrist_loaders(
    watchhand_root: str | Path,
    window_size: int = 200,
    train_subjects: Optional[List[int]] = None,
    val_subjects: Optional[List[int]] = None,
    studies: Optional[List[str]] = None,
    target_hz: int = 50,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """DataLoaders for Wrist rotation gestures only (3 classes, labels remapped 0–2).

    Classes: Flexion(0), Extension(1), Ulnar_deviation(2)
    """
    return make_watchhand_loaders(
        watchhand_root=watchhand_root,
        window_size=window_size,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        studies=studies,
        pose_ids=WRIST_POSE_IDS,
        target_hz=target_hz,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# ---------------------------------------------------------------------------
# Within-subject (same-person) split factories
# Block 0-7 = train, block 8-9 = val, all subjects shared between splits.
# Default split uses the 10 blocks in Study-1 (index 0..9).
# ---------------------------------------------------------------------------

# Default block split constants
WITHIN_SUBJECT_TRAIN_BLOCKS: List[int] = list(range(0, 8))  # blocks 0–7
WITHIN_SUBJECT_VAL_BLOCKS:   List[int] = [8, 9]             # blocks 8–9


def make_watchhand_within_subject_loaders(
    watchhand_root: str | Path,
    window_size: int = 200,
    subjects: Optional[List[int]] = None,
    studies: Optional[List[str]] = None,
    pose_ids: Optional[List[int]] = None,
    target_hz: int = 50,
    train_block_nums: Optional[List[int]] = None,
    val_block_nums: Optional[List[int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build within-subject train/val DataLoaders for WatchHand.

    Both splits use the same subject pool; they are distinguished by which
    session blocks (``block_nums``) are included.  Defaults: blocks 0–7
    for train, blocks 8–9 for val (10 blocks per subject in Study-1).

    Parameters
    ----------
    subjects : participant IDs to include (None = all)
    train_block_nums : block indices for training (default: 0–7)
    val_block_nums   : block indices for validation (default: 8–9)
    """
    train_blocks = train_block_nums if train_block_nums is not None else WITHIN_SUBJECT_TRAIN_BLOCKS
    val_blocks   = val_block_nums   if val_block_nums   is not None else WITHIN_SUBJECT_VAL_BLOCKS

    train_ds = WatchHandDataset(
        watchhand_root=watchhand_root,
        window_size=window_size,
        subjects=subjects,
        studies=studies,
        pose_ids=pose_ids,
        target_hz=target_hz,
        block_nums=train_blocks,
    )
    val_ds = WatchHandDataset(
        watchhand_root=watchhand_root,
        window_size=window_size,
        subjects=subjects,
        studies=studies,
        pose_ids=pose_ids,
        target_hz=target_hz,
        block_nums=val_blocks,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def make_watchhand_within_subject_simple_loaders(
    watchhand_root: str | Path,
    window_size: int = 200,
    subjects: Optional[List[int]] = None,
    studies: Optional[List[str]] = None,
    target_hz: int = 50,
    train_block_nums: Optional[List[int]] = None,
    val_block_nums: Optional[List[int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Within-subject DataLoaders for Simple gestures (5 classes, labels 0–4)."""
    return make_watchhand_within_subject_loaders(
        watchhand_root=watchhand_root, window_size=window_size,
        subjects=subjects, studies=studies, pose_ids=SIMPLE_POSE_IDS,
        target_hz=target_hz, train_block_nums=train_block_nums,
        val_block_nums=val_block_nums, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def make_watchhand_within_subject_asl_loaders(
    watchhand_root: str | Path,
    window_size: int = 200,
    subjects: Optional[List[int]] = None,
    studies: Optional[List[str]] = None,
    target_hz: int = 50,
    train_block_nums: Optional[List[int]] = None,
    val_block_nums: Optional[List[int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Within-subject DataLoaders for ASL gestures (10 classes, labels 0–9)."""
    return make_watchhand_within_subject_loaders(
        watchhand_root=watchhand_root, window_size=window_size,
        subjects=subjects, studies=studies, pose_ids=ASL_POSE_IDS,
        target_hz=target_hz, train_block_nums=train_block_nums,
        val_block_nums=val_block_nums, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def make_watchhand_within_subject_wrist_loaders(
    watchhand_root: str | Path,
    window_size: int = 200,
    subjects: Optional[List[int]] = None,
    studies: Optional[List[str]] = None,
    target_hz: int = 50,
    train_block_nums: Optional[List[int]] = None,
    val_block_nums: Optional[List[int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Within-subject DataLoaders for Wrist gestures (3 classes, labels 0–2)."""
    return make_watchhand_within_subject_loaders(
        watchhand_root=watchhand_root, window_size=window_size,
        subjects=subjects, studies=studies, pose_ids=WRIST_POSE_IDS,
        target_hz=target_hz, train_block_nums=train_block_nums,
        val_block_nums=val_block_nums, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
