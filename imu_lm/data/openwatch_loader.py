"""OpenWatch wrist-IMU gesture dataloader.

OpenWatch layout
----------------
    open_watch_without_augmentations/
      label_map.json
      id_<N>/<position>/<gesture>.csv

Each gesture CSV contains one fixed-length recording with columns:
    PPG_mean, AccX, AccY, AccZ, GyroX, GyroY, GyroZ

This loader treats each CSV as one supervised gesture sample, resamples it to
the configured model window length, applies the repo's window preprocessing,
and returns tensors shaped [C, T].
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.spectrogram import stft_encode
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.clip_voting import STRIDE_SAMPLES, WINDOW_SAMPLES, restricted_window_starts, sliding_window_starts
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)

OPENWATCH_ROOT = "/projects/b53017/vbc1591/openwatch_raw/open_watch_without_augmentations"
RAW_IMU_COLUMNS = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
PPG_COLUMNS = ["PPG_mean"]
_DUPLICATE_SUFFIX_RE = re.compile(r"\(\d+\)$")

# ---------------------------------------------------------------------------
# Release contamination maps (derived + verified 2026-08-15 by
# analysis/openwatch_forensics/derive_contamination_maps.py — regenerate with
# that script, do not edit by hand).
#
# 1. The "50 real participants" release contains byte-identical recordings
#    under different subject ids (id_41/id_42 are complete copies of id_39,
#    id_43 of id_40, id_48 of id_46, id_49 half-duplicated, plus 1-6-file
#    overlaps entangling 39/46/47/48/49). Splitting two ids from the same
#    group across train/val leaks identical samples.
# 2. `open_watch_augmentations` ids 51-85 are wrist-MIRRORED (+ locomotion-
#    overlaid) copies of real subjects — identified by max-lag correlation of
#    the mirror-invariant |acc| magnitude (> 0.999) in non-walking position
#    dirs, majority-voted over 8 gestures. Treating them as independent
#    subjects can put a mirrored copy of a held-out subject into train.
#
# `contamination_safe: true` in the objective cfg block activates both fixes:
# exact-file dedup at discovery + subject splits computed over CANONICAL
# identities (aug id -> source id -> duplicate-group representative).
# Default is off so every pre-existing run/eval stays bit-reproducible.
# ---------------------------------------------------------------------------
OPENWATCH_DUPLICATE_SUBJECT_GROUPS = [[39, 41, 42, 46, 47, 48, 49], [40, 43]]
OPENWATCH_AUG_SOURCE = {
    51: 1, 52: 2, 53: 3, 54: 4, 55: 5, 56: 7, 57: 8, 58: 9, 59: 10, 60: 11,
    61: 13, 62: 14, 63: 15, 64: 17, 65: 21, 66: 24, 67: 26, 68: 27, 69: 28,
    70: 29, 71: 31, 72: 32, 73: 36, 74: 37, 75: 38, 76: 40, 77: 45, 78: 39,
    79: 50, 80: 16, 81: 19, 82: 22, 83: 39, 84: 39, 85: 39,
}
_DUP_CANON = {sid: min(group) for group in OPENWATCH_DUPLICATE_SUBJECT_GROUPS for sid in group}


def canonical_subject(subject_id: int) -> int:
    """Collapse an OpenWatch subject id to its canonical real identity:
    augmented ids (51-85) map to their mirror-source subject, and members of
    a byte-duplicate group map to the group's lowest id."""
    sid = OPENWATCH_AUG_SOURCE.get(int(subject_id), int(subject_id))
    return _DUP_CANON.get(sid, sid)


@dataclass(frozen=True)
class OpenWatchRecord:
    path: str
    subject_id: int
    position: str
    gesture: str
    label: int


def _as_list(value: Optional[Any]) -> Optional[List[Any]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _load_label_map(root: Path) -> Dict[str, int]:
    path = root / "label_map.json"
    with open(path, "r", encoding="utf-8") as fh:
        label_map = json.load(fh)
    return {str(k): int(v) for k, v in label_map.items()}


def _subject_from_dir(name: str) -> Optional[int]:
    if not name.startswith("id_"):
        return None
    suffix = name[3:]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _canonical_gesture_name(stem: str) -> str:
    return _DUPLICATE_SUFFIX_RE.sub("", stem)


def _dedup_exact_records(records: List[OpenWatchRecord]) -> List[OpenWatchRecord]:
    """Drop byte-identical duplicate files, keeping the first occurrence in
    (subject_id, position, path) order — so the lowest-id member of each
    duplicate group is the one that survives, matching `canonical_subject`."""
    import hashlib

    ordered = sorted(records, key=lambda r: (r.subject_id, r.position, r.path))
    seen: set = set()
    kept: List[OpenWatchRecord] = []
    for r in ordered:
        with open(r.path, "rb") as fh:
            h = hashlib.md5(fh.read()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        kept.append(r)
    if len(kept) != len(records):
        logger.info("[openwatch] dedup_exact dropped %d byte-identical duplicate files (%d -> %d)",
                    len(records) - len(kept), len(records), len(kept))
    return kept


def discover_openwatch_records(
    root: str | Path = OPENWATCH_ROOT,
    subjects: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[str]] = None,
    gestures: Optional[Sequence[str]] = None,
    label_ids: Optional[Sequence[int]] = None,
    collapse_background: bool = False,
    background_label: str = "background",
    background_fraction: float = 1.0,
    seed: int = 42,
    dedup_exact: bool = False,
) -> Tuple[List[OpenWatchRecord], Dict[str, int]]:
    """Discover OpenWatch gesture CSV records under *root*.

    collapse_background=True switches from "filter down to `gestures`" to a
    "keyword vs. background" framing: every gesture in `gestures` keeps its own
    class (in the given order), and every OTHER gesture is collapsed into one
    extra `background_label` class instead of being dropped. Since OpenWatch has
    no dedicated idle/rest recordings, "background" here means "all non-target
    gesture recordings" — since that pool is far larger than any target class
    (hundreds vs. ~70-80 samples/class), `background_fraction` randomly
    subsamples it (seeded) so the background class doesn't drown out the
    targets.
    """
    root = Path(root)
    label_map = _load_label_map(root)

    subject_set = {int(s) for s in subjects} if subjects is not None else None
    position_set = {str(p) for p in positions} if positions is not None else None
    gesture_set = {str(g) for g in gestures} if gestures is not None else None
    label_set = {int(l) for l in label_ids} if label_ids is not None else None

    if collapse_background and not gesture_set:
        raise ValueError("[openwatch] collapse_background=True requires `gestures` "
                          "to define the target/foreground classes")

    records: List[OpenWatchRecord] = []
    for subject_dir in sorted(root.glob("id_*"), key=lambda p: p.name):
        subject_id = _subject_from_dir(subject_dir.name)
        if subject_id is None:
            continue
        if subject_set is not None and subject_id not in subject_set:
            continue

        for position_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            position = position_dir.name
            if position_set is not None and position not in position_set:
                continue
            for csv_path in sorted(position_dir.glob("*.csv")):
                gesture = _canonical_gesture_name(csv_path.stem)
                if gesture not in label_map:
                    logger.warning("[openwatch] gesture not in label_map, skipping: %s", csv_path)
                    continue
                label = int(label_map[gesture])
                is_target = gesture_set is not None and gesture in gesture_set
                if not collapse_background:
                    if gesture_set is not None and not is_target:
                        continue
                    if label_set is not None and label not in label_set:
                        continue
                records.append(
                    OpenWatchRecord(
                        path=str(csv_path),
                        subject_id=subject_id,
                        position=position,
                        gesture=gesture,
                        label=label,
                    )
                )

    if dedup_exact:
        # before background collapse/subsampling, so duplicates can't be
        # "kept" by the seeded background draw
        records = _dedup_exact_records(records)

    if collapse_background:
        target_order = list(dict.fromkeys(gestures))  # de-dup, keep given order
        target_records = [r for r in records if r.gesture in gesture_set]
        background_records = [r for r in records if r.gesture not in gesture_set]

        if background_fraction < 1.0:
            rng = np.random.RandomState(seed)
            keep_n = max(1, int(round(len(background_records) * background_fraction)))
            idx = rng.choice(len(background_records), size=keep_n, replace=False)
            background_records = [background_records[i] for i in sorted(idx)]

        name_to_new = {g: i for i, g in enumerate(target_order)}
        background_id = len(target_order)
        records = (
            [OpenWatchRecord(path=r.path, subject_id=r.subject_id, position=r.position,
                              gesture=r.gesture, label=name_to_new[r.gesture])
             for r in target_records]
            + [OpenWatchRecord(path=r.path, subject_id=r.subject_id, position=r.position,
                                gesture=r.gesture, label=background_id)
               for r in background_records]
        )
        label_map_out = {**name_to_new, background_label: background_id}
        logger.info(
            "[openwatch] discovered %d records under %s subjects=%s positions=%s "
            "targets=%d background=%d (fraction=%.3f) classes=%d",
            len(records), root, subjects, positions, len(target_records),
            len(background_records), background_fraction, len(label_map_out),
        )
        return records, label_map_out

    # Remap to a contiguous [0, num_used_classes) label space. A no-op when the
    # full label set is used (gestures/label_ids both None); when filtered down
    # to a subset (e.g. a micro-gesture-only split), this keeps class indices
    # dense so a classifier head can be sized to exactly len(used_labels)
    # instead of the full label_map.json range.
    used_labels = sorted({r.label for r in records})
    remap = {old: new for new, old in enumerate(used_labels)}
    records = [
        OpenWatchRecord(path=r.path, subject_id=r.subject_id, position=r.position,
                        gesture=r.gesture, label=remap[r.label])
        for r in records
    ]
    label_map_out = {name: remap[old] for name, old in label_map.items() if old in remap}

    logger.info(
        "[openwatch] discovered %d records under %s subjects=%s positions=%s classes=%d",
        len(records), root, subjects, positions, len(used_labels),
    )
    return records, label_map_out


def _read_csv_array(path: str, include_ppg: bool = False) -> np.ndarray:
    columns = (PPG_COLUMNS if include_ppg else []) + RAW_IMU_COLUMNS
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [[float(row[col]) for col in columns] for row in reader]
    return np.asarray(rows, dtype=np.float32)


def apply_unit_scale(x: np.ndarray, include_ppg: bool, acc_scale: float = 1.0, gyro_scale: float = 1.0) -> np.ndarray:
    """Raw integer sensor counts -> physical units (m/s^2, rad/s). Used by the dataset AND the
    evaluation scripts so training and scoring see identical input (cfg keys acc_scale/gyro_scale)."""
    acc_scale, gyro_scale = float(acc_scale), float(gyro_scale)
    if acc_scale == 1.0 and gyro_scale == 1.0:
        return x
    o = 1 if include_ppg else 0
    x = x.copy(); x[:, o:o + 3] *= np.float32(acc_scale); x[:, o + 3:o + 6] *= np.float32(gyro_scale)
    return x


def _to_flu(x: np.ndarray, include_ppg: bool) -> np.ndarray:
    """Convert raw watch axes to an FLU-like order.

    OpenWatch does not pin down device orientation in the README.  This optional
    transform follows the common smartwatch convention used elsewhere here:
    F = device +Y, L = device +X, U = device +Z.  PPG, when present, stays first.
    """
    offset = 1 if include_ppg else 0
    out = x.copy()
    out[:, offset + 0] = x[:, offset + 1]  # acc_F = AccY
    out[:, offset + 1] = x[:, offset + 0]  # acc_L = AccX
    out[:, offset + 2] = x[:, offset + 2]  # acc_U = AccZ
    out[:, offset + 3] = x[:, offset + 4]  # gyro_F = GyroY
    out[:, offset + 4] = x[:, offset + 3]  # gyro_L = GyroX
    out[:, offset + 5] = x[:, offset + 5]  # gyro_U = GyroZ
    return out


def _resample_to_len(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x.astype(np.float32, copy=False)
    return resample(x, target_len, axis=0).astype(np.float32)


def _split_subjects(
    subjects: Sequence[int],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[List[int], List[int], List[int]]:
    subjects = sorted({int(s) for s in subjects})
    rng = np.random.RandomState(seed)
    shuffled = np.asarray(subjects, dtype=np.int64)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    test = sorted(shuffled[:n_test].tolist())
    val = sorted(shuffled[n_test : n_test + n_val].tolist())
    train = sorted(shuffled[n_test + n_val :].tolist())
    return train, val, test


def _stratified_split_subjects_by_position(
    records: Sequence["OpenWatchRecord"],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Subject-level split stratified by each subject's rare-posture coverage
    (whether they have any `walking` and/or `standing_arm_down` data --
    `sitting`/`standing` are common enough across subjects not to need their
    own strata), approximating the OpenWatch paper's "randomized selection
    balanced per posture and activity condition" split. `_split_subjects`
    above (this repo's long-standing default, still used everywhere else) is
    a plain unstratified shuffle -- per-subject position coverage in this
    dataset is extremely uneven (only 3/50 subjects have all 4 postures;
    16/50 have only `sitting`), so an unstratified split can and does produce
    train/val posture distributions that look nothing alike (verified: our
    default seed=42 val split is 40% walking despite only 3/50 subjects total
    having any walking data at all).

    Stratifying by the *exact* position-coverage tuple creates strata too
    small (many size 1-4) for proportional rounding at a 10% val/test
    fraction to ever place a single member outside train (round(1*0.1)=0),
    which would silently force every rare-posture subject into train --
    the opposite failure mode. Using the coarser 4-way (has_walking x
    has_arm_down) stratification and *guaranteeing* at least 1 member in
    val/test for any stratum with >=2 members (instead of pure proportional
    rounding) keeps rare postures represented in every split without that
    degenerate all-in-train collapse.
    """
    subj_positions: Dict[int, set] = {}
    for r in records:
        subj_positions.setdefault(r.subject_id, set()).add(r.position)

    def _stratum(positions: set) -> Tuple[bool, bool]:
        return ("walking" in positions, "standing_arm_down" in positions)

    by_group: Dict[Tuple[bool, bool], List[int]] = {}
    for subj, positions in subj_positions.items():
        by_group.setdefault(_stratum(positions), []).append(subj)

    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for group in sorted(by_group.keys()):
        members = np.asarray(sorted(by_group[group]), dtype=np.int64)
        rng.shuffle(members)
        n = len(members)
        n_test = int(round(n * test_fraction))
        n_val = int(round(n * val_fraction))
        if n >= 2:
            n_test = max(n_test, 1) if test_fraction > 0 else n_test
            n_val = max(n_val, 1) if val_fraction > 0 else n_val
            # leave at least 1 for train if the guarantee above would empty it
            while n_test + n_val >= n and (n_test > 0 or n_val > 0):
                if n_val >= n_test and n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1
                else:
                    break
        test.extend(members[:n_test].tolist())
        val.extend(members[n_test : n_test + n_val].tolist())
        train.extend(members[n_test + n_val :].tolist())
    return sorted(train), sorted(val), sorted(test)


class OpenWatchDataset(Dataset):
    """One OpenWatch gesture CSV per sample.

    windowed=True switches to the OpenWatch paper's own unit of prediction
    (arXiv:2605.04791): one sample = one `window_samples`-long slice of the
    raw recording at `window_stride` (default 1s/100 samples @ stride 20 —
    ~11 windows per 300-sample recording). Every window inherits its
    recording's label (as in the paper's window-level metrics). This closes
    the train/eval mismatch documented in EXPERIMENTS.md §5d, where heads
    trained on whole recordings were evaluated on windows they had never
    seen the like of.

    Model-input handling for a window (the varlen alt-attn encoder accepts
    any multiple of patch_size natively — sinusoidal/2d pos-embed, pretrained
    on 64..512-sample windows):
    - window_zero_pad=True: feed the RAW slice untouched (no resampling, no
      time stretch) and zero-pad the time axis up to the next patch_size
      multiple (100 -> 112 = 7 patches) AFTER preprocessing, so the filter
      and z-score only ever see real samples and the pad stays exact zeros.
    - window_target_len=N: resample the slice to N samples instead.
    - default (neither): resample to the model's full fixed T (5.12x
      stretch) for encoders without varlen support.
    """

    def __init__(
        self,
        records: Sequence[OpenWatchRecord],
        cfg: Any,
        split_name: str = "train",
        include_ppg: bool = False,
        acc_scale: float = 1.0,
        gyro_scale: float = 1.0,
        coordinate_frame: str = "raw",
        target_len: Optional[int] = None,
        preload: bool = True,
        windowed: bool = False,
        window_samples: int = WINDOW_SAMPLES,
        window_stride: int = STRIDE_SAMPLES,
        window_target_len: Optional[int] = None,
        window_zero_pad: bool = False,
        window_pad_len: Optional[int] = None,
        window_restrict_len: Optional[int] = None,
    ):
        super().__init__()
        self.records = list(records)
        self.cfg = cfg
        self.split_name = split_name
        self.include_ppg = bool(include_ppg)
        self.acc_scale = float(acc_scale)      # raw counts -> m/s^2 (1.0 = leave as counts)
        self.gyro_scale = float(gyro_scale)    # raw counts -> rad/s
        self.coordinate_frame = str(coordinate_frame).lower()
        self.stats = PreprocessStats()
        self._T = int(target_len or compute_T_and_hop(cfg)[0])
        self._spec_enabled = bool((cfg_get(cfg, ["spectrogram"], {}) or {}).get("enabled", False))
        self.windowed = bool(windowed)
        self.window_samples = int(window_samples)
        self.window_stride = int(window_stride)
        self.window_zero_pad = bool(window_zero_pad)
        self.window_restrict_len = int(window_restrict_len) if window_restrict_len else None
        patch_size = int(cfg_get(cfg, ["vit1d", "patch_size"], 16))
        if window_pad_len:
            # explicit pad target (e.g. 128 = a 1.28s frame for a 1s window)
            if int(window_pad_len) % patch_size or int(window_pad_len) < self.window_samples:
                raise ValueError(f"window_pad_len={window_pad_len} must be a multiple of "
                                 f"patch_size={patch_size} and >= window_samples={self.window_samples}")
            self._pad_T = int(window_pad_len)
        else:
            # next patch multiple >= window_samples, e.g. 100 -> 112
            self._pad_T = ((self.window_samples + patch_size - 1) // patch_size) * patch_size
        self._win_T = int(window_target_len) if window_target_len else self._T
        self._cache: Optional[List[Tuple[np.ndarray, int]]] = [] if (preload and not self.windowed) else None

        if self.windowed:
            # Raw signals are tiny (~7KB each), so always cache them — the
            # window index needs every recording's length up front anyway.
            self._raw: List[np.ndarray] = [self._load_raw(rec) for rec in self.records]
            def _starts(n):
                if self.window_restrict_len:
                    return restricted_window_starts(
                        n, window=self.window_samples, stride=self.window_stride,
                        restrict_len=self.window_restrict_len)
                return sliding_window_starts(n, window=self.window_samples, stride=self.window_stride)

            self._window_index: List[Tuple[int, int]] = [
                (i, s) for i, x in enumerate(self._raw) for s in _starts(len(x))
            ]
        elif self._cache is not None:
            for rec in self.records:
                self._cache.append((self._load_window(rec), int(rec.label)))

        logger.info(
            "[openwatch] split=%s records=%d windows=%s T=%d win_T=%s include_ppg=%s frame=%s preload=%s windowed=%s",
            split_name, len(self.records),
            len(self._window_index) if self.windowed else "n/a",
            self._T, self._win_T if self.windowed else "n/a",
            self.include_ppg, self.coordinate_frame, preload, self.windowed,
        )

    def _load_raw(self, rec: OpenWatchRecord) -> np.ndarray:
        x = _read_csv_array(rec.path, include_ppg=self.include_ppg)
        x = apply_unit_scale(x, self.include_ppg, self.acc_scale, self.gyro_scale)   # raw counts -> m/s^2, rad/s
        if self.coordinate_frame == "flu":
            x = _to_flu(x, include_ppg=self.include_ppg)
        elif self.coordinate_frame not in {"raw", "device"}:
            raise ValueError(f"unsupported OpenWatch coordinate_frame={self.coordinate_frame}")
        return x

    def _load_window(self, rec: OpenWatchRecord) -> np.ndarray:
        return _resample_to_len(self._load_raw(rec), self._T)

    def __len__(self) -> int:
        return len(self._window_index) if self.windowed else len(self.records)

    def __getitem__(self, idx: int):
        if self.windowed:
            rec_idx, start = self._window_index[idx]
            seg = self._raw[rec_idx][start : start + self.window_samples]
            # zero-pad mode: feed the raw slice untouched; padding happens
            # AFTER preprocess/augment below so filter+z-score never see it
            xw = seg if self.window_zero_pad else _resample_to_len(seg, self._win_T)
            label = int(self.records[rec_idx].label)
        elif self._cache is not None:
            xw, label = self._cache[idx]
        else:
            rec = self.records[idx]
            xw, label = self._load_window(rec), int(rec.label)

        xproc = preprocess_window(xw, self.cfg, self.stats)
        if xproc is None:
            return None

        if self.split_name == "train":
            xproc = apply_augment(xproc.T, self.cfg).T

        if self.windowed and self.window_zero_pad and xproc.shape[1] < self._pad_T:
            xproc = np.pad(xproc, ((0, 0), (0, self._pad_T - xproc.shape[1])))

        if self._spec_enabled:
            out = stft_encode(torch.from_numpy(xproc).float(), self.cfg)
            x_tensor = out[1] if isinstance(out, tuple) else out
        else:
            x_tensor = torch.from_numpy(xproc).float()
        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


def make_openwatch_loaders(
    cfg: Any,
    root: str | Path = OPENWATCH_ROOT,
    train_subjects: Optional[Sequence[int]] = None,
    val_subjects: Optional[Sequence[int]] = None,
    test_subjects: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[str]] = None,
    gestures: Optional[Sequence[str]] = None,
    label_ids: Optional[Sequence[int]] = None,
    include_ppg: bool = False,
    acc_scale: float = 1.0,
    gyro_scale: float = 1.0,
    coordinate_frame: str = "raw",
    preload: bool = True,
    collapse_background: bool = False,
    background_label: str = "background",
    background_fraction: float = 1.0,
    windowed: bool = False,
    window_samples: int = WINDOW_SAMPLES,
    window_stride: int = STRIDE_SAMPLES,
    window_target_len: Optional[int] = None,
    window_zero_pad: bool = False,
    window_pad_len: Optional[int] = None,
    window_restrict_len: Optional[int] = None,
    eval_root: Optional[str | Path] = None,
    contamination_safe: bool = False,
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders for OpenWatch.

    eval_root, when set to a different release than `root`, draws val/test
    records from it while train records come from `root` — the augmented
    release is a training-time augmentation in the OpenWatch paper, so its
    models are trained on augmented data but evaluated on real recordings.
    The subject split is computed from the eval_root discovery (the real
    sessions — also what the paper stratifies on), so an aug-trained and a
    plain no-aug run share identical val/test sets and are directly
    comparable per-sample.

    contamination_safe=True activates both release-contamination fixes (see
    the constants block at the top of this module): byte-identical duplicate
    files are dropped at discovery, and the subject split is computed over
    CANONICAL identities via `canonical_subject` — so a duplicate-group
    member can never land on the other side of the split from its twin, and
    an augmented (mirrored) id can only train when its real source subject
    is train-side.
    """
    batch_size = int(cfg_get(cfg, ["data", "batch_size"], 256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers = int(cfg_get(cfg, ["data", "num_workers"], 0))
    pin_memory = bool(cfg_get(cfg, ["data", "pin_memory"], False))
    seed = int(cfg_get(cfg, ["splits", "seed"], cfg_get(cfg, ["trainer", "seed"], 0)))

    discover_kwargs = dict(
        subjects=None,
        positions=positions,
        gestures=gestures,
        label_ids=label_ids,
        collapse_background=collapse_background,
        background_label=background_label,
        background_fraction=background_fraction,
        seed=seed,
        dedup_exact=contamination_safe,
    )
    all_records, label_map = discover_openwatch_records(root=root, **discover_kwargs)
    if eval_root is not None and str(eval_root) != str(root):
        eval_records, eval_label_map = discover_openwatch_records(root=eval_root, **discover_kwargs)
        if eval_label_map != label_map:
            raise ValueError(
                f"[openwatch] label_map mismatch between root={root} ({label_map}) "
                f"and eval_root={eval_root} ({eval_label_map})"
            )
        logger.info("[openwatch] train records from %s, val/test records from eval_root=%s", root, eval_root)
    else:
        eval_records = all_records
    if collapse_background:
        logger.info("[openwatch] label_map=%s", label_map)

    # In contamination_safe mode the split is computed and applied over
    # canonical subject identities, so duplicate twins and mirrored copies
    # travel with their source subject.
    _canon = canonical_subject if contamination_safe else (lambda s: int(s))
    if contamination_safe:
        from dataclasses import replace as _dc_replace

        split_source_records = [
            _dc_replace(r, subject_id=_canon(r.subject_id)) for r in eval_records
        ]
    else:
        split_source_records = eval_records
    all_subjects = sorted({r.subject_id for r in split_source_records})

    if train_subjects is None and val_subjects is None and test_subjects is None:
        val_fraction = float(cfg_get(cfg, ["splits", "val_fraction"], 0.1))
        test_fraction = float(cfg_get(cfg, ["splits", "test_fraction"], 0.1))
        split_strategy = str(cfg_get(cfg, ["splits", "strategy"], "random")).lower()
        if split_strategy == "posture_stratified":
            train_subjects, val_subjects, test_subjects = _stratified_split_subjects_by_position(
                split_source_records, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction
            )
            logger.info("[openwatch] using posture_stratified split")
        else:
            train_subjects, val_subjects, test_subjects = _split_subjects(
                all_subjects, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction
            )
    elif train_subjects is None:
        held = set(int(s) for s in (list(val_subjects or []) + list(test_subjects or [])))
        train_subjects = [s for s in all_subjects if s not in held]

    split_subjects = {
        "train": {int(s) for s in train_subjects or []},
        "val": {int(s) for s in val_subjects or []},
        "test": {int(s) for s in test_subjects or []},
    }

    records_by_split: Dict[str, List[OpenWatchRecord]] = {
        "train": [r for r in all_records if _canon(r.subject_id) in split_subjects["train"]],
        "val": [r for r in eval_records if _canon(r.subject_id) in split_subjects["val"]],
        "test": [r for r in eval_records if _canon(r.subject_id) in split_subjects["test"]],
    }

    def _make(name: str, records: List[OpenWatchRecord], bs: int, shuffle: bool):
        if not records:
            return None
        ds = OpenWatchDataset(
            records=records,
            cfg=cfg,
            split_name=name,
            include_ppg=include_ppg,
            acc_scale=acc_scale,
            gyro_scale=gyro_scale,
            coordinate_frame=coordinate_frame,
            preload=preload,
            windowed=windowed,
            window_samples=window_samples,
            window_stride=window_stride,
            window_target_len=window_target_len,
            window_zero_pad=window_zero_pad,
            window_pad_len=window_pad_len,
            window_restrict_len=window_restrict_len,
        )
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=collate_skip_none,
            persistent_workers=num_workers > 0,
        )

    loaders: Dict[str, DataLoader] = {}
    for name, bs, shuffle in [
        ("train", batch_size, True),
        ("val", eval_batch_size, False),
        ("test", eval_batch_size, False),
    ]:
        loader = _make(name, records_by_split[name], bs, shuffle)
        if loader is not None:
            loaders[f"{name}_loader"] = loader
            logger.info(
                "[openwatch] %s_loader subjects=%s records=%d",
                name, sorted(split_subjects[name]), len(records_by_split[name]),
            )
    return loaders


def make_openwatch_loaders_from_cfg(
    cfg: Any, objective_key: str = "openwatch_gesture_supervised"
) -> Dict[str, DataLoader]:
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    ow_cfg = cfg_get(cfg, ["objective", objective_key], {}) or {}
    cfg_block = {**data_cfg, **ow_cfg}

    return make_openwatch_loaders(
        cfg=cfg,
        root=cfg_block.get("openwatch_root", cfg_block.get("root", OPENWATCH_ROOT)),
        train_subjects=_as_list(cfg_block.get("train_subjects")),
        val_subjects=_as_list(cfg_block.get("val_subjects")),
        test_subjects=_as_list(cfg_block.get("test_subjects")),
        positions=_as_list(cfg_block.get("positions")),
        gestures=_as_list(cfg_block.get("gestures")),
        label_ids=_as_list(cfg_block.get("label_ids")),
        include_ppg=bool(cfg_block.get("include_ppg", False)),
        acc_scale=float(cfg_block.get("acc_scale", 1.0)),
        gyro_scale=float(cfg_block.get("gyro_scale", 1.0)),
        coordinate_frame=str(cfg_block.get("coordinate_frame", "raw")),
        preload=bool(cfg_block.get("preload", True)),
        collapse_background=bool(cfg_block.get("collapse_background", False)),
        background_label=str(cfg_block.get("background_label", "background")),
        background_fraction=float(cfg_block.get("background_fraction", 1.0)),
        windowed=bool(cfg_block.get("windowed", False)),
        window_samples=int(cfg_block.get("window_samples", WINDOW_SAMPLES)),
        window_stride=int(cfg_block.get("window_stride", STRIDE_SAMPLES)),
        window_target_len=(int(cfg_block["window_target_len"]) if cfg_block.get("window_target_len") else None),
        window_zero_pad=bool(cfg_block.get("window_zero_pad", False)),
        window_pad_len=(int(cfg_block["window_pad_len"]) if cfg_block.get("window_pad_len") else None),
        window_restrict_len=(int(cfg_block["window_restrict_len"]) if cfg_block.get("window_restrict_len") else None),
        eval_root=cfg_block.get("eval_root"),
        contamination_safe=bool(cfg_block.get("contamination_safe", False)),
    )
