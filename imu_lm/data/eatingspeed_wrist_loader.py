"""eatingspeed_wrist_loader.py
PyTorch DataLoader for the "Eating Speed" wrist-IMU eating/drinking-gesture dataset.

Data source
-----------
    /projects/b53017/vbc1591/EatingSpeed/   (see 00_ReadMe.txt for the full description)

Three sub-datasets, each stored as four pickled Python lists (one entry per
participant), either already extracted as a folder or still zipped:

  FD-I    : 34 participants, free-living, ~251.70 h both-wrist IMU.
            Point-wise labels for ALL eating/drinking gestures all day.
            Official participant-level 7-fold CV split ships in
            ``FD1_7_fold_id_list.zip``.
  FD-II   : 27 participants, ~261.68 h. Hold-out dataset — only gestures
            *during meal sessions* are labelled; everything outside a meal
            is unannotated background (still label 0, but not exhaustively
            verified "other" the way FD-I is).
  MO      : 46 participants, meal-only sessions (~12.5 h combined per hand).
            Training-only split used by the source paper (no participant
            overlap with FD-I / FD-II) — there is no official test carve-out.

Per subset, per participant, per hand:
    X_{L,R}.pkl : list[np.ndarray[N_samples, 6]] float64 — [ax, ay, az, gx, gy, gz]
                  raw Shimmer3 accelerometer + gyroscope, native 64 Hz. The gyroscope is
                  stored in deg/s; ``load_subset`` converts it to rad/s (2026-09-17).
    Y_{L,R}.pkl : list[np.ndarray[N_samples]]    int/float — point-wise label
                  0 = other, 1 = eating gesture, 2 = drinking gesture.

Memory footprint — run on a compute node, not the login node
--------------------------------------------------------------
``load_subset`` fully deserializes the requested subset's 4 pickle lists (the
per-participant arrays can't be partially unpickled), and ``collect_windows``
then eagerly materializes every sliding window as a separate ``[6, T]`` copy
— for the default 2.56 s / 50 % hop settings, the windowed arrays end up
comparable in size to the raw pickles, not negligible. Measured peak RSS
(default windowing, ``hand="both"``, all participants, via ``/usr/bin/time -v``
on a compute node):
    FD-I  (34 participants, official fold 0, train+test) : ~15.2 GB peak
    FD-II (27 participants, single eval loader)           : ~16.7 GB peak
    MO    (46 participants, both hands)                   : <1 GB peak — always safe
This host's *login node* caps a whole user session to 6 GB RAM (cgroup
``memory.limit_in_bytes``), independent of the machine's actual free memory —
loading FD-I/FD-II with ``hand="both"`` will get SIGKILLed there even just
during ``load_subset`` (~5.9 GB / ~6.7 GB raw pickle data alone already
exceeds it). Either pass ``hand="L"`` or ``hand="R"`` (roughly half the
footprint end-to-end, verified to fit the login node), or submit via SLURM
with **at least 20-24 GB** requested — a plain ``--mem=16G`` request finishes
FD-I+FD-II but leaves almost no margin (measured 16.7 GB peak against a 16 GB
request). See ``test_a100.sh`` / ``launch_*.sh`` in the repo root for the
``srun``/``sbatch`` pattern already used here; no GPU is needed for data
loading, only ``--mem``.

Caveats carried over from 00_ReadMe.txt
----------------------------------------
  * "The left hand IMU data from X_L.pkl is the raw data before
    hand-mirroring processing." No axis-mirroring convention is documented
    here, so this loader does NOT mirror left-hand data — ``hand="both"``
    simply pools raw left- and right-wrist windows as independent samples.
    If you need left/right anatomically aligned (e.g. mirrored across the
    sagittal plane), check the reference implementation at
    https://github.com/Pituohai/Eating-Speed-Dataset before assuming one.
  * No sensor-orientation / body-frame convention (e.g. FLU) is documented
    for this device placement, so channels are kept in the raw Shimmer3
    sensor frame — no coordinate transform is applied.
  * FD-II labels 0/1 only in practice (no drinking gestures observed
    outside meals in this release); FD-I and MO contain 0/1/2.

Pipeline
--------
1. Load the 4 pickle lists for a subset (extracted folder preferred, zip
   fallback — never requires pre-extracting to disk).
2. Per participant, per selected hand: resample 64 Hz → target_hz with
   ``scipy.signal.resample_poly`` (signal) / nearest-index lookup (labels).
3. Slide fixed-length windows (``compute_T_and_hop`` from windowing.py, same
   convention as every other loader in this directory) and resolve each
   window's label via ``resolve_window_label`` (cfg-driven pure / majority /
   center policy — supports the 3-class label space directly).
4. Concatenate windows across the requested participants/hands into one
   ``[M, 6, T]`` array, wrapped by a small Dataset that applies the standard
   preprocess_window + apply_augment pipeline at __getitem__ time.

Shapes emitted per item
-----------------------
  x : [6, T]  float32  — [ax, ay, az, gx, gy, gz] (channels-first)
  y : []      int64    — 0 (other), 1 (eating), 2 (drinking)

Split strategies
----------------
  FD-I official 7-fold  : participant-level, matches the source paper's protocol
  Random                : pool all windows of one subset, random split
  FD-II / MO as eval     : load a whole subset as a single (unsplit) loader
"""

from __future__ import annotations

import io
import logging
import os
import pickle
import zipfile
from contextlib import contextmanager
from math import gcd
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.windowing import compute_T_and_hop, get_sample_rate_hz, resolve_window_label
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR_DEFAULT = "/projects/b53017/vbc1591/EatingSpeed"

NATIVE_HZ = 64
LABEL_NAMES = {0: "other", 1: "eating", 2: "drinking"}

SUBSETS = ("FD-I", "FD-II", "MO")
_FOLD_DIRNAME = "FD1_7_fold_id_list"
N_FDI_FOLDS = 7

IMU_CHANNELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
GYRO_DEG_TO_RAD = np.pi / 180.0   # raw pickles are deg/s; load_subset() returns rad/s (2026-09-17)
GYRO_BIAS_MAX_RAD_S = 20.0        # a gyro axis whose recording-long median exceeds this is a biased channel (see load_subset)


# ---------------------------------------------------------------------------
# Zip/extracted-folder-agnostic file access
# ---------------------------------------------------------------------------

@contextmanager
def _open_member_stream(root: str, subset: str, rel_path: str):
    """Open one file as a streaming file-like object — an already-extracted
    folder is opened directly; a zip archive is streamed via ``zf.open()``.

    Crucially this never materializes a redundant in-memory ``bytes`` copy of
    the (potentially multi-GB) file before deserializing it — both the plain
    ``open()`` and ``zf.open()`` paths hand ``pickle.load``/``np.load`` a
    lazily-read file object instead. ``rel_path`` is the path within the
    subset, e.g. ``"X_L.pkl"`` or ``"0/train.npy"`` (fold-id archive).
    """
    extracted_path = os.path.join(root, subset, rel_path)
    if os.path.isfile(extracted_path):
        with open(extracted_path, "rb") as f:
            yield f
        return

    zip_path = os.path.join(root, f"{subset}.zip")
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(
            f"[eatingspeed] neither {extracted_path} nor {zip_path} exist"
        )
    member = f"{subset}/{rel_path}"
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            # fall back to a suffix match in case of an unexpected internal layout
            candidates = [n for n in zf.namelist() if n.endswith(rel_path)]
            if not candidates:
                raise FileNotFoundError(f"[eatingspeed] {member} not found in {zip_path}")
            member = candidates[0]
        with zf.open(member) as fp:
            yield fp


def _read_member_bytes(root: str, subset: str, rel_path: str) -> bytes:
    """Read one (small) file's bytes in full — only used for the tiny fold-id
    ``.npy`` files, never for the multi-GB pickle lists (see
    ``_open_member_stream``)."""
    with _open_member_stream(root, subset, rel_path) as fp:
        return fp.read()


def _load_pickle_list(root: str, subset: str, filename: str) -> List[np.ndarray]:
    with _open_member_stream(root, subset, filename) as fp:
        data = pickle.load(fp)
    if not isinstance(data, list):
        raise TypeError(f"[eatingspeed] expected list in {subset}/{filename}, got {type(data)}")
    return data


def load_subset(root: str, subset: str, hand: str = "both") -> Dict[str, List[np.ndarray]]:
    """Load the per-participant lists for one subset — only for the requested hand(s).

    ``X_L``/``X_R``/``Y_L``/``Y_R`` are each ~2.7-2.9 GB for FD-I/FD-II, so
    this only deserializes the pickle(s) actually needed: pass
    ``hand="L"``/``"R"`` to load a single hand (~half the memory) or
    ``"both"`` to load everything (see the module docstring's memory-budget
    note — FD-I/FD-II with ``hand="both"`` need a compute-node allocation,
    not the shared login node).

    Returns a dict with keys ``X_{h}, Y_{h}`` for each requested hand ``h``,
    each ``list[np.ndarray]`` of length N_participants.
    """
    if subset not in SUBSETS:
        raise ValueError(f"[eatingspeed] subset must be one of {SUBSETS}, got {subset!r}")
    if hand not in ("L", "R", "both"):
        raise ValueError(f"[eatingspeed] hand must be 'L', 'R', or 'both', got {hand!r}")

    hands = ["L", "R"] if hand == "both" else [hand]
    out = {
        f"{prefix}_{h}": _load_pickle_list(root, subset, f"{prefix}_{h}.pkl")
        for h in hands
        for prefix in ("X", "Y")
    }
    n = len(out[f"X_{hands[0]}"])
    for name, lst in out.items():
        if len(lst) != n:
            raise ValueError(
                f"[eatingspeed] {subset}: length mismatch — X_{hands[0]} has {n}, {name} has {len(lst)}"
            )
    # The raw Shimmer3 pickles store the gyroscope in deg/s (still-block median |gyro| ~1.3-1.6, overall
    # p99 ~204); every other source in this repo is rad/s. Convert here so every caller -- live loaders and
    # cache builds alike -- sees rad/s. Per-window z-scoring is invariant to this constant, so z-score caches
    # built before 2026-09-17 are unaffected; physical-units caches built before then hold deg/s.
    for h in hands:
        for arr in out[f"X_{h}"]:
            arr[:, 3:6] *= GYRO_DEG_TO_RAD
    # Sensor-bias repair (2026-09-21). Two left-wrist FD-II recordings (participants 2 and 5) carry a constant offset of
    # about +70 rad/s (~4,000 deg/s) on gyro-x for the whole recording -- a stuck / mis-calibrated channel, not motion:
    # the other axes and the right wrist are normal and the signal on top of the offset has the usual +-2 rad/s
    # excursions. A per-window z-score removes such an offset, so every z-scored cache and model is unaffected; a
    # physical-units input would see the channel pinned at the clip value. Subtract the per-recording median of any
    # gyro axis whose |median| exceeds GYRO_BIAS_MAX_RAD_S (no real recording keeps a mean angular rate of 20 rad/s
    # for hours) and say so in the log.
    for h in hands:
        for i, arr in enumerate(out[f"X_{h}"]):
            med = np.nanmedian(arr[:, 3:6], axis=0)
            for ax in np.nonzero(np.abs(med) > GYRO_BIAS_MAX_RAD_S)[0]:
                arr[:, 3 + ax] -= med[ax]
                logger.warning("[eatingspeed] %s hand %s participant %d: gyro axis %d had a constant bias of %.1f rad/s "
                               "-- median subtracted", subset, h, i, ax, med[ax])
    logger.info("[eatingspeed] loaded subset=%s hand=%s: %d participants (gyro deg/s -> rad/s)", subset, hand, n)
    return out


def load_fdI_kfold_ids(root: str, fold_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Official FD-I participant-level 7-fold split (indices into the FD-I lists).

    Matches the validation protocol from
    https://arxiv.org/abs/2401.05376 shipped in ``FD1_7_fold_id_list.zip``.
    """
    if not (0 <= fold_idx < N_FDI_FOLDS):
        raise ValueError(f"[eatingspeed] fold_idx must be in [0, {N_FDI_FOLDS}), got {fold_idx}")
    train_bytes = _read_member_bytes(root, _FOLD_DIRNAME, f"{fold_idx}/train.npy")
    test_bytes  = _read_member_bytes(root, _FOLD_DIRNAME, f"{fold_idx}/test.npy")
    train_ids = np.load(io.BytesIO(train_bytes), allow_pickle=True)
    test_ids  = np.load(io.BytesIO(test_bytes),  allow_pickle=True)
    return train_ids, test_ids


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _resample_signal(X: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    """Anti-aliased polyphase resample of [N, C] signal → [N', C]."""
    if source_hz == target_hz:
        return X.astype(np.float32)
    g = gcd(int(source_hz), int(target_hz))
    up, down = target_hz // g, source_hz // g
    return resample_poly(X, up, down, axis=0).astype(np.float32)


def _resample_labels_nearest(y: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    """Nearest-index resample for categorical point-wise labels [N] → [N']."""
    if source_hz == target_hz:
        return y
    n_new = int(round(len(y) * target_hz / source_hz))
    idx = np.round(np.linspace(0, len(y) - 1, n_new)).astype(np.int64)
    return y[idx]


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _window_stream(
    X: np.ndarray,
    y: np.ndarray,
    T: int,
    hop: int,
    cfg,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide fixed windows over one participant's (resampled) stream.

    Returns
    -------
    Xw : [M, 6, T]  float32
    yw : [M]        int64
    """
    N = X.shape[0]
    if N < T:
        return np.empty((0, X.shape[1], T), dtype=np.float32), np.empty(0, dtype=np.int64)

    X_parts: List[np.ndarray] = []
    y_parts: List[int] = []
    for start in range(0, N - T + 1, hop):
        yw = y[start : start + T]
        label = resolve_window_label(yw, cfg)
        if label is None:
            continue
        X_parts.append(X[start : start + T].T)   # [C, T]
        y_parts.append(label)

    if not X_parts:
        return np.empty((0, X.shape[1], T), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.stack(X_parts, axis=0).astype(np.float32), np.asarray(y_parts, dtype=np.int64)


def collect_windows(
    subset_data: Dict[str, List[np.ndarray]],
    participant_ids,
    hand: str,
    cfg,
    target_hz: Optional[int] = None,
    max_participants: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample + window the requested participants/hand(s) of a loaded subset.

    Parameters
    ----------
    subset_data      : dict from ``load_subset`` (keys X_L, X_R, Y_L, Y_R)
    participant_ids  : iterable of integer indices into the subset's lists
    hand             : "L", "R", or "both" (pools left+right as independent
                       samples — see module docstring re: hand-mirroring)
    cfg              : experiment config — drives window length (compute_T_and_hop),
                       sample rate (get_sample_rate_hz), and label policy
                       (resolve_window_label)
    target_hz        : resample target; defaults to cfg's configured rate (50 Hz)
    max_participants : optional cap, useful for smoke tests

    Returns
    -------
    X : [M, 6, T]  float32
    y : [M]        int64
    """
    if hand not in ("L", "R", "both"):
        raise ValueError(f"[eatingspeed] hand must be 'L', 'R', or 'both', got {hand!r}")

    target_hz = int(target_hz) if target_hz is not None else int(get_sample_rate_hz(cfg))
    T, hop = compute_T_and_hop(cfg)

    participant_ids = list(participant_ids)
    if max_participants is not None:
        participant_ids = participant_ids[:max_participants]

    hands = ["L", "R"] if hand == "both" else [hand]

    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    for pid in tqdm(participant_ids, desc="[eatingspeed] windowing", unit="participant", dynamic_ncols=True):
        for h in hands:
            X_raw = np.asarray(subset_data[f"X_{h}"][pid], dtype=np.float64)   # [N, 6]
            y_raw = np.asarray(subset_data[f"Y_{h}"][pid])                     # [N]
            X_rs = _resample_signal(X_raw, NATIVE_HZ, target_hz)
            y_rs = _resample_labels_nearest(y_raw, NATIVE_HZ, target_hz).astype(np.int64)
            n = min(len(X_rs), len(y_rs))
            Xw, yw = _window_stream(X_rs[:n], y_rs[:n], T, hop, cfg)
            if len(Xw) > 0:
                X_chunks.append(Xw)
                y_chunks.append(yw)

    if not X_chunks:
        raise RuntimeError("[eatingspeed] no windows collected — check participant_ids/hand/cfg")

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    vals, counts = np.unique(y, return_counts=True)
    logger.info(
        "[eatingspeed] collected %d windows | label counts=%s",
        len(X), {int(v): int(c) for v, c in zip(vals, counts)},
    )
    return X, y


def pool_windows(*xy_pairs: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate several (X, y) window sets, e.g. to add MO into an FD-I train split."""
    Xs, ys = zip(*xy_pairs)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EatingSpeedDataset(Dataset):
    """Pre-windowed Eating Speed wrist-IMU dataset.

    Input  X : [M, 6, T]  float32  channels-first (C, T)
    Label  y : [M]        int64    0=other, 1=eating, 2=drinking
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cfg=None,
        multi_view: bool = False,
        split_name: str = "train",
    ) -> None:
        # asarray not astype: astype copies even when already float32 -- see
        # the note in nhanes_wrist_loader.NhanesDataset (fixed 2026-08-01).
        self._X          = np.asarray(X, dtype=np.float32)   # [M, 6, T]
        self._y           = torch.from_numpy(y).long()
        self._cfg         = cfg or {}
        self._multi_view  = multi_view
        # Gates augmentation to train only (bug fixed 2026-07-28); the
        # single-view path used to augment val too. Defaults to "train" so
        # existing callers are unchanged.
        self.split_name   = split_name
        self.stats        = PreprocessStats()

    def __len__(self) -> int:
        return len(self._X)

    def _preprocess(self, idx: int) -> np.ndarray:
        Xw    = self._X[idx].T                     # [T, 6]
        Xproc = preprocess_window(Xw, self._cfg, self.stats)
        if Xproc is None:
            Xproc = (Xw - Xw.mean(axis=0, keepdims=True)) / (Xw.std(axis=0, keepdims=True) + 1e-6)
            Xproc = Xproc.T.astype(np.float32)
        return Xproc                                # [C, T]

    def __getitem__(self, idx: int):
        Xproc = self._preprocess(idx)

        if self._multi_view:
            x1 = apply_augment(Xproc.T, self._cfg).T
            x2 = apply_augment(Xproc.T, self._cfg).T
            return (
                torch.from_numpy(x1).float(),
                torch.from_numpy(x2).float(),
                self._y[idx],
            )

        if self.split_name == "train":  # train-only (fixed 2026-07-28)
            Xproc = apply_augment(Xproc.T, self._cfg).T
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
    cfg=None,
    multi_view: bool = False,
    split_name: str = "train",
) -> DataLoader:
    ds = EatingSpeedDataset(X, y, cfg=cfg, multi_view=multi_view, split_name=split_name)
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

def make_eatingspeed_loaders_fdI_kfold(
    root: str = DATA_DIR_DEFAULT,
    fold_idx: int = 0,
    hand: str = "both",
    cfg=None,
    target_hz: Optional[int] = None,
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    include_mo_in_train: bool = False,
    max_participants: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Official FD-I participant-level 7-fold split (matches the source paper).

    Parameters
    ----------
    root                : EatingSpeed dataset root
    fold_idx            : which of the 7 official folds to use (0-6)
    hand                : "L", "R", or "both"
    include_mo_in_train  : if True, pool the full MO (meal-only, training-only)
                            subset into the training split — matches how the
                            source paper uses MO
    max_participants     : optional cap per split, useful for smoke tests

    Returns
    -------
    dict with keys ``'train_loader'``, ``'val_loader'``, ``'test_loader'``
    (val and test both point at the held-out fold — FD-I has no separate val
    carve-out in the official protocol).
    """
    eval_batch_size = eval_batch_size or batch_size

    fdI = load_subset(root, "FD-I", hand=hand)
    train_ids, test_ids = load_fdI_kfold_ids(root, fold_idx)

    X_train, y_train = collect_windows(
        fdI, train_ids, hand, cfg, target_hz=target_hz, max_participants=max_participants
    )
    X_test, y_test = collect_windows(
        fdI, test_ids, hand, cfg, target_hz=target_hz, max_participants=max_participants
    )

    if include_mo_in_train:
        mo = load_subset(root, "MO", hand=hand)
        n_mo = len(mo[f"X_{'L' if hand != 'R' else 'R'}"])
        X_mo, y_mo = collect_windows(
            mo, range(n_mo), hand, cfg, target_hz=target_hz, max_participants=max_participants
        )
        X_train, y_train = pool_windows((X_train, y_train), (X_mo, y_mo))

    logger.info(
        "[eatingspeed] FD-I fold=%d — train: %d windows | test: %d windows",
        fold_idx, len(X_train), len(X_test),
    )

    return {
        "train_loader": _make_loader(X_train, y_train, batch_size,      True,  num_workers, pin_memory, cfg=cfg),
        "val_loader":   _make_loader(X_test,  y_test,  eval_batch_size, False, num_workers, pin_memory, cfg=cfg, split_name="val"),
        "test_loader":  _make_loader(X_test,  y_test,  eval_batch_size, False, num_workers, pin_memory, cfg=cfg, split_name="test"),
    }


def make_eatingspeed_loaders_random(
    root: str = DATA_DIR_DEFAULT,
    subset: str = "FD-I",
    hand: str = "both",
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    cfg=None,
    target_hz: Optional[int] = None,
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_participants: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Random participant-level split of a single subset (FD-I, FD-II, or MO).

    Splits at the PARTICIPANT level (not window level) so no participant's
    data leaks across train/val/test. Useful for quick capacity checks;
    prefer ``make_eatingspeed_loaders_fdI_kfold`` for the rigorous, published
    FD-I protocol.
    """
    assert abs(sum(splits) - 1.0) < 1e-6, f"splits must sum to 1.0, got {sum(splits)}"

    eval_batch_size = eval_batch_size or batch_size
    data = load_subset(root, subset, hand=hand)
    n = len(data[f"X_{'L' if hand != 'R' else 'R'}"])

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(splits[0] * n))
    n_val   = int(round(splits[1] * n))

    train_ids = perm[:n_train]
    val_ids   = perm[n_train : n_train + n_val]
    test_ids  = perm[n_train + n_val :]

    X_train, y_train = collect_windows(data, train_ids, hand, cfg, target_hz=target_hz, max_participants=max_participants)
    X_val,   y_val   = collect_windows(data, val_ids,   hand, cfg, target_hz=target_hz, max_participants=max_participants)
    X_test,  y_test  = collect_windows(data, test_ids,  hand, cfg, target_hz=target_hz, max_participants=max_participants)

    logger.info(
        "[eatingspeed] %s random split — train: %d | val: %d | test: %d (windows)",
        subset, len(X_train), len(X_val), len(X_test),
    )

    return {
        "train_loader": _make_loader(X_train, y_train, batch_size,      True,  num_workers, pin_memory, cfg=cfg),
        "val_loader":   _make_loader(X_val,   y_val,   eval_batch_size, False, num_workers, pin_memory, cfg=cfg, split_name="val"),
        "test_loader":  _make_loader(X_test,  y_test,  eval_batch_size, False, num_workers, pin_memory, cfg=cfg, split_name="test"),
    }


def make_eatingspeed_eval_loader(
    root: str = DATA_DIR_DEFAULT,
    subset: str = "FD-II",
    hand: str = "both",
    cfg=None,
    target_hz: Optional[int] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_participants: Optional[int] = None,
) -> DataLoader:
    """Load an entire subset as a single unsplit DataLoader.

    Intended for FD-II (paper's hold-out evaluation set) or MO (paper's
    training-only set, if pooled elsewhere and evaluated here for sanity
    checks) — no train/val/test split is performed.
    """
    data = load_subset(root, subset, hand=hand)
    n = len(data[f"X_{'L' if hand != 'R' else 'R'}"])
    X, y = collect_windows(
        data, range(n), hand, cfg, target_hz=target_hz, max_participants=max_participants
    )
    logger.info("[eatingspeed] %s eval loader — %d windows", subset, len(X))
    return _make_loader(X, y, batch_size, False, num_workers, pin_memory, cfg=cfg)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    cfg = yaml.safe_load(open("configs/base.yaml"))

    loaders = make_eatingspeed_loaders_fdI_kfold(
        fold_idx=0, hand="both", cfg=cfg, max_participants=2,
    )
    for name, dl in loaders.items():
        xb, yb = next(iter(dl))
        print(name, xb.shape, yb.shape, "label counts:", torch.bincount(yb))
