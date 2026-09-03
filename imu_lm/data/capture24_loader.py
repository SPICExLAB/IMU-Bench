"""capture24_loader.py
PyTorch DataLoader for the Capture-24 free-living wrist-accelerometer HAR dataset
(Chan, Doherty et al., Oxford Wearables Group).

Data source
-----------
    /projects/b53017/vbc1591/capture24/capture24/
        P001.csv.gz ... P151.csv.gz   — one gzipped CSV per participant
        metadata.csv                  — pid, age band, sex
        annotation-label-dictionary.csv — maps raw free-text annotations to
                                          6 different activity-label taxonomies

151 participants, single wrist-worn Axivity AX3 **triaxial accelerometer only**
(no gyroscope — 3 channels, not 6), native 100 Hz, ~24-28 h continuous
free-living recording per participant (~4,000 person-hours total). Columns:
    time       : ISO datetime string, exactly 0.01 s apart for the WHOLE file
                 (verified: no timestamp gaps in this release — the signal is
                 already a uniformly-sampled 100 Hz grid throughout)
    x, y, z    : float, accelerometer reading in g (wrist frame, no documented
                 axis convention — kept raw, no coordinate transform applied)
    annotation : free-text Compendium-of-Physical-Activities-style label,
                 point-wise (changes whenever activity changes). A sample's
                 worth (~1-2%% of rows, verified on 5 participants) has this
                 as NaN — physically-present accelerometer data that simply
                 wasn't annotated; treated as unlabeled, not a recording gap.

Label taxonomies (see ``annotation-label-dictionary.csv``, joined on the raw
``annotation`` string)
------------------------------------------------------------------------------
    WillettsSpecific2018 (10 classes), WillettsMET2018 (11), DohertySpecific2018 (10),
    Willetts2018 (6), Doherty2018 (5), Walmsley2020 (4: sleep / sedentary / light /
    moderate-vigorous — the coarsest, most commonly used in recent HAR papers;
    default here).

Pipeline
--------
1. Per participant: read ``x, y, z, annotation`` (the ``time`` column is
   skipped entirely — parsing 10M+ timestamps per file is pure overhead once
   we know the grid is uniform; verified across every file sampled).
2. Map each row's raw ``annotation`` string to the selected taxonomy's integer
   class id via ``annotation-label-dictionary.csv``; unmapped/NaN → -1
   (unlabeled).
3. Resample 100 Hz → target_hz (``scipy.signal.resample_poly`` for the signal,
   nearest-index for the now-integer labels) — this is safe as one pass over
   the whole file precisely because there are no real timestamp gaps.
4. Slide windows only within contiguous *labeled* (id != -1) segments — same
   segment-respecting logic as ``OCDetect_dataloader.py`` (a gap boundary
   simply resets window accumulation), then resolve each window's label via
   ``resolve_window_label`` (cfg-driven pure/majority/center policy, same as
   every other windowed loader in this directory).

Multiprocessing note — uses "spawn", not "fork"
------------------------------------------------
``_collect_windows`` parallelizes per-file loading with a ``multiprocessing``
pool (each file is ~10M rows, ~10-15 s to load+window). It deliberately uses
``get_context("spawn")`` rather than ``fork``: this module imports torch,
which initializes an internal OpenMP thread pool at import time (sized to all
visible CPUs — 256 on this host); forking a process with that many live
threads is a well-known deadlock hazard (a lock held by one of those threads
at the moment of ``fork()`` can stay permanently locked in the child). This
was not theoretical — an earlier ``fork``-based version of this loader
deadlocked in exactly this way during testing (workers sat blocked on pool
IPC pipes with zero real files open). It was also hit again on this host's
*login node*, which caps a whole user session to 6 GB RAM: multiple worker
processes each briefly needing ~1 GB+ while pandas parses a 10M-row CSV can
exceed that cap, and Python's ``multiprocessing.Pool`` does not reliably
recover when a worker is OOM-killed mid-task (the main process hangs waiting
for a result that will never arrive). Verified clean on a compute node with
``--mem=32G``: e.g. 4 files loaded in ~13 s. If testing on the login node,
keep ``max_files`` small (1-2 concurrent workers' worth); for anything larger,
submit via SLURM (see ``test_a100.sh`` in the repo root) — no GPU needed.

Shapes emitted per item
-----------------------
  x : [3, T]  float32  — [acc_x, acc_y, acc_z] (channels-first; note: only 3
                          channels, no gyro — see caveat below)
  y : []      int64    — class id into the selected taxonomy's vocabulary
                          (``label_names(taxonomy)`` gives id → name)

Caveat: ``random_rotation`` (imu_lm.data.augmentations.transform) hard-codes
``x[:, 3:6]`` assuming 6-channel acc+gyro input and will raise on this
dataset's 3-channel windows — leave ``random_rotation.enabled=False`` in cfg
for Capture-24. Every other augmentation in that module is channel-count
agnostic and safe to use.

Split strategies
----------------
  LOPO (Leave-One-Participant-Out) : held-out participant → val + test; rest → train
  Random                           : pool all windows, random 80/10/10 split
"""

from __future__ import annotations

import glob
import logging
import multiprocessing as _mp
import os
import re
from functools import lru_cache
from math import gcd
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.windowing import compute_T_and_hop, resolve_window_label
from imu_lm.utils.helpers import cfg_get

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR_DEFAULT = "/projects/b53017/vbc1591/capture24/capture24"

NATIVE_HZ = 100
ACC_COLS = ["x", "y", "z"]

TAXONOMIES = (
    "WillettsSpecific2018", "WillettsMET2018", "DohertySpecific2018",
    "Willetts2018", "Doherty2018", "Walmsley2020",
)
DEFAULT_TAXONOMY = "Walmsley2020"
UNLABELED_ID = -1


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _list_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "P*.csv.gz")))
    if not files:
        raise FileNotFoundError(f"[capture24] no P*.csv.gz files found under {data_dir}")
    return files


def select_capture24_files(
    data_dir: str,
    max_files: Optional[int] = None,
    seed: int = 42,
) -> List[str]:
    """Deterministically select up to *max_files* Capture-24 participant files.

    Capture-24 is ~4,000 person-hours across 151 files; loading everything to
    RAM just to sub-sample a few dozen hours is wasteful. This pre-caps the
    file set BEFORE windowing, matching ``OCDetect_dataloader.select_ocd_files``.
    """
    files = _list_files(data_dir)
    if max_files is not None and 0 < int(max_files) < len(files):
        perm = np.random.default_rng(seed).permutation(len(files))
        kept = [files[i] for i in perm[: int(max_files)]]
        files = sorted(kept)
        _log.info("[capture24] pre-capped to %d/%d files (seed=%d)",
                   len(files), len(_list_files(data_dir)), seed)
    return files


def _participant_id_from_path(path: str) -> str:
    """Parse the participant ID from a Capture-24 filename, e.g. P001.csv.gz → 'P001'."""
    name = os.path.basename(path)
    m = re.match(r"(P\d+)\.csv\.gz$", name)
    if m is None:
        raise ValueError(f"[capture24] cannot parse participant ID from: {name}")
    return m.group(1)


def list_participants(data_dir: str) -> List[str]:
    return [_participant_id_from_path(f) for f in _list_files(data_dir)]


# ---------------------------------------------------------------------------
# Annotation → label-taxonomy mapping
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def load_annotation_dictionary(root: str) -> pd.DataFrame:
    """Load & cache ``annotation-label-dictionary.csv`` (206 rows — tiny)."""
    path = os.path.join(root, "annotation-label-dictionary.csv")
    return pd.read_csv(path)


def label_names(root: str, taxonomy: str) -> List[str]:
    """Sorted class-name vocabulary for one taxonomy — index is the class id."""
    if taxonomy not in TAXONOMIES:
        raise ValueError(f"[capture24] taxonomy must be one of {TAXONOMIES}, got {taxonomy!r}")
    ann_dict = load_annotation_dictionary(root)
    return sorted(ann_dict[f"label:{taxonomy}"].unique())


def build_annotation_to_id_map(root: str, taxonomy: str) -> Dict[str, int]:
    """Raw ``annotation`` free-text string → integer class id for *taxonomy*.

    Unmapped / NaN annotations are simply absent from this dict — look them
    up with ``.get(ann, UNLABELED_ID)`` (or ``pandas.Series.map`` + ``fillna``).
    """
    ann_dict = load_annotation_dictionary(root)
    names = label_names(root, taxonomy)
    name_to_id = {name: i for i, name in enumerate(names)}
    col = f"label:{taxonomy}"
    return {
        row["annotation"]: name_to_id[row[col]]
        for _, row in ann_dict.iterrows()
    }


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _resample_signal(X: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    if source_hz == target_hz:
        return X.astype(np.float32)
    g = gcd(int(source_hz), int(target_hz))
    up, down = target_hz // g, source_hz // g
    return resample_poly(X, up, down, axis=0).astype(np.float32)


def _resample_labels_nearest(y: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    if source_hz == target_hz:
        return y
    n_new = int(round(len(y) * target_hz / source_hz))
    idx = np.round(np.linspace(0, len(y) - 1, n_new)).astype(np.int64)
    return y[idx]


# ---------------------------------------------------------------------------
# Per-file loading + windowing
# ---------------------------------------------------------------------------

def _load_recording(path: str, ann_to_id: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load one participant CSV. The ``time`` column is skipped — every file
    in this release is a uniformly-sampled 100 Hz grid with no real gaps
    (verified), so timestamps carry no extra information for windowing.

    Returns
    -------
    acc      : [N, 3]  float32 — [x, y, z] in g
    label_id : [N]     int32   — class id for the configured taxonomy, or
                                 UNLABELED_ID (-1) where annotation is NaN /
                                 not present in the dictionary
    """
    df = pd.read_csv(
        path,
        usecols=ACC_COLS + ["annotation"],
        dtype={c: "float32" for c in ACC_COLS},
        low_memory=False,
    )
    acc = df[ACC_COLS].to_numpy(dtype=np.float32)
    label_id = df["annotation"].map(ann_to_id).fillna(UNLABELED_ID).to_numpy(dtype=np.int32)
    return acc, label_id


def _process_one_file(
    args: Tuple[str, Dict[str, int], int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray, Optional[str], Optional[str]]:
    """Multiprocessing worker: load, resample, and window one participant file."""
    path, ann_to_id, target_hz, window_samples, hop_samples, cfg_dict = args
    try:
        acc, label_id = _load_recording(path, ann_to_id)
        acc_rs = _resample_signal(acc, NATIVE_HZ, target_hz)
        label_rs = _resample_labels_nearest(label_id, NATIVE_HZ, target_hz)
    except Exception as exc:
        return (
            np.empty((0, 3, window_samples), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            str(exc),
            os.path.basename(path),
        )

    # all_windows (pretraining): keep unlabeled / transition windows too (label -> -1).
    # Capture-24 is ~3,900 h of wear time but only ~2,550 h carry a usable annotation,
    # so the default labelled-segment windowing silently drops ~1/3 of the signal.
    all_windows = bool(cfg_dict.get("_capture24_all_windows", False))
    usable     = np.ones_like(label_rs, dtype=bool) if all_windows else (label_rs != UNLABELED_ID)
    changes    = np.diff(usable.astype(np.int8), prepend=0, append=0)
    seg_starts = np.where(changes ==  1)[0]
    seg_ends   = np.where(changes == -1)[0]

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    for start, end in zip(seg_starts, seg_ends):
        seg_len = int(end - start)
        if seg_len < window_samples:
            continue
        seg_acc = acc_rs[start:end]
        seg_lbl = label_rs[start:end]
        for w_start in range(0, seg_len - window_samples + 1, hop_samples):
            yw = seg_lbl[w_start : w_start + window_samples]
            label = resolve_window_label(yw, cfg_dict)
            if label is None:
                if not all_windows:
                    continue
                label = UNLABELED_ID
            X_parts.append(seg_acc[w_start : w_start + window_samples].T)   # [3, T]
            y_parts.append(label)

    if not X_parts:
        return (
            np.empty((0, 3, window_samples), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            None, None,
        )
    return (
        np.stack(X_parts, axis=0).astype(np.float32),
        np.asarray(y_parts, dtype=np.int64),
        None, None,
    )


def _collect_windows(
    files: List[str],
    root: str,
    taxonomy: str,
    cfg,
    target_hz: Optional[int] = None,
    max_files: Optional[int] = None,
    all_windows: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load + window *files*, respecting unlabeled (id == -1) regions as segment breaks
    (unless ``all_windows`` — then every window is kept and unresolved labels are -1).

    Returns
    -------
    X : [M, 3, T]  float32
    y : [M]        int64
    """
    if max_files is not None:
        files = files[:max_files]

    T, hop = compute_T_and_hop(cfg)
    target_hz = int(target_hz) if target_hz is not None else NATIVE_HZ

    ann_to_id = build_annotation_to_id_map(root, taxonomy)
    cfg_dict = dict(cfg) if cfg is not None else {}
    cfg_dict["_capture24_all_windows"] = bool(all_windows)

    args_list = [(f, ann_to_id, target_hz, T, hop, cfg_dict) for f in files]
    n_workers = min(_mp.cpu_count(), 8, len(files))

    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []

    # Use "spawn", not "fork": torch initializes an internal OpenMP thread pool
    # (sized to all visible CPUs) at import time, and forking a process with
    # that many live threads is a well-known deadlock hazard — a lock held by
    # one of those threads at the moment of fork() can be permanently stuck in
    # the child. Confirmed by hitting exactly this hang while testing (workers
    # sat blocked on pool IPC pipes with zero file descriptors open, i.e. dead
    # before doing any real work). "spawn" re-execs a clean interpreter per
    # worker instead, at the cost of a few seconds' import overhead per pool.
    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for X_c, y_c, err, fname in tqdm(
            pool.imap(_process_one_file, args_list),
            total=len(files),
            desc="[capture24] loading",
            unit="file",
            dynamic_ncols=True,
        ):
            if err is not None:
                _log.warning("[capture24] skipping %s — %s", fname, err)
            if len(X_c) > 0:
                X_chunks.append(X_c)
                y_chunks.append(y_c)

    if not X_chunks:
        raise RuntimeError("[capture24] no windows collected — check data_dir and taxonomy")

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    vals, counts = np.unique(y, return_counts=True)
    names = label_names(root, taxonomy)
    _log.info(
        "[capture24] collected %d windows | label counts=%s",
        len(X), {names[int(v)]: int(c) for v, c in zip(vals, counts)},
    )
    return X, y


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Capture24Dataset(Dataset):
    """Pre-windowed Capture-24 wrist-accelerometer dataset.

    Input  X : [M, 3, T]  float32  channels-first (C, T) — acc only, no gyro
    Label  y : [M]        int64    class id in the selected taxonomy
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cfg: dict | None = None,
        multi_view: bool = False,
        split_name: str = "train",
    ) -> None:
        # asarray not astype: astype copies even when already float32 -- see
        # the note in nhanes_wrist_loader.NhanesDataset (fixed 2026-08-01).
        self._X          = np.asarray(X, dtype=np.float32)   # [M, 3, T]
        self._y          = torch.from_numpy(y).long()
        self._cfg        = cfg or {}
        self._multi_view = multi_view
        # Gates augmentation to train only (bug fixed 2026-07-28) -- see
        # __getitem__. Defaults to "train" so existing positional callers keep
        # augmenting; the loader factories below pass the real split.
        self.split_name  = split_name
        self.stats       = PreprocessStats()

    def __len__(self) -> int:
        return len(self._X)

    def _preprocess(self, idx: int) -> np.ndarray:
        Xw    = self._X[idx].T                     # [T, 3]
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

        # Train-only (bug fixed 2026-07-28): previously unconditional, so val
        # windows were augmented -- inconsistent with nymeria/openwatch/
        # realdata (which gate on split_name) and a source of noisy,
        # run-to-run-varying combined-MAE val loss. The multi_view branch
        # above is deliberately left unconditional: it exists only for
        # contrastive training, where two augmented views ARE the objective.
        if self.split_name == "train":
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
    cfg: dict | None = None,
    multi_view: bool = False,
    split_name: str = "train",
) -> DataLoader:
    ds = Capture24Dataset(X, y, cfg=cfg, multi_view=multi_view, split_name=split_name)
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

def make_capture24_loaders_lopo(
    data_dir: str = DATA_DIR_DEFAULT,
    test_participant: str = None,
    taxonomy: str = DEFAULT_TAXONOMY,
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    target_hz: Optional[int] = None,
    cfg: dict | None = None,
) -> Dict[str, DataLoader]:
    """Leave-One-Participant-Out (LOPO) split.

    The held-out ``test_participant`` (e.g. ``"P001"``) provides both val and
    test loaders. All other participants are pooled for training.
    """
    all_files = _list_files(data_dir)
    all_ids   = {_participant_id_from_path(f) for f in all_files}
    if test_participant not in all_ids:
        raise ValueError(f"[capture24] test_participant must be one of {sorted(all_ids)}, got {test_participant!r}")

    eval_batch_size = eval_batch_size or batch_size

    train_files = [f for f in all_files if _participant_id_from_path(f) != test_participant]
    test_files  = [f for f in all_files if _participant_id_from_path(f) == test_participant]

    _log.info(
        "[capture24] LOPO test_participant=%s | %d train files, %d test files",
        test_participant, len(train_files), len(test_files),
    )

    X_train, y_train = _collect_windows(train_files, data_dir, taxonomy, cfg, target_hz=target_hz)
    X_eval,  y_eval  = _collect_windows(test_files,  data_dir, taxonomy, cfg, target_hz=target_hz)

    return {
        "train_loader": _make_loader(X_train, y_train, batch_size,      True,  num_workers, pin_memory, cfg=cfg),
        "val_loader":   _make_loader(X_eval,  y_eval,  eval_batch_size, False, num_workers, pin_memory, cfg=cfg, split_name="val"),
        "test_loader":  _make_loader(X_eval,  y_eval,  eval_batch_size, False, num_workers, pin_memory, cfg=cfg, split_name="test"),
    }


def make_capture24_loaders_random(
    data_dir: str = DATA_DIR_DEFAULT,
    taxonomy: str = DEFAULT_TAXONOMY,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    target_hz: Optional[int] = None,
    multi_view: bool = False,
    max_files: Optional[int] = None,
    cfg: dict | None = None,
    all_windows: bool = False,
) -> Dict[str, DataLoader]:
    """Random participant-level split pooling all participants.

    Splits at the PARTICIPANT level (not window level) so no participant's
    data leaks across train/val/test. Useful for quick capacity checks;
    prefer ``make_capture24_loaders_lopo`` for subject-independent evaluation.
    """
    assert abs(sum(splits) - 1.0) < 1e-6, f"splits must sum to 1.0, got {sum(splits)}"

    eval_batch_size = eval_batch_size or batch_size
    all_files = select_capture24_files(data_dir, max_files=max_files, seed=seed)

    rng  = np.random.default_rng(seed)
    perm = rng.permutation(len(all_files))
    n_train = int(round(splits[0] * len(all_files)))
    n_val   = int(round(splits[1] * len(all_files)))

    train_files = [all_files[i] for i in perm[:n_train]]
    val_files   = [all_files[i] for i in perm[n_train : n_train + n_val]]
    test_files  = [all_files[i] for i in perm[n_train + n_val :]]

    X_train, y_train = _collect_windows(train_files, data_dir, taxonomy, cfg, target_hz=target_hz, all_windows=all_windows)
    X_val,   y_val   = _collect_windows(val_files,   data_dir, taxonomy, cfg, target_hz=target_hz, all_windows=all_windows)
    X_test,  y_test  = _collect_windows(test_files,  data_dir, taxonomy, cfg, target_hz=target_hz, all_windows=all_windows)

    _log.info(
        "[capture24] random split — train: %d files/%d windows | val: %d files/%d windows | test: %d files/%d windows",
        len(train_files), len(X_train), len(val_files), len(X_val), len(test_files), len(X_test),
    )

    return {
        "train_loader": _make_loader(X_train, y_train, batch_size,      True,  num_workers, pin_memory, cfg=cfg, multi_view=multi_view),
        "val_loader":   _make_loader(X_val,   y_val,   eval_batch_size, False, num_workers, pin_memory, cfg=cfg, split_name="val"),
        "test_loader":  _make_loader(X_test,  y_test,  eval_batch_size, False, num_workers, pin_memory, cfg=cfg, split_name="test"),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO)

    cfg = yaml.safe_load(open("configs/base.yaml"))

    files = select_capture24_files(DATA_DIR_DEFAULT, max_files=3, seed=0)
    print("files:", [os.path.basename(f) for f in files])
    print("Walmsley2020 classes:", label_names(DATA_DIR_DEFAULT, "Walmsley2020"))

    X, y = _collect_windows(files, DATA_DIR_DEFAULT, DEFAULT_TAXONOMY, cfg)
    print("X:", X.shape, "y:", y.shape)
