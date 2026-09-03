"""NHANES PAX (Physical Activity Monitor) loader.

NHANES accelerometry cycles G (2011-2012) and H (2013-2014): wrist-worn
ActiGraph GT3X+ triaxial accelerometer, ~14,693 participants combined
(6,917 cycle G + 7,776 cycle H), ~2.3TB raw, downloaded to
/projects/b53017/vbc1591/nhanes_pax/{pax_g,pax_h}/{SEQN}.tar.bz2.

Each participant archive contains one `*.sensor.csv` file per ~hour of wear
(columns: HEADER_TIMESTAMP,X,Y,Z, ~80Hz, pre-calibrated to g units -- the
first file per participant is typically a short leading partial hour) plus
one `{SEQN}_Logs.csv` (per-day data-quality flags; usually empty/no issues
in practice). Files are read directly out of the .tar.bz2 (never bulk-
extracted to disk -- 2.3TB is too large for that to be sane) and
concatenated in chronological order (parsed from each member's embedded
timestamp, not tar member order, since some archives are not stored sorted).

Acc-only (3ch, no gyro) -- same modality as Capture24, reuses the same
AccOnlyPadTo6ChDataset convention in combined_mae_multidata_loader.py.

No activity labels exist in this raw release (NHANES PAX doesn't ship
per-timestamp activity annotations), so windows are cut by sliding directly
over each participant's continuous concatenated signal, with a light
near-zero-variance filter as a crude non-wear/dead-sensor guard. This is an
accepted simplification for a first integration -- proper GGIR-style
non-wear detection (long stationary runs, temperature/light channels where
available) is a real gap, not attempted here.
"""
from __future__ import annotations

import io
import logging
import multiprocessing as _mp
import os
import re
import tarfile
from typing import Dict, List, Optional, Tuple

# Must happen BEFORE numpy/pandas import: without this, each worker process
# in _collect_windows's Pool independently lets its BLAS backend (OpenBLAS/MKL)
# size its internal thread pool to the full visible core count (52 on this
# node) -- observed in practice as one worker process alone spawning 150+
# OS threads (confirmed via `pstree -p`), which starves the shared login
# node. One arithmetic-heavy call (e.g. np.interp in _resample_signal) is
# enough to trigger it. This has no effect on torch's own threading (set
# separately, if at all) since torch isn't used in these worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kw):
        return x

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.helpers import cfg_get

_log = logging.getLogger(__name__)

NATIVE_HZ = 80
DEFAULT_CYCLES = ("pax_g", "pax_h")
_FNAME_TS_RE = re.compile(r"\.(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-\d{3}-P\d+\.sensor\.csv$")


def _list_files(data_dir: str, cycles=DEFAULT_CYCLES) -> List[str]:
    """Sorted list of every participant .tar.bz2 archive across the given cycles."""
    files: List[str] = []
    for cyc in cycles:
        cyc_dir = os.path.join(data_dir, cyc)
        if not os.path.isdir(cyc_dir):
            continue
        files.extend(
            os.path.join(cyc_dir, f) for f in os.listdir(cyc_dir) if f.endswith(".tar.bz2")
        )
    return sorted(files)


def select_nhanes_files(
    root: str, max_files: Optional[int] = None, seed: int = 42, cycles=DEFAULT_CYCLES,
) -> List[str]:
    """Deterministically select up to *max_files* participant archives.

    Same shuffle-then-cap-then-resort convention as
    OCDetect_dataloader.select_ocd_files / capture24_loader.select_capture24_files:
    full sorted list shuffled with a fixed seed (spreads the kept set across
    both cycles rather than biasing toward pax_g), first max_files kept, then
    re-sorted so downstream collection order is stable and reproducible.
    """
    files = _list_files(root, cycles)
    if max_files is not None and 0 < int(max_files) < len(files):
        perm = np.random.default_rng(seed).permutation(len(files))
        kept = [files[i] for i in perm[: int(max_files)]]
        files = sorted(kept)
    return files


def _participant_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.splitext(os.path.basename(path))[0])[0]  # "62161.tar.bz2" -> "62161"


def list_participants(data_dir: str, cycles=DEFAULT_CYCLES) -> List[str]:
    return [_participant_id_from_path(f) for f in _list_files(data_dir, cycles)]


def _load_participant(path: str) -> Optional[np.ndarray]:
    """Read one participant .tar.bz2 archive, concatenate every *.sensor.csv
    member in chronological order, return [N, 3] float32 acc (g units).
    None if the archive has no readable sensor data."""
    import pandas as pd

    try:
        with tarfile.open(path, mode="r:bz2") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".sensor.csv")]
            if not members:
                return None
            members.sort(key=lambda m: (_FNAME_TS_RE.search(m.name) or [None, m.name]).group(1)
                          if _FNAME_TS_RE.search(m.name) else m.name)

            chunks = []
            for m in members:
                f = tf.extractfile(m)
                if f is None:
                    continue
                df = pd.read_csv(io.BytesIO(f.read()), usecols=["X", "Y", "Z"], dtype="float32")
                chunks.append(df.to_numpy(dtype=np.float32))
    except Exception as exc:  # pragma: no cover -- corrupt archive, unreadable member, etc.
        _log.warning("[nhanes] failed to read %s: %s", path, exc)
        return None

    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


def _resample_signal(X: np.ndarray, source_hz: int, target_hz: int) -> np.ndarray:
    if source_hz == target_hz:
        return X
    n_out = int(round(X.shape[0] * target_hz / source_hz))
    t_src = np.linspace(0.0, 1.0, X.shape[0], endpoint=False)
    t_dst = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.stack([np.interp(t_dst, t_src, X[:, c]) for c in range(X.shape[1])], axis=1).astype(np.float32)


def _process_one_file(
    args: Tuple[str, int, int, int, float],
) -> Tuple[np.ndarray, Optional[str], Optional[str]]:
    """Multiprocessing worker: load, resample, and window one participant archive."""
    path, target_hz, window_samples, hop_samples, min_wear_var = args
    try:
        X = _load_participant(path)
        if X is None or len(X) < window_samples:
            return np.empty((0, 3, window_samples), dtype=np.float32), None, None
        X_rs = _resample_signal(X, NATIVE_HZ, target_hz)
    except Exception as exc:
        return np.empty((0, 3, window_samples), dtype=np.float32), str(exc), os.path.basename(path)

    n = len(X_rs)
    if n < window_samples:
        return np.empty((0, 3, window_samples), dtype=np.float32), None, None

    n_wins = (n - window_samples) // hop_samples + 1
    idxs = np.arange(n_wins) * hop_samples
    idx_mat = idxs[:, None] + np.arange(window_samples)
    wins = X_rs[idx_mat]  # [n_wins, window_samples, 3]

    # Crude non-wear/dead-sensor guard: drop windows whose accel variance is
    # ~0 (flat-lined signal -- off-body or a stuck sensor). NOT a real
    # GGIR-style non-wear algorithm -- a documented simplification.
    #
    # Bug fixed 2026-07-28: this used to be
    #     wins.reshape(len(wins), -1).var(axis=1)
    # which pools variance over time AND the 3 channels together, so the
    # between-axis mean spread dominates and swamps the threshold. A perfectly
    # flat-lined stationary window (1 g on one axis, 0 on the other two --
    # exactly the off-body/stuck-sensor case this guard exists to catch) has a
    # pooled variance of ~0.222 g^2, i.e. >2000x above min_wear_var=1e-4, so it
    # was always kept. The guard only ever fired when all three axes were
    # constant AND equal to each other. Per-channel variance (max over
    # channels, so a window survives if ANY axis actually moves) is what the
    # docstring always described.
    win_var = wins.var(axis=1).max(axis=1)   # [n_wins] -- per-channel var, then max over channels
    wins = wins[win_var > min_wear_var]

    if len(wins) == 0:
        return np.empty((0, 3, window_samples), dtype=np.float32), None, None
    return wins.transpose(0, 2, 1).astype(np.float32), None, None  # [M, 3, T]


def _collect_windows(
    files: List[str],
    target_hz: int,
    window_samples: int,
    hop_samples: int,
    min_wear_var: float = 1e-4,
    max_windows: Optional[int] = None,
    per_file_max: Optional[int] = None,
) -> np.ndarray:
    """`per_file_max` (optional): keep at most this many windows per participant,
    taken at an even stride across the whole recording (~8 days), so a
    `max_windows` budget is spread over ALL participants instead of being filled
    by the first ~25-40 in permutation order (which is what a plain sequential
    fill does: 4,000 h == ~30 people out of 14,693 -- no population diversity).

    `max_windows` (optional): stop accumulating once this many windows have
    been collected, terminating the pool early instead of processing every
    remaining file. NHANES participants can each yield 100k+ windows (up to
    ~8-9 days of continuous 50%-overlap wear), so collecting the FULL
    unbounded set before applying any downstream max_hours-style cap can
    exhaust memory long before the cap is ever applied -- confirmed directly
    (OOM-killed at 101/132 participants under a 64GB budget during this
    loader's first real smoke test). This makes the cap effective DURING
    collection, not just after."""
    if not files:
        return np.empty((0, 3, window_samples), dtype=np.float32)

    args_list = [(f, target_hz, window_samples, hop_samples, min_wear_var) for f in files]
    n_workers = min(_mp.cpu_count(), int(os.environ.get("NHANES_WORKERS", "8")), len(files))   # 8 workers = 11.6 s/participant -> 42 h for all 13k

    # Preallocate the destination up front and fill it as chunks arrive, rather
    # than accumulating every chunk and merging at the end (rewritten
    # 2026-07-31 after a 5.6 h build died at the merge step).
    #
    # History, because both earlier versions looked correct and both failed at
    # the very END, after all the irreversible decompression work:
    #   v1  np.concatenate(X_chunks) -- holds the whole chunk list AND the new
    #       output alive at once, so peak = 2x final.
    #   v2  np.empty(total) then copy-and-free chunk-by-chunk -- still needs the
    #       full allocation WHILE the chunks are all still resident, i.e. peak
    #       is still ~2x at the moment of allocation. Observed exactly that:
    #       "Unable to allocate 772. GiB" with 839 GB of chunks already held,
    #       against a 1200 GB cgroup limit.
    #   v3  (this) allocate ONCE up front at the max_windows budget, then write
    #       each chunk straight in and drop it. Peak = (bytes actually filled)
    #       + one chunk, because np.empty is lazily committed on Linux -- so
    #       only the pages we really touch ever count against RSS.
    #
    # Requires max_windows to size the buffer. Without it we cannot know the
    # final length in advance, so fall back to the v1 accumulate-then-merge
    # path -- fine for the small callers that omit the cap, and the docstring
    # warns against omitting it at NHANES scale.
    X_out: Optional[np.ndarray] = None
    if max_windows is not None:
        X_out = np.empty((int(max_windows), 3, window_samples), dtype=np.float32)
        _log.info("[nhanes] preallocated %.1f GB output buffer for up to %d windows "
                  "(lazily committed; only the filled prefix counts against RSS)",
                  X_out.nbytes / 1e9, int(max_windows))
    X_chunks: List[np.ndarray] = []
    total = 0
    # "spawn", not "fork": same rationale as capture24_loader.py -- forking a
    # process with torch's internal OpenMP thread pool already initialized is
    # a known deadlock hazard (confirmed by capture24's own comment; not
    # re-tested here, just inherited defensively).
    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for X_c, err, fname in tqdm(
            pool.imap(_process_one_file, args_list),
            total=len(files), desc="[nhanes] loading", unit="participant", dynamic_ncols=True,
        ):
            if err is not None:
                _log.warning("[nhanes] skipping %s -- %s", fname, err)
            if per_file_max is not None and len(X_c) > int(per_file_max):
                sel = np.linspace(0, len(X_c) - 1, int(per_file_max)).round().astype(np.int64)
                X_c = X_c[sel]
            if len(X_c) > 0:
                if X_out is not None:
                    take = min(len(X_c), int(max_windows) - total)
                    if take > 0:
                        X_out[total:total + take] = X_c[:take]
                    total += take
                else:
                    X_chunks.append(X_c)
                    total += len(X_c)
            del X_c          # drop this participant's windows immediately
            if max_windows is not None and total >= max_windows:
                _log.info("[nhanes] hit max_windows=%d budget, stopping collection early "
                          "(processed a subset of the %d requested files)", max_windows, len(files))
                pool.terminate()
                break

    if X_out is not None:
        if total == 0:
            raise RuntimeError("[nhanes] no windows collected -- check data_dir/cycles/max_files")
        X = X_out[:total]          # view, not a copy: no extra memory
        _log.info("[nhanes] collected %d windows from up to %d files (%.1f GB filled of "
                  "%.1f GB buffer)", len(X), len(files), X.nbytes / 1e9, X_out.nbytes / 1e9)
        return X

    if not X_chunks:
        raise RuntimeError("[nhanes] no windows collected -- check data_dir/cycles/max_files")
    X = np.concatenate(X_chunks, axis=0)
    _log.info("[nhanes] collected %d windows from up to %d files (%.1f GB, unbounded path)",
              len(X), len(files), X.nbytes / 1e9)
    return X


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NhanesDataset(Dataset):
    """Pre-windowed NHANES PAX wrist accelerometer dataset (acc-only, 3ch)."""

    def __init__(self, X: np.ndarray, cfg: dict | None = None,
                 split_name: str = "train") -> None:
        # split_name gates augmentation to train only -- the same fix applied to
        # capture24/eatingspeed/OCDetect/CachedWindowDataset on 2026-07-28; this
        # loader was missed in that sweep.
        self.split_name = split_name
        # np.asarray, NOT X.astype(np.float32) (fixed 2026-08-01). astype
        # defaults to copy=True and so duplicates the array even when it is
        # ALREADY float32 -- which every caller here passes. Invisible at small
        # scale, fatal at NHANES scale: it killed a 4.5 h cache build with
        # "Unable to allocate 745. GiB" while the identical 745 GiB source
        # array was still live (745 + 745 vs a 900 GB cgroup limit). asarray
        # is a no-op when the dtype already matches, and still converts when it
        # does not.
        self._X = np.asarray(X, dtype=np.float32)  # [M, 3, T]
        self._cfg = cfg or {}
        self.stats = PreprocessStats()

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        Xw = self._X[idx].T  # [T, 3]
        Xproc = preprocess_window(Xw, self._cfg, self.stats)
        if Xproc is None:
            Xproc = (Xw - Xw.mean(axis=0, keepdims=True)) / (Xw.std(axis=0, keepdims=True) + 1e-6)
            Xproc = Xproc.T.astype(np.float32)
        if self.split_name == "train":
            Xproc = apply_augment(Xproc.T, self._cfg).T
        return torch.from_numpy(Xproc).float(), torch.tensor(0, dtype=torch.long)


def _make_loader(X: np.ndarray, batch_size: int, shuffle: bool, num_workers: int,
                  pin_memory: bool, cfg: dict | None = None,
                  split_name: str = "train") -> DataLoader:
    ds = NhanesDataset(X, cfg=cfg, split_name=split_name)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False, persistent_workers=(num_workers > 0),
    )


def make_nhanes_loaders_random(
    root: str,
    cycles=DEFAULT_CYCLES,
    splits: Tuple[float, ...] = (0.9, 0.1),
    batch_size: int = 256,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    target_hz: Optional[int] = None,
    max_files: Optional[int] = None,
    max_train_windows: Optional[int] = None,
    max_val_windows: Optional[int] = None,
    cfg: dict | None = None,
    spread_cap: bool = False,
) -> Dict[str, DataLoader]:
    """Random PARTICIPANT-level split (no participant's windows leak across
    train/val/test) -- same convention as capture24_loader.make_capture24_loaders_random.
    """
    assert abs(sum(splits) - 1.0) < 1e-6, f"splits must sum to 1.0, got {sum(splits)}"
    eval_batch_size = eval_batch_size or batch_size
    all_files = select_nhanes_files(root, max_files=max_files, seed=seed, cycles=cycles)
    if not all_files:
        raise RuntimeError(f"[nhanes] no participant archives found under {root} (cycles={cycles})")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_files))
    n_train = int(round(splits[0] * len(all_files)))
    n_val = len(all_files) - n_train  # 2-way split by default (train/val only) -- any
                                       # remainder from a 3-element splits still goes to val
    train_files = [all_files[i] for i in perm[:n_train]]
    val_files = [all_files[i] for i in perm[n_train:]]

    T, hop = compute_T_and_hop(cfg)
    target_hz_eff = int(target_hz) if target_hz is not None else 50

    # max_train_windows/max_val_windows cap collection DURING the loop (see
    # _collect_windows' docstring). Unlike every other capped source in this
    # pool (e.g. ocdetect, whose val is left uncapped since its overall
    # volume is tiny), NHANES's per-participant window density is so much
    # higher than any other source (single participants can contribute
    # 100k+ windows) that leaving val uncapped let NHANES alone make up 92.8%
    # of the ENTIRE 12-source val set in a real build -- val_loss would have
    # been, in effect, a NHANES-only metric, silently defeating the point of
    # a multidata val set for early stopping. Cap both splits.
    def _per_file(max_w, n_files):
        # spread the budget over every participant (10% slack: some participants
        # yield fewer windows than their share after the non-wear filter)
        if not spread_cap or max_w is None or n_files == 0:
            return None
        k = int(np.ceil(1.1 * float(max_w) / n_files))
        _log.info("[nhanes] spread_cap: %d windows budget over %d participants -> <= %d windows "
                  "(%.2f h) per participant, evenly strided", int(max_w), n_files, k, k * hop / 50 / 3600)
        return k
    X_train = _collect_windows(train_files, target_hz_eff, T, hop, max_windows=max_train_windows,
                               per_file_max=_per_file(max_train_windows, len(train_files)))
    X_val = _collect_windows(val_files, target_hz_eff, T, hop, max_windows=max_val_windows,
                             per_file_max=_per_file(max_val_windows, len(val_files)))

    _log.info(
        "[nhanes] random split -- train: %d files/%d windows | val: %d files/%d windows",
        len(train_files), len(X_train), len(val_files), len(X_val),
    )

    return {
        "train_loader": _make_loader(X_train, batch_size, True, num_workers, pin_memory, cfg=cfg),
        "val_loader": _make_loader(X_val, eval_batch_size, False, num_workers, pin_memory, cfg=cfg, split_name="val"),
    }
