from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:  # required for parquet IO
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover - hard fail
    raise ImportError("pyarrow is required for dataset loading") from exc

from imu_lm.data.splits import SessionKey, build_session_index, make_splits
from imu_lm.data.windowing import compute_T_and_hop, resolve_window_label
from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.spectrogram import stft_encode
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)


def _build_window_specs(
    session_keys: List[SessionKey], n_rows_map: Dict[SessionKey, int], cfg: Any
) -> Tuple[List[SessionKey], np.ndarray, np.ndarray, Dict[str, int]]:
    """Precompute compact window specs using numpy arrays.

    Returns:
        kept_keys: deduplicated list of SessionKeys that have windows
        sess_idx:  int32 array, index into kept_keys for each window
        starts:    int32 array, start row for each window
        counters:  dict with sessions/windows counts
    """
    T, hop = compute_T_and_hop(cfg)
    kept_keys: List[SessionKey] = []

    # First pass: count total windows to preallocate arrays
    total_win = 0
    for key in session_keys:
        N = n_rows_map.get(key, 0)
        if N >= T:
            total_win += (N - T) // hop + 1

    sess_idx = np.empty(total_win, dtype=np.int32) #pre allocate arrays
    starts = np.empty(total_win, dtype=np.int32) #pre allocate arrays

    # Second pass: fill arrays
    offset = 0
    for key in session_keys:
        N = n_rows_map.get(key, 0)
        if N < T:
            continue
        ki = len(kept_keys)
        kept_keys.append(key)
        n_win = (N - T) // hop + 1
        sess_idx[offset : offset + n_win] = ki
        starts[offset : offset + n_win] = np.arange(0, N - T + 1, hop, dtype=np.int32)
        offset += n_win

    counters = {"sessions": len(kept_keys), "windows": total_win}
    return kept_keys, sess_idx, starts, counters


@dataclass
class _SessionData:
    X: np.ndarray
    y: np.ndarray
    t: Optional[np.ndarray]


class WindowDataset(Dataset):
    def __init__(self, parquet_path: str, session_index: pd.DataFrame, session_keys: List[SessionKey], cfg: Any, split_name: str = "train"):
        self.parquet_path = parquet_path
        self.cfg = cfg
        self.split_name = split_name
        self.stats = PreprocessStats()

        # Cache config values read on every __getitem__
        self._dataset_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
        self._subject_col = cfg_get(cfg, ["data", "subject_column"], "subject_id")
        self._session_col = cfg_get(cfg, ["data", "session_column"], "session_id")
        self._label_col = cfg_get(cfg, ["data", "label_column"], "dataset_activity_id")
        self._time_col = cfg_get(cfg, ["data", "time_column"], None)
        self._sensor_cols = cfg_get(cfg, ["data", "sensor_columns"], []) or []
        self._drop_na = bool(cfg_get(cfg, ["data", "drop_na"], False))
        self._T, self._hop = compute_T_and_hop(cfg)
        self._handle_gaps = bool(cfg_get(cfg, ["windowing", "handle_gaps"], False))
        self._gap_method = cfg_get(cfg, ["windowing", "gap_method"], "interpolate")
        self._max_gap_ns = float(cfg_get(cfg, ["windowing", "max_gap_ms"], 200.0)) * 1e6
        self._spec_enabled = bool((cfg_get(cfg, ["spectrogram"], {}) or {}).get("enabled", False))

        # Build load columns list once
        self._load_cols = list(self._sensor_cols) + [self._label_col]
        if self._time_col:
            self._load_cols.append(self._time_col)
        self._load_cols.extend([self._dataset_col, self._subject_col, self._session_col])

        # Single pass over session_index for both gap_counts and n_rows
        max_gaps_per_session = int(cfg_get(cfg, ["windowing", "max_gaps_per_session"], 1_000_000_000))
        gap_counts: Dict[SessionKey, int] = {}
        n_rows_map: Dict[SessionKey, int] = {}
        for _, r in session_index.iterrows():
            k = SessionKey(r["dataset"], str(r["subject_id"]), str(r["session_id"]))
            gap_counts[k] = int(r.get("gap_count", 0))
            n_rows_map[k] = int(r["n_rows"])
        filtered_keys = [k for k in session_keys if gap_counts.get(k, 0) <= max_gaps_per_session]

        self._keys, self._sess_idx, self._starts, counters = _build_window_specs(
            filtered_keys, n_rows_map, cfg
        )

        logger.info(
            "WindowDataset split=%s sessions=%d windows=%d (filtered=%d)",
            split_name,
            counters.get("sessions", 0),
            counters.get("windows", 0),
            len(session_keys) - len(filtered_keys),
        )

        # Preload all sessions into RAM once at construction time.
        # Total raw sensor data is small (~80 MB for 3.3M rows × 6 channels),
        # so this eliminates all parquet I/O during training.
        import time as _time
        pa_ds = ds.dataset(self.parquet_path, format="parquet")
        self._session_store: Dict[SessionKey, _SessionData] = {}
        for i, key in enumerate(self._keys):
            t0 = _time.perf_counter()
            filt = (
                (ds.field(self._dataset_col) == key.dataset)
                & (ds.field(self._subject_col) == key.subject_id)
                & (ds.field(self._session_col) == key.session_id)
            )
            table = pa_ds.to_table(columns=self._load_cols, filter=filt)
            df = table.to_pandas()
            if self._time_col:
                df = df.sort_values(self._time_col).reset_index(drop=True)
            if self._drop_na:
                df = df.dropna(subset=self._sensor_cols)
            X = df[self._sensor_cols].to_numpy(dtype=np.float32)
            y = df[self._label_col].to_numpy()
            t = df[self._time_col].to_numpy() if self._time_col else None
            self._session_store[key] = _SessionData(X=X, y=y, t=t)
            elapsed = _time.perf_counter() - t0
            logger.info(
                "[preload %d/%d] %s/%s/%s  rows=%d  %.1fs",
                i + 1, len(self._keys),
                key.dataset, key.subject_id, key.session_id,
                len(X), elapsed,
            )
        total_mb = sum(s.X.nbytes for s in self._session_store.values()) / 1e6
        logger.info(
            "WindowDataset split=%s preloaded %d sessions into RAM (%.1f MB)",
            split_name, len(self._session_store), total_mb,
        )

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        key = self._keys[self._sess_idx[idx]]
        start = int(self._starts[idx])
        sd = self._session_store[key]
        X, y, t = sd.X, sd.y, sd.t
        T = self._T
        Xw = X[start : start + T]
        yw = y[start : start + T]

        label = resolve_window_label(yw, self.cfg)
        if label is None:
            return None

        # gap gating at fetch time
        if self._handle_gaps and t is not None:
            dt = np.diff(t[start : start + T])
            if np.any(dt > self._max_gap_ns):
                if self._gap_method in {"drop", "split_segment"}:
                    return None

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        Xproc = apply_augment(Xproc.T, self.cfg).T

        if self._spec_enabled:
            out = stft_encode(torch.from_numpy(Xproc).float(), self.cfg)
            x_tensor = out[1] if isinstance(out, tuple) else out
            if x_tensor.dim() != 3:
                return None
        else:
            x_tensor = torch.from_numpy(Xproc).float()

        y_tensor = torch.tensor(label, dtype=torch.long)
        return x_tensor, y_tensor


class SessionGroupedBatchSampler:
    """Batch sampler that groups windows by session for I/O efficiency.

    Instead of random access across all windows (causing constant parquet
    cache misses on the single-entry _SessionCache), this sampler yields all
    windows from one session before moving to the next.  Sessions are shuffled
    each epoch for randomness; windows within each session are also shuffled.
    """

    def __init__(self, sess_idx: np.ndarray, batch_size: int,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # sess_idx is contiguous by session from _build_window_specs
        _, self._group_starts, self._group_counts = np.unique(
            sess_idx, return_index=True, return_counts=True
        )
        self._n_sessions = len(self._group_starts)
        self._total = int(sess_idx.shape[0])

    def set_epoch(self, epoch: int):
        """For DDP compatibility."""
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        if self.shuffle:
            session_order = rng.permutation(self._n_sessions)
        else:
            session_order = np.arange(self._n_sessions)

        batch: list[int] = []
        for si in session_order:
            start = int(self._group_starts[si])
            count = int(self._group_counts[si])
            indices = np.arange(start, start + count)
            if self.shuffle:
                rng.shuffle(indices)
            for idx in indices:
                batch.append(int(idx))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch

        self.epoch += 1

    def __len__(self):
        if self.drop_last:
            return self._total // self.batch_size
        return (self._total + self.batch_size - 1) // self.batch_size


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0)


def _make_loader(dataset: WindowDataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    persistent = num_workers > 0
    if shuffle:
        batch_sampler = SessionGroupedBatchSampler(
            dataset._sess_idx, batch_size, shuffle=True, drop_last=False,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_skip_none,
            persistent_workers=persistent,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_skip_none,
        persistent_workers=persistent,
    )


def make_loaders(cfg: Any, dataset_filter=None) -> Dict[str, DataLoader]:
    """Construct DataLoaders for train/val/probe splits.

    dataset_filter: optional list of dataset names to restrict session scan.
                    When provided, builds probe_* loaders. Otherwise builds train/val/test.
    """

    parquet_path = cfg_get(cfg, ["paths", "dataset_path"])
    batch_size = int(cfg_get(cfg, ["data", "batch_size"], 256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers = int(cfg_get(cfg, ["data", "num_workers"], 0))
    pin_memory = bool(cfg_get(cfg, ["data", "pin_memory"], False))

    is_probe = dataset_filter is not None

    # In pretrain mode, honour data.train_on_datasets if provided.
    scan_filter = dataset_filter
    if not is_probe:
        train_on = cfg_get(cfg, ["data", "train_on_datasets"], None)
        if train_on is not None:
            scan_filter = list(train_on)

    mode = "probe" if is_probe else "pretrain"
    logger.info("[%s] build_session_index path=%s dataset_filter=%s", mode, parquet_path, scan_filter)
    session_index = build_session_index(parquet_path, cfg, dataset_filter=scan_filter)

    # Filter to a single task when probe_task is set (e.g. "skin_care", "MakeCereal")
    probe_task = cfg_get(cfg, ["probe", "probe_task"], None) if is_probe else None
    if probe_task:
        session_col = cfg_get(cfg, ["data", "session_column"], "session_id")
        mask = session_index[session_col].str.lower().str.startswith(probe_task.lower())
        session_index = session_index[mask].reset_index(drop=True)
        logger.info("[probe] probe_task=%s filtered sessions=%d", probe_task, len(session_index))

    dataset_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
    datasets_in_run = sorted(session_index[dataset_col].unique().tolist())
    logger.info("[%s] sessions=%d datasets=%d: %s", mode, len(session_index), len(datasets_in_run), datasets_in_run)
    splits = make_splits(session_index, cfg)

    loaders: Dict[str, DataLoader] = {}
    def add(name: str, keys: List[SessionKey], bs: int, shuf: bool):
        if not keys:
            return
        wds = WindowDataset(parquet_path, session_index, keys, cfg, split_name=name)
        loaders[f"{name}_loader"] = _make_loader(wds, bs, shuf, num_workers, pin_memory)
        logger.info("[%s] %s_loader: sessions=%d windows=%d batch_size=%d", mode, name, len(keys), len(wds), bs)

    if is_probe:
        add("probe_train", splits["probe_train_keys"], batch_size, True)
        add("probe_val", splits["probe_val_keys"], eval_batch_size, False)
        add("probe_test", splits["probe_test_keys"], eval_batch_size, False)
    else:
        add("train", splits["train_keys"], batch_size, True)
        add("val", splits["val_keys"], eval_batch_size, False)
        add("test", splits["test_keys"], eval_batch_size, False)

    return loaders


def make_per_participant_loaders(cfg: Any, participant_id: str = None) -> Dict[str, DataLoader]:
    """Stratified class-balanced probe loaders. If participant_id is None, pools all participants."""
    parquet_path = cfg_get(cfg, ["paths", "dataset_path"])
    batch_size = int(cfg_get(cfg, ["data", "batch_size"], 256))
    eval_bs = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    nw = int(cfg_get(cfg, ["data", "num_workers"], 0))
    pin = bool(cfg_get(cfg, ["data", "pin_memory"], False))
    ratios = cfg_get(cfg, ["splits", "probe_ratios"], [0.7, 0.1, 0.2])
    seed = int(cfg_get(cfg, ["splits", "seed"], 0))
    label_col = cfg_get(cfg, ["data", "label_column"], "dataset_activity_id")
    subj_col = cfg_get(cfg, ["data", "subject_column"], "subject_id")
    probe_ds = cfg_get(cfg, ["splits", "probe_dataset"], None)

    si = build_session_index(parquet_path, cfg, dataset_filter=[probe_ds] if probe_ds else None)
    if participant_id is not None:
        si = si[si[subj_col].astype(str) == str(participant_id)].reset_index(drop=True)
        if si.empty:
            raise RuntimeError(f"No sessions for participant={participant_id}")

    ds_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
    sess_col = cfg_get(cfg, ["data", "session_column"], "session_id")

    # Apply probe_task session filter (same as make_loaders)
    probe_task = cfg_get(cfg, ["probe", "probe_task"], None)
    if probe_task:
        mask = si[sess_col].astype(str).str.lower().str.startswith(probe_task.lower())
        si = si[mask].reset_index(drop=True)
        logger.info("[probe] probe_task=%s filtered sessions=%d", probe_task, len(si))
    keys = [SessionKey(r[ds_col], str(r[subj_col]), str(r[sess_col])) for _, r in si.iterrows()]
    full = WindowDataset(parquet_path, si, keys, cfg, split_name="probe")
    n = len(full)

    pa_ds_obj = ds.dataset(parquet_path, format="parquet")
    T = full._T
    _lcache: Dict[tuple, np.ndarray] = {}
    wlabels = np.empty(n, dtype=np.int64)
    for i in range(n):
        k = full._keys[full._sess_idx[i]]
        ck = (k.dataset, k.subject_id, k.session_id)
        if ck not in _lcache:
            f = (ds.field(full._dataset_col) == k.dataset) & (ds.field(full._subject_col) == k.subject_id) & (ds.field(full._session_col) == k.session_id)
            _lcache[ck] = pa_ds_obj.to_table(columns=[label_col], filter=f)[label_col].to_numpy(zero_copy_only=False)
        wlabels[i] = int(_lcache[ck][int(full._starts[i]) + T // 2])

    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for lbl in np.unique(wlabels):
        ci = np.where(wlabels == lbl)[0]
        rng.shuffle(ci)  # randomize within each class before splitting
        b = (np.cumsum(ratios) * len(ci)).astype(int)
        train_idx.extend(ci[:b[0]].tolist())
        val_idx.extend(ci[b[0]:b[1]].tolist())
        test_idx.extend(ci[b[1]:].tolist())

    # Sort each split by session (once) so SessionGroupedBatchSampler sees contiguous groups
    def _sort_by_session(idxs):
        arr = np.array(idxs)
        sess = np.array([full._sess_idx[i] for i in arr], dtype=np.int32)
        return arr[np.argsort(sess, kind="stable")].tolist(), sess[np.argsort(sess, kind="stable")]

    persistent = nw > 0
    loaders: Dict[str, DataLoader] = {}
    for name, sl, bs, shuf in [("probe_train", train_idx, batch_size, True), ("probe_val", val_idx, eval_bs, False), ("probe_test", test_idx, eval_bs, False)]:
        if sl:
            sl_sorted, sub_sess_idx = _sort_by_session(sl)
            subset = Subset(full, sl_sorted)
            if shuf:
                batch_sampler = SessionGroupedBatchSampler(
                    sub_sess_idx, bs, shuffle=True, drop_last=False,
                )
                loaders[f"{name}_loader"] = DataLoader(
                    subset, batch_sampler=batch_sampler, num_workers=nw,
                    pin_memory=pin, collate_fn=collate_skip_none,
                    persistent_workers=persistent,
                )
            else:
                loaders[f"{name}_loader"] = DataLoader(
                    subset, batch_size=bs, shuffle=False, num_workers=nw,
                    pin_memory=pin, drop_last=False, collate_fn=collate_skip_none,
                    persistent_workers=persistent,
                )
    logger.info("[probe] participant=%s classes=%d train=%d val=%d test=%d", participant_id, len(np.unique(wlabels)), len(train_idx), len(val_idx), len(test_idx))
    return loaders
