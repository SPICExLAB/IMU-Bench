from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # required dependency for parquet IO
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover - hard fail
    raise ImportError("pyarrow is required for dataset loading") from exc


from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionKey:
    dataset: str
    subject_id: str
    session_id: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.dataset, self.subject_id, self.session_id)


def build_session_index(
    parquet_path: str,
    cfg: Any,
    batch_limit: Optional[int] = None,
    dataset_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Build a session-level index with streaming aggregation (no full pandas load).

    ``batch_limit`` can be used by smoke tests to stop after N batches.
    ``dataset_filter`` restricts scan to these dataset names if provided.
    """

    dataset_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
    subject_col = cfg_get(cfg, ["data", "subject_column"], "subject_id")
    session_col = cfg_get(cfg, ["data", "session_column"], "session_id")
    time_col = cfg_get(cfg, ["data", "time_column"], None)
    label_col = cfg_get(cfg, ["data", "label_column"], None)
    handle_gaps = bool(cfg_get(cfg, ["windowing", "handle_gaps"], False))
    if not handle_gaps:
        time_col = None

    max_gap_ms = float(cfg_get(cfg, ["windowing", "max_gap_ms"], 200.0))
    gap_ns = int(max_gap_ms * 1e6)

    cols = [dataset_col, subject_col, session_col]
    if time_col:
        cols.append(time_col)
    if label_col:
        cols.append(label_col)

    dset = ds.dataset(parquet_path, format="parquet")
    filt = None
    if dataset_filter:
        filt = ds.field(dataset_col).isin(dataset_filter)
    scanner = dset.scanner(columns=cols, batch_size=1_000_000, filter=filt)

    n_rows: Dict[str, int] = {}
    t_min: Dict[str, int] = {}
    t_max: Dict[str, int] = {}
    gap_count: Dict[str, int] = {}
    label_mode: Dict[str, Any] = {}

    prev_key0: Optional[str] = None
    prev_t0: Optional[int] = None

    batch_count = 0
    total_rows = 0
    for batch in scanner.to_batches():
        batch_count += 1
        if len(batch) == 0:
            continue
        total_rows += len(batch)
        if batch_count % 5 == 0:
            logger.info("build_session_index: scanned_batches=%d total_rows=%d", batch_count, total_rows)

        d = pc.cast(batch[dataset_col], pa.string())
        s = pc.cast(batch[subject_col], pa.string())
        e = pc.cast(batch[session_col], pa.string())
        d_np = np.array(d)
        s_np = np.array(s)
        e_np = np.array(e)
        key_np = np.char.add(np.char.add(np.char.add(d_np, "|"), np.char.add(s_np, "|")), e_np)
        key = pa.array(key_np, type=pa.string())

        gb = pa.table({"key": key}).group_by("key").aggregate([("key", "count")])
        keys_b = gb["key"].to_pylist()
        cnts_b = gb["key_count"].to_numpy()
        for k, c in zip(keys_b, cnts_b):
            n_rows[k] = n_rows.get(k, 0) + int(c)

        if label_col and label_col in batch.schema.names:
            lbl = batch[label_col].to_numpy(zero_copy_only=False)
            key_arr = np.array(key.to_pylist(), dtype=object)
            for k in np.unique(key_arr):
                mask = key_arr == k
                lbls = lbl[mask]
                vals, cnts = np.unique(lbls, return_counts=True)
                if k not in label_mode:
                    label_mode[k] = {}
                for v, c in zip(vals, cnts):
                    label_mode[k][v] = label_mode[k].get(v, 0) + int(c)

        if time_col:
            t = pc.cast(batch[time_col], pa.int64()).to_numpy(zero_copy_only=False)

            tb = pa.table({"key": key, "t": pa.array(t, type=pa.int64())})
            gb2 = tb.group_by("key").aggregate([("t", "min"), ("t", "max")])
            keys2 = gb2["key"].to_pylist()
            tmins = gb2["t_min"].to_numpy()
            tmaxs = gb2["t_max"].to_numpy()
            for k, mn, mx in zip(keys2, tmins, tmaxs):
                mn = int(mn); mx = int(mx)
                t_min[k] = mn if k not in t_min else min(t_min[k], mn)
                t_max[k] = mx if k not in t_max else max(t_max[k], mx)

            key_np = np.array(key.to_pylist(), dtype=object)
            if len(t) >= 2:
                dt = t[1:] - t[:-1]
                same = (key_np[1:] == key_np[:-1])
                gaps = (dt > gap_ns) & same
                if gaps.any():
                    gap_keys = key_np[1:][gaps]
                    uniq, counts = np.unique(gap_keys, return_counts=True)
                    for k, c in zip(uniq, counts):
                        gap_count[k] = gap_count.get(k, 0) + int(c)

            if len(key_np) == 0 or len(t) == 0:
                continue

            if prev_key0 is not None:
                if key_np[0] == prev_key0 and (t[0] - prev_t0) > gap_ns:
                    gap_count[prev_key0] = gap_count.get(prev_key0, 0) + 1

            prev_key0 = key_np[-1]
            prev_t0 = int(t[-1])

        if batch_limit is not None and batch_count >= batch_limit:
            break

    rows = []
    for k, N in n_rows.items():
        ds_, subj_, sess_ = k.split("|", 2)
        rec = {
            dataset_col: ds_,
            subject_col: subj_,
            session_col: sess_,
            "n_rows": int(N),
            "gap_count": int(gap_count.get(k, 0)),
        }
        if time_col:
            rec["t_min"] = int(t_min.get(k, 0))
            rec["t_max"] = int(t_max.get(k, 0))
        if label_col and k in label_mode:
            mode_label = max(label_mode[k], key=label_mode[k].get)
            rec[label_col] = mode_label
        rows.append(rec)

    return pd.DataFrame.from_records(rows)


def _group_key(series: pd.Series, key: str) -> pd.Series:
    vals = series.fillna("").astype(str)
    return vals if key else series


def _to_session_keys(df: pd.DataFrame, dataset_col: str, subject_col: str, session_col: str) -> List[SessionKey]:
    return [SessionKey(r[dataset_col], r[subject_col], r[session_col]) for _, r in df.iterrows()]


def _split_by_ratio(groups: List[np.ndarray], ratios: List[float]) -> List[List[np.ndarray]]:
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    counts = len(groups)
    boundaries = np.cumsum(ratios) * counts
    idxs = [0] + [int(round(b)) for b in boundaries]
    splits: List[List[np.ndarray]] = []
    for i in range(len(ratios)):
        splits.append(groups[idxs[i]: idxs[i + 1]])
    return splits


def make_splits(session_index: pd.DataFrame, cfg: Any) -> Dict[str, List[SessionKey]]:
    """Create deterministic splits according to cfg.data.splits."""

    splits_cfg = cfg_get(cfg, ["splits"], {}) or {}
    dataset_col = cfg_get(cfg, ["data", "dataset_column"], "dataset")
    subject_col = cfg_get(cfg, ["data", "subject_column"], "subject_id")
    session_col = cfg_get(cfg, ["data", "session_column"], "session_id")

    probe_dataset = splits_cfg.get("probe_dataset")
    group_key = splits_cfg.get("group_key") or session_col
    shuffle = bool(splits_cfg.get("shuffle", False))
    val_ratio = float(splits_cfg.get("val_ratio", 0.0))
    test_ratio = float(splits_cfg.get("test_ratio", 0.0))
    probe_ratios = splits_cfg.get("probe_ratios", [0.7, 0.1, 0.2])
    seed = int(splits_cfg.get("seed", 0))
    probe_stratify = bool(splits_cfg.get("probe_stratify_by_label", False))
    label_col = cfg_get(cfg, ["data", "label_column"], "global_activity_id")

    rng = np.random.RandomState(seed)

    # Predefined split carried in the session id (2026-09-18, user-study watch task): sessions named
    # "<recording>__train" / "<recording>__val" are assigned to that split verbatim -- used when each
    # recording is split in time rather than by subject. Every session must carry one of the suffixes.
    if splits_cfg.get("split_by_session_suffix"):
        sid = session_index[session_col].astype(str)
        is_tr, is_va = sid.str.endswith("__train"), sid.str.endswith("__val")
        if not (is_tr | is_va).all():
            bad = sid[~(is_tr | is_va)].unique()[:5].tolist()
            raise ValueError(f"split_by_session_suffix: sessions without __train/__val suffix, e.g. {bad}")
        empty = pd.DataFrame(columns=session_index.columns)
        keys = lambda d: _to_session_keys(d, dataset_col, subject_col, session_col)
        logger.info("split_by_session_suffix: train sessions=%d val sessions=%d", int(is_tr.sum()), int(is_va.sum()))
        return {"train_keys": keys(session_index[is_tr]), "val_keys": keys(session_index[is_va]),
                "test_keys": keys(empty), "probe_train_keys": keys(empty), "probe_val_keys": keys(empty),
                "probe_test_keys": keys(empty)}

    def group_sessions(df: pd.DataFrame) -> List[np.ndarray]:
        groups = []
        by = _group_key(df[group_key] if group_key in df else df[session_col], group_key)
        for _, g in df.groupby(by):
            groups.append(g)
        if shuffle:
            rng.shuffle(groups)
        return groups

    # Non-probe splits (train/val/test)
    non_probe_df = session_index[session_index[dataset_col] != probe_dataset]
    non_probe_groups = group_sessions(non_probe_df)
    val_count = int(round(len(non_probe_groups) * val_ratio))
    test_count = int(round(len(non_probe_groups) * test_ratio))
    test_groups = non_probe_groups[:test_count]
    val_groups = non_probe_groups[test_count:test_count + val_count]
    train_groups = non_probe_groups[test_count + val_count:]

    train_df = pd.concat(train_groups, ignore_index=True) if train_groups else pd.DataFrame(columns=session_index.columns)
    val_df = pd.concat(val_groups, ignore_index=True) if val_groups else pd.DataFrame(columns=session_index.columns)
    test_df = pd.concat(test_groups, ignore_index=True) if test_groups else pd.DataFrame(columns=session_index.columns)

    # Probe splits
    probe_df = session_index[session_index[dataset_col] == probe_dataset]

    if probe_stratify:
        if label_col not in probe_df:
            raise ValueError(f"label_column {label_col} missing; cannot stratify probe splits")

        def group_label_hist(df: pd.DataFrame) -> List[Tuple[pd.DataFrame, Dict[int, int]]]:
            groups = []
            by = _group_key(df[group_key] if group_key in df else df[session_col], group_key)
            for _, g in df.groupby(by):
                hist = g[label_col].value_counts().to_dict()
                groups.append((g, hist))
            if shuffle:
                rng.shuffle(groups)
            return groups

        groups = group_label_hist(probe_df)

        # Greedy assignment to cover classes across splits
        target_counts = [int(round(len(groups) * r)) for r in probe_ratios]
        # ensure at least 1 group if ratio > 0 and available groups
        for i, r in enumerate(probe_ratios):
            if r > 0 and target_counts[i] == 0 and len(groups) > 0:
                target_counts[i] = 1
        # normalize if sum drifts
        while sum(target_counts) > len(groups):
            target_counts[target_counts.index(max(target_counts))] -= 1

        class_sets = [set(), set(), set()]
        split_groups: List[List[pd.DataFrame]] = [[], [], []]

        # Sort groups by number of unique labels (desc) to place rich groups first
        groups_sorted = sorted(groups, key=lambda x: len(x[1]), reverse=True)

        for g_df, hist in groups_sorted:
            labels = set(hist.keys())
            # pick split that improves coverage and is under target
            best_split = None
            best_gain = -1
            for i in range(3):
                if len(split_groups[i]) >= target_counts[i]:
                    continue
                gain = len(labels - class_sets[i])
                if gain > best_gain:
                    best_gain = gain
                    best_split = i
            if best_split is None:
                # all targets filled; put into smallest split by size
                sizes = [len(sg) for sg in split_groups]
                best_split = int(np.argmin(sizes))
            split_groups[best_split].append(g_df)
            class_sets[best_split].update(labels)

        pr_train_df = pd.concat(split_groups[0], ignore_index=True) if split_groups[0] else pd.DataFrame(columns=session_index.columns)
        pr_val_df = pd.concat(split_groups[1], ignore_index=True) if split_groups[1] else pd.DataFrame(columns=session_index.columns)
        pr_test_df = pd.concat(split_groups[2], ignore_index=True) if split_groups[2] else pd.DataFrame(columns=session_index.columns)

        logger.info(
            "probe group split (stratified): total_groups=%d train=%d val=%d test=%d ratios=%s",
            len(groups_sorted),
            len(split_groups[0]),
            len(split_groups[1]),
            len(split_groups[2]),
            probe_ratios,
        )

        # Warn if any class missing in a split
        all_labels = set(probe_df[label_col].unique().tolist())
        for name, cls_set in zip(["probe_train", "probe_val", "probe_test"], class_sets):
            missing = all_labels - cls_set
            if missing:
                logger.warning("probe stratify: split=%s missing labels=%s", name, sorted(list(missing)))
    else:
        probe_groups = group_sessions(probe_df)
        pr_train, pr_val, pr_test = _split_by_ratio(probe_groups, probe_ratios)

        pr_train_df = pd.concat(pr_train, ignore_index=True) if pr_train else pd.DataFrame(columns=session_index.columns)
        pr_val_df = pd.concat(pr_val, ignore_index=True) if pr_val else pd.DataFrame(columns=session_index.columns)
        pr_test_df = pd.concat(pr_test, ignore_index=True) if pr_test else pd.DataFrame(columns=session_index.columns)

        logger.info(
            "probe group split: total_groups=%d train=%d val=%d test=%d ratios=%s",
            len(probe_groups),
            len(pr_train),
            len(pr_val),
            len(pr_test),
            probe_ratios,
        )

    result = {
        "train_keys": _to_session_keys(train_df, dataset_col, subject_col, session_col),
        "val_keys": _to_session_keys(val_df, dataset_col, subject_col, session_col),
        "test_keys": _to_session_keys(test_df, dataset_col, subject_col, session_col),
        "probe_train_keys": _to_session_keys(pr_train_df, dataset_col, subject_col, session_col),
        "probe_val_keys": _to_session_keys(pr_val_df, dataset_col, subject_col, session_col),
        "probe_test_keys": _to_session_keys(pr_test_df, dataset_col, subject_col, session_col),
    }

    def log_split(name: str, df: pd.DataFrame):
        if df.empty:
            logger.info("split=%s empty", name)
            return
        rows = df["n_rows"].sum() if "n_rows" in df else len(df)
        logger.info("split=%s sessions=%d rows=%s", name, len(df), rows)

    def log_labels(name: str, df: pd.DataFrame, topk: int = 10):
        if df.empty or label_col not in df:
            logger.info("split=%s label stats unavailable", name)
            return
        vc = df[label_col].value_counts()
        logger.info(
            "split=%s labels: n_classes=%d top%d=%s",
            name,
            vc.size,
            topk,
            dict(vc.head(topk)),
        )

    log_split("train", train_df)
    log_split("val", val_df)
    log_split("test", test_df)
    log_split("probe_train", pr_train_df)
    log_split("probe_val", pr_val_df)
    log_split("probe_test", pr_test_df)

    log_labels("probe_train", pr_train_df)
    log_labels("probe_val", pr_val_df)
    log_labels("probe_test", pr_test_df)

    return result
