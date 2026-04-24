"""Combined Nymeria-FLU + parquet-realdata variable-length MAE loader.

Each yielded batch is padded to T_max (default 10.24 s at 50 Hz = 512 frames)
and carries a patch-level padding mask so the MAE encoder/decoder can ignore
tail padding while still operating on a fixed maximum token grid.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

try:
    import pyarrow.dataset as ds
except Exception as exc:  # pragma: no cover
    raise ImportError("pyarrow is required for parquet dataset loading") from exc

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.data.nymeria_wrist_loader_FLU import _load_nymeria_session_flu
from imu_lm.data.splits import SessionKey, build_session_index, make_splits
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)


@dataclass
class _SourceSession:
    X: np.ndarray
    source: str


def _load_parquet_sessions(
    parquet_path: str,
    session_index,
    session_keys: List[SessionKey],
    cfg: Any,
) -> List[_SourceSession]:
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    dataset_col = data_cfg.get("dataset_column", "dataset")
    subject_col = data_cfg.get("subject_column", "subject_id")
    session_col = data_cfg.get("session_column", "session_id")
    time_col = data_cfg.get("time_column", None)
    sensor_cols = data_cfg.get("sensor_columns", []) or []
    drop_na = bool(data_cfg.get("drop_na", False))

    max_gaps_per_session = int(cfg_get(cfg, ["windowing", "max_gaps_per_session"], 1_000_000_000))
    gap_counts: Dict[SessionKey, int] = {}
    n_rows_map: Dict[SessionKey, int] = {}
    for _, row in session_index.iterrows():
        key = SessionKey(row["dataset"], str(row["subject_id"]), str(row["session_id"]))
        gap_counts[key] = int(row.get("gap_count", 0))
        n_rows_map[key] = int(row["n_rows"])

    filtered_keys = [
        key for key in session_keys
        if gap_counts.get(key, 0) <= max_gaps_per_session and n_rows_map.get(key, 0) > 0
    ]

    load_cols = list(sensor_cols)
    if time_col:
        load_cols.append(time_col)
    load_cols.extend([dataset_col, subject_col, session_col])

    pa_ds = ds.dataset(parquet_path, format="parquet")
    sessions: List[_SourceSession] = []
    for key in tqdm(filtered_keys, desc="Loading realdata sessions (var-len)", unit="session", dynamic_ncols=True):
        filt = (
            (ds.field(dataset_col) == key.dataset)
            & (ds.field(subject_col) == key.subject_id)
            & (ds.field(session_col) == key.session_id)
        )
        table = pa_ds.to_table(columns=load_cols, filter=filt)
        df = table.to_pandas()
        if time_col:
            df = df.sort_values(time_col).reset_index(drop=True)
        if drop_na:
            df = df.dropna(subset=sensor_cols)
        X = df[sensor_cols].to_numpy(dtype=np.float32)
        if len(X) == 0:
            continue
        sessions.append(_SourceSession(X=X, source="realdata"))

    return sessions


class CombinedFLUVarLenMAEDataset(IterableDataset):
    """Infinite combined MAE dataset with per-batch random valid length and padding."""

    def __init__(
        self,
        flu_train_files: List[str],
        parquet_sessions: List[_SourceSession],
        batch_size: int,
        T_min: int,
        T_max: int,
        patch_size: int,
        cfg: Any,
        split_name: str = "train",
        seed: int = 42,
        target_hz: int = 50,
        add_gravity: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.T_min = T_min
        self.T_max = T_max
        self.patch_size = patch_size
        self.cfg = cfg
        self.split_name = split_name
        self.seed = seed
        self.target_hz = target_hz
        self.add_gravity = add_gravity
        self.allowed_Ts = list(range(T_min, T_max + 1, patch_size))
        self.num_channels = len((cfg_get(cfg, ["data", "sensor_columns"], []) or []))
        self.max_patches_per_channel = T_max // patch_size

        self._sessions: List[_SourceSession] = []
        for path in tqdm(flu_train_files, desc=f"Loading {split_name} Nymeria FLU sessions", unit="session", dynamic_ncols=True):
            result = _load_nymeria_session_flu(path, target_hz=target_hz, add_gravity=add_gravity)
            if result is None:
                continue
            X, _ = result
            self._sessions.append(_SourceSession(X=X, source="nymeria_flu"))
        self._sessions.extend(parquet_sessions)

        if not self._sessions:
            raise RuntimeError("CombinedFLUVarLenMAEDataset: no valid sessions loaded.")

        logger.info(
            "[combined_flu_var_len] split=%s sessions=%d T=[%d,%d] batch=%d",
            split_name, len(self._sessions), T_min, T_max, batch_size,
        )

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (worker_info.id if worker_info is not None else 0)
        rng = np.random.default_rng(worker_seed)
        stats = PreprocessStats()

        while True:
            T_valid = int(rng.choice(self.allowed_Ts))
            valid_sessions = [session for session in self._sessions if len(session.X) >= T_valid]
            if not valid_sessions:
                continue

            x_list: List[np.ndarray] = []
            pad_masks: List[np.ndarray] = []
            attempts = 0
            while len(x_list) < self.batch_size:
                attempts += 1
                if attempts > self.batch_size * 50:
                    break

                session = valid_sessions[int(rng.integers(len(valid_sessions)))]
                start = int(rng.integers(0, len(session.X) - T_valid + 1))
                Xw = session.X[start : start + T_valid]

                Xproc = preprocess_window(Xw, self.cfg, stats)
                if Xproc is None:
                    Xproc = Xw.T.astype(np.float32)
                Xproc = apply_augment(Xproc.T, self.cfg).T.astype(np.float32)  # [C, T_valid]

                x_pad = np.zeros((self.num_channels, self.T_max), dtype=np.float32)
                x_pad[:, :T_valid] = Xproc

                valid_patches = T_valid // self.patch_size
                time_pad = np.arange(self.max_patches_per_channel) >= valid_patches
                patch_pad = np.repeat(time_pad[None, :], self.num_channels, axis=0).reshape(-1)

                x_list.append(x_pad)
                pad_masks.append(patch_pad)

            if len(x_list) < self.batch_size:
                continue

            x_tensor = torch.from_numpy(np.stack(x_list, axis=0)).float()
            dummy_y = torch.zeros(self.batch_size, 1, dtype=torch.float32)
            padding_mask = torch.from_numpy(np.stack(pad_masks, axis=0)).bool()
            yield x_tensor, dummy_y, padding_mask


def make_combined_flu_var_len_loaders(
    cfg: Any,
    train_pkl_files: List[str],
    val_pkl_files: List[str],
) -> Dict[str, Optional[DataLoader]]:
    vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
    win_cfg = cfg_get(cfg, ["windowing"], {}) or {}
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    trainer_cfg = cfg_get(cfg, ["trainer"], {}) or {}
    obj_cfg = cfg_get(cfg, ["objective", "mae_combined_flu_var_len"], {}) or {}

    patch_size = int(vit_cfg.get("patch_size", 16))
    fps = float(win_cfg.get("fps", 50.0))
    min_secs = float(win_cfg.get("min_window_seconds", 1.28))
    max_secs = float(win_cfg.get("max_window_seconds", 10.24))
    T_min = max(int(round(min_secs * fps / patch_size)) * patch_size, patch_size)
    T_max = int(round(max_secs * fps / patch_size)) * patch_size

    batch_size = int(data_cfg.get("batch_size", trainer_cfg.get("batch_size", 32)))
    num_workers = int(data_cfg.get("num_workers", 4))
    seed = int(cfg_get(cfg, ["trainer", "seed"], 42))
    add_gravity = bool(obj_cfg.get("add_gravity", True))

    parquet_path = cfg_get(cfg, ["paths", "dataset_path"])
    train_on = cfg_get(cfg, ["data", "train_on_datasets"], None)
    session_index = build_session_index(
        parquet_path,
        cfg,
        dataset_filter=list(train_on) if train_on else None,
    )
    parquet_splits = make_splits(session_index, cfg)
    train_parquet_sessions = _load_parquet_sessions(parquet_path, session_index, parquet_splits["train_keys"], cfg)
    val_parquet_sessions = _load_parquet_sessions(parquet_path, session_index, parquet_splits["val_keys"], cfg)

    loaders: Dict[str, Optional[DataLoader]] = {}
    for split, pkl_files, parq_sessions, split_seed in [
        ("train", train_pkl_files, train_parquet_sessions, seed),
        ("val", val_pkl_files, val_parquet_sessions, seed + 1000),
    ]:
        dataset = CombinedFLUVarLenMAEDataset(
            flu_train_files=pkl_files,
            parquet_sessions=parq_sessions,
            batch_size=batch_size,
            T_min=T_min,
            T_max=T_max,
            patch_size=patch_size,
            cfg=cfg,
            split_name=split,
            seed=split_seed,
            add_gravity=add_gravity,
        )
        loaders[f"{split}_loader"] = DataLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers if split == "train" else min(num_workers, 2),
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return loaders