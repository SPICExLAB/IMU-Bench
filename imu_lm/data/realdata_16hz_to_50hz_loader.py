"""Realdata 16Hz→50Hz degradation loader.

Wraps the standard parquet WindowDataset but applies a frequency-degradation
augmentation to every session at load time:

  original 50 Hz  →  downsample to 16 Hz  →  linear-interpolate back to 50 Hz

The intent is to simulate lower-quality sensor data and study whether a ViT1D
encoder pretrained on this degraded signal transfers better to low-rate devices.

Usage (config snippet):
  objective:
    type: mae_realdata_16hz_to_50hz
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import resample_poly
from math import gcd

from imu_lm.data.loaders import (
    WindowDataset,
    _make_loader,
    _SessionData,
)
from imu_lm.data.splits import SessionKey, build_session_index, make_splits
from imu_lm.utils.helpers import cfg_get

try:
    import pyarrow.dataset as ds
except Exception as exc:
    raise ImportError("pyarrow is required for dataset loading") from exc

logger = logging.getLogger(__name__)

_SOURCE_HZ = 50
_TARGET_HZ = 16


def _downsample_and_upsample(X: np.ndarray, source_hz: int, low_hz: int, target_hz: int) -> np.ndarray:
    """Downsample X [N, C] from source_hz → low_hz, then resample back to target_hz.

    Uses polyphase resampling (scipy.signal.resample_poly) for accurate
    anti-aliased downsampling and band-limited upsampling.
    """
    # --- down: source_hz → low_hz ---
    g_down = gcd(source_hz, low_hz)
    up_d, down_d = low_hz // g_down, source_hz // g_down
    X_low = resample_poly(X, up_d, down_d, axis=0).astype(np.float32)

    # --- up: low_hz → target_hz ---
    g_up = gcd(low_hz, target_hz)
    up_u, down_u = target_hz // g_up, low_hz // g_up
    X_out = resample_poly(X_low, up_u, down_u, axis=0).astype(np.float32)

    # Clip to original length (polyphase can produce 1–2 extra samples)
    return X_out[: len(X)]


class Realdata16HzTo50HzDataset(WindowDataset):
    """WindowDataset subclass that applies 50→16→50 Hz degradation at session load time."""

    def __init__(self, parquet_path: str, session_index, session_keys: List[SessionKey], cfg: Any, split_name: str = "train"):
        # Call parent __init__ which loads all sessions into self._session_store
        super().__init__(parquet_path, session_index, session_keys, cfg, split_name=split_name)

        # Apply degradation to every loaded session in-place
        for key, sd in self._session_store.items():
            X_deg = _downsample_and_upsample(sd.X, _SOURCE_HZ, _TARGET_HZ, _SOURCE_HZ)
            self._session_store[key] = _SessionData(X=X_deg, y=sd.y, t=sd.t)

        logger.info(
            "Realdata16HzTo50HzDataset split=%s: applied 50→%dHz→50Hz degradation to %d sessions",
            split_name, _TARGET_HZ, len(self._session_store),
        )


def make_realdata_16hz_to_50hz_loaders(cfg: Any, dataset_filter=None) -> Dict:
    """Construct train/val/test DataLoaders with 50→16→50 Hz degraded parquet data.

    Drop-in replacement for ``make_loaders`` with the degradation applied.
    """
    from torch.utils.data import DataLoader

    parquet_path = cfg_get(cfg, ["paths", "dataset_path"])
    batch_size = int(cfg_get(cfg, ["data", "batch_size"], 256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers = int(cfg_get(cfg, ["data", "num_workers"], 0))
    pin_memory = bool(cfg_get(cfg, ["data", "pin_memory"], False))

    # Honor data.train_on_datasets if provided (same as make_loaders)
    scan_filter = dataset_filter
    if scan_filter is None:
        train_on = cfg_get(cfg, ["data", "train_on_datasets"], None)
        if train_on is not None:
            scan_filter = list(train_on)

    logger.info("[pretrain] build_session_index (16→50Hz) path=%s filter=%s", parquet_path, scan_filter)
    session_index = build_session_index(parquet_path, cfg, dataset_filter=scan_filter)
    splits = make_splits(session_index, cfg)

    loaders: Dict[str, DataLoader] = {}

    def add(name: str, keys: List[SessionKey], bs: int, shuf: bool):
        if not keys:
            return
        wds = Realdata16HzTo50HzDataset(parquet_path, session_index, keys, cfg, split_name=name)
        loaders[f"{name}_loader"] = _make_loader(wds, bs, shuf, num_workers, pin_memory)
        logger.info("[pretrain] %s_loader: sessions=%d windows=%d batch_size=%d", name, len(keys), len(wds), bs)

    add("train", splits["train_keys"], batch_size, True)
    add("val",   splits["val_keys"],   eval_batch_size, False)
    add("test",  splits["test_keys"],  eval_batch_size, False)

    return loaders
