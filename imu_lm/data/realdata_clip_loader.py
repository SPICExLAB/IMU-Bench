"""realdata_clip_loader.py
CLIP-style contrastive pretraining dataloader for the realdata parquet dataset.

Returns (x [6, T] float32, text str) pairs where text is constructed as:
    sentence_template.format(dataset_activity_label)
e.g. "The person is walking" when the activity label is "walking".

Design
------
  _build_label_map         — reads two label columns from the parquet once,
                             returns {activity_id: sentence_str}

  RealdataCLIPDataset      — WindowDataset subclass; __getitem__ returns
                             (x_tensor, text_str) instead of (x_tensor, label_int)

  _collate_clip            — collate_fn: stacks x tensors, keeps texts as list[str]

  make_realdata_clip_loaders — factory that returns train/val/test DataLoaders;
                               the label map is built once and shared across splits
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import pyarrow.dataset as ds_pa
except ImportError as exc:
    raise ImportError("pyarrow is required for dataset loading") from exc

from imu_lm.data.loaders import (
    WindowDataset,
)
from imu_lm.data.splits import SessionKey, build_session_index, make_splits
from imu_lm.data.windowing import resolve_window_label
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.data.augmentations.transform import apply_augment
from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)


def _resample_roundtrip(x: np.ndarray, native_hz: float, low_hz: float) -> np.ndarray:
    """[T, C] -> downsample to low_hz -> upsample back to native_hz -> [T, C]
    (output length == input length T, by construction).

    Mimics the effective bandwidth-limiting that HumanML3D-derived synthetic
    IMU has: its real information content is capped by AMASS mocap's native
    20 Hz frame rate even though the working pipeline upsamples it to
    native_hz (50 Hz) before deriving acc/gyro. Applying this same
    downsample-then-upsample round trip to REAL sensor windows at zero-shot
    eval time removes their genuine >low_hz-Nyquist spectral content, so
    train (HumanML3D) and test (real datasets) data share the same effective
    bandwidth cap instead of the model being trained on inherently-smooth
    data and evaluated on genuinely-higher-bandwidth data.
    """
    from scipy.signal import resample
    T = x.shape[0]
    if T < 4:
        return x
    duration_s = T / native_hz
    n_low = max(2, int(round(duration_s * low_hz)))
    x_low = resample(x, n_low, axis=0)
    x_up  = resample(x_low, T, axis=0)
    return x_up.astype(x.dtype)


# ---------------------------------------------------------------------------
# Label-map builder
# ---------------------------------------------------------------------------

def _build_label_map(
    parquet_path: str,
    label_col: str,
    label_name_col: str,
    sentence_template: str,
    dataset_filter: Optional[List[str]] = None,
    dataset_col: str = "dataset",
) -> Dict[int, str]:
    """Read the two label columns and return {activity_id → sentence} mapping.

    Only the two label columns are loaded from the parquet (fast columnar read).
    The activity name is lower-cased and underscores are replaced with spaces
    before formatting so "The person is walking fast" looks natural.
    """
    pa_dataset = ds_pa.dataset(parquet_path, format="parquet")
    filt = None
    if dataset_filter:
        filt = ds_pa.field(dataset_col).isin(dataset_filter)
    table = pa_dataset.to_table(columns=[label_col, label_name_col], filter=filt)
    df_map = (
        table.to_pandas()[[label_col, label_name_col]]
        .drop_duplicates(subset=[label_col])
    )

    id_to_text: Dict[int, str] = {}
    for _, row in df_map.iterrows():
        try:
            activity_id = int(row[label_col])
            activity_name = (
                str(row[label_name_col]).strip().lower().replace("_", " ")
            )
            id_to_text[activity_id] = sentence_template.format(activity_name)
        except Exception:
            pass

    logger.info(
        "[RealdataCLIP] label map: %d unique activities. Sample: %s",
        len(id_to_text),
        sorted(id_to_text.values())[:6],
    )
    return id_to_text


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealdataCLIPDataset(WindowDataset):
    """WindowDataset subclass for CLIP contrastive training on real labeled data.

    __getitem__ returns (x [6, T] float32, text str) instead of (x, label_int).

    Parameters
    ----------
    id_to_text : pre-built {dataset_activity_id → sentence} dict from
                 _build_label_map(); shared across train/val/test splits so the
                 parquet label columns are read only once.
    sentence_template : fallback template used for any label_id not in id_to_text.
    roundtrip_low_hz : if set (e.g. 20.0), every window is downsampled to this
                 rate then upsampled back to native_hz before preprocessing —
                 see _resample_roundtrip. Default None preserves prior
                 behavior exactly (no resampling).
    native_hz : the rate this dataset's windows are stored at (default 50.0,
                 this benchmark's canonical rate for all real datasets).
    """

    def __init__(
        self,
        parquet_path: str,
        session_index,
        session_keys: List[SessionKey],
        cfg: Any,
        split_name: str = "train",
        id_to_text: Optional[Dict[int, str]] = None,
        sentence_template: str = "The person is {}",
        roundtrip_low_hz: Optional[float] = None,
        native_hz: float = 50.0,
    ) -> None:
        super().__init__(parquet_path, session_index, session_keys, cfg, split_name)
        self._id_to_text = id_to_text or {}
        self._sentence_template = sentence_template
        self._roundtrip_low_hz = roundtrip_low_hz
        self._native_hz = native_hz

    # ------------------------------------------------------------------
    # Public accessor so make_realdata_clip_loaders can read unique texts
    # ------------------------------------------------------------------
    @property
    def id_to_text(self) -> Dict[int, str]:
        return self._id_to_text

    # ------------------------------------------------------------------
    # Override __getitem__ to return (x, text_str)
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, str]]:
        key   = self._keys[self._sess_idx[idx]]
        start = int(self._starts[idx])
        sd    = self._session_store[key]
        T     = self._T

        Xw = sd.X[start : start + T]
        yw = sd.y[start : start + T]

        # Resolve integer label (center / majority / pure — from config)
        label = resolve_window_label(yw, self.cfg)
        if label is None:
            return None

        # Gap gating (mirrors WindowDataset logic)
        if self._handle_gaps and sd.t is not None:
            dt = np.diff(sd.t[start : start + T])
            if np.any(dt > self._max_gap_ns):
                if self._gap_method in {"drop", "split_segment"}:
                    return None

        if self._roundtrip_low_hz is not None:
            Xw = _resample_roundtrip(Xw, self._native_hz, self._roundtrip_low_hz)

        Xproc = preprocess_window(Xw, self.cfg, self.stats)
        if Xproc is None:
            return None

        if self.split_name == "train":
            Xproc = apply_augment(Xproc.T, self.cfg).T    # [C, T]
        x_tensor = torch.from_numpy(Xproc).float()

        text = self._id_to_text.get(
            label, self._sentence_template.format(str(label))
        )
        return x_tensor, text


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _collate_clip(
    batch: list,
) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """Stack x tensors and keep texts as a plain list.  Drops None items."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, texts = zip(*batch)
    return torch.stack(xs, 0), list(texts)


# ---------------------------------------------------------------------------
# Public loader factory
# ---------------------------------------------------------------------------

def make_realdata_clip_loaders(
    cfg: Any,
    sentence_template: str = "The person is {}",
    dataset_filter: Optional[List[str]] = None,
) -> Dict[str, DataLoader]:
    """Build train / val / test DataLoaders for CLIP contrastive pretraining.

    The label-to-text mapping is constructed once (shared across splits).

    Returns
    -------
    dict with keys 'train_loader', 'val_loader', 'test_loader'
    Each loader's underlying dataset exposes ``.id_to_text`` for the full
    activity → sentence mapping used during training.
    """
    parquet_path    = cfg_get(cfg, ["paths", "dataset_path"])
    batch_size      = int(cfg_get(cfg, ["data", "batch_size"],      256))
    eval_batch_size = int(cfg_get(cfg, ["data", "eval_batch_size"], batch_size))
    num_workers     = int(cfg_get(cfg, ["data", "num_workers"],     0))
    pin_memory      = bool(cfg_get(cfg, ["data", "pin_memory"],     False))

    label_col      = cfg_get(cfg, ["data", "label_column"],      "dataset_activity_id")
    label_name_col = cfg_get(cfg, ["data", "label_name_column"], "dataset_activity_label")
    dataset_col    = cfg_get(cfg, ["data", "dataset_column"],    "dataset")

    # Determine which datasets to include
    scan_filter = dataset_filter
    if scan_filter is None:
        train_on = cfg_get(cfg, ["data", "train_on_datasets"], None)
        if train_on is not None:
            scan_filter = list(train_on)

    # Build label map once — shared across all splits
    id_to_text = _build_label_map(
        parquet_path, label_col, label_name_col,
        sentence_template, scan_filter, dataset_col,
    )

    logger.info(
        "[realdata_clip] build_session_index path=%s filter=%s",
        parquet_path, scan_filter,
    )
    session_index = build_session_index(parquet_path, cfg, dataset_filter=scan_filter)
    splits        = make_splits(session_index, cfg)

    loaders: Dict[str, DataLoader] = {}

    def _add(name: str, keys: List[SessionKey], bs: int, shuffle: bool) -> None:
        if not keys:
            return
        dataset = RealdataCLIPDataset(
            parquet_path  = parquet_path,
            session_index = session_index,
            session_keys  = keys,
            cfg           = cfg,
            split_name    = name,
            id_to_text    = id_to_text,
            sentence_template = sentence_template,
        )
        if shuffle:
            # Random shuffle across ALL windows (not session-grouped) so each batch
            # contains diverse activity labels — critical for InfoNCE to get non-zero
            # gradients. SessionGroupedBatchSampler groups same-session windows
            # (same activity) together, which zeros out the contrastive gradient.
            # All data is preloaded to RAM so random access has no I/O cost.
            loader = DataLoader(
                dataset,
                batch_size       = bs,
                shuffle          = True,
                drop_last        = True,
                num_workers      = num_workers,
                pin_memory       = pin_memory,
                collate_fn       = _collate_clip,
                persistent_workers = (num_workers > 0),
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size   = bs,
                shuffle      = False,
                num_workers  = num_workers,
                pin_memory   = pin_memory,
                drop_last    = False,
                collate_fn   = _collate_clip,
                persistent_workers = (num_workers > 0),
            )
        loaders[f"{name}_loader"] = loader
        logger.info(
            "[realdata_clip] %s_loader: sessions=%d windows=%d bs=%d",
            name, len(keys), len(dataset), bs,
        )

    _add("train", splits["train_keys"],            batch_size,      True)
    _add("val",   splits["val_keys"],              eval_batch_size, False)
    _add("test",  splits.get("test_keys", []),     eval_batch_size, False)

    return loaders
