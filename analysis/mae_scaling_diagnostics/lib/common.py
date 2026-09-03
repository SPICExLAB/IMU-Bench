"""Shared helpers for the MAE data-scaling diagnostic scripts.

Every script in analysis/mae_scaling_diagnostics/ imports from here instead of
duplicating config-loading / builder-invocation logic. Nothing here writes to
runs/, configs/, or any canonical checkpoint -- outputs always go to
analysis/mae_scaling_diagnostics/out/.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, "analysis", "mae_scaling_diagnostics", "out")
os.makedirs(OUT_DIR, exist_ok=True)

# The six data-scaling pretraining configs actually used in EXPERIMENTS.md §9,
# in ascending order. Every one of these shares an identical `sources:` block
# (fixed per-source sample_fraction/max_files/etc.) -- only `overall_fraction`
# differs -- confirmed by diff during the initial research pass.
FRACTION_CONFIGS: Dict[str, Tuple[float, str]] = {
    "0.1pct": (0.001, "configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_data0.1pct.yaml"),
    "1pct":   (0.01,  "configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_data1pct.yaml"),
    "5pct":   (0.05,  "configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_data5pct.yaml"),
    "25pct":  (0.25,  "configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_data25pct.yaml"),
    "50pct":  (0.5,   "configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_data50pct.yaml"),
    "100pct": (1.0,   "configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml"),
}

# Corresponding run directories on disk (only best.pt exists for each -- see plan).
RUN_DIRS: Dict[str, str] = {
    "0.1pct": "runs/vit1d_ssl_mae_varlen_altattn_multidata_data0.1pct",
    "1pct":   "runs/vit1d_ssl_mae_varlen_altattn_multidata_data1pct",
    "5pct":   "runs/vit1d_ssl_mae_varlen_altattn_multidata_data5pct",
    "25pct":  "runs/vit1d_ssl_mae_varlen_altattn_multidata_data25pct",
    "50pct":  "runs/vit1d_ssl_mae_varlen_altattn_multidata_data50pct",
    "100pct": "runs/vit1d_ssl_mae_varlen_altattn_multidata",
}

BASE_CONFIG = "configs/base.yaml"


def load_merged_cfg(model_config_rel: str, base_config_rel: str = BASE_CONFIG) -> Dict[str, Any]:
    """Reproduce exactly what scripts/run_pretrain.py does: base.yaml deep-updated
    by the model config. Uses absolute paths so this works regardless of cwd."""
    from imu_lm.utils.helpers import deep_update, load_yaml

    base_cfg = load_yaml(os.path.join(REPO_ROOT, base_config_rel))
    model_cfg = load_yaml(os.path.join(REPO_ROOT, model_config_rel))
    return deep_update(base_cfg, model_cfg)


def mm_cfg_of(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from imu_lm.utils.helpers import cfg_get
    return cfg_get(cfg, ["objective", "mae_multidata_varlen_altattn_flu"], {}) or {}


def build_all_sources_once(cfg: Dict[str, Any], vl_max: Optional[int] = None):
    """Build every enabled source ONCE, with provenance, ignoring overall_fraction
    entirely (i.e. at the shared 100%-of-sources baseline every fraction config
    is built from -- all 6 data{X}pct configs have byte-identical `sources:`
    blocks, only `overall_fraction` differs). This is the expensive step
    (SMPL FK for nymeria/totalcapture/motionx/sonic, HumanML3D clip loading,
    OCDetect/capture24 file parsing) -- callers should do it exactly once and
    replay cheap index-only subsampling for each of the 6 fractions afterward,
    instead of rebuilding from raw per fraction.

    Returns (train_parts, val_parts, names) where train_parts[i]/val_parts[i]
    are the XOnlyDataset-wrapped per-source datasets for names[i] (some val
    entries may be None), in imu_lm.data.combined_mae_multidata_loader.ALL_SOURCES
    order (only enabled, non-empty sources included).
    """
    from imu_lm.data import combined_mae_multidata_loader as cm
    from imu_lm.utils.helpers import cfg_get

    mm_cfg = mm_cfg_of(cfg)
    sources_cfg = mm_cfg.get("sources", {}) or {}
    if vl_max is None:
        vl_cfg = cfg_get(cfg, ["data", "variable_length"], {}) or {}
        vl_max = int(vl_cfg.get("max_samples", 512))

    names: List[str] = []
    train_parts = []
    val_parts = []
    for name in cm.ALL_SOURCES:
        src_cfg = sources_cfg.get(name, {}) or {}
        if not bool(src_cfg.get("enabled", True)):
            continue
        train_ds, val_ds = cm._BUILDERS[name](cfg, src_cfg, vl_max, collect_provenance=True)
        if train_ds is None:
            continue
        names.append(name)
        train_parts.append(train_ds)
        val_parts.append(val_ds)
    return train_parts, val_parts, names


def fraction_indices_for(total_n: int, fraction: float, seed: int = 42) -> np.ndarray:
    """Exactly reproduces combined_mae_multidata_loader._fraction_subset's index
    draw (pooled mode, the mode every non-`_stratified` data{X}pct config uses)."""
    if fraction >= 1.0 or total_n == 0:
        return np.arange(total_n)
    k = max(1, round(total_n * fraction))
    return np.sort(np.random.default_rng(seed).choice(total_n, size=k, replace=False))


def concat_offsets(parts) -> np.ndarray:
    """Cumulative sizes for a list of datasets, matching torch ConcatDataset's
    own bookkeeping -- offsets[i] is the global start index of parts[i]."""
    sizes = [len(p) for p in parts]
    return np.concatenate([[0], np.cumsum(sizes)])[:-1]
