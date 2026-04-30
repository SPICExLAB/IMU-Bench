from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

from imu_lm.utils.helpers import cfg_get

logger = logging.getLogger(__name__)


CANONICAL_SAMPLE_RATE_HZ = 50.0


def get_sample_rate_hz(cfg: Any) -> float:
    """Resolve the sample rate used to convert window seconds to samples.

    Defaults to the canonical 50 Hz used across the existing codebase, but
    allows dataset-specific configs to opt into another rate when a loader
    emits a different temporal resolution.
    """

    return float(
        cfg_get(
            cfg,
            ["windowing", "sample_rate_hz"],
            cfg_get(cfg, ["data", "sample_rate_hz"], CANONICAL_SAMPLE_RATE_HZ),
        )
    )


def compute_T_and_hop(cfg: Any) -> Tuple[int, int]:
    """Compute window length (T) and hop in samples.

    Uses the configured sample rate when present, otherwise defaults to 50 Hz.
    """

    window_seconds = float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56))
    hop_ratio = float(cfg_get(cfg, ["windowing", "window_hop_ratio"], 0.5))
    sample_rate_hz = get_sample_rate_hz(cfg)
    T = int(round(window_seconds * sample_rate_hz))
    hop = max(1, int(round(T * hop_ratio)))
    return T, hop


def resolve_window_label(yw: np.ndarray, cfg: Any) -> Optional[int]:
    """Resolve a window label according to policy.

    Returns None if the window should be skipped.
    """

    policy = cfg_get(cfg, ["windowing", "label_policy"], "pure")
    majority_threshold = float(cfg_get(cfg, ["windowing", "majority_threshold"], 0.8))
    unknown_label_id = cfg_get(cfg, ["data", "unknown_label_id"], None)

    if yw.size == 0:
        return None

    if policy == "pure":
        if np.all(yw == yw[0]):
            return int(yw[0])
        return None

    if policy == "majority":
        vals, counts = np.unique(yw, return_counts=True)
        idx = counts.argmax()
        p = counts[idx] / float(len(yw))
        if p >= majority_threshold:
            return int(vals[idx])
        return None

    if policy == "center":
        return int(yw[len(yw) // 2])

    if policy == "unknown_if_mixed":
        if np.all(yw == yw[0]):
            return int(yw[0])
        return int(unknown_label_id) if unknown_label_id is not None else None

    raise ValueError(f"Unsupported label_policy={policy}")
