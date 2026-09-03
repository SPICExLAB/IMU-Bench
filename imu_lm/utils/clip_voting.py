"""Window/clip-level aggregation matching the OpenWatch paper's own protocol
(Bonazzi & Ahmed et al., arXiv:2605.04791): "windows of length W=100 samples
(1s at 100Hz) were extracted with stride 20 samples... An early-stop rule
assigns the clip label once any class appears in k=3 consecutive windows. If
no class meets the threshold, majority voting is used as a fallback."
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

NATIVE_HZ = 100.0
WINDOW_SAMPLES = 100
STRIDE_SAMPLES = 20
CONSECUTIVE_K = 3


def sliding_window_starts(length: int, window: int = WINDOW_SAMPLES, stride: int = STRIDE_SAMPLES) -> List[int]:
    """Start indices of consecutive-run windows over a length-`length` signal.
    Falls back to a single window covering the whole signal if it's shorter
    than `window` samples."""
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    return starts or [0]


# The paper's own arithmetic: "windows ... were extracted with stride 20
# samples FROM A RESTRICTED SUB-INTERVAL of the clip, yielding 6 overlapping
# windows per clip" — (L-100)/20+1 = 6 gives L = 200, i.e. a 2.0s
# sub-interval of the 3.0s recordings. WHERE the sub-interval sits is not
# stated; we center it (avoid clip boundaries symmetrically), which for a
# 300-sample recording gives starts 50,70,...,150. This finally explains the
# 6-vs-11 windows/clip discrepancy EXPERIMENTS.md §5d chased with stride=40.
RESTRICT_SAMPLES = 200


def restricted_window_starts(
    length: int,
    window: int = WINDOW_SAMPLES,
    stride: int = STRIDE_SAMPLES,
    restrict_len: int = RESTRICT_SAMPLES,
) -> List[int]:
    """Paper-protocol starts: slide over a CENTERED `restrict_len` sub-interval.
    Falls back to full-range sliding when the signal is shorter than the
    sub-interval."""
    if length <= restrict_len:
        return sliding_window_starts(length, window=window, stride=stride)
    lo = (length - restrict_len) // 2
    starts = list(range(lo, lo + restrict_len - window + 1, stride))
    return starts or [lo]


def aggregate_clip_label(window_preds: Sequence[int], k: int = CONSECUTIVE_K) -> int:
    """Early-stop rule: first run of `k` consecutive identical predictions
    (in temporal order) wins; falls back to plain majority vote otherwise."""
    run_label, run_len = None, 0
    for p in window_preds:
        if p == run_label:
            run_len += 1
        else:
            run_label, run_len = p, 1
        if run_len >= k:
            return int(run_label)

    vals, counts = np.unique(np.asarray(window_preds), return_counts=True)
    return int(vals[np.argmax(counts)])
