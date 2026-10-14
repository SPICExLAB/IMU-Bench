"""Gyroscope unit check for physical-units caches (added 2026-09-17).

Every source is expected to deliver the gyroscope in rad/s. A source that silently delivers deg/s is
57.3x too large -- the EatingSpeed pickles did exactly that and nothing noticed, because per-window
z-scoring removes a constant per-channel scale. Physical-units pipelines do not, so a cache built
without normalisation is checked before anything is written.

Test: in windows where the device is at rest (|acc| nearly constant: coefficient of variation < 1 %),
the true angular rate is ~0, so the median |gyro| is the sensor noise floor: 0.02-0.03 rad/s for every
rad/s source here, 1.3-1.6 for deg/s. The overall 99th percentile is a second test that needs no still
windows: consumer IMUs saturate at +-2000 deg/s = 34.9 rad/s.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

STILL_CV_MAX = 0.01            # std(|acc|)/mean(|acc|) inside a window
STILL_GYRO_MEDIAN_MAX = 0.3    # rad/s; deg/s sources sit at ~1.3-1.6
P99_GYRO_MAX = 40.0            # rad/s; above the +-2000 dps (34.9 rad/s) sensor range


class GyroUnitError(RuntimeError):
    pass


def _window_array(item: Any) -> Optional[np.ndarray]:
    x = item[0] if isinstance(item, (tuple, list)) else item
    if x is None:
        return None
    x = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    return x.astype(np.float32) if x.ndim == 2 and x.shape[0] >= 6 else None   # [C, T]


def check_gyro_units(dataset, name: str, n_windows: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Sample windows from ``dataset`` (items ``[C, T]`` or ``(x, ...)``) and raise GyroUnitError if the
    gyroscope channels (3:6) look like deg/s. Returns the statistics for the cache's unit marker."""
    n = len(dataset)
    idx = np.random.default_rng(seed).choice(n, size=min(n, n_windows), replace=False)
    still_med, all_abs = [], []
    for i in idx:
        x = _window_array(dataset[int(i)])
        if x is None:
            continue
        acc = np.linalg.norm(x[:3], axis=0)
        gyr = np.linalg.norm(x[3:6], axis=0)
        valid = np.isfinite(acc) & np.isfinite(gyr) & (np.abs(x[:6]).sum(0) > 0)   # drop padding
        if valid.sum() < 16:
            continue
        acc, gyr = acc[valid], gyr[valid]
        all_abs.append(gyr)
        m = float(np.abs(acc.mean()))
        if m > 1e-6 and float(acc.std()) / m < STILL_CV_MAX:
            still_med.append(float(np.median(gyr)))
    if not all_abs:
        raise GyroUnitError(f"[unit-check] {name}: no usable windows to check")
    p99 = float(np.percentile(np.concatenate(all_abs), 99))
    med = float(np.median(still_med)) if still_med else None
    stats = {"n_checked": len(all_abs), "n_still": len(still_med), "still_gyro_median": med, "gyro_p99": p99}
    problems = []
    if med is not None and len(still_med) >= 20 and med > STILL_GYRO_MEDIAN_MAX:
        problems.append(f"median |gyro| in {len(still_med)} still windows = {med:.3f} (> {STILL_GYRO_MEDIAN_MAX} rad/s)")
    if p99 > P99_GYRO_MAX:
        problems.append(f"99th-percentile |gyro| = {p99:.1f} (> {P99_GYRO_MAX} rad/s, beyond any sensor range)")
    if problems:
        raise GyroUnitError(f"[unit-check] {name}: gyroscope does not look like rad/s -- " + "; ".join(problems)
                            + ". Convert the source to rad/s in its loader before caching.")
    return stats
