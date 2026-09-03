"""Artefact removal + sensor-range model for SYNTHETIC wrist IMU (mocap -> FK -> finite differences).

Why: 2nd-order finite differences of mocap positions turn every tracking glitch / frame
jump into an acceleration spike (MotionX: 27 % of cached windows peaked above 10 g, p90 peak
2,150 m/s^2, max ~98,000 m/s^2; SONIC: 2 % of windows above 16 g, 89 % of them 1-sample
spikes). A real MEMS wrist IMU can't produce those: it saturates at +-16 g / +-2000 dps.

Pipeline (per clip, after synthesis, before windowing):
  1. artefact samples = |acc| > acc_max  or  |gyro| > gyro_max   (physically impossible)
  2. short runs (<= max_spike_len samples) -> linearly interpolated from the neighbours
     (isolated glitch, the underlying motion is continuous)
  3. long runs -> treated as a tracking discontinuity: the samples are dropped and the clip
     is SPLIT there (keeps real motion on both sides, never stitches across a teleport)
  4. clamp everything to the sensor range (saturation, like a real IMU on an impact)
Segments shorter than `min_len` are discarded.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

G = 9.80665
ACC_MAX = 16.0 * G            # m/s^2  (+-16 g range of the common wrist IMU chips)
GYRO_MAX = 2000.0 * np.pi / 180.0   # rad/s (+-2000 dps)


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))   # [start, end)


def sanitize_synthetic_imu(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    acc_max: float = ACC_MAX,
    gyro_max: float = GYRO_MAX,
    max_spike_len: int = 3,
    min_len: int = 64,
) -> Tuple[List[Tuple[np.ndarray, Optional[np.ndarray]]], dict]:
    """X [N, 6] (acc xyz, gyro xyz), optional y [N, ...] aligned labels.
    Returns (segments, stats); segments = [(X_seg, y_seg or None), ...]."""
    X = np.asarray(X, dtype=np.float32)
    N = len(X)
    if N == 0:
        return [], {"n": 0}
    bad = (np.linalg.norm(X[:, :3], axis=1) > acc_max) | (np.linalg.norm(X[:, 3:6], axis=1) > gyro_max)
    bad |= ~np.isfinite(X).all(axis=1)
    n_spike = n_cut = 0
    cut_runs: List[Tuple[int, int]] = []
    if bad.any():
        Xf = X.copy()
        for s, e in _runs(bad):
            if e - s <= max_spike_len and s > 0 and e < N:
                n_spike += e - s
                for c in range(X.shape[1]):
                    Xf[s:e, c] = np.interp(np.arange(s, e), [s - 1, e], [Xf[s - 1, c], Xf[e, c]])
            else:
                n_cut += e - s
                cut_runs.append((s, e))
        X = Xf
    # split at long runs / edge runs
    bounds, prev = [], 0
    for s, e in cut_runs:
        bounds.append((prev, s)); prev = e
    bounds.append((prev, N))
    segs = []
    for s, e in bounds:
        if e - s >= min_len:
            Xs = X[s:e].copy()
            Xs[:, :3] = np.clip(Xs[:, :3], -acc_max, acc_max)
            Xs[:, 3:6] = np.clip(Xs[:, 3:6], -gyro_max, gyro_max)
            segs.append((Xs, None if y is None else y[s:e]))
    stats = {"n": N, "spike_samples_interpolated": n_spike, "cut_samples": n_cut,
             "segments": len(segs), "kept_samples": int(sum(len(s[0]) for s in segs))}
    return segs, stats


if __name__ == "__main__":      # self-test
    rng = np.random.RandomState(0)
    X = rng.normal(0, 3, (1000, 6)).astype(np.float32); X[:, 2] += 9.8
    X[100, 0] = 5000.0; X[101, 1] = -900.0                 # 2-sample glitch -> interpolated
    X[500:520, :3] = 4000.0                                 # 20-sample teleport -> split
    X[990:1000, 3] = 100.0                                  # tail run -> dropped
    y = np.arange(1000)
    segs, st = sanitize_synthetic_imu(X, y)
    assert st["segments"] == 2 and st["spike_samples_interpolated"] == 2 and st["cut_samples"] == 30, st
    assert len(segs[0][0]) == 500 and len(segs[1][0]) == 470 and segs[1][1][0] == 520
    assert np.abs(segs[0][0][100:102, :2]).max() < 20 and np.abs(segs[0][0][:, :3]).max() <= ACC_MAX
    print("sanitize self-test ok:", st)
