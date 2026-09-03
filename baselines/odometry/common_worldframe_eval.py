"""Shared scoring for the TLIO / Air-IO odometry baselines, evaluated in the
EXACT metric definition used by "4b. Odometry -- canonical world-frame
variant (Nymeria pelvis motion)" (see
scripts/eval_odom_nymeria_worldframe.py::reconstruct_and_score /
eval_metrics for the reference implementation this mirrors).

Why this module exists
-----------------------
"Ours"/Scratch/MobilePoser in this benchmark all predict velocity on a fixed
33-point grid per 2.56s window (Pt=8 patches x 4 sub-patches = Ns=32
segments, patch_size=16 @ 50Hz). TLIO's native output is a single 3D
displacement for the whole window; Air-IO's CodeNetMotion outputs a shorter
sequence (~15 steps for a 128-sample window, set by its conv strides) via its
own `get_label` alignment. Neither matches the 33-point grid directly.

Rather than distorting either baseline's architecture to force an exact
32-segment output, we let each model predict at its own native temporal
resolution and then *interpolate the reconstructed trajectory* onto the same
33-point grid "Ours" is scored on before applying the identical
ate/drift/vel_mse formulas. This keeps the metric definitions byte-for-byte
identical to Section 4b while being honest about each baseline's coarser
native resolution (a real, documented property of the adaptation, not a bug).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

WINDOW_SECONDS_DEFAULT = 2.56
PATCH_SIZE_DEFAULT = 16
T_DEFAULT = 128
NS_DEFAULT = 32  # Pt=8 patches * 4 sub-patches, matching "Ours" (T=128, patch_size=16)


def resample_grid(pos_full: np.ndarray, n_target: int = NS_DEFAULT + 1) -> np.ndarray:
    """pos_full: [T+1, 3] at native per-sample resolution -> [n_target, 3]
    evenly-spaced exact subsample (native grid is always >= as fine)."""
    T = pos_full.shape[0] - 1
    idxs = np.round(np.linspace(0, T, n_target)).astype(int)
    return pos_full[idxs]


def interp_trajectory(traj_pred: np.ndarray, n_target: int = NS_DEFAULT + 1) -> np.ndarray:
    """traj_pred: [M, 3], M evenly-spaced points spanning the whole window
    (traj_pred[0] must be the window-start origin) -> [n_target, 3] resampled
    via per-axis linear interpolation onto an evenly-spaced n_target grid."""
    M = traj_pred.shape[0]
    if M == n_target:
        return traj_pred
    t_src = np.linspace(0.0, 1.0, M)
    t_tgt = np.linspace(0.0, 1.0, n_target)
    out = np.empty((n_target, 3), dtype=np.float64)
    for d in range(3):
        out[:, d] = np.interp(t_tgt, t_src, traj_pred[:, d])
    return out


def score_window(
    traj_pred: np.ndarray,
    pos_full_gt: np.ndarray,
    window_seconds: float = WINDOW_SECONDS_DEFAULT,
    n_grid: int = NS_DEFAULT + 1,
) -> Tuple[float, float, float, float]:
    """Score one window. traj_pred: [M,3] predicted trajectory (cumsum of the
    model's own predicted velocities, M = model's native step count + 1,
    traj_pred[0] == origin). pos_full_gt: [T+1,3] native-resolution GT
    trajectory for this window (from prepare_nymeria_windows.py).

    Returns (vel_mse, ate, final_err, path_len) -- identical formulas to
    eval_odom_nymeria_worldframe.py::reconstruct_and_score/eval_metrics.
    """
    pos_gt = resample_grid(pos_full_gt, n_grid).astype(np.float64)
    traj = interp_trajectory(traj_pred.astype(np.float64), n_grid)
    dt = window_seconds / (n_grid - 1)
    v_pred = np.diff(traj, axis=0) / dt
    v_gt = np.diff(pos_gt, axis=0) / dt
    vel_mse = float(np.mean((v_pred - v_gt) ** 2))
    ate = float(np.sqrt(np.mean(np.sum((traj - pos_gt) ** 2, axis=1))))
    ferr = float(np.linalg.norm(traj[-1] - pos_gt[-1]))
    plen = float(np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)))
    return vel_mse, ate, ferr, plen


def aggregate(vel_mses: List[float], ates: List[float], ferrs: List[float], plens: List[float]) -> dict:
    plens_a = np.asarray(plens)
    ferrs_a = np.asarray(ferrs)
    moving = plens_a > 0.1
    drift = float(np.mean(ferrs_a[moving] / plens_a[moving]) * 100.0) if moving.any() else float("nan")
    return {
        "vel_mse": float(np.mean(vel_mses)),
        "ate": float(np.mean(ates)),
        "drift": drift,
    }
