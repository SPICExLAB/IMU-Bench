#!/usr/bin/env python3
"""Compare synthetic vs real right-wrist IMU (FLU frame) from a single Nymeria pkl.

Both signals are expressed in the FLU (Forward-Left-Up) sensor body frame at 50 Hz.

Real path:   wrist_imu_data → CPF local → world → FLU
Syn  path:   xsens_pose[joint9] → finite-diff → world → FLU

Usage
-----
    python scripts/vis_syn_vs_real_wrist_imu.py \
        --pkl  /mnt/nas2/processed_data_body_Ty_rwrist/<file>.pkl \
        --out  data_vis/syn_vs_real_wrist_imu.png \
        --window-idx 20
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_HZ = 50
WINDOW_LEN = 128   # 2.56 s @ 50 Hz

CH_NAMES = ["acc_F", "acc_L", "acc_U", "gyro_F", "gyro_L", "gyro_U"]

# Same R_WRIST_FROM_FLU used by both loaders
R_WRIST_FROM_FLU = np.array(
    [[-1.0, 0.0, 0.0],
     [ 0.0, 0.0, 1.0],
     [ 0.0, 1.0, 0.0]],
    dtype=np.float64,
)

_XSENS_RIGHT_WRIST_JOINT = 9


# ---------------------------------------------------------------------------
# Real IMU loader (mirrors nymeria_real_wrist_full_body_loader_FLU)
# ---------------------------------------------------------------------------

def _load_real(raw: dict, add_gravity: bool) -> np.ndarray | None:
    """Returns [N, 6] float32 in FLU frame, or None if data missing."""
    wrist_imu = raw.get("wrist_imu_data", {})
    wrist_gt  = raw.get("wrist_gt_data",  {})

    if not raw.get("has_wrist_imu", False):
        return None
    for key in ("accel", "gyro"):
        if key not in wrist_imu:
            return None
    if "orientation" not in wrist_gt:
        return None

    body_fps_raw  = float(raw.get("imu_sampling_rate_hz") or
                          raw.get("pose_sampling_rate_hz") or TARGET_HZ)
    wrist_fps_raw = float(raw.get("wrist_imu_sampling_rate_hz", body_fps_raw))
    wrist_ds = max(1, round(wrist_fps_raw / TARGET_HZ))

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    acc_local  = _np(wrist_imu["accel"]).astype(np.float64)[::wrist_ds]
    gyro_local = _np(wrist_imu["gyro"]).astype(np.float64)[::wrist_ds]

    orient_raw = wrist_gt["orientation"]
    try:
        import pypose as pp
        if isinstance(orient_raw, pp.LieTensor):
            R_world = orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
        else:
            raise TypeError
    except (ImportError, TypeError):
        R_world = ScipyRot.from_quat(_np(orient_raw).astype(np.float64)[::wrist_ds]).as_matrix()

    N = min(acc_local.shape[0], gyro_local.shape[0], R_world.shape[0])
    acc_local, gyro_local, R_world = acc_local[:N], gyro_local[:N], R_world[:N]

    # CPF local → world
    acc_world  = np.einsum("nij,nj->ni", R_world, acc_local)
    gyro_world = np.einsum("nij,nj->ni", R_world, gyro_local)

    if not add_gravity:
        acc_world += np.array([0.0, -9.81, 0.0])

    # world → FLU
    R_world_from_FLU = np.einsum("nij,jk->nik", R_world, R_WRIST_FROM_FLU)
    R_FLU_from_world = R_world_from_FLU.transpose(0, 2, 1)

    acc_flu  = np.einsum("nij,nj->ni", R_FLU_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_FLU_from_world, gyro_world)

    return np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic IMU loader (mirrors nymeria_syn_wrist_loader_FLU)
# ---------------------------------------------------------------------------

def _load_syn(raw: dict, add_gravity: bool) -> np.ndarray | None:
    """Returns [N, 6] float32 in FLU frame, or None if data missing."""
    xsens_pose = raw.get("gt_data", {}).get("xsens_pose")
    if xsens_pose is None:
        return None

    if hasattr(xsens_pose, "detach"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)   # [N, 23, 4, 4]

    fps_raw = float(raw.get("imu_sampling_rate_hz") or
                    raw.get("pose_sampling_rate_hz") or TARGET_HZ)
    ds = max(1, round(fps_raw / TARGET_HZ))
    xsens_pose = xsens_pose[::ds]
    N = xsens_pose.shape[0]
    if N < 3:
        return None
    fps = float(TARGET_HZ)

    j   = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, j, :3, 3].copy()    # [N, 3]
    rot = xsens_pose[:, j, :3, :3].copy()   # [N, 3, 3] R_world_from_wrist

    # Linear acceleration — central finite difference
    acc_world = np.zeros((N, 3), dtype=np.float64)
    acc_world[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc_world[0]    = acc_world[1]
    acc_world[-1]   = acc_world[-2]

    if add_gravity:
        acc_world[:, 1] += 9.81

    # Angular velocity — dR = R_{i+1} @ R_i^T → rotvec * fps
    dR = np.einsum("nij,nkj->nik", rot[1:], rot[:-1])  # [N-1, 3, 3]
    gyro_raw = ScipyRot.from_matrix(dR).as_rotvec().astype(np.float64) * fps
    gyro_world = np.zeros((N, 3), dtype=np.float64)
    gyro_world[:-1] = gyro_raw
    gyro_world[-1]  = gyro_raw[-1]

    # world → FLU
    R_world_from_FLU = np.einsum("nij,jk->nik", rot, R_WRIST_FROM_FLU)
    R_FLU_from_world = R_world_from_FLU.transpose(0, 2, 1)

    acc_flu  = np.einsum("nij,nj->ni", R_FLU_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_FLU_from_world, gyro_world)

    return np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    real: np.ndarray,   # [W, 6]
    syn:  np.ndarray,   # [W, 6]
    out_path: str,
    pkl_name: str,
    window_start: int,
    plot_hz: float = None,
):
    if plot_hz is None:
        plot_hz = TARGET_HZ
    W = real.shape[0]
    t = np.arange(W) / float(plot_hz)

    fig, axes = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
    hz_str = f"{plot_hz:.0f} Hz" + (f" (downsampled from {TARGET_HZ} Hz)" if plot_hz != TARGET_HZ else "")
    fig.suptitle(
        f"Synthetic vs Real right-wrist IMU (FLU frame)  —  {pkl_name}\n"
        f"window start={window_start} ({window_start/TARGET_HZ:.2f} s), "
        f"length={W} ({W/plot_hz:.2f} s)  |  {hz_str}",
        fontsize=11, fontweight="bold",
    )

    # Per-channel stats for annotation
    for c, (ax, name) in enumerate(zip(axes, CH_NAMES)):
        ax.plot(t, real[:, c], color="steelblue", lw=1.2, label="Real", zorder=3)
        ax.plot(t, syn[:, c],  color="tomato",    lw=1.2, label="Syn",  zorder=2,
                linestyle="--")

        rmse = float(np.sqrt(np.mean((real[:, c] - syn[:, c]) ** 2)))
        corr = float(np.corrcoef(real[:, c], syn[:, c])[0, 1])
        ax.set_ylabel(f"{name}\nRMSE={rmse:.3f}\nr={corr:.3f}", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", lw=0.4, alpha=0.5)

    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.8)
    axes[-1].set_xlabel("Time (s)", fontsize=10)

    # Global stats panel
    rmse_all = float(np.sqrt(np.mean((real - syn) ** 2)))
    corr_all = float(np.corrcoef(real.flatten(), syn.flatten())[0, 1])
    fig.text(
        0.01, 0.01,
        f"Overall  RMSE={rmse_all:.4f}   r={corr_all:.4f}",
        fontsize=9, color="dimgray",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare synthetic vs real Nymeria right-wrist IMU (FLU frame)."
    )
    p.add_argument(
        "--pkl", required=True,
        help="Path to a Nymeria pkl file with both wrist_imu_data and xsens_pose.",
    )
    p.add_argument(
        "--out", default="data_vis/syn_vs_real_wrist_imu.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--window-idx", type=int, default=20,
        help="Index of the 128-sample window to visualise (0-based).",
    )
    p.add_argument(
        "--no-gravity", action="store_true",
        help="Disable gravity addition (not recommended; both pipelines default to add_gravity=True).",
    )
    p.add_argument(
        "--full-session", action="store_true",
        help="Plot the full session rather than a single window.",
    )
    p.add_argument(
        "--t-start-s", type=float, default=0.0,
        help="Start time in seconds (applied after --full-session or --window-idx selection).",
    )
    p.add_argument(
        "--duration-s", type=float, default=None,
        help="Duration in seconds to plot. If omitted, plots to the end of the selected range.",
    )
    p.add_argument(
        "--downsample-hz", type=float, default=None,
        help="Downsample to this rate before plotting (e.g. 20). Default: no downsampling.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    add_gravity = not args.no_gravity

    print(f"Loading: {args.pkl}")
    with open(args.pkl, "rb") as fh:
        raw = pickle.load(fh)

    print("Extracting real wrist IMU …")
    real_full = _load_real(raw, add_gravity=add_gravity)
    if real_full is None:
        sys.exit(
            "ERROR: pkl is missing real wrist IMU data "
            "(has_wrist_imu=False or wrist_imu_data/wrist_gt_data keys missing)."
        )

    print("Synthesizing wrist IMU from xsens_pose …")
    syn_full = _load_syn(raw, add_gravity=add_gravity)
    if syn_full is None:
        sys.exit("ERROR: pkl is missing 'gt_data.xsens_pose'.")

    N = min(len(real_full), len(syn_full))
    real_full = real_full[:N]
    syn_full  = syn_full[:N]
    print(f"Session length: {N} frames ({N/TARGET_HZ:.1f} s) at {TARGET_HZ} Hz")

    if args.full_session:
        real_win = real_full
        syn_win  = syn_full
        window_start = 0
    else:
        window_start = args.window_idx * WINDOW_LEN
        if window_start + WINDOW_LEN > N:
            print(
                f"Warning: window_idx={args.window_idx} out of range "
                f"(max {(N - WINDOW_LEN) // WINDOW_LEN}), using start=0"
            )
            window_start = 0
        real_win = real_full[window_start : window_start + WINDOW_LEN]
        syn_win  = syn_full [window_start : window_start + WINDOW_LEN]

    # Optional time-range clip (works with both --full-session and --window-idx)
    if args.t_start_s > 0 or args.duration_s is not None:
        i0 = int(args.t_start_s * TARGET_HZ)
        i1 = (int((args.t_start_s + args.duration_s) * TARGET_HZ)
              if args.duration_s is not None else len(real_win))
        i0 = max(0, min(i0, len(real_win) - 1))
        i1 = max(i0 + 1, min(i1, len(real_win)))
        real_win = real_win[i0:i1]
        syn_win  = syn_win [i0:i1]
        window_start = window_start + i0

    # Optional downsampling
    plot_hz = TARGET_HZ
    if args.downsample_hz is not None and args.downsample_hz < TARGET_HZ:
        step = int(round(TARGET_HZ / args.downsample_hz))
        real_win  = real_win [::step]
        syn_win   = syn_win  [::step]
        plot_hz   = TARGET_HZ / step

    pkl_name = os.path.basename(args.pkl)
    plot_comparison(real_win, syn_win, args.out, pkl_name, window_start, plot_hz)

    # Print per-channel summary
    print("\nChannel RMSE / Pearson r (real vs syn):")
    print(f"  {'Channel':<10}  {'RMSE':>8}  {'r':>7}")
    for c, name in enumerate(CH_NAMES):
        rmse = float(np.sqrt(np.mean((real_win[:, c] - syn_win[:, c]) ** 2)))
        corr = float(np.corrcoef(real_win[:, c], syn_win[:, c])[0, 1])
        print(f"  {name:<10}  {rmse:>8.4f}  {corr:>7.4f}")
    rmse_all = float(np.sqrt(np.mean((real_win - syn_win) ** 2)))
    corr_all = float(np.corrcoef(real_win.flatten(), syn_win.flatten())[0, 1])
    print(f"  {'Overall':<10}  {rmse_all:>8.4f}  {corr_all:>7.4f}")


if __name__ == "__main__":
    main()
