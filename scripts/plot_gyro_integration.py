"""Compare gyro-integrated wrist orientation against GT wrist orientation.

For IMUPoser pkls:
    GT orientation  = real_imu_data["real_ori"][:, 1]   (R_world_from_sensor, Z-up)
    Gyro            = derived from consecutive GT orientation differences
                      (no raw wrist gyro in the pkl — angular velocity back-computed from GT)

For Nymeria pkls:
    GT orientation  = wrist_gt_data["orientation"]       (quaternion or rotmat)
    Gyro            = wrist_imu_data["gyro"]             (raw sensor-frame angular velocity, rad/s)

In both cases the integrator starts from R_gt[0] and steps forward with:
    R_int[t+1] = R_int[t] @ Exp(omega_sensor[t] / fps)

For IMUPoser the integration error is near-zero (gyro is derived from GT), so the plot
acts as a sanity check / numerical precision reference.
For Nymeria (raw gyro) the plot shows real integration drift over the sequence.

Outputs (saved to --out-dir):
    gyro_vs_gt_geodesic.png  – per-frame geodesic error between integrated & GT
    gyro_vs_gt_euler.png     – roll/pitch/yaw time-series of GT vs integrated (ZYX, degrees)
    gyro_signal.png          – raw angular velocity signal (rad/s) that was integrated

Usage
-----
    python scripts/plot_gyro_integration.py \\
        --pkl   /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00001.pkl \\
        [--fps   30]          # override sample rate (default: read from pkl)
        [--out-dir data_vis]
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotmat_to_euler_zyx_deg(R: np.ndarray) -> np.ndarray:
    """[N, 3, 3] → [N, 3] Euler ZYX angles in degrees (yaw, pitch, roll)."""
    return ScipyRot.from_matrix(R).as_euler("ZYX", degrees=True)


def _geodesic_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> np.ndarray:
    """[N, 3, 3] x [N, 3, 3] → [N] geodesic error in degrees."""
    R_diff = np.einsum("nij,njk->nik", R_pred, R_gt.transpose(0, 2, 1))
    trace  = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos    = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _integrate_gyro(R0: np.ndarray, omega: np.ndarray, fps: float) -> np.ndarray:
    """Euler-integrate angular velocity to get orientation.

    Parameters
    ----------
    R0    : [3, 3]  initial orientation (R_world_from_sensor)
    omega : [N, 3]  angular velocity in sensor frame at each frame (rad/s)
    fps   : float   sample rate

    Returns
    -------
    R_int : [N+1, 3, 3]  integrated orientations (first entry = R0)
    """
    N = len(omega)
    R_int = np.zeros((N + 1, 3, 3), dtype=np.float64)
    R_int[0] = R0
    dt = 1.0 / fps
    for t in range(N):
        rotvec = omega[t] * dt        # axis-angle increment (sensor frame)
        dR = ScipyRot.from_rotvec(rotvec).as_matrix()
        R_int[t + 1] = R_int[t] @ dR
    return R_int


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_imuposer(pkl_path: str, fps: float):
    """Load IMUPoser pkl. Returns (R_gt, omega_sensor, fps)."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    actual_fps = float(raw.get("sampling_rate_hz", fps) or fps)
    ri = raw.get("real_imu_data", {})

    if "real_ori" not in ri:
        raise ValueError("No real_ori in real_imu_data — not an IMUPoser pkl")

    R_gt = np.asarray(ri["real_ori"], dtype=np.float64)[:, 1]   # [N, 3, 3]
    N = len(R_gt)

    # Derive angular velocity from consecutive GT orientations
    dR = np.einsum("nij,njk->nik", R_gt[:-1].transpose(0, 2, 1), R_gt[1:])   # [N-1, 3, 3]
    omega = np.zeros((N, 3), dtype=np.float64)
    omega[:-1] = ScipyRot.from_matrix(dR).as_rotvec() * actual_fps
    omega[-1]  = omega[-2]

    return R_gt, omega, actual_fps


def load_nymeria(pkl_path: str, fps: float):
    """Load Nymeria pkl. Returns (R_gt, omega_sensor, fps)."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    wrist_imu = raw.get("wrist_imu_data", {})
    wrist_gt  = raw.get("wrist_gt_data",  {})

    if "gyro" not in wrist_imu or "orientation" not in wrist_gt:
        raise ValueError("Missing wrist_imu_data/wrist_gt_data — not a Nymeria pkl")

    gyro_raw = wrist_imu["gyro"]
    if hasattr(gyro_raw, "detach"):
        gyro_raw = gyro_raw.detach().cpu().numpy()
    omega = np.asarray(gyro_raw, dtype=np.float64)   # [N, 3]  sensor-frame rad/s

    orient_raw = wrist_gt["orientation"]
    try:
        import pypose as pp
        if isinstance(orient_raw, pp.LieTensor):
            R_gt = orient_raw.matrix().detach().cpu().numpy().astype(np.float64)
        else:
            raise TypeError
    except (ImportError, TypeError):
        orient_np = np.asarray(
            orient_raw.detach().cpu().numpy() if hasattr(orient_raw, "detach") else orient_raw,
            dtype=np.float64,
        )
        if orient_np.ndim == 2 and orient_np.shape[1] == 4:
            R_gt = ScipyRot.from_quat(orient_np).as_matrix().astype(np.float64)
        else:
            R_gt = orient_np   # assume already [N, 3, 3]

    actual_fps = float(raw.get("wrist_imu_sampling_rate_hz", fps) or fps)

    # Align lengths
    N = min(len(R_gt), len(omega))
    return R_gt[:N], omega[:N], actual_fps


def load_pkl(pkl_path: str, fps: float):
    """Auto-detect format and load."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    if "wrist_imu_data" in raw and "wrist_gt_data" in raw:
        R_gt, omega, actual_fps = load_nymeria(pkl_path, fps)
        fmt = "nymeria"
    elif "real_imu_data" in raw:
        R_gt, omega, actual_fps = load_imuposer(pkl_path, fps)
        fmt = "imuposer"
    else:
        raise ValueError(f"Unknown pkl format: {pkl_path}")
    return R_gt, omega, actual_fps, fmt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_geodesic(geo: np.ndarray, fps: float, out_path: str, title_suffix: str):
    t = np.arange(len(geo)) / fps
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, geo, color="steelblue", lw=0.8, alpha=0.85, label="Geodesic error")
    ax.axhline(geo.mean(), color="r", lw=1.4, ls="--",
               label=f"Mean: {geo.mean():.2f}°  |  Median: {np.median(geo):.2f}°  |  Max: {geo.max():.2f}°")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Geodesic error (°)")
    ax.set_title(f"Gyro-integrated vs GT wrist orientation — geodesic error{title_suffix}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)


def plot_euler(euler_gt: np.ndarray, euler_int: np.ndarray, fps: float, out_path: str, title_suffix: str):
    """Plot GT vs integrated roll/pitch/yaw (ZYX convention) in degrees."""
    t = np.arange(len(euler_gt)) / fps
    labels = ["Yaw (Z)", "Pitch (Y)", "Roll (X)"]
    colors_gt  = ["steelblue",   "tomato",   "seagreen"]
    colors_int = ["cornflowerblue", "lightsalmon", "lightgreen"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for i, (ax, lbl, cgt, cint) in enumerate(zip(axes, labels, colors_gt, colors_int)):
        ax.plot(t, euler_gt[:, i],  color=cgt,  lw=1.2, label="GT")
        ax.plot(t, euler_int[:, i], color=cint, lw=1.0, ls="--", alpha=0.85, label="Integrated")
        err = np.abs(euler_gt[:, i] - euler_int[:, i])
        ax2 = ax.twinx()
        ax2.fill_between(t, err, alpha=0.15, color="gray", label=f"Error  (mean {err.mean():.2f}°)")
        ax2.set_ylabel("Error (°)", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.set_ylabel(f"{lbl} (°)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"GT vs Gyro-integrated wrist orientation — Euler ZYX{title_suffix}", y=1.01)
    fig.tight_layout()
    _save(fig, out_path)


def plot_gyro_signal(omega: np.ndarray, fps: float, out_path: str, title_suffix: str):
    t = np.arange(len(omega)) / fps
    labels = ["ω_x (roll-rate)", "ω_y (pitch-rate)", "ω_z (yaw-rate)"]
    colors = ["steelblue", "tomato", "seagreen"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    for ax, lbl, col, idx in zip(axes, labels, colors, range(3)):
        ax.plot(t, omega[:, idx], color=col, lw=0.8, alpha=0.85)
        ax.set_ylabel(f"{lbl}\n(rad/s)")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Wrist angular velocity (sensor frame){title_suffix}", y=1.01)
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Plot gyro-integrated vs GT wrist orientation")
    ap.add_argument("--pkl",     required=True, help="Path to IMUPoser or Nymeria pkl")
    ap.add_argument("--fps",     type=float, default=30.0, help="Sample rate override")
    ap.add_argument("--out-dir", default="data_vis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading {args.pkl} ...")
    R_gt, omega, actual_fps, fmt = load_pkl(args.pkl, args.fps)
    N = len(R_gt)
    print(f"  Format : {fmt}")
    print(f"  Frames : {N}  @  {actual_fps:.1f} Hz  ({N/actual_fps:.1f} s)")
    print(f"  Gyro source : {'raw sensor gyro' if fmt == 'nymeria' else 'derived from consecutive GT rotations'}")

    # Integrate from initial GT orientation
    # R_int_full[t] = estimated orientation at frame t (same index as R_gt[t])
    # R_int_full[0] = R_gt[0] (exact by construction)
    print("Integrating gyro ...")
    R_int_full = _integrate_gyro(R_gt[0], omega, actual_fps)   # [N+1, 3, 3]
    R_int = R_int_full[:N]                                       # [N, 3, 3]  same index as R_gt

    # Metrics
    geo = _geodesic_deg(R_int, R_gt)    # [N]
    print(f"  Geodesic error — mean: {geo.mean():.3f}°  median: {np.median(geo):.3f}°  max: {geo.max():.3f}°")

    seq_name = os.path.splitext(os.path.basename(args.pkl))[0]
    title_suffix = f"\n{seq_name}  ({N/actual_fps:.0f} s @ {actual_fps:.0f} Hz)"

    # Euler angles
    euler_gt  = _rotmat_to_euler_zyx_deg(R_gt)    # [N, 3] yaw/pitch/roll
    euler_int = _rotmat_to_euler_zyx_deg(R_int)   # [N, 3]

    plot_geodesic(geo,  actual_fps,
                  out_path=os.path.join(args.out_dir, f"{seq_name}_gyro_vs_gt_geodesic.png"),
                  title_suffix=title_suffix)
    plot_euler(euler_gt, euler_int, actual_fps,
               out_path=os.path.join(args.out_dir, f"{seq_name}_gyro_vs_gt_euler.png"),
               title_suffix=title_suffix)
    plot_gyro_signal(omega, actual_fps,
                     out_path=os.path.join(args.out_dir, f"{seq_name}_gyro_signal.png"),
                     title_suffix=title_suffix)

    print("Done.")


if __name__ == "__main__":
    main()
