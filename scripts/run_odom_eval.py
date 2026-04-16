"""Odometry evaluation for the ViT1D wrist model (encoder + ViT1DOdomDecoder).

Loads a single Nymeria pkl sequence, runs the trained model over the full
sequence, integrates the per-patch displacement predictions into a trajectory,
and saves comparison plots against ground-truth root translation.

Output PNGs (written to --out-dir)
-----------------------------------
    trajectory_topdown.png   — GT vs predicted path in the XZ (floor) plane
    trajectory_3d.png        — GT vs predicted XYZ trajectories, one axis per row
    displacement_timeseries.png — per-patch GT vs predicted displacements (X, Y, Z)
    error_summary.png        — per-axis MAE bar chart + speed correlation scatter

Usage
-----
    python scripts/run_odom_eval.py \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_odom_nymeria.yaml \\
        --run          vit1d_odom_nymeria \\
        --pkl          /mnt/nas2/.../some_seq.pkl \\
        [--ckpt        best]          # "best" | "latest" | absolute path
        [--max-frames  5000]          # 0 = full sequence
        [--fps         50]
        [--out-dir     data_vis/odom_eval]
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3-D projection
from scipy.spatial.transform import Rotation as ScipyRot

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DOdomDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

_XSENS_RIGHT_WRIST_JOINT = 9   # joint index used for IMU synthesis
_XSENS_ROOT_JOINT        = 0   # pelvis / root (displacement labels)


# ---------------------------------------------------------------------------
# Data extraction (global-frame IMU + GT root position)
# ---------------------------------------------------------------------------

def load_pkl_odom(pkl_path: str, fps: int = 50, max_frames: int = 0):
    """Extract global-frame wrist IMU and world-frame root position.

    Returns
    -------
    X        : [N, 6]  float32   — global-frame acc + gyro (right wrist)
    pos_root : [N, 3]  float32   — world-frame root (pelvis) position  (metres)
    actual_fps : float
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    xsens_pose = raw["gt_data"]["xsens_pose"]
    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float32)   # [N, 23, 4, 4]

    actual_fps = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or fps
    )
    N = xsens_pose.shape[0]
    if max_frames > 0:
        N = min(N, max_frames)
        xsens_pose = xsens_pose[:N]

    # ---- Global-frame wrist acc ------------------------------------------
    jw  = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, jw, :3, 3]    # [N, 3]  wrist world position
    R   = xsens_pose[:, jw, :3, :3]   # [N, 3, 3]  world←sensor

    acc_global       = np.zeros((N, 3), dtype=np.float32)
    acc_global[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (actual_fps ** 2)
    acc_global[0]    = acc_global[1]
    acc_global[-1]   = acc_global[-2]

    # ---- Sensor-frame wrist gyro (matches training loader) ---------------
    dR           = np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])
    omega_sensor = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec()
        .reshape(N - 1, 3).astype(np.float32) * actual_fps
    )
    gyro         = np.zeros((N, 3), dtype=np.float32)
    gyro[:-1]    = omega_sensor
    gyro[-1]     = omega_sensor[-1]

    X = np.concatenate([acc_global, gyro], axis=1)   # [N, 6]

    # ---- GT root position ------------------------------------------------
    jr       = _XSENS_ROOT_JOINT
    pos_root = xsens_pose[:, jr, :3, 3].astype(np.float32)  # [N, 3]

    return X, pos_root, actual_fps


# ---------------------------------------------------------------------------
# Inference: per-patch displacements → integrated trajectory
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    X: np.ndarray,                # [N, 6]
    pos_root: np.ndarray,         # [N, 3]  GT (needed for patch-level labels)
    encoder: ViT1DEncoder,
    odom_decoder: ViT1DOdomDecoder,
    cfg,
    device: torch.device,
    fps: int = 50,
):
    """Run the model over the full sequence using non-overlapping windows.

    Returns
    -------
    pred_disp_patches : [M, 3]  float32   — predicted displacement per patch
    gt_disp_patches   : [M, 3]  float32   — GT displacement per patch
    pred_traj         : [M+1, 3] float32  — integrated predicted trajectory
    gt_traj           : [M+1, 3] float32  — integrated GT trajectory (from pos_root)
    patch_times       : [M]     float64   — centre time of each patch (seconds)
    """
    T, _hop    = compute_T_and_hop(cfg)
    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    stats      = PreprocessStats()
    N          = len(X)

    # Pad to exact multiple of T; add one extra frame to pos_root_pad so that
    # the last patch can access pos_root_pad[f0 + patch_size] without going OOB.
    pad   = (T - N % T) % T
    X_pad = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X
    pos_root_pad = np.concatenate([pos_root, np.tile(pos_root[-1:], (pad + 1, 1))])

    encoder.eval()
    odom_decoder.eval()

    pred_disps = []
    gt_disps   = []
    patch_times_list = []

    n_windows = len(X_pad) // T
    for w in range(n_windows):
        start = w * T
        Xw    = X_pad[start:start + T]   # [T, 6]

        Xproc = preprocess_window(Xw, cfg, stats)
        if Xproc is None:
            # Fill with zeros so index stays aligned
            P_t = T // patch_size
            pred_disps.extend([np.zeros(3, dtype=np.float32)] * P_t)
            gt_disps.extend([np.zeros(3, dtype=np.float32)] * P_t)
            patch_times_list.extend([
                (start + p * patch_size + patch_size / 2) / fps
                for p in range(P_t)
            ])
            continue

        x = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)  # [1, 6, T]

        tokens, _npc = encoder.forward_tokens(x)     # [1, C*Pt, D]
        pred          = odom_decoder(tokens)           # [1, Pt, 3]
        pred_np       = pred.squeeze(0).cpu().numpy() # [Pt, 3]

        P_t = pred_np.shape[0]
        for p in range(P_t):
            f0 = start + p * patch_size
            f1 = f0 + patch_size
            gt_d = pos_root_pad[f1] - pos_root_pad[f0]
            pred_disps.append(pred_np[p].astype(np.float32))
            gt_disps.append(gt_d.astype(np.float32))
            patch_times_list.append((f0 + patch_size / 2) / fps)

    pred_disp_patches = np.array(pred_disps, dtype=np.float32)   # [M, 3]
    gt_disp_patches   = np.array(gt_disps,   dtype=np.float32)   # [M, 3]
    patch_times       = np.array(patch_times_list, dtype=np.float64)

    # --- Integrate displacements into trajectories starting at GT origin ---
    origin = pos_root[0]
    pred_traj = np.empty((len(pred_disp_patches) + 1, 3), dtype=np.float32)
    gt_traj   = np.empty((len(gt_disp_patches)   + 1, 3), dtype=np.float32)
    pred_traj[0] = origin
    gt_traj[0]   = origin
    for i in range(len(pred_disp_patches)):
        pred_traj[i + 1] = pred_traj[i] + pred_disp_patches[i]
        gt_traj[i + 1]   = gt_traj[i]   + gt_disp_patches[i]

    return pred_disp_patches, gt_disp_patches, pred_traj, gt_traj, patch_times


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_disp, gt_disp, pred_traj, gt_traj):
    """Return a dict of scalar evaluation metrics."""
    # Per-patch displacement error
    disp_err = np.abs(pred_disp - gt_disp)
    disp_mae = disp_err.mean(0)            # [3]   per axis
    disp_mae_total = disp_err.mean()

    # Absolute Trajectory Error: L2 distance between integrated trajectories
    pos_err    = np.linalg.norm(pred_traj - gt_traj, axis=1)  # [M+1]
    ate        = pos_err.mean()
    ate_max    = pos_err.max()
    ate_final  = pos_err[-1]

    # Per-axis trajectory MAE
    traj_mae   = np.abs(pred_traj - gt_traj).mean(0)           # [3]

    return {
        "disp_mae_x": float(disp_mae[0]),
        "disp_mae_y": float(disp_mae[1]),
        "disp_mae_z": float(disp_mae[2]),
        "disp_mae":   float(disp_mae_total),
        "ate_mean":   float(ate),
        "ate_max":    float(ate_max),
        "ate_final":  float(ate_final),
        "traj_mae_x": float(traj_mae[0]),
        "traj_mae_y": float(traj_mae[1]),
        "traj_mae_z": float(traj_mae[2]),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_trajectory_topdown(pred_traj, gt_traj, out_dir):
    """Top-down (X-Z plane) trajectory comparison."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_traj[:, 0],   gt_traj[:, 2],   "b-",  linewidth=1.4, label="GT",        alpha=0.85)
    ax.plot(pred_traj[:, 0], pred_traj[:, 2], "r--", linewidth=1.2, label="Predicted", alpha=0.85)
    # Mark start/end
    ax.scatter(*gt_traj[0, [0, 2]],   color="blue",  marker="o", s=60, zorder=5, label="Start GT")
    ax.scatter(*gt_traj[-1, [0, 2]],  color="navy",  marker="x", s=80, zorder=5, label="End GT")
    ax.scatter(*pred_traj[0, [0, 2]], color="red",   marker="o", s=60, zorder=5, label="Start Pred")
    ax.scatter(*pred_traj[-1, [0, 2]],color="darkred",marker="x",s=80, zorder=5, label="End Pred")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title("Top-Down Trajectory  (XZ plane)\nGT vs Predicted")
    ax.legend(fontsize=9); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "trajectory_topdown.png"))


def plot_trajectory_3d(pred_traj, gt_traj, out_dir):
    """3-D trajectory overlay."""
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(*gt_traj.T,   "b-",  linewidth=1.2, label="GT",        alpha=0.8)
    ax.plot(*pred_traj.T, "r--", linewidth=1.0, label="Predicted", alpha=0.8)
    ax.scatter(*gt_traj[0],    color="blue",    marker="o", s=50, zorder=5)
    ax.scatter(*gt_traj[-1],   color="navy",    marker="x", s=70, zorder=5)
    ax.scatter(*pred_traj[0],  color="red",     marker="o", s=50, zorder=5)
    ax.scatter(*pred_traj[-1], color="darkred", marker="x", s=70, zorder=5)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("3-D Trajectory  —  GT vs Predicted")
    ax.legend()
    _save(fig, os.path.join(out_dir, "trajectory_3d.png"))


def plot_xyz_over_time(pred_traj, gt_traj, patch_times, out_dir, fps=50):
    """X, Y, Z trajectory components vs time (three stacked subplots)."""
    # patch_times[i] is the centre of patch i — shift by half-patch so it
    # aligns with traj indices (which have one extra point at t=0)
    traj_times = np.concatenate([[0.0], patch_times])

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    labels = ("X (m)", "Y (m)", "Z (m)")
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(traj_times, gt_traj[:, i],   "b-",  linewidth=1.2, label="GT",        alpha=0.85)
        ax.plot(traj_times, pred_traj[:, i], "r--", linewidth=1.0, label="Predicted", alpha=0.85)
        ax.set_ylabel(lbl); ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Root Translation  —  GT vs Predicted")
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "trajectory_xyz_timeseries.png"))


def plot_displacement_timeseries(pred_disp, gt_disp, patch_times, out_dir):
    """Per-patch displacement (X, Y, Z) comparison over time."""
    max_pts = min(len(patch_times), 600)
    t  = patch_times[:max_pts]
    pd = pred_disp[:max_pts]
    gd = gt_disp[:max_pts]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    labels = ("ΔX (m/patch)", "ΔY (m/patch)", "ΔZ (m/patch)")
    for i, (ax, lbl) in enumerate(zip(axes, labels)):
        ax.plot(t, gd[:, i], "b-",  linewidth=1.2, label="GT",        alpha=0.85)
        ax.plot(t, pd[:, i], "r--", linewidth=1.0, label="Predicted", alpha=0.85)
        ax.set_ylabel(lbl); ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("Per-Patch Root Displacement  —  GT vs Predicted")
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "displacement_timeseries.png"))


def plot_error_summary(pred_disp, gt_disp, pred_traj, gt_traj, metrics, out_dir):
    """Two-panel summary: per-axis displacement MAE + trajectory ATE over time."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Panel 1: per-axis displacement MAE bar chart --------------------
    ax = axes[0]
    axis_labels = ["X", "Y", "Z"]
    maes = [metrics["disp_mae_x"], metrics["disp_mae_y"], metrics["disp_mae_z"]]
    colors = ["#4477aa", "#228833", "#aa3377"]
    bars = ax.bar(axis_labels, maes, color=colors, width=0.5)
    for bar, v in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(maes) * 0.02,
                f"{v*100:.2f} cm", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("MAE (m)"); ax.set_title("Per-Axis Displacement MAE")
    ax.set_ylim(0, max(maes) * 1.18); ax.grid(axis="y", alpha=0.3)
    ax.text(0.98, 0.97, f"Overall MAE: {metrics['disp_mae']*100:.2f} cm",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # ---- Panel 2: ATE (L2 position error) over time ---------------------
    ax = axes[1]
    ate_seq = np.linalg.norm(pred_traj - gt_traj, axis=1)
    ax.plot(ate_seq, "g-", linewidth=1.2, alpha=0.9)
    ax.axhline(metrics["ate_mean"], color="r", linestyle="--",
               label=f"Mean ATE: {metrics['ate_mean']*100:.1f} cm")
    ax.axhline(metrics["ate_final"], color="orange", linestyle=":",
               label=f"Final ATE: {metrics['ate_final']*100:.1f} cm")
    ax.set_xlabel("Patch index"); ax.set_ylabel("ATE (m)")
    ax.set_title("Absolute Trajectory Error over Time")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "error_summary.png"))


def plot_speed_correlation(pred_disp, gt_disp, out_dir):
    """Scatter: GT displacement magnitude vs predicted displacement magnitude.
    
    A perfect predictor would lie on the diagonal.
    """
    gt_speed   = np.linalg.norm(gt_disp,   axis=1)   # [M]
    pred_speed = np.linalg.norm(pred_disp, axis=1)   # [M]

    # Subsample to avoid overloaded scatter plot
    step = max(1, len(gt_speed) // 1000)
    gt_s = gt_speed[::step]
    pr_s = pred_speed[::step]

    # Pearson correlation
    corr = float(np.corrcoef(gt_speed, pred_speed)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_s, pr_s, s=5, alpha=0.4, color="steelblue")
    lim = max(gt_s.max(), pr_s.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.2, label="Ideal")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("GT speed (m / patch)"); ax.set_ylabel("Pred speed (m / patch)")
    ax.set_title(f"Speed Correlation  (r = {corr:.3f})")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "speed_correlation.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser(
        description="Evaluate ViT1D odometry decoder on a Nymeria sequence"
    )
    ap.add_argument("--config",       required=True, help="Base config YAML")
    ap.add_argument("--model-config", required=True, help="Model config YAML (vit1d_odom_nymeria.yaml)")
    ap.add_argument("--run",          required=True, help="Run name under runs/<run>")
    ap.add_argument("--pkl",          required=True, help="Path to a single Nymeria pkl file")
    ap.add_argument("--ckpt",         default="best", help="Checkpoint: best | latest | abs path")
    ap.add_argument("--max-frames",   type=int, default=0, help="Max frames to eval (0 = full)")
    ap.add_argument("--fps",          type=int, default=50)
    ap.add_argument("--out-dir",      default="data_vis/odom_eval")
    args = ap.parse_args()

    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg = deep_update(base_cfg, model_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Checkpoint -------------------------------------------------------
    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    run_dir   = os.path.join(runs_root, args.run)
    if args.ckpt in ("best", "latest"):
        ckpt_path = os.path.join(run_dir, "checkpoints", f"{args.ckpt}.pt")
    else:
        ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    state    = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = state["cfg"]

    # ---- Build model from checkpoint config -------------------------------
    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    T_win, _   = compute_T_and_hop(ckpt_cfg)
    vit_cfg    = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))

    odom_decoder = ViT1DOdomDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        cfg=ckpt_cfg,
    ).to(device)
    odom_decoder.load_state_dict(state["odom_decoder"])
    odom_decoder.eval()

    logger.info(
        "Encoder: embed_dim=%d  in_channels=%d  patch_size=%d  num_patches=%d",
        encoder.embed_dim, encoder.in_channels, patch_size, T_win // patch_size,
    )

    # ---- Data extraction --------------------------------------------------
    logger.info("Loading: %s", os.path.basename(args.pkl))
    X, pos_root, actual_fps = load_pkl_odom(args.pkl, fps=args.fps, max_frames=args.max_frames)
    fps = int(round(actual_fps))
    N   = len(X)
    logger.info("Frames: %d  (%.1f s @ %d Hz)", N, N / fps, fps)

    # ---- Inference --------------------------------------------------------
    logger.info("Running inference ...")
    pred_disp, gt_disp, pred_traj, gt_traj, patch_times = run_inference(
        X, pos_root, encoder, odom_decoder, cfg, device, fps=fps
    )

    # ---- Metrics ----------------------------------------------------------
    metrics = compute_metrics(pred_disp, gt_disp, pred_traj, gt_traj)

    print("\n=== Odometry Evaluation ===")
    print(f"  Per-patch displacement MAE  (X):  {metrics['disp_mae_x']*100:6.2f} cm")
    print(f"  Per-patch displacement MAE  (Y):  {metrics['disp_mae_y']*100:6.2f} cm")
    print(f"  Per-patch displacement MAE  (Z):  {metrics['disp_mae_z']*100:6.2f} cm")
    print(f"  Per-patch displacement MAE total: {metrics['disp_mae']*100:6.2f} cm")
    print(f"  Absolute Trajectory Error (mean): {metrics['ate_mean']*100:6.2f} cm")
    print(f"  Absolute Trajectory Error  (max): {metrics['ate_max']*100:6.2f} cm")
    print(f"  Absolute Trajectory Error (final):{metrics['ate_final']*100:6.2f} cm")

    # ---- Save plots -------------------------------------------------------
    logger.info("Saving plots to %s ...", args.out_dir)
    plot_trajectory_topdown(pred_traj, gt_traj, args.out_dir)
    plot_trajectory_3d(pred_traj, gt_traj, args.out_dir)
    plot_xyz_over_time(pred_traj, gt_traj, patch_times, args.out_dir, fps=fps)
    plot_displacement_timeseries(pred_disp, gt_disp, patch_times, args.out_dir)
    plot_error_summary(pred_disp, gt_disp, pred_traj, gt_traj, metrics, args.out_dir)
    plot_speed_correlation(pred_disp, gt_disp, args.out_dir)

    logger.info("Done. All plots written to %s", args.out_dir)


if __name__ == "__main__":
    main()
