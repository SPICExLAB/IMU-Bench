"""Dataset-level odometry evaluation for the ViT1D wrist model.

Runs inference over every pkl in a test directory and reports:

  ATE   — Absolute Trajectory Error
            Mean L2 distance (m) between the integrated predicted trajectory
            and the GT trajectory, computed per frame (patch boundary) and
            averaged across all frames of all sequences.

  RTE-1s — Relative Translation Error over 1-second sub-sequences
            For every consecutive window of Δ patches (≈ 1 s), measure the
            L2 error between predicted and GT relative displacement.
            RTE-1s = mean over all such windows and all sequences.

Both metrics are reported per-sequence and aggregated (mean ± std) over the
full test set.  A summary PNG is written to --out-dir.

Usage
-----
    python scripts/run_odom_eval_dataset.py \\
        --config       configs/base.yaml \\
        --model-config configs/vit1d_odom_nymeria.yaml \\
        --run          vit1d_odom_nymeria \\
        --test-dir     /mnt/nas2/processed_data_test_body_Ty \\
        [--ckpt        best]          # best | latest | abs path
        [--max-frames  0]             # per-sequence frame cap  (0 = full)
        [--fps         50]
        [--rte-window  1.0 2.56]      # one or more RTE sub-sequence durations in seconds
        [--out-dir     data_vis/odom_eval_dataset]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as ScipyRot
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DOdomDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)

_XSENS_RIGHT_WRIST_JOINT = 9
_XSENS_ROOT_JOINT        = 0

# FLU frame rotation matrix (right wrist T-pose)
_R_WRIST_FROM_FLU = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


# ---------------------------------------------------------------------------
# Data loading (mirrors run_odom_eval.py::load_pkl_odom)
# ---------------------------------------------------------------------------

def load_pkl_odom(pkl_path: str, fps: int = 50, max_frames: int = 0):
    """Return (X [N,6], pos_root [N,3], actual_fps)."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    xsens_pose = raw["gt_data"]["xsens_pose"]
    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float32)

    actual_fps = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or fps
    )
    N = xsens_pose.shape[0]
    if max_frames > 0:
        N = min(N, max_frames)
        xsens_pose = xsens_pose[:N]

    jw  = _XSENS_RIGHT_WRIST_JOINT
    pos = xsens_pose[:, jw, :3, 3]
    R   = xsens_pose[:, jw, :3, :3]

    acc_global       = np.zeros((N, 3), dtype=np.float32)
    acc_global[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (actual_fps ** 2)
    acc_global[0]    = acc_global[1]
    acc_global[-1]   = acc_global[-2]

    dR           = np.einsum("nij,njk->nik", R[:-1].transpose(0, 2, 1), R[1:])
    omega_sensor = (
        ScipyRot.from_matrix(dR.reshape(-1, 3, 3)).as_rotvec()
        .reshape(N - 1, 3).astype(np.float32) * actual_fps
    )
    gyro      = np.zeros((N, 3), dtype=np.float32)
    gyro[:-1] = omega_sensor
    gyro[-1]  = omega_sensor[-1]

    X        = np.concatenate([acc_global, gyro], axis=1)
    pos_root = xsens_pose[:, _XSENS_ROOT_JOINT, :3, 3].astype(np.float32)
    return X, pos_root, actual_fps


# ---------------------------------------------------------------------------
# Data loading — Nymeria real wrist FLU (for local-velocity model, out_dim=12)
# ---------------------------------------------------------------------------

def load_pkl_odom_flu(pkl_path: str, target_hz: int = 50,
                      add_gravity: bool = True, max_frames: int = 0):
    """Load Nymeria pkl for the local-velocity FLU model.

    Returns
    -------
    X_flu          : [N, 6]    float32  -- FLU acc + gyro (real wrist IMU)
    pos_root       : [N, 3]    float32  -- pelvis world position
    R_world_from_X : [N, 3, 3] float32  -- wrist FLU orientation in world
    target_hz      : float
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    wrist_imu  = raw.get("wrist_imu_data", {})
    wrist_gt   = raw.get("wrist_gt_data",  {})
    xsens_pose = raw["gt_data"]["xsens_pose"]

    if hasattr(xsens_pose, "numpy"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)

    body_fps_raw  = float(
        raw.get("pose_sampling_rate_hz") or raw.get("imu_sampling_rate_hz") or target_hz
    )
    wrist_fps_raw = float(raw.get("wrist_imu_sampling_rate_hz", body_fps_raw))
    body_ds  = max(1, round(body_fps_raw  / target_hz))
    wrist_ds = max(1, round(wrist_fps_raw / target_hz))

    xsens_pose = xsens_pose[::body_ds]

    def _to_np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    acc_local  = _to_np(wrist_imu["accel"]).astype(np.float64)[::wrist_ds]
    gyro_local = _to_np(wrist_imu["gyro"]).astype(np.float64)[::wrist_ds]
    orient_raw = wrist_gt["orientation"]

    try:
        import pypose as pp
        _is_lietensor = isinstance(orient_raw, pp.LieTensor)
    except ImportError:
        _is_lietensor = False

    if _is_lietensor:
        R_world_from_wrist = (
            orient_raw.matrix().detach().cpu().numpy().astype(np.float64)[::wrist_ds]
        )
    else:
        orient_np          = _to_np(orient_raw).astype(np.float64)[::wrist_ds]
        R_world_from_wrist = ScipyRot.from_quat(orient_np).as_matrix().astype(np.float64)

    N = min(xsens_pose.shape[0], acc_local.shape[0])
    if max_frames > 0:
        N = min(N, max_frames)

    xsens_pose         = xsens_pose[:N]
    acc_local          = acc_local[:N]
    gyro_local         = gyro_local[:N]
    R_world_from_wrist = R_world_from_wrist[:N]

    R_world_from_X = np.einsum("nij,jk->nik", R_world_from_wrist, _R_WRIST_FROM_FLU)
    R_X_from_world = R_world_from_X.transpose(0, 2, 1)

    acc_world = np.einsum("nij,nj->ni", R_world_from_wrist, acc_local)
    if not add_gravity:
        acc_world = acc_world + np.array([0.0, -9.81, 0.0])
    acc_flu  = np.einsum("nij,nj->ni", R_X_from_world, acc_world)

    gyro_world = np.einsum("nij,nj->ni", R_world_from_wrist, gyro_local)
    gyro_flu   = np.einsum("nij,nj->ni", R_X_from_world, gyro_world)

    X_flu    = np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)
    pos_root = xsens_pose[:, _XSENS_ROOT_JOINT, :3, 3].astype(np.float32)
    return X_flu, pos_root, R_world_from_X.astype(np.float32), float(target_hz)


# ---------------------------------------------------------------------------
# Inference  (mirrors run_odom_eval.py::run_inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(X, pos_root, encoder, odom_decoder, cfg, device, fps):
    """Return (pred_traj [M+1,3], gt_traj [M+1,3], pred_vels [M,3], gt_vels [M,3]).

    pred_vels / gt_vels are world-frame velocities in m/s (displacement / patch_duration).
    """
    T, _       = compute_T_and_hop(cfg)
    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    dt_patch   = patch_size / fps
    stats      = PreprocessStats()
    N          = len(X)

    pad          = (T - N % T) % T
    X_pad        = np.concatenate([X, np.zeros((pad, 6), dtype=np.float32)]) if pad else X
    pos_root_pad = np.concatenate([pos_root, np.tile(pos_root[-1:], (pad + 1, 1))])

    pred_disps = []
    gt_disps   = []

    for w in range(len(X_pad) // T):
        start = w * T
        Xw    = X_pad[start:start + T]
        Xproc = preprocess_window(Xw, cfg, stats)
        P_t   = T // patch_size
        if Xproc is None:
            pred_disps.extend([np.zeros(3, dtype=np.float32)] * P_t)
            gt_disps.extend([np.zeros(3, dtype=np.float32)] * P_t)
            continue

        x      = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)
        tokens, _ = encoder.forward_tokens(x)
        pred_np   = odom_decoder(tokens).squeeze(0).cpu().numpy()  # [Pt, 3]

        for p in range(pred_np.shape[0]):
            f0 = start + p * patch_size
            f1 = f0 + patch_size
            pred_disps.append(pred_np[p].astype(np.float32))
            gt_disps.append((pos_root_pad[f1] - pos_root_pad[f0]).astype(np.float32))

    pred_d = np.array(pred_disps, dtype=np.float32)  # [M, 3]
    gt_d   = np.array(gt_disps,   dtype=np.float32)  # [M, 3]

    # Integrate to patch-boundary trajectories
    origin    = pos_root[0]
    M         = len(pred_d)
    pred_traj = np.empty((M + 1, 3), dtype=np.float32)
    gt_traj   = np.empty((M + 1, 3), dtype=np.float32)
    pred_traj[0] = gt_traj[0] = origin
    for i in range(M):
        pred_traj[i + 1] = pred_traj[i] + pred_d[i]
        gt_traj[i + 1]   = gt_traj[i]   + gt_d[i]

    pred_vels = pred_d / dt_patch   # [M, 3] m/s
    gt_vels   = gt_d   / dt_patch   # [M, 3] m/s
    return pred_traj, gt_traj, pred_vels, gt_vels


# ---------------------------------------------------------------------------
# Inference — local-velocity model (out_dim=12)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference_local_vel(
    X_flu, pos_root, R_world_from_X_gt,
    encoder, odom_decoder, cfg, device, fps,
):
    """Inference for the local-velocity model (out_dim=12).

    Algorithm
    ---------
    1. Integrate gyro (FLU body frame) from the GT initial orientation to
       estimate R_world_from_X at every frame (Euler integration).
    2. For each predicted sub-patch velocity v_flu[p,k]:
           delta_world = R_world_from_X[t_mid] @ v_flu[p,k] * (Q/fps)
    3. Accumulate delta_world to build a trajectory at sub-patch resolution
       (one step = Q frames, e.g. 4 frames = 0.08 s at 50 Hz).

    Returns (pred_traj [M+1,3], gt_traj [M+1,3], pred_vels [M,3], gt_vels [M,3])
    at sub-patch boundaries.  pred_vels / gt_vels are world-frame velocities (m/s).
    """
    T, _       = compute_T_and_hop(cfg)
    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    Q          = patch_size // 4          # frames per sub-patch
    dt_Q       = Q / fps                  # seconds per sub-patch
    dt         = 1.0 / fps
    stats      = PreprocessStats()
    N          = len(X_flu)

    # --- Gyro integration -------------------------------------------------
    # gyro_flu = X_flu[:, 3:6]  (body-frame angular velocity, rad/s)
    gyro_flu = X_flu[:, 3:6].astype(np.float64)
    R_int    = np.empty((N, 3, 3), dtype=np.float64)
    R_int[0] = R_world_from_X_gt[0].astype(np.float64)   # GT initial orientation
    for t in range(1, N):
        # Body-frame integration: R[t] = R[t-1] @ exp(gyro_body * dt)
        dR     = ScipyRot.from_rotvec(gyro_flu[t - 1] * dt).as_matrix()
        R_int[t] = R_int[t - 1] @ dR

    # --- Sliding-window inference -----------------------------------------
    pad   = (T - N % T) % T
    X_pad = np.concatenate([X_flu, np.zeros((pad, 6), dtype=np.float32)]) if pad else X_flu
    P_t   = T // patch_size

    pred_deltas      = []
    gt_deltas        = []
    pred_vels_world  = []   # world-frame velocity predictions (m/s)
    gt_vels_world    = []   # world-frame velocity GT (m/s)

    for w in range(len(X_pad) // T):
        start  = w * T
        Xw     = X_pad[start : start + T]
        Xproc  = preprocess_window(Xw, cfg, stats)

        if Xproc is None:
            for p in range(P_t):
                for k in range(4):
                    pred_deltas.append(np.zeros(3, dtype=np.float32))
                    pred_vels_world.append(np.zeros(3, dtype=np.float32))
                    t_s = start + p * patch_size + k * Q
                    t_e = min(t_s + Q, N - 1)
                    gt_d = (pos_root[t_e] - pos_root[max(0, t_s)]).astype(np.float32)
                    gt_deltas.append(gt_d)
                    gt_vels_world.append(gt_d / dt_Q)
            continue

        x        = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)
        tokens, _ = encoder.forward_tokens(x)
        vel_pred  = odom_decoder(tokens).squeeze(0).cpu().numpy().reshape(P_t, 4, 3)
        # [P_t, 4, 3] : vel_pred[p, k] = predicted velocity in FLU for sub-patch k of patch p

        for p in range(P_t):
            for k in range(4):
                t_s   = start + p * patch_size + k * Q
                t_e   = t_s + Q
                t_mid = t_s + Q // 2

                R_at_mid = R_int[min(t_mid, N - 1)]
                v_flu_k  = vel_pred[p, k].astype(np.float64)
                v_world_pred = (R_at_mid @ v_flu_k).astype(np.float32)   # m/s world frame
                delta_pred   = (v_world_pred * dt_Q).astype(np.float32)

                pred_deltas.append(delta_pred)
                pred_vels_world.append(v_world_pred)

                if t_e < N and t_s < N:
                    gt_d = (pos_root[t_e] - pos_root[t_s]).astype(np.float32)
                else:
                    gt_d = np.zeros(3, dtype=np.float32)
                gt_deltas.append(gt_d)
                gt_vels_world.append((gt_d / dt_Q).astype(np.float32))

    # --- Integrate sub-patch deltas into trajectory -----------------------
    origin   = pos_root[0]
    M        = len(pred_deltas)
    pred_traj = np.empty((M + 1, 3), dtype=np.float32)
    gt_traj   = np.empty((M + 1, 3), dtype=np.float32)
    pred_traj[0] = gt_traj[0] = origin
    for i in range(M):
        pred_traj[i + 1] = pred_traj[i] + pred_deltas[i]
        gt_traj[i + 1]   = gt_traj[i]   + gt_deltas[i]

    pred_vels = np.array(pred_vels_world, dtype=np.float32)  # [M, 3]
    gt_vels   = np.array(gt_vels_world,   dtype=np.float32)  # [M, 3]
    return pred_traj, gt_traj, pred_vels, gt_vels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ate(pred_traj: np.ndarray, gt_traj: np.ndarray) -> np.ndarray:
    """Per-frame ATE [M+1] in metres."""
    return np.linalg.norm(pred_traj - gt_traj, axis=1)


def compute_rte(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    delta: int,
) -> np.ndarray:
    """Relative Translation Error for windows of `delta` patches.

    RTE[i] = ||( pred_traj[i+delta] - pred_traj[i] )
               - ( gt_traj[i+delta]   - gt_traj[i]   )||
    """
    if len(pred_traj) <= delta:
        return np.array([], dtype=np.float32)
    pred_rel = pred_traj[delta:] - pred_traj[:-delta]   # [M+1-delta, 3]
    gt_rel   = gt_traj[delta:]   - gt_traj[:-delta]
    return np.linalg.norm(pred_rel - gt_rel, axis=1)    # [M+1-delta]


def compute_drift_rate(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    """Drift rate = mean_ATE / total_GT_path_length  (dimensionless, multiply by 100 for %).

    Measures how much the trajectory drifts relative to the distance actually
    traveled.  A value of 0.05 means the average position error is 5 % of the
    total GT path length.
    """
    gt_path_len = float(np.linalg.norm(np.diff(gt_traj, axis=0), axis=1).sum())
    if gt_path_len < 1e-6:
        return float("nan")
    mean_ate = float(np.linalg.norm(pred_traj - gt_traj, axis=1).mean())
    return mean_ate / gt_path_len


def compute_vel_metrics(
    pred_vels: np.ndarray, gt_vels: np.ndarray
) -> tuple:
    """Velocity error metrics (world-frame, m/s).

    Returns (mse, rmse, mean_L2) where:
      mse     : mean squared error averaged over all steps and all 3 axes
      rmse    : sqrt(mse)
      mean_L2 : mean L2 norm of the per-step velocity error vector
    """
    err    = pred_vels - gt_vels          # [M, 3]
    mse    = float((err ** 2).mean())
    rmse   = float(np.sqrt(mse))
    mean_l2 = float(np.linalg.norm(err, axis=1).mean())
    return mse, rmse, mean_l2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved  %s", path)


def plot_ate_distribution(seq_ate_means: np.ndarray, out_dir: str):
    """Histogram of per-sequence mean ATE."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(seq_ate_means * 100, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(float(seq_ate_means.mean()) * 100, color="red", linestyle="--",
               label=f"Mean = {seq_ate_means.mean()*100:.1f} cm")
    ax.set_xlabel("Per-sequence mean ATE (cm)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Mean ATE across Test Sequences")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "ate_distribution.png"))


def plot_rte_distribution(seq_rte_means: np.ndarray, rte_window_s: float, out_dir: str):
    """Histogram of per-sequence mean RTE-{window}s."""
    label = f"{rte_window_s:.2f}".rstrip('0').rstrip('.')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(seq_rte_means * 100, bins=30, color="tomato", edgecolor="white", alpha=0.85)
    ax.axvline(float(seq_rte_means.mean()) * 100, color="navy", linestyle="--",
               label=f"Mean = {seq_rte_means.mean()*100:.1f} cm")
    ax.set_xlabel(f"Per-sequence mean RTE-{label}s (cm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Mean RTE-{label}s across Test Sequences")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, f"rte_{label}s_distribution.png"))


def plot_summary_bars(
    ate_mean: float, ate_std: float,
    rte_results: list,   # list of (label_str, mean, std)
    out_dir: str,
):
    """Bar chart of ATE + all RTE windows with error bars."""
    labels = ["ATE"] + [r[0] for r in rte_results]
    means  = [ate_mean * 100] + [r[1] * 100 for r in rte_results]
    stds   = [ate_std  * 100] + [r[2] * 100 for r in rte_results]
    rte_colors = ["tomato", "darkorange", "orchid", "goldenrod"]
    colors = ["steelblue"] + rte_colors[:len(rte_results)]
    fig, ax = plt.subplots(figsize=(max(5, 2 + 1.5 * len(labels)), 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=6,
                  width=0.5, error_kw=dict(elinewidth=1.5))
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(means) * 0.03,
                f"{v:.1f} cm", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Error (cm)")
    ax.set_title("Dataset-Level Odometry Metrics\n(mean ± std across sequences)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, os.path.join(out_dir, "summary_metrics.png"))


def plot_drift_rate_distribution(seq_drift_rates: np.ndarray, out_dir: str):
    """Histogram of per-sequence drift rate (%)."""
    vals = seq_drift_rates[~np.isnan(seq_drift_rates)] * 100
    if len(vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=30, color="mediumseagreen", edgecolor="white", alpha=0.85)
    ax.axvline(float(vals.mean()), color="red", linestyle="--",
               label=f"Mean = {vals.mean():.2f} %")
    ax.set_xlabel("Drift rate (%)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Drift Rate across Test Sequences")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "drift_rate_distribution.png"))


def plot_vel_error_distribution(
    pred_vels: np.ndarray, gt_vels: np.ndarray, out_dir: str
):
    """Histogram of per-step velocity L2 error (m/s)."""
    l2_err = np.linalg.norm(pred_vels - gt_vels, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(l2_err, bins=50, color="darkorchid", edgecolor="white", alpha=0.85)
    ax.axvline(float(l2_err.mean()), color="red", linestyle="--",
               label=f"Mean = {l2_err.mean():.3f} m/s")
    ax.set_xlabel("Per-step velocity L2 error (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Velocity Error across All Steps")
    ax.legend(); ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "vel_error_distribution.png"))


def plot_ate_vs_duration(
    seq_durations: np.ndarray, seq_ate_means: np.ndarray, out_dir: str
):
    """Scatter: sequence duration vs mean ATE (longer seqs drift more?)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(seq_durations, seq_ate_means * 100, s=20, alpha=0.6, color="steelblue")
    ax.set_xlabel("Sequence duration (s)")
    ax.set_ylabel("Mean ATE (cm)")
    ax.set_title("ATE vs Sequence Duration")
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, "ate_vs_duration.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser(
        description="Dataset-level odometry evaluation (ATE + RTE-1s)"
    )
    ap.add_argument("--config",       required=True,  help="Base config YAML")
    ap.add_argument("--model-config", required=True,  help="Model config YAML")
    ap.add_argument("--run",          required=True,  help="Run name under runs/<run>")
    ap.add_argument("--test-dir",     required=True,  help="Directory with test pkl files")
    ap.add_argument("--ckpt",         default="best", help="best | latest | abs path")
    ap.add_argument("--max-frames",   type=int,   default=0,   help="Per-sequence cap (0=all)")
    ap.add_argument("--fps",          type=int,   default=50)
    ap.add_argument("--rte-window",   type=float, nargs="+", default=[1.0, 5.0],
                    help="One or more RTE window durations in seconds (e.g. 1.0 5.0)")
    ap.add_argument("--out-dir",      default="data_vis/odom_eval_dataset")
    args = ap.parse_args()

    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg       = deep_update(base_cfg, model_cfg)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ---- Load checkpoint --------------------------------------------------
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

    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    T_win, _   = compute_T_and_hop(ckpt_cfg)
    vit_cfg    = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))

    # Auto-detect out_dim from checkpoint (3 = global disp, 12 = local vel 4×3)
    out_dim = int(state["odom_decoder"]["head.weight"].shape[0])
    odom_decoder = ViT1DOdomDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        cfg=ckpt_cfg,
        out_dim=out_dim,
    ).to(device)
    odom_decoder.load_state_dict(state["odom_decoder"])
    odom_decoder.eval()

    # Local-velocity model: out_dim=12 → FLU loader + gyro-integrated trajectory
    use_local_vel = (out_dim == 12)
    step_frames   = patch_size // 4 if use_local_vel else patch_size
    if use_local_vel:
        _obj_type    = cfg_get(ckpt_cfg, ["objective", "type"], "odom_nymeria_real_wrist_flu")
        _ocfg        = cfg_get(ckpt_cfg, ["objective", _obj_type], {}) or {}
        _add_gravity = bool(_ocfg.get("add_gravity", True))
        logger.info(
            "Local-velocity model (out_dim=12): FLU loader + gyro integration. "
            "step_frames=%d  add_gravity=%s", step_frames, _add_gravity,
        )

    # ---- Gather pkl files -------------------------------------------------
    pkl_files = sorted(glob.glob(os.path.join(args.test_dir, "*.pkl")))
    if not pkl_files:
        raise SystemExit(f"No pkl files found in {args.test_dir}")
    logger.info("Found %d test sequences", len(pkl_files))

    # Resolve deltas (patch steps) for each requested RTE window, using the fps
    # of the first successfully loaded sequence.  All sequences share the same fps.
    # We'll refine per-sequence below but use a representative value for display.
    rte_windows  = args.rte_window          # list of floats, e.g. [1.0, 2.56]

    # ---- Per-sequence evaluation ------------------------------------------
    seq_names       = []
    seq_ate_means   = []
    seq_drift_rates = []
    # rte_accum[i] = list of per-sequence mean RTE for window rte_windows[i]
    rte_seq_means = [[] for _ in rte_windows]
    seq_durations = []
    seq_failed    = []

    all_ate_vals  = []
    all_rte_vals  = [[] for _ in rte_windows]
    all_pred_vels = []   # world-frame velocity predictions (m/s) for all sequences
    all_gt_vels   = []   # world-frame velocity GT (m/s) for all sequences

    fps_used = args.fps   # will be updated from the first sequence

    for pkl_path in tqdm(pkl_files, desc="Evaluating sequences", unit="seq"):
        seq_name = os.path.basename(pkl_path).replace(".pkl", "")
        try:
            if use_local_vel:
                X_data, pos_root, R_world_from_X, actual_fps = load_pkl_odom_flu(
                    pkl_path, target_hz=args.fps, add_gravity=_add_gravity,
                    max_frames=args.max_frames,
                )
            else:
                X_data, pos_root, actual_fps = load_pkl_odom(
                    pkl_path, args.fps, args.max_frames
                )
        except Exception as exc:
            logger.warning("Skipping %s: %s", seq_name, exc)
            seq_failed.append(seq_name)
            continue

        fps      = int(round(actual_fps))
        fps_used = fps

        if use_local_vel:
            pred_traj, gt_traj, pred_vels, gt_vels = run_inference_local_vel(
                X_data, pos_root, R_world_from_X, encoder, odom_decoder, ckpt_cfg, device, fps
            )
        else:
            pred_traj, gt_traj, pred_vels, gt_vels = run_inference(
                X_data, pos_root, encoder, odom_decoder, ckpt_cfg, device, fps
            )

        ate_seq = compute_ate(pred_traj, gt_traj)   # [M+1]

        seq_names.append(seq_name)
        seq_ate_means.append(float(ate_seq.mean()))
        seq_drift_rates.append(compute_drift_rate(pred_traj, gt_traj))
        seq_durations.append(len(X_data) / fps)
        all_ate_vals.append(ate_seq)
        all_pred_vels.append(pred_vels)
        all_gt_vels.append(gt_vels)

        for wi, win_s in enumerate(rte_windows):
            delta   = max(1, round(win_s * fps / step_frames))
            rte_seq = compute_rte(pred_traj, gt_traj, delta)
            if len(rte_seq) > 0:
                rte_seq_means[wi].append(float(rte_seq.mean()))
                all_rte_vals[wi].append(rte_seq)

    if not seq_names:
        raise SystemExit("No sequences evaluated successfully.")

    seq_ate_means   = np.array(seq_ate_means)
    seq_drift_rates = np.array(seq_drift_rates, dtype=np.float64)
    seq_durations   = np.array(seq_durations)

    all_ate_flat    = np.concatenate(all_ate_vals)
    ate_global_mean = float(all_ate_flat.mean())
    ate_global_std  = float(all_ate_flat.std())

    # Velocity metrics
    all_pred_vels_flat = np.concatenate(all_pred_vels)   # [total_steps, 3]
    all_gt_vels_flat   = np.concatenate(all_gt_vels)     # [total_steps, 3]
    vel_mse, vel_rmse, vel_mean_l2 = compute_vel_metrics(all_pred_vels_flat, all_gt_vels_flat)

    # Actual window durations (may differ slightly from target due to rounding)
    rte_actual_s = [
        max(1, round(w * fps_used / step_frames)) * step_frames / fps_used
        for w in rte_windows
    ]

    # Per-window global stats
    rte_stats = []   # list of (label, global_mean, global_std, seq_means_arr)
    for wi, win_s in enumerate(rte_windows):
        act_s    = rte_actual_s[wi]
        label    = f"RTE-{act_s:.2f}s".rstrip('0').rstrip('.')
        label    = label if label.endswith('s') else label + 's'
        flat     = np.concatenate(all_rte_vals[wi]) if all_rte_vals[wi] else np.array([np.nan])
        seq_arr  = np.array(rte_seq_means[wi]) if rte_seq_means[wi] else np.array([np.nan])
        rte_stats.append((label, float(flat.mean()), float(flat.std()), seq_arr))

    # ---- Console summary --------------------------------------------------
    hdr = "=" * 72
    print(f"\n{hdr}")
    print(f"  Dataset-level odometry evaluation — {len(seq_names)} sequences")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  patch_size : {patch_size}  fps : {fps_used}")
    win_str = "  ".join(f"{a:.3f} s" for a in rte_actual_s)
    print(f"  RTE windows: {win_str}")
    print(hdr)
    drift_valid = seq_drift_rates[~np.isnan(seq_drift_rates)]
    drift_mean  = float(np.nanmean(seq_drift_rates)) if len(drift_valid) else float("nan")
    drift_std   = float(np.nanstd(seq_drift_rates))  if len(drift_valid) else float("nan")

    print(f"\n  {'Metric':<36} {'Value':>14}")
    print(f"  {'-'*52}")
    print(f"  {'ATE (cm)':<36} {ate_global_mean*100:>14.2f}  ± {ate_global_std*100:.2f}")
    for label, g_mean, g_std, seq_arr in rte_stats:
        print(f"  {f'{label} (cm)':<36} {g_mean*100:>14.2f}  ± {g_std*100:.2f}")
    print(f"  {'Drift rate (%)':<36} {drift_mean*100:>14.3f}  ± {drift_std*100:.3f}")
    print(f"  {'-'*52}")
    print(f"  {'Velocity MSE (m²/s²)':<36} {vel_mse:>14.6f}")
    print(f"  {'Velocity RMSE (m/s)':<36} {vel_rmse:>14.6f}")
    print(f"  {'Velocity mean-L2 (m/s)':<36} {vel_mean_l2:>14.6f}")
    print(f"\n  Per-sequence stats:")
    print(f"    ATE:  mean={seq_ate_means.mean()*100:.2f} cm  "
          f"std={seq_ate_means.std()*100:.2f} cm  "
          f"min={seq_ate_means.min()*100:.2f} cm  "
          f"max={seq_ate_means.max()*100:.2f} cm")
    for label, _, _, seq_arr in rte_stats:
        if seq_arr.size and not np.isnan(seq_arr).all():
            print(f"    {label}: mean={np.nanmean(seq_arr)*100:.2f} cm  "
                  f"std={np.nanstd(seq_arr)*100:.2f} cm  "
                  f"min={np.nanmin(seq_arr)*100:.2f} cm  "
                  f"max={np.nanmax(seq_arr)*100:.2f} cm")
    if len(drift_valid):
        print(f"    Drift rate: mean={drift_mean*100:.3f} %  "
              f"std={drift_std*100:.3f} %  "
              f"min={np.nanmin(seq_drift_rates)*100:.3f} %  "
              f"max={np.nanmax(seq_drift_rates)*100:.3f} %")

    if seq_failed:
        print(f"\n  ⚠ Skipped {len(seq_failed)} sequences: "
              f"{seq_failed[:5]}{'...' if len(seq_failed) > 5 else ''}")
    print()

    # ---- Plots ------------------------------------------------------------
    plot_ate_distribution(seq_ate_means, args.out_dir)
    for label, _, _, seq_arr in rte_stats:
        if seq_arr.size and not np.isnan(seq_arr).all():
            plot_rte_distribution(seq_arr, float(label.replace('RTE-','').replace('s','')),
                                  args.out_dir)
    plot_drift_rate_distribution(seq_drift_rates, args.out_dir)
    plot_summary_bars(
        ate_mean=seq_ate_means.mean(),
        ate_std=seq_ate_means.std(),
        rte_results=[(lbl, g_mean, g_std) for lbl, g_mean, g_std, _ in rte_stats],
        out_dir=args.out_dir,
    )
    plot_ate_vs_duration(seq_durations, seq_ate_means, args.out_dir)
    plot_vel_error_distribution(all_pred_vels_flat, all_gt_vels_flat, args.out_dir)

    logger.info("All plots written to %s", args.out_dir)


if __name__ == "__main__":
    main()
