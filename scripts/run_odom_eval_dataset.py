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
# Inference  (mirrors run_odom_eval.py::run_inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(X, pos_root, encoder, odom_decoder, cfg, device, fps):
    """Return (pred_traj [M+1,3], gt_traj [M+1,3]) patch-boundary trajectories."""
    T, _       = compute_T_and_hop(cfg)
    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
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

    return pred_traj, gt_traj


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
    ap.add_argument("--rte-window",   type=float, nargs="+", default=[1.0, 2.56],
                    help="One or more RTE window durations in seconds (e.g. 1.0 2.56)")
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

    odom_decoder = ViT1DOdomDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        cfg=ckpt_cfg,
    ).to(device)
    odom_decoder.load_state_dict(state["odom_decoder"])
    odom_decoder.eval()

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
    seq_names     = []
    seq_ate_means = []
    # rte_accum[i] = list of per-sequence mean RTE for window rte_windows[i]
    rte_seq_means = [[] for _ in rte_windows]
    seq_durations = []
    seq_failed    = []

    all_ate_vals  = []
    all_rte_vals  = [[] for _ in rte_windows]

    fps_used = args.fps   # will be updated from the first sequence

    for pkl_path in tqdm(pkl_files, desc="Evaluating sequences", unit="seq"):
        seq_name = os.path.basename(pkl_path).replace(".pkl", "")
        try:
            X, pos_root, actual_fps = load_pkl_odom(pkl_path, args.fps, args.max_frames)
        except Exception as exc:
            logger.warning("Skipping %s: %s", seq_name, exc)
            seq_failed.append(seq_name)
            continue

        fps      = int(round(actual_fps))
        fps_used = fps

        pred_traj, gt_traj = run_inference(
            X, pos_root, encoder, odom_decoder, cfg, device, fps
        )

        ate_seq = compute_ate(pred_traj, gt_traj)   # [M+1]

        seq_names.append(seq_name)
        seq_ate_means.append(float(ate_seq.mean()))
        seq_durations.append(len(X) / fps)
        all_ate_vals.append(ate_seq)

        for wi, win_s in enumerate(rte_windows):
            delta   = max(1, round(win_s * fps / patch_size))
            rte_seq = compute_rte(pred_traj, gt_traj, delta)
            if len(rte_seq) > 0:
                rte_seq_means[wi].append(float(rte_seq.mean()))
                all_rte_vals[wi].append(rte_seq)

    if not seq_names:
        raise SystemExit("No sequences evaluated successfully.")

    seq_ate_means = np.array(seq_ate_means)
    seq_durations = np.array(seq_durations)

    all_ate_flat    = np.concatenate(all_ate_vals)
    ate_global_mean = float(all_ate_flat.mean())
    ate_global_std  = float(all_ate_flat.std())

    # Actual window durations (may differ slightly from target due to rounding)
    rte_actual_s = [
        max(1, round(w * fps_used / patch_size)) * patch_size / fps_used
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
    print(f"\n  {'Metric':<32} {'Global mean':>14}  {'Global std':>12}")
    print(f"  {'-'*62}")
    print(f"  {'ATE (cm)':<32} {ate_global_mean*100:>14.2f}  {ate_global_std*100:>12.2f}")
    for label, g_mean, g_std, seq_arr in rte_stats:
        print(f"  {f'{label} (cm)':<32} {g_mean*100:>14.2f}  {g_std*100:>12.2f}")
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
    plot_summary_bars(
        ate_mean=seq_ate_means.mean(),
        ate_std=seq_ate_means.std(),
        rte_results=[(lbl, g_mean, g_std) for lbl, g_mean, g_std, _ in rte_stats],
        out_dir=args.out_dir,
    )
    plot_ate_vs_duration(seq_durations, seq_ate_means, args.out_dir)

    logger.info("All plots written to %s", args.out_dir)


if __name__ == "__main__":
    main()
