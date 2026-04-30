"""run_local_vel_eval_dataset.py
Dataset-level evaluation for the `local_vel_imuposer_flu` model.

The model predicts pelvis velocity in the wrist FLU frame:
    y: [B, N_patches, 4, 3]  (4 sub-patches × 3-D velocity per patch, m/s)
    decoder outputs: [B, N_patches, 12]  (flattened)

To build a world-frame trajectory from predicted velocities the orientation of
the wrist FLU frame must be rotated back to world.  Two orientation sources are
evaluated:

  GT-R   — GT FK-derived R_world_from_FLU (label-math check, ~0 error)
  Real   — raw real_ori sensor wrist orientation (from the pkl)
  Gyro   — gyro-integrated R_world_from_FLU (realistic inference scenario)

Metrics reported per sequence and globally (ATE, RTE-1s, RTE-2.56s).
Per-sequence trajectory comparison plots are saved to --out-dir.

Usage
-----
    python scripts/run_local_vel_eval_dataset.py \
        --config       configs/base.yaml \
        --model-config configs/vit1d_local_vel_imuposer_flu.yaml \
        --run          vit1d_local_vel_imuposer_flu \
        --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
        [--ckpt        best] \
        [--max-seqs    20]   \
        [--rte-window  1.0 2.56] \
        [--out-dir     data_vis/local_vel_imuposer_flu_eval]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRot
from tqdm import tqdm

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.imuposer_wrist_local_odom_loader_FLU import (
    R_WRIST_FROM_FLU,
    _IMUPOSER_R_WRIST,
    load_session_local_disp_flu,
)
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DOdomDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gyro integration
# ---------------------------------------------------------------------------

def integrate_gyro(gyro_flu: np.ndarray, R0_wfx: np.ndarray, fps: float) -> np.ndarray:
    """R_world_from_FLU at every frame via right-multiply body-frame increments.

    R[t+1] = R[t] @ Rot(gyro_flu[t] * dt)
    """
    N, dt = gyro_flu.shape[0], 1.0 / fps
    R = np.zeros((N, 3, 3), dtype=np.float64)
    R[0] = R0_wfx
    for t in range(N - 1):
        R[t + 1] = R[t] @ ScipyRot.from_rotvec(gyro_flu[t] * dt).as_matrix()
    return R


# ---------------------------------------------------------------------------
# Load real IMU wrist orientation (R_world_from_FLU) from pkl
# ---------------------------------------------------------------------------

def _load_real_ori_wrist(pkl_path: str) -> tuple[np.ndarray, int]:
    """Return (R_world_from_FLU [N,3,3], raw_N) without resampling."""
    with open(pkl_path, "rb") as fh:
        seq_data = pickle.load(fh, encoding="latin1")
    ri = seq_data.get("real_imu_data", {})
    real_ori = np.asarray(ri.get("real_ori"), dtype=np.float64)  # [N, 5, 3, 3]
    R_ws = real_ori[:, _IMUPOSER_R_WRIST]                        # [N, 3, 3]
    # T-pose offset: R_world_from_FLU = R_world_from_sensor @ R_WRIST_FROM_FLU
    R_wfx = np.einsum("nij,jk->nik", R_ws, R_WRIST_FROM_FLU.astype(np.float64))
    return R_wfx, real_ori.shape[0]


# ---------------------------------------------------------------------------
# Inference: predict velocity labels, integrate trajectory
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    pkl_path: str,
    encoder: ViT1DEncoder,
    odom_decoder: ViT1DOdomDecoder,
    cfg: dict,
    device: torch.device,
    target_hz: int = 50,
):
    """Load a single pkl and return per-sub-patch trajectories.

    Returns a dict with keys:
        t_sec     : [n_sub+1]   – time axis (sub-patch boundaries)
        t_mid_sec : [n_sub]     – sub-patch midpoint times
        pos_gt    : [n_sub+1, 3] – GT pelvis world position
        pos_gtR   : [n_sub+1, 3] – pred vel + GT R_world_from_FLU
        pos_realR : [n_sub+1, 3] – pred vel + real IMU wrist orientation
        pos_gyro  : [n_sub+1, 3] – pred vel + gyro-integrated R_world_from_FLU
        err_gtR   : [n_sub+1]   – 3D error GT-R
        err_realR : [n_sub+1]   – 3D error Real
        err_gyro  : [n_sub+1]   – 3D error Gyro
        vel_gt    : [n_sub, 3]  – GT FLU-frame velocity per sub-patch (m/s)
        vel_pred  : [n_sub, 3]  – predicted FLU-frame velocity per sub-patch (m/s)
    or None on failure.
    """
    result = load_session_local_disp_flu(pkl_path, target_hz=target_hz)
    if result is None:
        return None
    X_flu, pos_root, R_X_fw = result          # [N,6], [N,3], [N,3,3] R_X_from_world (f32)
    N = X_flu.shape[0]

    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    Q          = patch_size // 4               # frames per sub-patch
    dt_Q       = Q / target_hz                 # seconds per sub-patch
    T_win, _   = compute_T_and_hop(cfg)
    stats      = PreprocessStats()
    fps        = float(target_hz)

    # ------------------------------------------------------------------
    # Collect predicted velocities over the full sequence (windowed)
    # ------------------------------------------------------------------
    # pred_vel_buf[t] accumulates FLU-frame velocity predictions for frame t (sub-patch midpoint)
    # We store predictions keyed by sub-patch (p_global, k)
    num_patches = N // patch_size
    pred_vel = np.zeros((num_patches, 4, 3), dtype=np.float64)  # [M_total, 4, 3]
    pred_count = np.zeros((num_patches, 4), dtype=np.int32)     # for averaging overlapping windows

    # Slide windows of T_win frames, hop = T_win (non-overlapping for label consistency)
    for w_start in range(0, N - T_win + 1, T_win):
        Xw    = X_flu[w_start : w_start + T_win]  # [T_win, 6]
        Xproc = preprocess_window(Xw, cfg, stats)
        if Xproc is None:
            continue
        x      = torch.from_numpy(Xproc).float().unsqueeze(0).to(device)   # [1, 6, T_win]
        tokens, _ = encoder.forward_tokens(x)
        # pred: [1, Pt, 12]
        pred_flat = odom_decoder(tokens).squeeze(0).cpu().numpy()          # [Pt, 12]
        pred_w    = pred_flat.reshape(pred_flat.shape[0], 4, 3)            # [Pt, 4, 3]
        p_base    = w_start // patch_size
        for p_loc, pv in enumerate(pred_w):
            p_glob = p_base + p_loc
            if p_glob >= num_patches:
                continue
            pred_vel[p_glob] += pv
            pred_count[p_glob] += 1

    # Average over overlapping windows (non-overlapping here so count=1)
    mask = pred_count > 0
    for p in range(num_patches):
        for k in range(4):
            if mask[p, k]:
                pred_vel[p, k] /= pred_count[p, k]

    # ------------------------------------------------------------------
    # Build sub-patch boundary arrays
    # ------------------------------------------------------------------
    n_sub    = num_patches * 4
    t_starts = np.array([p * patch_size + k * Q
                         for p in range(num_patches) for k in range(4)], dtype=np.int32)
    t_mids   = t_starts + Q // 2
    t_bounds = np.append(t_starts, t_starts[-1] + Q)
    t_sec    = t_bounds / fps

    pos_gt = pos_root[t_bounds].astype(np.float64)   # [n_sub+1, 3]

    # ------------------------------------------------------------------
    # GT velocity labels (matching loader formula)
    # ------------------------------------------------------------------
    R_X_fw_f64 = R_X_fw.astype(np.float64)   # R_X_from_world [N,3,3]
    vel_gt = np.zeros((n_sub, 3), dtype=np.float64)
    for i in range(n_sub):
        t_s, t_e, t_mid = int(t_starts[i]), int(t_starts[i]) + Q, int(t_mids[i])
        if t_e >= N:
            break
        v_w = (pos_root[t_e].astype(np.float64) - pos_root[t_s].astype(np.float64)) / dt_Q
        vel_gt[i] = R_X_fw_f64[t_mid] @ v_w

    # GT-R orientations (R_world_from_FLU = R_X_fw^T)
    R_gt_wfx = R_X_fw.transpose(0, 2, 1).astype(np.float64)    # [N, 3, 3]

    # Real IMU wrist orientation
    try:
        R_real_wfx, _ = _load_real_ori_wrist(pkl_path)
        # Clip to N frames (raw and resampled may differ by 1 frame)
        if R_real_wfx.shape[0] > N:
            R_real_wfx = R_real_wfx[:N]
        elif R_real_wfx.shape[0] < N:
            R_real_wfx = np.concatenate([R_real_wfx,
                                          np.tile(R_real_wfx[-1:], (N - R_real_wfx.shape[0], 1, 1))])
    except Exception:
        R_real_wfx = R_gt_wfx  # fallback to GT

    # Gyro-integrated orientation
    gyro_flu   = X_flu[:, 3:6].astype(np.float64)
    R_gyro_wfx = integrate_gyro(gyro_flu, R_gt_wfx[0], fps)

    # ------------------------------------------------------------------
    # Integrate trajectories from predicted velocities
    # ------------------------------------------------------------------
    pred_flat_vel = pred_vel.reshape(n_sub, 3)   # [n_sub, 3] FLU vel

    def _integrate(R_wfx):
        pos = np.zeros((n_sub + 1, 3), dtype=np.float64)
        pos[0] = pos_gt[0]
        for i in range(n_sub):
            tm = min(int(t_mids[i]), N - 1)
            pos[i + 1] = pos[i] + R_wfx[tm] @ pred_flat_vel[i] * dt_Q
        return pos

    pos_gtR   = _integrate(R_gt_wfx)
    pos_realR = _integrate(R_real_wfx)
    pos_gyro  = _integrate(R_gyro_wfx)

    err_gtR   = np.linalg.norm(pos_gt - pos_gtR,   axis=1)
    err_realR = np.linalg.norm(pos_gt - pos_realR,  axis=1)
    err_gyro  = np.linalg.norm(pos_gt - pos_gyro,   axis=1)

    return dict(
        t_sec=t_sec,
        t_mid_sec=t_mids / fps,
        pos_gt=pos_gt,
        pos_gtR=pos_gtR,
        pos_realR=pos_realR,
        pos_gyro=pos_gyro,
        err_gtR=err_gtR,
        err_realR=err_realR,
        err_gyro=err_gyro,
        vel_gt=vel_gt,
        vel_pred=pred_flat_vel,
    )


# ---------------------------------------------------------------------------
# Per-sequence plot
# ---------------------------------------------------------------------------

def plot_sequence(
    name: str,
    data: dict,
    out_dir: str,
):
    """Save a 4-panel trajectory comparison figure for one sequence."""
    t     = data["t_sec"]
    gt    = data["pos_gt"]
    gtR   = data["pos_gtR"]
    realR = data["pos_realR"]
    gyro  = data["pos_gyro"]
    e_gtR   = data["err_gtR"]
    e_realR = data["err_realR"]
    e_gyro  = data["err_gyro"]

    kw_gt   = dict(color="green",  lw=1.5, label="GT")
    kw_gtR  = dict(color="blue",   lw=1.0, ls="--",
                   label=f"Pred+FK-R   ATE={e_gtR.mean():.4f}m")
    kw_real = dict(color="orange", lw=1.0, ls=":",
                   label=f"Pred+RealOri ATE={e_realR.mean():.4f}m")
    kw_gyro = dict(color="red",    lw=1.0, ls="-.",
                   label=f"Pred+Gyro   ATE={e_gyro.mean():.4f}m")

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        f"Local-vel FLU odom evaluation — {name}\n"
        f"sub-patches={len(t)-1}  duration={t[-1]:.1f}s",
        fontsize=10,
    )

    # Panel 1: top-down XZ (Y-up SMPL ground plane)
    ax1 = fig.add_subplot(2, 2, 1, aspect="equal")
    ax1.plot(gt[:,0],   gt[:,2],   **kw_gt)
    ax1.plot(gtR[:,0],  gtR[:,2],  **kw_gtR)
    ax1.plot(realR[:,0],realR[:,2],**kw_real)
    ax1.plot(gyro[:,0], gyro[:,2], **kw_gyro)
    ax1.plot(gt[0,0], gt[0,2], "g^", ms=8, zorder=5, label="Start")
    ax1.set(xlabel="World X (m)", ylabel="World Z (m)", title="Top-down (XZ, Y-up)")
    ax1.legend(fontsize=6.5); ax1.grid(alpha=0.3)

    # Panel 2: height (Y) over time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, gt[:,1],   **kw_gt)
    ax2.plot(t, gtR[:,1],  **kw_gtR)
    ax2.plot(t, realR[:,1],**kw_real)
    ax2.plot(t, gyro[:,1], **kw_gyro)
    ax2.set(xlabel="Time (s)", ylabel="World Y (m)", title="Height over time")
    ax2.legend(fontsize=6.5); ax2.grid(alpha=0.3)

    # Panel 3: 3D error curves
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, e_gtR,   color="blue",   lw=1.0, label=f"Pred+FK-R   (label check)")
    ax3.plot(t, e_realR, color="orange", lw=1.0, label="Pred+RealOri")
    ax3.plot(t, e_gyro,  color="red",    lw=1.0, label="Pred+Gyro")
    ax3.set(xlabel="Time (s)", ylabel="3D error (m)", title="Cumulative position error")
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    # Panel 4: ATE bar summary
    ax4 = fig.add_subplot(2, 2, 4)
    labels  = ["Pred+FK-R\n(label chk)", "Pred+RealOri", "Pred+Gyro"]
    vals    = [e_gtR.mean()*100, e_realR.mean()*100, e_gyro.mean()*100]
    colors  = ["blue", "orange", "red"]
    bars    = ax4.bar(labels, vals, color=colors, alpha=0.75, width=0.4)
    for bar, v in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.03,
                 f"{v:.2f} cm", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set(ylabel="Mean ATE (cm)", title="ATE by orientation source")
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "trajectories", f"{name}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-sequence velocity comparison plot
# ---------------------------------------------------------------------------

def plot_velocity(name: str, data: dict, out_dir: str):
    """Save a 4-panel FLU-frame velocity comparison (GT vs predicted)."""
    t_mid  = data["t_mid_sec"]         # [n_sub]
    v_gt   = data["vel_gt"]            # [n_sub, 3]  FLU: (F, L, U)
    v_pred = data["vel_pred"]          # [n_sub, 3]

    speed_gt   = np.linalg.norm(v_gt,   axis=1)
    speed_pred = np.linalg.norm(v_pred, axis=1)
    vel_mae    = np.abs(v_gt - v_pred).mean(axis=0)   # per-component
    speed_mae  = np.abs(speed_gt - speed_pred).mean()

    components = [("F (forward)", 0, "tab:blue"),
                  ("L (left)",    1, "tab:orange"),
                  ("U (up)",      2, "tab:green")]

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        f"FLU-frame velocity: GT vs predicted — {name}\n"
        f"Component MAE  F={vel_mae[0]*100:.2f} cm/s  "
        f"L={vel_mae[1]*100:.2f} cm/s  U={vel_mae[2]*100:.2f} cm/s",
        fontsize=10,
    )

    for ax, (label, idx, color) in zip(axes[:3], components):
        ax.plot(t_mid, v_gt[:,idx],   color=color, lw=1.2, label="GT")
        ax.plot(t_mid, v_pred[:,idx], color=color, lw=1.0, ls="--", alpha=0.8,
                label=f"Pred  MAE={vel_mae[idx]*100:.2f} cm/s")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set(ylabel=f"v_{label} (m/s)")
        ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)

    axes[3].plot(t_mid, speed_gt,   color="purple", lw=1.2, label="GT speed")
    axes[3].plot(t_mid, speed_pred, color="purple", lw=1.0, ls="--", alpha=0.8,
                 label=f"Pred speed  MAE={speed_mae*100:.2f} cm/s")
    axes[3].set(xlabel="Time (s)", ylabel="‖v_FLU‖ (m/s)", title="Speed magnitude")
    axes[3].legend(fontsize=7.5, loc="upper right"); axes[3].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "velocities", f"{name}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_summary_bars(
    ate_means: dict,   # {"GT-R": arr, "Real": arr, "Gyro": arr}
    out_dir: str,
):
    keys   = list(ate_means.keys())
    colors = ["blue", "orange", "red"]
    means  = [np.nanmean(ate_means[k]) * 100 for k in keys]
    stds   = [np.nanstd(ate_means[k])  * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(keys, means, yerr=stds, color=colors, capsize=6, width=0.45, alpha=0.8,
                  error_kw=dict(elinewidth=1.5))
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + max(means)*0.03,
                f"{v:.2f} cm", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set(ylabel="Mean ATE (cm)",
           title="Dataset-level ATE by orientation source\n(mean ± std across sequences)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "summary_ate_by_ori.png"))


def plot_ate_distribution(ate_means: dict, out_dir: str):
    keys   = list(ate_means.keys())
    colors = ["blue", "orange", "red"]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5), sharey=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, key, color in zip(axes, keys, colors):
        vals = ate_means[key] * 100
        ax.hist(vals, bins=20, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(float(np.nanmean(vals)), color="black", ls="--",
                   label=f"Mean={np.nanmean(vals):.1f} cm")
        ax.set(xlabel="Mean ATE (cm)", ylabel="Count", title=f"{key}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Per-sequence ATE distribution by orientation source", fontsize=11)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "ate_distribution.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run",          required=True)
    ap.add_argument("--test-dir",     required=True)
    ap.add_argument("--ckpt",         default="best")
    ap.add_argument("--target-hz",    type=int, default=50)
    ap.add_argument("--max-seqs",     type=int, default=0, help="0 = all")
    ap.add_argument("--rte-window",   type=float, nargs="+", default=[1.0, 2.56])
    ap.add_argument("--out-dir",      default="data_vis/local_vel_imuposer_flu_eval")
    args = ap.parse_args()

    base_cfg  = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg       = deep_update(base_cfg, model_cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "velocities"), exist_ok=True)

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

    T_win, _   = compute_T_and_hop(ckpt_cfg)
    vit_cfg    = cfg_get(ckpt_cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))

    encoder = ViT1DEncoder(ckpt_cfg).to(device)
    encoder.load_state_dict(state["model"])
    encoder.eval()

    odom_decoder = ViT1DOdomDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels,
        cfg=ckpt_cfg,
        out_dim=12,   # 4 sub-patches x 3-D FLU velocity
    ).to(device)
    odom_decoder.load_state_dict(state["odom_decoder"])
    odom_decoder.eval()

    # ---- Gather pkl files -------------------------------------------------
    pkl_files = sorted(glob.glob(os.path.join(args.test_dir, "*.pkl")))
    if not pkl_files:
        raise SystemExit(f"No pkl files found in {args.test_dir}")
    if args.max_seqs > 0:
        pkl_files = pkl_files[:args.max_seqs]
    logger.info("Evaluating %d sequences", len(pkl_files))

    # ---- Per-sequence loop ------------------------------------------------
    ate_collections = {"GT-R (chk)": [], "Real ori": [], "Gyro": []}
    seq_failed = []

    for pkl_path in tqdm(pkl_files, desc="Sequences", unit="seq"):
        name = os.path.basename(pkl_path).replace(".pkl", "")
        try:
            data = run_inference(
                pkl_path, encoder, odom_decoder, cfg, device, args.target_hz
            )
        except Exception as exc:
            logger.warning("FAILED %s: %s", name, exc)
            seq_failed.append(name)
            continue
        if data is None:
            seq_failed.append(name)
            continue

        ate_collections["GT-R (chk)"].append(float(data["err_gtR"].mean()))
        ate_collections["Real ori"].append(float(data["err_realR"].mean()))
        ate_collections["Gyro"].append(float(data["err_gyro"].mean()))

        plot_sequence(name, data, args.out_dir)
        plot_velocity(name, data, args.out_dir)

    if not ate_collections["Gyro"]:
        raise SystemExit("No sequences evaluated successfully.")

    ate_np = {k: np.array(v) for k, v in ate_collections.items()}

    # ---- Console summary --------------------------------------------------
    hdr = "=" * 70
    print(f"\n{hdr}")
    print(f"  local_vel_imuposer_flu — dataset ATE summary")
    print(f"  {len(ate_np['Gyro'])} sequences  |  ckpt: {ckpt_path}")
    print(hdr)
    print(f"\n  {'Ori source':<24} {'Mean ATE':>12}  {'Std':>10}  {'Min':>10}  {'Max':>10}")
    print(f"  {'-'*68}")
    for k, arr in ate_np.items():
        print(f"  {k:<24} {arr.mean()*100:>10.2f}cm  {arr.std()*100:>8.2f}cm  "
              f"{arr.min()*100:>8.2f}cm  {arr.max()*100:>8.2f}cm")
    if seq_failed:
        print(f"\n  Skipped {len(seq_failed)} sequences: {seq_failed[:5]}")
    print()

    # ---- Summary plots ----------------------------------------------------
    plot_summary_bars(ate_np, args.out_dir)
    plot_ate_distribution(ate_np, args.out_dir)
    logger.info("Done. Results in %s", args.out_dir)


if __name__ == "__main__":
    main()
