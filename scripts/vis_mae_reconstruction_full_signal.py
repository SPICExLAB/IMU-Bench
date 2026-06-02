#!/usr/bin/env python3
"""
Full reconstructed vs original signal comparison (no band filtering).

For each dataset (wisdm, pamap2):
  - Pick N example windows
  - Show all 6 channels: original (solid) vs reconstructed (dashed)
  - Bottom row: residual (recon - orig)

Saves one PNG per dataset + one combined figure.

Usage:
    python scripts/vis_mae_reconstruction_full_signal.py \
        --run  vit1d_realdata_pretrain \
        --out-dir data_vis/
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DDecoder
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.utils.helpers import cfg_get

FS       = 50.0
WIN      = 128
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CH_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CH_COLORS = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9", "#9b59b6", "#1abc9c"]

DATASET_COLORS = {"wisdm": "#e377c2", "pamap2": "#1f77b4"}
TIME_AXIS = np.arange(WIN) / FS


# ── Model loading ─────────────────────────────────────────────────────────────

def _remap(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace(".self_attn.", ".attn.").replace(".linear1.", ".ff.0.").replace(".linear2.", ".ff.3.")
        out[k] = v
    return out


def load_model(run_dir, device):
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"),
                      map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    T    = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    ps   = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))

    enc = ViT1DEncoder(cfg).to(device)
    enc.load_state_dict(_remap(ckpt["model"]), strict=True)
    enc.eval()

    dec = ViT1DDecoder(embed_dim=enc.embed_dim,
                       out_channels=enc.in_channels,
                       target_T=T, cfg=cfg).to(device)
    dec.load_state_dict(ckpt["decoder"], strict=True)
    dec.eval()

    return enc, dec, cfg, ps, T


# ── Data ─────────────────────────────────────────────────────────────────────

def load_windows(parquet, dataset, cfg, n, rng):
    needed = ["dataset", "session_id", "timestamp_ns"] + IMU_COLS
    df = pd.read_parquet(parquet, columns=needed,
                         filters=[("dataset", "==", dataset)])
    df = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)
    T   = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    hop = max(1, T // 2)
    wins = []
    for _, grp in df.groupby("session_id"):
        vals = grp[IMU_COLS].values.astype(np.float32)
        for start in range(0, len(vals) - T + 1, hop):
            proc = preprocess_window(vals[start:start + T], cfg)
            if proc is not None:
                wins.append(proc.astype(np.float32))   # [6, T]
        if len(wins) >= n * 3:
            break
    arr = np.stack(wins)
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    print(f"  {dataset}: {len(wins)} windows available, sampling {len(idx)}")
    return arr[idx]


@torch.no_grad()
def reconstruct(windows, enc, dec, device, ps):
    N, C, T = windows.shape
    Pt, out = T // ps, []
    for s in range(0, N, 64):
        x    = torch.from_numpy(windows[s:s+64]).to(device)
        toks, npc = enc.forward_tokens(x)
        mask = torch.zeros(x.shape[0], C * Pt, dtype=torch.bool, device=device)
        out.append(dec(toks, mask, npc, C).cpu().numpy())
    return np.concatenate(out, axis=0)


# ── Per-dataset figure ────────────────────────────────────────────────────────

def plot_dataset(orig, recon, dataset, n_examples, out_path):
    """
    Layout: n_examples columns × 7 rows
      rows 0-5 : channels acc_x … gyro_z  (original solid, recon dashed)
      row  6   : residual (recon - orig), mean over channels
    """
    rng  = np.random.default_rng(1)
    idxs = rng.choice(len(orig), size=min(n_examples, len(orig)), replace=False)
    ds_color = DATASET_COLORS.get(dataset, "#333333")

    n_rows = len(CH_NAMES) + 1   # 6 channels + 1 residual
    fig, axes = plt.subplots(n_rows, n_examples,
                             figsize=(5 * n_examples, 2.2 * n_rows),
                             sharex=True)
    if n_examples == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f"Dataset: {dataset}  —  Full signal: original vs reconstructed",
                 fontsize=12, fontweight="bold", color=ds_color, y=1.01)

    overall_mse = float(np.mean((recon[idxs] - orig[idxs]) ** 2))

    for col, idx in enumerate(idxs):
        o = orig [idx]   # [6, T]
        r = recon[idx]   # [6, T]
        mse = float(np.mean((r - o) ** 2))

        # Channel rows
        for row, (ch_name, ch_color) in enumerate(zip(CH_NAMES, CH_COLORS)):
            ax = axes[row, col]
            ax.plot(TIME_AXIS, o[row], color=ch_color, lw=1.4, label="original")
            ax.plot(TIME_AXIS, r[row], color=ch_color, lw=1.4, ls="--",
                    alpha=0.85, label="reconstructed")
            ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

            if col == 0:
                ax.set_ylabel(ch_name, fontsize=8, fontweight="bold", color=ch_color)
            if row == 0:
                ax.set_title(f"Example {col+1}  (MSE={mse:.5f})",
                             fontsize=8, color="#444444")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

        # Residual row
        ax_res = axes[-1, col]
        residual = r - o    # [6, T]
        for ci, ch_color in enumerate(CH_COLORS):
            ax_res.plot(TIME_AXIS, residual[ci], color=ch_color, lw=0.9, alpha=0.7)
        ax_res.axhline(0, color="k", lw=0.8, ls="--")
        ax_res.grid(alpha=0.2); ax_res.tick_params(labelsize=7)
        ax_res.set_xlabel("Time (s)", fontsize=8)
        if col == 0:
            ax_res.set_ylabel("Residual\n(recon − orig)", fontsize=8,
                              fontweight="bold", color="#666666")

    fig.text(0.5, -0.01,
             f"Overall MSE ({n_examples} examples): {overall_mse:.6f}",
             ha="center", fontsize=9, color="#333333")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── Combined figure (mean ± std over all windows) ────────────────────────────

def plot_combined(results, out_path):
    """
    For each dataset: mean ± std of original and reconstructed across all N windows.
    Layout: 6 rows (channels) × 2 columns (datasets).
    """
    datasets = list(results.keys())
    fig, axes = plt.subplots(len(CH_NAMES), len(datasets),
                             figsize=(7 * len(datasets), 2.2 * len(CH_NAMES)),
                             sharex=True)
    if len(datasets) == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("MAE Reconstruction — Mean ± Std over all windows\n"
                 "Solid = original   |   Dashed = reconstructed",
                 fontsize=12, fontweight="bold", y=1.01)

    for di, ds in enumerate(datasets):
        orig  = results[ds]["orig"]    # [N, 6, T]
        recon = results[ds]["recon"]   # [N, 6, T]
        ds_color = DATASET_COLORS.get(ds, "#333333")
        mse = float(np.mean((recon - orig) ** 2))

        for row, (ch_name, ch_color) in enumerate(zip(CH_NAMES, CH_COLORS)):
            ax = axes[row, di]

            o_mean = orig [:, row, :].mean(axis=0)
            o_std  = orig [:, row, :].std (axis=0)
            r_mean = recon[:, row, :].mean(axis=0)
            r_std  = recon[:, row, :].std (axis=0)

            ax.plot(TIME_AXIS, o_mean, color=ch_color, lw=1.6, label="original")
            ax.fill_between(TIME_AXIS, o_mean - o_std, o_mean + o_std,
                            color=ch_color, alpha=0.15)
            ax.plot(TIME_AXIS, r_mean, color=ch_color, lw=1.6, ls="--",
                    alpha=0.85, label="reconstructed")
            ax.fill_between(TIME_AXIS, r_mean - r_std, r_mean + r_std,
                            color=ch_color, alpha=0.08)

            ax.grid(alpha=0.2); ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(f"{ds}  (MSE={mse:.5f}  N={len(orig)})",
                             fontsize=9, fontweight="bold", color=ds_color)
                ax.legend(fontsize=7, loc="upper right")
            if di == 0:
                ax.set_ylabel(ch_name, fontsize=8, fontweight="bold",
                              color=ch_color)
            if row == len(CH_NAMES) - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",       default="vit1d_realdata_pretrain")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--parquet",   default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--datasets",  nargs="+", default=["wisdm", "pamap2"])
    p.add_argument("--n",         type=int, default=200, help="Windows to reconstruct")
    p.add_argument("--examples",  type=int, default=4,
                   help="Example windows shown in per-dataset figure")
    p.add_argument("--out-dir",   default="data_vis")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    rng    = np.random.default_rng(42)

    run_dir = os.path.join(args.runs_root, args.run)
    print(f"Loading model from {run_dir} ...")
    enc, dec, cfg, ps, T = load_model(run_dir, device)

    results = {}
    for ds in args.datasets:
        print(f"\nLoading {ds} ...")
        wins  = load_windows(args.parquet, ds, cfg, args.n, rng)
        print(f"  Reconstructing {len(wins)} windows ...")
        recon = reconstruct(wins, enc, dec, device, ps)
        results[ds] = {"orig": wins, "recon": recon}

        mse = float(np.mean((recon - wins) ** 2))
        print(f"  Overall MSE: {mse:.6f}")

        out_path = os.path.join(args.out_dir, f"mae_full_signal_{ds}.png")
        print(f"  Plotting per-dataset figure ...")
        plot_dataset(wins, recon, ds, args.examples, out_path)

    print(f"\nPlotting combined mean±std figure ...")
    combined_path = os.path.join(args.out_dir, "mae_full_signal_combined.png")
    plot_combined(results, combined_path)


if __name__ == "__main__":
    main()
