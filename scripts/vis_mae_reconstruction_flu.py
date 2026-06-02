#!/usr/bin/env python3
"""
MAE reconstruction visualizer for FLU-trained models (e.g. vit1d_mae_realdata_ocdetect_flu).

Supports two data sources:
  - "ocd"   : OCDetect wrist-IMU (Android → FLU transform applied on load)
  - any HAR dataset name (wisdm, pamap2, …) : realdata parquet

For each dataset it plots:
  1. Per-dataset figure  : n_examples columns × 6 channel rows + residual row
  2. Per-band MSE bar    : 0-1, 1-3, 3-8, 8-25 Hz comparison across datasets
  3. Combined mean±std   : all datasets side-by-side

Usage:
    python scripts/vis_mae_reconstruction_flu.py \
        --run    vit1d_mae_realdata_ocdetect_flu \
        --data   ocd wisdm pamap2 \
        --out-dir data_vis/flu_recon/
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
from imu_lm.data.OCDetect_dataloader import _list_files, _collect_windows
from imu_lm.utils.helpers import cfg_get

FS       = 50.0
WIN      = 128
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TIME_AXIS = np.arange(WIN) / FS

CH_NAMES_PARQUET = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CH_NAMES_OCD     = ["acc_F", "acc_L", "acc_U", "gyro_F", "gyro_L", "gyro_U"]
CH_COLORS        = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9", "#9b59b6", "#1abc9c"]

BANDS       = [(0, 1), (1, 3), (3, 8), (8, 25)]
BAND_LABELS = ["0–1 Hz", "1–3 Hz", "3–8 Hz", "8–25 Hz"]
BAND_COLORS = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9"]

DS_COLORS = {
    "ocd": "#8e44ad", "wisdm": "#e377c2", "pamap2": "#1f77b4",
    "samosa": "#ff7f0e", "recofit": "#2ca02c", "shoaib": "#d62728",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(run_dir: str, device: torch.device):
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    step = ckpt.get("step", "?")
    print(f"  Checkpoint step: {step}")

    T  = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    ps = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))

    enc = ViT1DEncoder(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    enc.eval()

    dec = ViT1DDecoder(
        embed_dim=enc.embed_dim,
        out_channels=enc.in_channels,
        target_T=T,
        cfg=cfg,
    ).to(device)
    dec.load_state_dict(ckpt["decoder"], strict=True)
    dec.eval()

    print(f"  Encoder: embed_dim={enc.embed_dim}, in_channels={enc.in_channels}, patch_size={ps}")
    return enc, dec, cfg, ps, T


# ── Data loading ──────────────────────────────────────────────────────────────

def load_windows_ocd(data_dir: str, cfg: dict, n: int, rng: np.random.Generator,
                     max_files: int = 10):
    """Load OCDetect windows with FLU transform + z-norm."""
    files = _list_files(data_dir)
    print(f"  OCDetect: {len(files)} files found, loading first {max_files}")
    X_raw, _ = _collect_windows(files, apply_flu_transform=True,
                                 max_files=max_files)  # [M, 6, 128]
    perm = rng.permutation(len(X_raw))
    X_raw = X_raw[perm[:min(n * 5, len(X_raw))]]

    wins = []
    for i in range(len(X_raw)):
        proc = preprocess_window(X_raw[i].T, cfg)   # [128,6] → preprocess → [6,128]
        if proc is not None:
            wins.append(proc.astype(np.float32))
        if len(wins) >= n:
            break

    arr = np.stack(wins)
    print(f"  OCDetect: {len(wins)} windows after preprocessing")
    return arr  # [N, 6, 128]


def load_windows_parquet(parquet: str, dataset: str, cfg: dict, n: int,
                          rng: np.random.Generator):
    """Load windows from master parquet (wisdm, pamap2, …)."""
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
                wins.append(proc.astype(np.float32))
        if len(wins) >= n * 3:
            break
    arr = np.stack(wins)
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    print(f"  {dataset}: {len(wins)} windows available, using {len(idx)}")
    return arr[idx]


# ── Reconstruction ────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct(windows: np.ndarray, enc, dec, device, ps: int) -> np.ndarray:
    N, C, T = windows.shape
    Pt, out = T // ps, []
    for s in range(0, N, 64):
        x    = torch.from_numpy(windows[s:s + 64]).to(device)
        toks, npc = enc.forward_tokens(x)
        mask = torch.zeros(x.shape[0], C * Pt, dtype=torch.bool, device=device)
        out.append(dec(toks, mask, npc, C).cpu().numpy())
    return np.concatenate(out, axis=0)


# ── Frequency band filtering ──────────────────────────────────────────────────

def bandpass(signal: np.ndarray, lo: float, hi: float, fs: float = FS) -> np.ndarray:
    T     = signal.shape[-1]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    F     = np.fft.rfft(signal, axis=-1)
    m     = (freqs >= lo) & (freqs < hi)
    F_out = np.zeros_like(F)
    F_out[..., m] = F[..., m]
    return np.fft.irfft(F_out, n=T, axis=-1)


# ── Plot 1: full signal per dataset ──────────────────────────────────────────

def plot_full_signal(orig, recon, ds_name, n_examples, out_path, ch_names):
    rng  = np.random.default_rng(1)
    idxs = rng.choice(len(orig), size=min(n_examples, len(orig)), replace=False)
    ds_color = DS_COLORS.get(ds_name, "#333333")

    n_rows = len(ch_names) + 1
    fig, axes = plt.subplots(n_rows, n_examples,
                             figsize=(5 * n_examples, 2.2 * n_rows),
                             sharex=True)
    if n_examples == 1:
        axes = axes[:, np.newaxis]

    overall_mse = float(np.mean((recon[idxs] - orig[idxs]) ** 2))
    fig.suptitle(f"{ds_name}  —  original (solid) vs reconstructed (dashed)"
                 f"  |  overall MSE={overall_mse:.5f}",
                 fontsize=11, fontweight="bold", color=ds_color, y=1.01)

    for col, idx in enumerate(idxs):
        o, r = orig[idx], recon[idx]
        mse  = float(np.mean((r - o) ** 2))
        for row, (ch_name, ch_color) in enumerate(zip(ch_names, CH_COLORS)):
            ax = axes[row, col]
            ax.plot(TIME_AXIS, o[row], color=ch_color, lw=1.4, label="original")
            ax.plot(TIME_AXIS, r[row], color=ch_color, lw=1.4, ls="--",
                    alpha=0.85, label="reconstructed")
            ax.grid(alpha=0.2); ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(ch_name, fontsize=8, fontweight="bold", color=ch_color)
            if row == 0:
                ax.set_title(f"Example {col+1}  (MSE={mse:.5f})", fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

        ax_res = axes[-1, col]
        for ci, ch_color in enumerate(CH_COLORS):
            ax_res.plot(TIME_AXIS, r[ci] - o[ci], color=ch_color, lw=0.9, alpha=0.7)
        ax_res.axhline(0, color="k", lw=0.8, ls="--")
        ax_res.grid(alpha=0.2); ax_res.tick_params(labelsize=7)
        ax_res.set_xlabel("Time (s)", fontsize=8)
        if col == 0:
            ax_res.set_ylabel("Residual\n(recon−orig)", fontsize=8, color="#666666")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── Plot 2: per-band MSE bar chart across datasets ────────────────────────────

def plot_band_mse(results: dict, out_path: str):
    datasets = list(results.keys())
    n_bands  = len(BANDS)
    x        = np.arange(n_bands)
    width    = 0.8 / max(len(datasets), 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    for di, ds_name in enumerate(datasets):
        orig  = results[ds_name]["orig"]
        recon = results[ds_name]["recon"]
        mses  = []
        for lo, hi in BANDS:
            o_b = bandpass(orig,  lo, hi)
            r_b = bandpass(recon, lo, hi)
            mses.append(float(np.mean((r_b - o_b) ** 2)))
        color = DS_COLORS.get(ds_name, f"C{di}")
        ax.bar(x + di * width, mses, width=width * 0.9,
               label=ds_name, color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(BAND_LABELS, fontsize=10)
    ax.set_ylabel("MSE (reconstructed vs original)", fontsize=10)
    ax.set_title("Per-frequency-band reconstruction MSE\n"
                 "Lower freq bands weighted more in training loss",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── Plot 3: combined mean±std ─────────────────────────────────────────────────

def plot_combined_mean_std(results: dict, out_path: str, ch_names_map: dict):
    datasets = list(results.keys())
    fig, axes = plt.subplots(6, len(datasets),
                             figsize=(7 * len(datasets), 2.2 * 6),
                             sharex=True)
    if len(datasets) == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("MAE Reconstruction — Mean ± Std over all windows\n"
                 "Solid = original  |  Dashed = reconstructed",
                 fontsize=12, fontweight="bold", y=1.01)

    for di, ds_name in enumerate(datasets):
        orig     = results[ds_name]["orig"]
        recon    = results[ds_name]["recon"]
        ch_names = ch_names_map.get(ds_name, CH_NAMES_PARQUET)
        ds_color = DS_COLORS.get(ds_name, "#333333")
        mse      = float(np.mean((recon - orig) ** 2))

        for row, (ch_name, ch_color) in enumerate(zip(ch_names, CH_COLORS)):
            ax = axes[row, di]
            o_mean = orig [:, row, :].mean(axis=0)
            o_std  = orig [:, row, :].std(axis=0)
            r_mean = recon[:, row, :].mean(axis=0)
            r_std  = recon[:, row, :].std(axis=0)

            ax.plot(TIME_AXIS, o_mean, color=ch_color, lw=1.6, label="original")
            ax.fill_between(TIME_AXIS, o_mean - o_std, o_mean + o_std,
                            color=ch_color, alpha=0.15)
            ax.plot(TIME_AXIS, r_mean, color=ch_color, lw=1.6, ls="--",
                    alpha=0.85, label="reconstructed")
            ax.fill_between(TIME_AXIS, r_mean - r_std, r_mean + r_std,
                            color=ch_color, alpha=0.08)
            ax.grid(alpha=0.2); ax.tick_params(labelsize=7)

            if row == 0:
                ax.set_title(f"{ds_name}  (MSE={mse:.5f}  N={len(orig)})",
                             fontsize=9, fontweight="bold", color=ds_color)
                ax.legend(fontsize=7, loc="upper right")
            if di == 0:
                ax.set_ylabel(ch_name, fontsize=8, fontweight="bold", color=ch_color)
            if row == 5:
                ax.set_xlabel("Time (s)", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",       default="vit1d_mae_realdata_ocdetect_flu")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--data",      nargs="+", default=["ocd", "wisdm", "pamap2"],
                   help="Datasets to visualize: 'ocd' for OCDetect, or any parquet dataset name")
    p.add_argument("--parquet",   default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--ocd-dir",       default="/mnt/nas2/yiquan_data/preprocessed")
    p.add_argument("--max-ocd-files", type=int, default=10,
                   help="Max OCDetect files to load (default 10, ~5k windows)")
    p.add_argument("--n",         type=int, default=200,
                   help="Number of windows per dataset to reconstruct")
    p.add_argument("--examples",  type=int, default=4,
                   help="Example windows shown in per-dataset figure")
    p.add_argument("--out-dir",   default="data_vis/flu_recon")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    rng    = np.random.default_rng(42)
    os.makedirs(args.out_dir, exist_ok=True)

    run_dir = os.path.join(args.runs_root, args.run)
    enc, dec, cfg, ps, T = load_model(run_dir, device)

    results      = {}
    ch_names_map = {}

    for ds in args.data:
        print(f"\nLoading {ds} ...")
        if ds == "ocd":
            wins     = load_windows_ocd(args.ocd_dir, cfg, args.n, rng,
                                        max_files=args.max_ocd_files)
            ch_names = CH_NAMES_OCD
        else:
            wins     = load_windows_parquet(args.parquet, ds, cfg, args.n, rng)
            ch_names = CH_NAMES_PARQUET

        print(f"  Reconstructing {len(wins)} windows ...")
        recon = reconstruct(wins, enc, dec, device, ps)
        results[ds]      = {"orig": wins, "recon": recon}
        ch_names_map[ds] = ch_names

        mse = float(np.mean((recon - wins) ** 2))
        print(f"  Overall MSE: {mse:.6f}")

        # Per-band MSE summary
        for (lo, hi), label in zip(BANDS, BAND_LABELS):
            o_b = bandpass(wins,  lo, hi)
            r_b = bandpass(recon, lo, hi)
            print(f"    {label:>10}: MSE={np.mean((r_b - o_b)**2):.6f}")

        out_path = os.path.join(args.out_dir, f"recon_full_{ds}.png")
        plot_full_signal(wins, recon, ds, args.examples, out_path, ch_names)

    print(f"\nPlotting per-band MSE comparison ...")
    plot_band_mse(results, os.path.join(args.out_dir, "recon_band_mse.png"))

    print(f"\nPlotting combined mean±std ...")
    plot_combined_mean_std(results,
                           os.path.join(args.out_dir, "recon_combined.png"),
                           ch_names_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
