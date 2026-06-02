#!/usr/bin/env python3
"""
Time-domain signal comparison: original vs MAE-reconstructed, per frequency band.

For each dataset (wisdm, pamap2):
  - Pick one representative window
  - Bandpass filter it to each band (0-1, 1-3, 3-8, 8-25 Hz)
  - Plot original (solid) vs reconstructed (dashed) for acc and gyro

Usage:
    python scripts/vis_mae_reconstruction_signal.py \
        --run  vit1d_realdata_pretrain \
        --out  data_vis/mae_signal_comparison.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

BANDS       = [(0, 1), (1, 3), (3, 8), (8, 25)]
BAND_LABELS = ["0–1 Hz", "1–3 Hz", "3–8 Hz", "8–25 Hz"]
BAND_COLORS = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9"]

DATASET_COLORS = {"wisdm": "#e377c2", "pamap2": "#1f77b4"}

TIME_AXIS = np.arange(WIN) / FS   # seconds


# ── Model loading (same as vis_mae_reconstruction_freq.py) ───────────────────

def _remap(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace(".self_attn.", ".attn.").replace(".linear1.", ".ff.0.").replace(".linear2.", ".ff.3.")
        out[k] = v
    return out


def _compute_T(cfg):
    win_sec = cfg_get(cfg, ["windowing", "window_seconds"], 2.56)
    return int(round(float(win_sec) * FS))


def load_model(run_dir, device):
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"),
                      map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    T    = _compute_T(cfg)
    vit_cfg    = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))

    encoder = ViT1DEncoder(cfg).to(device)
    encoder.load_state_dict(_remap(ckpt["model"]), strict=True)
    encoder.eval()

    decoder = ViT1DDecoder(embed_dim=encoder.embed_dim,
                           out_channels=encoder.in_channels,
                           target_T=T, cfg=cfg).to(device)
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    decoder.eval()

    return encoder, decoder, cfg, patch_size, T


# ── Data ─────────────────────────────────────────────────────────────────────

def load_windows(parquet, dataset, cfg, n, rng):
    needed = ["dataset", "session_id", "timestamp_ns"] + IMU_COLS
    df = pd.read_parquet(parquet, columns=needed,
                         filters=[("dataset", "==", dataset)])
    df = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)
    T    = _compute_T(cfg)
    hop  = max(1, T // 2)
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
    print(f"  {dataset}: {len(wins)} windows, using {len(idx)}")
    return arr[idx]


@torch.no_grad()
def reconstruct(windows, encoder, decoder, device, patch_size):
    N, C, T = windows.shape
    Pt = T // patch_size
    out = []
    BS  = 64
    for s in range(0, N, BS):
        x    = torch.from_numpy(windows[s:s+BS]).to(device)
        B    = x.shape[0]
        toks, npc = encoder.forward_tokens(x)
        mask = torch.zeros(B, C * Pt, dtype=torch.bool, device=device)
        r    = decoder(toks, mask, npc, C)
        out.append(r.cpu().numpy())
    return np.concatenate(out, axis=0)


# ── Band filtering ────────────────────────────────────────────────────────────

def bandpass(signal: np.ndarray, lo: float, hi: float, fs: float = FS) -> np.ndarray:
    """signal [C, T] → band-filtered [C, T] via FFT zeroing."""
    T     = signal.shape[-1]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    F     = np.fft.rfft(signal, axis=-1)
    mask  = (freqs >= lo) & (freqs < hi)
    F_out = np.zeros_like(F)
    F_out[..., mask] = F[..., mask]
    return np.fft.irfft(F_out, n=T, axis=-1)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_signal_comparison(results: dict, out_path: str, n_examples: int = 3):
    """
    Layout:
      Rows   : freq bands (0-1, 1-3, 3-8, 8-25 Hz)
      Columns: datasets × sensor groups (acc / gyro)
    Each cell shows n_examples overlaid traces of orig (solid) vs recon (dashed).
    """
    datasets  = list(results.keys())
    n_ds      = len(datasets)
    n_bands   = len(BANDS)
    n_cols    = n_ds * 2      # acc + gyro per dataset

    fig, axes = plt.subplots(
        n_bands, n_cols,
        figsize=(4.5 * n_cols, 2.8 * n_bands),
        sharex=True,
    )
    fig.suptitle(
        "MAE Reconstruction — Original (solid) vs Reconstructed (dashed)\n"
        "per frequency band and sensor group",
        fontsize=13, fontweight="bold", y=1.01,
    )

    rng = np.random.default_rng(0)

    for di, ds_name in enumerate(datasets):
        orig  = results[ds_name]["orig"]   # [N, 6, T]
        recon = results[ds_name]["recon"]  # [N, 6, T]
        ds_color = DATASET_COLORS.get(ds_name, "#333333")

        # Pick a few example indices
        idxs = rng.choice(len(orig), size=min(n_examples, len(orig)), replace=False)

        for bi, ((lo, hi), blabel, bcolor) in enumerate(
                zip(BANDS, BAND_LABELS, BAND_COLORS)):
            for si, (ch_slice, sensor_name) in enumerate([
                (slice(0, 3), "Acc"),
                (slice(3, 6), "Gyro"),
            ]):
                col = di * 2 + si
                ax  = axes[bi, col]

                for k, idx in enumerate(idxs):
                    alpha = 0.9 - k * 0.2
                    o_filt = bandpass(orig [idx, ch_slice, :], lo, hi)  # [3, T]
                    r_filt = bandpass(recon[idx, ch_slice, :], lo, hi)  # [3, T]
                    # mean over the 3 channels for cleaner display
                    ax.plot(TIME_AXIS, o_filt.mean(axis=0),
                            color=bcolor, lw=1.2, alpha=alpha,
                            label="original" if k == 0 else None)
                    ax.plot(TIME_AXIS, r_filt.mean(axis=0),
                            color=bcolor, lw=1.2, alpha=alpha,
                            ls="--",
                            label="reconstructed" if k == 0 else None)

                # per-cell MSE annotation
                o_all = bandpass(orig [idxs][:, ch_slice, :], lo, hi)
                r_all = bandpass(recon[idxs][:, ch_slice, :], lo, hi)
                mse   = float(np.mean((r_all - o_all) ** 2))
                ax.text(0.98, 0.96, f"MSE={mse:.5f}",
                        transform=ax.transAxes, fontsize=6.5,
                        ha="right", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

                ax.grid(alpha=0.25); ax.tick_params(labelsize=7)

                # Column header (top row only)
                if bi == 0:
                    ax.set_title(f"{ds_name} — {sensor_name}",
                                 fontsize=9, fontweight="bold", color=ds_color)

                # Row label (first column of each row)
                if col == 0:
                    ax.set_ylabel(f"{blabel}\n(band-filtered)",
                                  fontsize=8, fontweight="bold", color=bcolor)

                # X-axis label (bottom row only)
                if bi == n_bands - 1:
                    ax.set_xlabel("Time (s)", fontsize=8)

                # Legend (top-left cell only)
                if bi == 0 and col == 0:
                    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",       default="vit1d_realdata_pretrain")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--parquet",   default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--datasets",  nargs="+", default=["wisdm", "pamap2"])
    p.add_argument("--n",         type=int, default=200, help="Windows to reconstruct")
    p.add_argument("--examples",  type=int, default=3,
                   help="Example traces to overlay per cell (default 3)")
    p.add_argument("--out",       default="data_vis/mae_signal_comparison.png")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    rng    = np.random.default_rng(42)

    run_dir = os.path.join(args.runs_root, args.run)
    print(f"Loading model from {run_dir} ...")
    encoder, decoder, cfg, patch_size, T = load_model(run_dir, device)
    print(f"  embed_dim={encoder.embed_dim}, patch_size={patch_size}, T={T}")

    results = {}
    for ds in args.datasets:
        print(f"\nLoading {ds} ...")
        wins  = load_windows(args.parquet, ds, cfg, args.n, rng)
        print(f"  Reconstructing ...")
        recon = reconstruct(wins, encoder, decoder, device, patch_size)
        results[ds] = {"orig": wins, "recon": recon}

        # Print per-band MSE summary
        print(f"  Per-band MSE — {ds}")
        for (lo, hi), label in zip(BANDS, BAND_LABELS):
            o = bandpass(wins,  lo, hi)
            r = bandpass(recon, lo, hi)
            print(f"    {label:>10}: MSE={np.mean((r-o)**2):.6f}")

    print(f"\nPlotting ...")
    plot_signal_comparison(results, args.out, n_examples=args.examples)


if __name__ == "__main__":
    main()
