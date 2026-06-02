#!/usr/bin/env python3
"""
Frequency-domain analysis of MAE reconstruction quality.

Loads pretrained ViT1D MAE model, passes unmasked IMU windows through
encoder+decoder, and compares the reconstructed vs original signal
in the frequency domain (PSD and band energy fractions).

Datasets: wisdm, pamap2 (from parquet)

Usage:
    python scripts/vis_mae_reconstruction_freq.py \
        --run    vit1d_realdata_pretrain \
        --out    data_vis/mae_freq_analysis.png
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

DATASET_COLORS = {
    "wisdm":  "#e377c2",
    "pamap2": "#1f77b4",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def _remap_state_dict(sd: dict) -> dict:
    """Remap old nn.TransformerEncoderLayer key names to current custom module names.

    self_attn.* → attn.*
    linear1.*   → ff.0.*
    linear2.*   → ff.3.*
    """
    out = {}
    for k, v in sd.items():
        k = k.replace(".self_attn.", ".attn.")
        k = k.replace(".linear1.",   ".ff.0.")
        k = k.replace(".linear2.",   ".ff.3.")
        out[k] = v
    return out


def load_model(run_dir: str, device: torch.device):
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    encoder = ViT1DEncoder(cfg).to(device)
    encoder.load_state_dict(_remap_state_dict(ckpt["model"]), strict=True)
    encoder.eval()

    # Build decoder and load weights
    vit_cfg  = cfg_get(cfg, ["vit1d"], {}) or {}
    dec_cfg  = vit_cfg.get("decoder", {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    T, _ = _compute_T(cfg)

    decoder = ViT1DDecoder(
        embed_dim   = encoder.embed_dim,
        out_channels= encoder.in_channels,
        target_T    = T,
        cfg         = cfg,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    decoder.eval()

    return encoder, decoder, cfg, patch_size, T


def _compute_T(cfg) -> tuple[int, int]:
    win_sec = cfg_get(cfg, ["windowing", "window_seconds"], 2.56)
    hop_rat = cfg_get(cfg, ["windowing", "window_hop_ratio"], 0.5)
    T   = int(round(float(win_sec) * FS))
    hop = max(1, int(round(T * float(hop_rat))))
    return T, hop


# ── Data loading ──────────────────────────────────────────────────────────────

def load_windows(parquet: str, dataset: str, cfg, n: int, rng) -> np.ndarray:
    """Load n random preprocessed windows [n, 6, T] for a dataset."""
    needed = ["dataset", "session_id", "timestamp_ns"] + IMU_COLS
    df = pd.read_parquet(parquet, columns=needed,
                         filters=[("dataset", "==", dataset)])
    if df.empty:
        raise ValueError(f"Dataset '{dataset}' not found in parquet.")
    df = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)

    T, hop = _compute_T(cfg)
    windows = []
    for _, grp in df.groupby("session_id"):
        vals = grp[IMU_COLS].values.astype(np.float32)  # [L, 6]
        L = len(vals)
        if L < T:
            continue
        for start in range(0, L - T + 1, hop):
            w = vals[start : start + T]               # [T, 6]
            proc = preprocess_window(w, cfg)
            if proc is None:
                continue
            windows.append(proc.astype(np.float32))   # already [6, T]
        if len(windows) >= n * 3:
            break

    if not windows:
        raise RuntimeError(f"No valid windows loaded for {dataset}")

    arr = np.stack(windows)
    idx = rng.choice(len(arr), size=min(n, len(arr)), replace=False)
    print(f"  {dataset}: {len(windows)} windows available, using {len(idx)}")
    return arr[idx]


# ── Reconstruction ────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct(windows: np.ndarray, encoder, decoder, device,
                patch_size: int, batch_size: int = 64) -> np.ndarray:
    """Pass unmasked windows through encoder+decoder → reconstructed [N, 6, T]."""
    N, C, T = windows.shape
    Pt = T // patch_size
    all_recon = []

    for start in range(0, N, batch_size):
        x = torch.from_numpy(windows[start : start + batch_size]).to(device)
        B = x.shape[0]

        # Encode all patches (no masking)
        tokens, num_patches_per_channel = encoder.forward_tokens(x)

        # Decode: mask = all False (nothing masked)
        mask = torch.zeros(B, C * Pt, dtype=torch.bool, device=device)
        recon = decoder(
            visible_tokens          = tokens,
            mask                    = mask,
            num_patches_per_channel = num_patches_per_channel,
            num_channels            = C,
        )  # [B, C, T]

        all_recon.append(recon.cpu().numpy())

    return np.concatenate(all_recon, axis=0)   # [N, C, T]


# ── Spectral analysis ─────────────────────────────────────────────────────────

def compute_psd(windows: np.ndarray):
    """windows [N, C, T] → (freqs, mean_psd [C, F], std_psd [C, F])."""
    power = np.abs(np.fft.rfft(windows, axis=2)) ** 2
    freqs = np.fft.rfftfreq(windows.shape[2], d=1.0 / FS)
    return freqs, power.mean(axis=0), power.std(axis=0)


def compute_band_fractions(windows: np.ndarray):
    """windows [N, C, T] → mean [n_bands], std [n_bands] fraction of total energy."""
    freqs = np.fft.rfftfreq(windows.shape[2], d=1.0 / FS)
    power = np.abs(np.fft.rfft(windows, axis=2)) ** 2   # [N, C, F]
    power_ch = power.mean(axis=1)                         # [N, F]
    band_e = np.zeros((len(windows), len(BANDS)), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band_e[:, i] = power_ch[:, mask].sum(axis=1)
    total = band_e.sum(axis=1, keepdims=True) + 1e-8
    fracs = band_e / total
    return fracs.mean(axis=0), fracs.std(axis=0)


def band_mse(orig: np.ndarray, recon: np.ndarray) -> dict:
    """Per-band time-domain MSE: bandpass filter orig and recon with FFT zeroing,
    then compute mean squared error on the filtered signal.
    Returns dict band_label → MSE (float).
    """
    T     = orig.shape[2]
    freqs = np.fft.rfftfreq(T, d=1.0 / FS)
    mses  = {}
    for (lo, hi), label in zip(BANDS, BAND_LABELS):
        band_mask = (freqs >= lo) & (freqs < hi)
        if not band_mask.any():
            continue
        # Bandpass-filter both signals
        F_orig  = np.fft.rfft(orig,  axis=2)
        F_recon = np.fft.rfft(recon, axis=2)
        F_orig [:, :, ~band_mask] = 0
        F_recon[:, :, ~band_mask] = 0
        sig_orig  = np.fft.irfft(F_orig,  n=T, axis=2)
        sig_recon = np.fft.irfft(F_recon, n=T, axis=2)
        mse = float(np.mean((sig_recon - sig_orig) ** 2))
        mses[label] = mse
    return mses


# ── Plotting ──────────────────────────────────────────────────────────────────

def smooth(x, w=3):
    return np.convolve(x, np.ones(w) / w, mode="same")


def plot_results(results: dict, out_path: str):
    """
    results = {
        dataset_name: {
            "orig":  windows [N,6,T],
            "recon": windows [N,6,T],
        }
    }
    Layout per dataset (2 rows):
      row 0: acc PSD orig vs recon  | gyro PSD orig vs recon | band fractions orig
      row 1: acc spectral error     | gyro spectral error     | band fractions recon
    """
    datasets = list(results.keys())
    n_ds     = len(datasets)
    freqs    = np.fft.rfftfreq(WIN, d=1.0 / FS)

    fig = plt.figure(figsize=(16, 5.5 * n_ds))
    fig.suptitle("MAE Reconstruction: Frequency-Domain Analysis\n"
                 "Solid = original  |  Dashed = reconstructed",
                 fontsize=13, fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(n_ds * 2, 3, figure=fig, hspace=0.55, wspace=0.35,
                           width_ratios=[2, 2, 1.3])

    for di, ds_name in enumerate(datasets):
        orig  = results[ds_name]["orig"]
        recon = results[ds_name]["recon"]
        color = DATASET_COLORS.get(ds_name, "#333333")

        _, psd_orig_mean,  psd_orig_std  = compute_psd(orig)
        _, psd_recon_mean, psd_recon_std = compute_psd(recon)
        frac_orig_mean,  frac_orig_std   = compute_band_fractions(orig)
        frac_recon_mean, frac_recon_std  = compute_band_fractions(recon)
        mses = band_mse(orig, recon)

        for col, (ch_slice, ch_label) in enumerate([
            (slice(0, 3), "Accelerometer"),
            (slice(3, 6), "Gyroscope"),
        ]):
            # ── row 0: PSD overlay ──────────────────────────────────────────
            ax = fig.add_subplot(gs[di * 2, col])
            m_o = smooth(psd_orig_mean[ch_slice].mean(axis=0))
            s_o = smooth(psd_orig_std[ch_slice].mean(axis=0))
            m_r = smooth(psd_recon_mean[ch_slice].mean(axis=0))
            s_r = smooth(psd_recon_std[ch_slice].mean(axis=0))

            ax.semilogy(freqs, m_o, color=color,    lw=1.8, label="original")
            ax.semilogy(freqs, m_r, color=color,    lw=1.8, ls="--", label="reconstructed")
            ax.fill_between(freqs,
                            np.maximum(m_o - s_o, 1e-10), m_o + s_o,
                            color=color, alpha=0.15)
            ax.fill_between(freqs,
                            np.maximum(m_r - s_r, 1e-10), m_r + s_r,
                            color=color, alpha=0.08)
            for v in [1, 3, 8]:
                ax.axvline(v, color="k", lw=0.5, ls="--", alpha=0.35)
            ax.set_xlim(0, 25); ax.tick_params(labelsize=7); ax.grid(alpha=0.25)
            if di == 0:
                ax.set_title(f"{ch_label} PSD", fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{ds_name}\nPSD", fontsize=8, fontweight="bold",
                              color=color, labelpad=3)
            ax.legend(fontsize=6, loc="upper right")
            ax.set_xticklabels([])

            # ── row 1: per-band MSE (time-domain, band-filtered) ───────────
            ax2 = fig.add_subplot(gs[di * 2 + 1, col])
            T_sig = orig.shape[2]
            freqs_b = np.fft.rfftfreq(T_sig, d=1.0 / FS)
            mse_per_band = []
            for lo, hi in BANDS:
                bm = (freqs_b >= lo) & (freqs_b < hi)
                if not bm.any():
                    mse_per_band.append(0.0)
                    continue
                Fo = np.fft.rfft(orig  [:, ch_slice, :], axis=2).copy()
                Fr = np.fft.rfft(recon [:, ch_slice, :], axis=2).copy()
                Fo[:, :, ~bm] = 0;  Fr[:, :, ~bm] = 0
                so = np.fft.irfft(Fo, n=T_sig, axis=2)
                sr = np.fft.irfft(Fr, n=T_sig, axis=2)
                mse_per_band.append(float(np.mean((sr - so) ** 2)))
            bars2 = ax2.bar(BAND_LABELS, mse_per_band, color=BAND_COLORS,
                            edgecolor="k", lw=0.4)
            for bar, val in zip(bars2, mse_per_band):
                ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=6)
            ax2.tick_params(labelsize=7, axis="x", rotation=20)
            ax2.tick_params(labelsize=7, axis="y")
            ax2.grid(axis="y", alpha=0.3)
            ax2.set_ylabel("MSE", fontsize=7, color=color)
            if di == 0 and col == 0:
                ax2.set_title("Per-band time-domain MSE\n(band-filtered orig vs recon)",
                              fontsize=9, fontweight="bold")

        # ── band fraction comparison (col 2, row 0) ──────────────────────────
        ax_b = fig.add_subplot(gs[di * 2, 2])
        x_pos = np.arange(len(BANDS))
        w     = 0.35
        bars_o = ax_b.bar(x_pos - w/2, frac_orig_mean  * 100, w,
                          yerr=frac_orig_std  * 100, label="original",
                          color=BAND_COLORS, alpha=0.85, edgecolor="k", lw=0.4,
                          error_kw={"elinewidth": 0.8, "capsize": 2})
        bars_r = ax_b.bar(x_pos + w/2, frac_recon_mean * 100, w,
                          yerr=frac_recon_std * 100, label="reconstructed",
                          color=BAND_COLORS, alpha=0.45, edgecolor="k", lw=0.4,
                          hatch="//",
                          error_kw={"elinewidth": 0.8, "capsize": 2})
        ax_b.set_xticks(x_pos); ax_b.set_xticklabels(BAND_LABELS, fontsize=7, rotation=20)
        ax_b.set_ylabel("% energy", fontsize=7)
        ax_b.tick_params(labelsize=7); ax_b.grid(axis="y", alpha=0.3)
        ax_b.legend(fontsize=6, loc="upper right")
        if di == 0:
            ax_b.set_title("Band energy %\n(orig vs recon)", fontsize=9, fontweight="bold")

        # ── per-band overall MSE (both channels combined, col 2 row 1) ─────
        ax_e = fig.add_subplot(gs[di * 2 + 1, 2])
        mse_vals  = list(mses.values())
        mse_names = list(mses.keys())
        bars_e = ax_e.barh(mse_names, mse_vals,
                           color=BAND_COLORS, edgecolor="k", lw=0.4)
        for bar, val in zip(bars_e, mse_vals):
            ax_e.text(val * 1.02, bar.get_y() + bar.get_height() / 2,
                      f"{val:.4f}", va="center", fontsize=7)
        ax_e.set_xlabel("MSE", fontsize=7)
        ax_e.tick_params(labelsize=7); ax_e.grid(axis="x", alpha=0.3)
        if di == 0:
            ax_e.set_title("Per-band MSE\n(all channels combined)",
                           fontsize=9, fontweight="bold")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run",      default="vit1d_realdata_pretrain",
                   help="Run name under runs_root (default: vit1d_realdata_pretrain)")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--parquet",  default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--datasets", nargs="+", default=["wisdm", "pamap2"])
    p.add_argument("--n",        type=int, default=500, help="Windows per dataset")
    p.add_argument("--out",      default="data_vis/mae_freq_analysis.png")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    rng    = np.random.default_rng(42)

    run_dir = os.path.join(args.runs_root, args.run)
    print(f"Loading model from: {run_dir}")
    encoder, decoder, cfg, patch_size, T = load_model(run_dir, device)
    print(f"  Encoder: embed_dim={encoder.embed_dim}, "
          f"layers={len(encoder.encoder.layers)}, patch_size={patch_size}, T={T}")

    results = {}
    for ds in args.datasets:
        print(f"\nLoading {ds} ...")
        windows = load_windows(args.parquet, ds, cfg, args.n, rng)
        print(f"  Reconstructing {len(windows)} windows ...")
        recon = reconstruct(windows, encoder, decoder, device, patch_size)
        results[ds] = {"orig": windows, "recon": recon}

        # Print band energy comparison table
        frac_o, _ = compute_band_fractions(windows)
        frac_r, _ = compute_band_fractions(recon)
        print(f"\n  Band energy fractions — {ds}")
        print(f"  {'Band':>10}  {'Original':>10}  {'Recon':>10}  {'Error':>10}")
        print(f"  {'-'*46}")
        for label, fo, fr in zip(BAND_LABELS, frac_o, frac_r):
            err = abs(fr - fo) / (fo + 1e-8)
            print(f"  {label:>10}  {fo*100:>9.1f}%  {fr*100:>9.1f}%  {err*100:>9.1f}%")

        mses_print = band_mse(windows, recon)
        print(f"\n  Per-band time-domain MSE:")
        for band, mse in mses_print.items():
            print(f"    {band:>10}: {mse:.6f}")

    print(f"\nPlotting ...")
    plot_results(results, args.out)


if __name__ == "__main__":
    main()
