#!/usr/bin/env python3
"""
Reconstruction quality comparison between pretrained ViT1D models.

For each model × dataset: computes per-band MSE on masked patches.
Outputs a bar chart and console table.

Usage:
    python scripts/compare_reconstruction.py \
        --runs  vit1d_mae_realdata_ocdetect_flu_v3 \
                vit1d_mae_realdata_ocdetect_flu_v3_MoE \
        --data  ocd wisdm pamap2 \
        --out-dir data_vis/recon_compare/
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
from imu_lm.objectives.masked_1d import generate_patch_mask, _patch_mask_to_signal_mask
from imu_lm.utils.helpers import cfg_get

FS        = 50.0
IMU_COLS  = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
BANDS     = [(0, 1), (1, 3), (3, 8), (8, 25)]
BAND_LBLS = ["0–1 Hz", "1–3 Hz", "3–8 Hz", "8–25 Hz"]
BAND_WEIGHTS = [4.0, 3.0, 2.0, 1.0]   # same as v3/v3_MoE config


# ── Helpers ───────────────────────────────────────────────────────────────────

def bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    T     = x.shape[-1]
    freqs = np.fft.rfftfreq(T, d=1.0 / FS)
    F     = np.fft.rfft(x, axis=-1)
    m     = (freqs >= lo) & (freqs < hi)
    F_out = np.zeros_like(F)
    F_out[..., m] = F[..., m]
    return np.fft.irfft(F_out, n=T, axis=-1)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(run_dir: str, device: torch.device):
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"),
                      map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    T    = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    ps   = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))

    enc = ViT1DEncoder(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    enc.eval()

    dec = ViT1DDecoder(embed_dim=enc.embed_dim, out_channels=enc.in_channels,
                       target_T=T, cfg=cfg).to(device)
    dec.load_state_dict(ckpt["decoder"], strict=True)
    dec.eval()

    step = ckpt.get("step", "?")
    print(f"  {os.path.basename(run_dir)}  step={step}  freq_moe={enc._use_freq_moe}")
    return enc, dec, cfg, ps, T


# ── Data loading ──────────────────────────────────────────────────────────────

def load_windows_ocd(data_dir, cfg, n, rng, max_files=10):
    files = _list_files(data_dir)
    print(f"  ocd: loading {min(max_files, len(files))} files ...")
    X_raw, _ = _collect_windows(files, apply_flu_transform=True, max_files=max_files)
    perm = rng.permutation(len(X_raw))
    wins = []
    for i in perm:
        proc = preprocess_window(X_raw[i].T, cfg)
        if proc is not None:
            wins.append(proc.astype(np.float32))
        if len(wins) >= n:
            break
    return np.stack(wins)


def load_windows_parquet(parquet, dataset, cfg, n, rng):
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
    return arr[idx]


# ── MAE-style evaluation (same as val/mae_freq_loss) ─────────────────────────

def _bandpass_torch(x: torch.Tensor, lo: float, hi: float, fs: float = 50.0) -> torch.Tensor:
    T     = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=x.device)
    X_f   = torch.fft.rfft(x.float(), dim=-1)
    keep  = (freqs >= lo) & (freqs < hi)
    X_out = torch.zeros_like(X_f)
    X_out[..., keep] = X_f[..., keep]
    return torch.fft.irfft(X_out, n=T, dim=-1).to(x.dtype)


@torch.no_grad()
def eval_mae_loss(windows: np.ndarray, enc, dec, cfg, device,
                  mask_ratio: float = 0.75,
                  strategy: str = "composite",
                  band_weights: list = None) -> dict:
    """Evaluate using the exact same method as val/mae_freq_loss:
      - 75% masking (composite strategy)
      - encoder sees only visible patches
      - MSE computed only on masked patches, per frequency band
      - returns weighted loss + per-band MSEs
    """
    if band_weights is None:
        band_weights = BAND_WEIGHTS

    N, C, T = windows.shape
    ps = enc.patch_size
    Pt = T // ps
    weight_sum = sum(band_weights)

    band_accs  = {lbl: [] for lbl in BAND_LBLS}
    total_accs = []

    for s in range(0, N, 64):
        x = torch.from_numpy(windows[s:s + 64]).to(device)
        B = x.shape[0]

        patch_mask = generate_patch_mask(B, Pt, C, mask_ratio, strategy, device)

        vis_toks, vis_pad = enc(x, mask=patch_mask, return_keep_padding_mask=True)
        recon = dec(vis_toks, patch_mask, Pt, C, visible_padding_mask=vis_pad)

        signal_mask = _patch_mask_to_signal_mask(patch_mask, Pt, C, ps)
        T_r = recon.shape[-1]
        if T_r < signal_mask.shape[-1]:
            signal_mask = signal_mask[:, :, :T_r]
            x = x[:, :, :T_r]

        total = torch.zeros(1, device=device)
        for (lo, hi), lbl, w in zip(BANDS, BAND_LBLS, band_weights):
            x_b = _bandpass_torch(x, lo, hi)
            r_b = _bandpass_torch(recon, lo, hi)
            mse = torch.nn.functional.mse_loss(r_b[signal_mask], x_b[signal_mask])
            band_accs[lbl].append(float(mse.item()))
            total = total + (w / weight_sum) * mse
        total_accs.append(float(total.item()))

    result = {lbl: float(np.mean(band_accs[lbl])) for lbl in BAND_LBLS}
    result["mae_freq_loss"] = float(np.mean(total_accs))
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(all_results, run_names, datasets, out_path):
    """
    all_results[run][ds] = {band_label: mse, ...}
    One row per dataset, grouped bars per band.
    """
    metrics   = BAND_LBLS + ["mae_freq_loss"]
    n_metrics = len(metrics)
    n_ds      = len(datasets)
    colors    = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    short     = [r.replace("vit1d_mae_realdata_ocdetect_flu_", "") for r in run_names]

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    for di, ds in enumerate(datasets):
        ax    = axes[di]
        x     = np.arange(n_metrics)
        width = 0.8 / len(run_names)

        for ri, run in enumerate(run_names):
            mses = all_results[run].get(ds)
            if mses is None:
                continue
            vals = [mses[m] for m in metrics]
            bars = ax.bar(x + ri * width, vals, width * 0.9,
                          label=short[ri], color=colors[ri % len(colors)], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=6, rotation=45)

        ax.set_xticks(x + width * (len(run_names) - 1) / 2)
        ax.set_xticklabels(metrics, fontsize=8, rotation=20, ha="right")
        ax.set_title(ds, fontsize=10, fontweight="bold")
        ax.set_ylabel("MSE" if di == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        if di == 0:
            ax.legend(fontsize=8)

    fig.suptitle("MAE Reconstruction Comparison — same masking as val/mae_freq_loss (lower = better)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def print_table(all_results, run_names, datasets):
    metrics = BAND_LBLS + ["mae_freq_loss"]
    short   = [r.replace("vit1d_mae_realdata_ocdetect_flu_", "") for r in run_names]
    col_w   = 12

    print("\n" + "=" * (14 + col_w * len(run_names) * len(metrics)))
    for ds in datasets:
        print(f"\n── {ds} ──")
        header = f"{'Band':<14}" + "".join(f"{s:<{col_w}}" for s in short)
        print(header)
        print("-" * len(header))
        for m in metrics:
            row = f"{m:<14}"
            for run in run_names:
                v = (all_results[run].get(ds) or {}).get(m)
                row += f"{'N/A' if v is None else f'{v:.6f}':<{col_w}}"
            print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["vit1d_mae_realdata_ocdetect_flu_v3",
                            "vit1d_mae_realdata_ocdetect_flu_v3_MoE"])
    p.add_argument("--runs-root",     default="runs")
    p.add_argument("--data",          nargs="+", default=["ocd", "wisdm", "pamap2"])
    p.add_argument("--parquet",       default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--ocd-dir",       default="/mnt/nas2/yiquan_data/preprocessed")
    p.add_argument("--max-ocd-files", type=int, default=10)
    p.add_argument("--n",             type=int, default=500)
    p.add_argument("--out-dir",       default="data_vis/recon_compare")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    rng    = np.random.default_rng(42)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all models
    models = {}
    cfgs   = {}
    for run in args.runs:
        print(f"\nLoading {run} ...")
        enc, dec, cfg, ps, T = load_model(os.path.join(args.runs_root, run), device)
        models[run] = (enc, dec, ps, T)
        cfgs[run]   = cfg

    base_cfg = cfgs[args.runs[0]]

    # Load data once (shared across models)
    print("\nLoading data ...")
    data = {}
    for ds in args.data:
        if ds == "ocd":
            wins = load_windows_ocd(args.ocd_dir, base_cfg, args.n, rng, args.max_ocd_files)
        else:
            wins = load_windows_parquet(args.parquet, ds, base_cfg, args.n, rng)
        data[ds] = wins
        print(f"  {ds}: {len(wins)} windows")

    # Evaluate per model × dataset using MAE-style masked loss
    all_results = {run: {} for run in args.runs}

    for run in args.runs:
        enc, dec, ps, T = models[run]
        cfg = cfgs[run]
        # Read band weights from model's own config
        bw = list(cfg_get(cfg, ["objective", "freq_mae", "band_weights"],
                          BAND_WEIGHTS) or BAND_WEIGHTS)
        print(f"\n── {run}  (band_weights={bw}) ──")
        for ds, wins in data.items():
            result = eval_mae_loss(wins, enc, dec, cfg, device, band_weights=bw)
            all_results[run][ds] = result
            print(f"  {ds}: mae_freq_loss={result['mae_freq_loss']:.5f}  "
                  + "  ".join(f"{l}={result[l]:.5f}" for l in BAND_LBLS))

    print_table(all_results, args.runs, args.data)

    plot_comparison(all_results, args.runs, args.data,
                    os.path.join(args.out_dir, "recon_comparison.png"))

    # Save CSV
    rows = []
    for run in args.runs:
        for ds in args.data:
            mses = all_results[run].get(ds) or {}
            row  = {"model": run, "dataset": ds}
            row.update(mses)
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, "recon_results.csv"), index=False
    )
    print(f"Saved CSV → {os.path.join(args.out_dir, 'recon_results.csv')}")


if __name__ == "__main__":
    main()
