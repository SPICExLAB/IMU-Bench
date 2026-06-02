#!/usr/bin/env python3
"""
Compare linear probing (SAMOSA) + MAE reconstruction quality
for multiple pretrained ViT1D models at a given step.

Usage:
    # Compare all 4 models at step 80000 (max common step)
    python scripts/compare_all_models.py --step 80000

    # Compare at latest best.pt
    python scripts/compare_all_models.py

    # Custom runs / step
    python scripts/compare_all_models.py --step 5000 \
        --runs vit1d_mae_realdata_ocdetect_flu_v3 \
               vit1d_mae_realdata_ocdetect_flu_v3_MoE
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DDecoder
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.data.OCDetect_dataloader import _list_files, _collect_windows
from imu_lm.objectives.masked_1d import generate_patch_mask, _patch_mask_to_signal_mask
from imu_lm.utils.helpers import cfg_get

FS       = 50.0
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
BANDS     = [(0, 1), (1, 3), (3, 8), (8, 25)]
BAND_LBLS = ["0-1Hz", "1-3Hz", "3-8Hz", "8-25Hz"]
BAND_WEIGHTS_DEFAULT = [4.0, 3.0, 2.0, 1.0]

DEFAULT_RUNS = [
    "vit1d_mae_realdata_ocdetect_flu_v3",
    "vit1d_mae_realdata_ocdetect_flu_v3_MoE",
    "vit1d_mae_realdata_ocdetect_flu_v4_MoE",
    "vit1d_mae_realdata_ocdetect_flu_v5_MoE",
]

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(run_dir: str, step: int, device: torch.device):
    if step is not None:
        ckpt_path = os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt")
    else:
        ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    T    = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))

    enc = ViT1DEncoder(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    dec = ViT1DDecoder(enc.embed_dim, enc.in_channels, T, cfg).to(device)
    dec.load_state_dict(ckpt["decoder"], strict=True)
    dec.eval()

    actual_step = ckpt.get("step", "?")
    bw = list(cfg_get(cfg, ["objective", "freq_mae", "band_weights"],
                      BAND_WEIGHTS_DEFAULT) or BAND_WEIGHTS_DEFAULT)
    short = os.path.basename(run_dir).replace("vit1d_mae_realdata_ocdetect_flu_", "")
    print(f"  {short}  step={actual_step}  bert_mask={enc._use_bert_mask}  freq_moe={enc._use_freq_moe}")
    return enc, dec, cfg, bw, short


# ── Bandpass (torch) ──────────────────────────────────────────────────────────

def bandpass_t(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    T     = x.shape[-1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / FS, device=x.device)
    X_f   = torch.fft.rfft(x.float(), dim=-1)
    keep  = (freqs >= lo) & (freqs < hi)
    X_out = torch.zeros_like(X_f)
    X_out[..., keep] = X_f[..., keep]
    return torch.fft.irfft(X_out, n=T, dim=-1).to(x.dtype)


# ── Full reconstruction (no masking) ─────────────────────────────────────────

@torch.no_grad()
def eval_full_reconstruction(windows: np.ndarray, enc, dec, device) -> dict:
    """Encoder sees ALL tokens (no masking). MSE on full signal per band."""
    N, C, T = windows.shape
    ps = enc.patch_size
    Pt = T // ps
    band_acc  = {l: [] for l in BAND_LBLS}
    total_acc = []

    for s in range(0, N, 32):
        x = torch.from_numpy(windows[s:s + 32]).to(device)
        B = x.shape[0]
        toks, npc = enc.forward_tokens(x)               # all 48 tokens, no masking
        empty_mask = torch.zeros(B, C * Pt, dtype=torch.bool, device=device)
        recon = dec(toks, empty_mask, npc, C)            # [B, C, T]

        tot = 0.0
        for (lo, hi), lbl in zip(BANDS, BAND_LBLS):
            xb  = bandpass_t(x,     lo, hi)
            rb  = bandpass_t(recon, lo, hi)
            mse = float(F.mse_loss(rb, xb))
            band_acc[lbl].append(mse)
            tot += mse
        total_acc.append(float(F.mse_loss(recon, x)))   # overall MSE on full signal

    return {l: float(np.mean(band_acc[l])) for l in BAND_LBLS} | \
           {"overall_mse": float(np.mean(total_acc))}


# ── Masked reconstruction (same as val/mae_freq_loss) ─────────────────────────

@torch.no_grad()
def eval_reconstruction(windows: np.ndarray, enc, dec, bw: list, device,
                         mask_ratio: float = 0.75) -> dict:
    N, C, T = windows.shape
    ps = enc.patch_size
    Pt = T // ps
    ws = sum(bw)
    band_acc  = {l: [] for l in BAND_LBLS}
    total_acc = []

    for s in range(0, N, 32):
        x  = torch.from_numpy(windows[s:s + 32]).to(device)
        B  = x.shape[0]
        pm = generate_patch_mask(B, Pt, C, mask_ratio, "composite", device)

        if getattr(enc, "_use_bert_mask", False):
            toks  = enc(x, mask=pm, return_keep_padding_mask=False)
            recon = dec(toks, torch.zeros_like(pm), Pt, C)
        else:
            toks, vpm = enc(x, mask=pm, return_keep_padding_mask=True)
            recon = dec(toks, pm, Pt, C, visible_padding_mask=vpm)

        sm  = _patch_mask_to_signal_mask(pm, Pt, C, ps)
        T_r = recon.shape[-1]
        if T_r < sm.shape[-1]:
            sm = sm[:, :, :T_r]
            x  = x[:, :, :T_r]

        tot = torch.zeros(1, device=device)
        for (lo, hi), lbl, w in zip(BANDS, BAND_LBLS, bw):
            xb  = bandpass_t(x,     lo, hi)
            rb  = bandpass_t(recon, lo, hi)
            mse = F.mse_loss(rb[sm], xb[sm])
            band_acc[lbl].append(float(mse))
            tot = tot + (w / ws) * mse
        total_acc.append(float(tot))

    return {l: float(np.mean(band_acc[l])) for l in BAND_LBLS} | \
           {"mae_freq_loss": float(np.mean(total_acc))}


# ── Linear probing (SAMOSA) ───────────────────────────────────────────────────

def load_samosa(parquet: str, cfg: dict):
    needed = ["dataset", "subject_id", "session_id", "timestamp_ns",
              "dataset_activity_id", "dataset_activity_label"] + IMU_COLS
    df = pd.read_parquet(parquet, columns=needed,
                         filters=[("dataset", "==", "samosa")])
    df = df.sort_values(["subject_id", "session_id", "timestamp_ns"]).reset_index(drop=True)
    T   = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    hop = max(1, T // 2)
    wins, labels, subjs = [], [], []
    for (subj, sess), grp in df.groupby(["subject_id", "session_id"]):
        vals = grp[IMU_COLS].values.astype(np.float32)
        acts = grp["dataset_activity_id"].values
        for start in range(0, len(vals) - T + 1, hop):
            proc = preprocess_window(vals[start:start + T], cfg)
            if proc is not None:
                wins.append(proc.astype(np.float32))
                labels.append(int(acts[start + T // 2]))
                subjs.append(subj)
    return np.stack(wins), np.array(labels), np.array(subjs)


def subject_split(X, y, subjects, val_subjects):
    val_m = np.isin(subjects, val_subjects)
    X_tr, y_tr = X[~val_m], y[~val_m]
    X_val, y_val = X[val_m], y[val_m]
    common = set(np.unique(y_tr)) & set(np.unique(y_val))
    tr_m   = np.isin(y_tr,  list(common))
    val_m2 = np.isin(y_val, list(common))
    return X_tr[tr_m], y_tr[tr_m], X_val[val_m2], y_val[val_m2]


@torch.no_grad()
def extract_features(X: np.ndarray, enc, device, batch_size=256) -> np.ndarray:
    feats = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i + batch_size]).to(device)
        feats.append(enc(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


def linear_probe(X_tr, y_tr, X_val, y_val) -> float:
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr)
        return balanced_accuracy_score(y_val, clf.predict(X_val))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_recon_data(parquet, base_cfg, ocd_dir, n, rng, max_ocd_files=10):
    # OCDetect
    files = _list_files(ocd_dir)
    X_ocd, _ = _collect_windows(files, apply_flu_transform=True, max_files=max_ocd_files)
    perm = rng.permutation(len(X_ocd))
    ocd_wins = []
    for i in perm:
        proc = preprocess_window(X_ocd[i].T, base_cfg)
        if proc is not None:
            ocd_wins.append(proc.astype(np.float32))
        if len(ocd_wins) >= n:
            break
    ocd_wins = np.stack(ocd_wins)

    # WISDM from parquet
    df = pd.read_parquet(parquet,
                         columns=["dataset", "session_id", "timestamp_ns"] + IMU_COLS,
                         filters=[("dataset", "==", "wisdm")])
    df = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)
    wisdm_wins = []
    for _, grp in df.groupby("session_id"):
        vals = grp[IMU_COLS].values.astype(np.float32)
        for s in range(0, len(vals) - 128 + 1, 64):
            proc = preprocess_window(vals[s:s + 128], base_cfg)
            if proc is not None:
                wisdm_wins.append(proc.astype(np.float32))
        if len(wisdm_wins) >= n * 3:
            break
    idx = rng.choice(len(wisdm_wins), min(n, len(wisdm_wins)), replace=False)
    wisdm_wins = np.stack(wisdm_wins)[idx]

    print(f"  ocd: {len(ocd_wins)}  wisdm: {len(wisdm_wins)}")
    return {"ocd": ocd_wins, "wisdm": wisdm_wins}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(probe_results, recon_results, shorts, step_label, out_dir,
                 summary_key="mae_freq_loss"):
    os.makedirs(out_dir, exist_ok=True)
    datasets_r = list(next(iter(recon_results.values())).keys())
    metrics_r  = BAND_LBLS + ["mae_freq_loss"]

    # ── Figure 1: linear probing bar chart ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(shorts) * 1.5), 4.5))
    vals    = [probe_results[s] for s in shorts]
    bars    = ax.bar(shorts, vals,
                     color=[COLORS[i % len(COLORS)] for i in range(len(shorts))],
                     alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2 + 0.05)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_title(f"SAMOSA Linear Probing — {step_label}", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probing_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/probing_comparison.png")

    # ── Figure 2: reconstruction comparison (per dataset, grouped bars) ───────
    fig, axes = plt.subplots(1, len(datasets_r),
                             figsize=(6 * len(datasets_r), 4.5), sharey=False)
    if len(datasets_r) == 1:
        axes = [axes]
    x     = np.arange(len(metrics_r))
    width = 0.8 / len(shorts)

    for di, ds in enumerate(datasets_r):
        ax = axes[di]
        for ri, short in enumerate(shorts):
            vals_r = [recon_results[short][ds][m] for m in metrics_r]
            ax.bar(x + ri * width, vals_r, width * 0.9,
                   label=short, color=COLORS[ri % len(COLORS)], alpha=0.85)
        ax.set_xticks(x + width * (len(shorts) - 1) / 2)
        ax.set_xticklabels(metrics_r, fontsize=8, rotation=20, ha="right")
        ax.set_title(f"{ds}", fontsize=10, fontweight="bold")
        ax.set_ylabel("MSE" if di == 0 else "")
        ax.grid(axis="y", alpha=0.3)
        if di == 0:
            ax.legend(fontsize=8)

    fig.suptitle(f"Reconstruction Quality (masked patches, same as val/mae_freq_loss) — {step_label}",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reconstruction_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/reconstruction_comparison.png")

    # ── Figure 3: combined side-by-side (probing + mae_freq_loss) ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Probing
    ax = axes[0]
    vals = [probe_results[s] for s in shorts]
    bars = ax.bar(shorts, vals,
                  color=[COLORS[i % len(COLORS)] for i in range(len(shorts))],
                  alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Linear Probing (SAMOSA)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Reconstruction (averaged over datasets)
    ax = axes[1]
    avg_recon = {s: np.mean([recon_results[s][ds][summary_key]
                              for ds in datasets_r]) for s in shorts}
    vals_r = [avg_recon[s] for s in shorts]
    bars   = ax.bar(shorts, vals_r,
                    color=[COLORS[i % len(COLORS)] for i in range(len(shorts))],
                    alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals_r):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    recon_title = "Full Reconstruction MSE (no mask)" if summary_key == "overall_mse" \
                  else "Masked Reconstruction (mae_freq_loss)"
    ax.set_title(f"{recon_title}\navg over datasets", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"{summary_key} (lower = better)")
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Model Comparison — {step_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/combined_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--step",  type=int, default=None,
                   help="Checkpoint step (default: auto = max common step)")
    p.add_argument("--parquet",       default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--ocd-dir",       default="/mnt/nas2/yiquan_data/preprocessed")
    p.add_argument("--max-ocd-files", type=int, default=10)
    p.add_argument("--n-recon",       type=int, default=300,
                   help="Windows per dataset for reconstruction eval")
    p.add_argument("--val-subjects",  nargs="+", default=None)
    p.add_argument("--full-recon",    action="store_true",
                   help="Use full reconstruction (no masking) instead of masked MAE evaluation")
    p.add_argument("--out-dir",       default="data_vis/model_comparison")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    rng    = np.random.default_rng(42)
    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-detect max common step
    step = args.step
    if step is None:
        import glob, re
        common = None
        for run in args.runs:
            paths  = glob.glob(os.path.join(args.runs_root, run, "checkpoints", "step_*.pt"))
            steps  = {int(re.search(r"step_(\d+)\.pt", p).group(1)) for p in paths}
            common = steps if common is None else common & steps
        if common:
            step = max(common)
            print(f"Auto-selected step={step} (max common across all runs)")
        else:
            print("No common step found — using best.pt")

    step_label = f"step_{step}" if step else "best"

    # Load all models
    models = {}
    for run in args.runs:
        print(f"\nLoading {run} ...")
        enc, dec, cfg, bw, short = load_model(
            os.path.join(args.runs_root, run), step, device
        )
        models[short] = {"enc": enc, "dec": dec, "cfg": cfg, "bw": bw}

    shorts   = list(models.keys())
    base_cfg = models[shorts[0]]["cfg"]

    # ── Reconstruction evaluation ──────────────────────────────────────────────
    print("\nLoading reconstruction data ...")
    recon_data = load_recon_data(args.parquet, base_cfg, args.ocd_dir,
                                 args.n_recon, rng, args.max_ocd_files)

    recon_results = {}
    mode_label = "full (no mask)" if args.full_recon else "masked (75%)"
    summary_key = "overall_mse"  if args.full_recon else "mae_freq_loss"
    print(f"\nEvaluating reconstruction [{mode_label}] ...")
    for short, m in models.items():
        recon_results[short] = {}
        for ds, wins in recon_data.items():
            if args.full_recon:
                r = eval_full_reconstruction(wins, m["enc"], m["dec"], device)
                print(f"  {short} / {ds}: overall_mse={r['overall_mse']:.5f}  " +
                      "  ".join(f"{l}={r[l]:.5f}" for l in BAND_LBLS))
            else:
                r = eval_reconstruction(wins, m["enc"], m["dec"], m["bw"], device)
                print(f"  {short} / {ds}: mae_freq_loss={r['mae_freq_loss']:.5f}  " +
                      "  ".join(f"{l}={r[l]:.5f}" for l in BAND_LBLS))
            recon_results[short][ds] = r

    # ── Linear probing (SAMOSA) ────────────────────────────────────────────────
    print("\nLoading SAMOSA ...")
    X, y, subjects = load_samosa(args.parquet, base_cfg)
    all_subjs  = sorted(np.unique(subjects))
    val_subjs  = args.val_subjects or all_subjs[-4:]
    print(f"  Val subjects: {val_subjs}")
    X_tr, y_tr, X_val, y_val = subject_split(X, y, subjects, val_subjs)
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

    probe_results = {}
    print("\nLinear probing ...")
    for short, m in models.items():
        F_tr  = extract_features(X_tr,  m["enc"], device)
        F_val = extract_features(X_val, m["enc"], device)
        bal   = linear_probe(F_tr, y_tr, F_val, y_val)
        probe_results[short] = bal
        print(f"  {short}: bal_acc={bal:.4f}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"{'Model':<20} {'Bal Acc':>9}  " +
          "  ".join(f"{ds+' recon':>14}" for ds in recon_data))
    print("-" * 65)
    for short in shorts:
        recon_str = "  ".join(
            f"{recon_results[short][ds][summary_key]:>14.5f}"
            for ds in recon_data
        )
        print(f"{short:<20} {probe_results[short]:>9.4f}  {recon_str}")
    print("=" * 65)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    rows = []
    for short in shorts:
        row = {"model": short, "step": step_label,
               "probe_bal_acc": probe_results[short]}
        for ds in recon_data:
            for k, v in recon_results[short][ds].items():
                row[f"recon_{ds}_{k}"] = v
        rows.append(row)
    csv_path = os.path.join(args.out_dir, f"comparison_{step_label}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_results(probe_results, recon_results, shorts, step_label, args.out_dir,
                 summary_key=summary_key)


if __name__ == "__main__":
    main()
