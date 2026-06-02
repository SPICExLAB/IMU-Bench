#!/usr/bin/env python3
"""
Linear-probing comparison of pretrained ViT1D models on HAR datasets.

For each model × dataset:
  1. Load frozen pretrained encoder
  2. Extract mean-pooled features from the parquet dataset
  3. Fit LogisticRegression on train features
  4. Report balanced accuracy on val set

Outputs:
  - Console table
  - Bar chart PNG  (--out-dir)
  - CSV of results (--out-dir)

Usage:
    python scripts/compare_pretrain_models.py \
        --runs  vit1d_mae_realdata_ocdetect_flu_v3 \
                vit1d_mae_realdata_ocdetect_flu_v3_MoE \
        --datasets wisdm pamap2 utd_mhad samosa recofit \
        --out-dir  data_vis/model_compare/
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.utils.helpers import cfg_get

FS       = 50.0
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_encoder(run_dir: str, device: torch.device) -> tuple:
    ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    step = ckpt.get("step", "?")

    enc = ViT1DEncoder(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    print(f"  Loaded {os.path.basename(run_dir)}  step={step}"
          f"  embed_dim={enc.embed_dim}  freq_moe={enc._use_freq_moe}")
    return enc, cfg, step


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset_windows(parquet: str, dataset: str, cfg: dict, rng: np.random.Generator):
    """Load all windows for a dataset from the master parquet, return (X, y, subjects)."""
    needed = ["dataset", "session_id", "subject_id", "timestamp_ns",
              "dataset_activity_id"] + IMU_COLS
    df = pd.read_parquet(parquet, columns=needed,
                         filters=[("dataset", "==", dataset)])
    if df.empty:
        return None, None, None

    df = df.sort_values(["subject_id", "session_id", "timestamp_ns"]).reset_index(drop=True)
    T   = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    hop = max(1, T // 2)

    wins, labels, subjs = [], [], []
    for (subj, sess), grp in df.groupby(["subject_id", "session_id"]):
        vals  = grp[IMU_COLS].values.astype(np.float32)
        lbls  = grp["dataset_activity_id"].values
        for start in range(0, len(vals) - T + 1, hop):
            proc = preprocess_window(vals[start:start + T], cfg)
            if proc is not None:
                wins.append(proc.astype(np.float32))   # [6, T]
                center = start + T // 2
                labels.append(int(lbls[center]))
                subjs.append(subj)

    if not wins:
        return None, None, None

    X = np.stack(wins)        # [N, 6, T]
    y = np.array(labels)
    s = np.array(subjs)
    print(f"    {dataset}: {len(X)} windows, {len(np.unique(y))} classes, "
          f"{len(np.unique(s))} subjects")
    return X, y, s


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(X: np.ndarray, enc: ViT1DEncoder,
                     device: torch.device, batch_size: int = 128) -> np.ndarray:
    feats = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i + batch_size]).to(device)
        f = enc(x)          # [B, D] — mean-pooled, no mask
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


# ── Train / val split by subject ─────────────────────────────────────────────

def subject_split(X, y, subjects, val_ratio=0.2, seed=42):
    rng    = np.random.default_rng(seed)
    unique = np.unique(subjects)
    n_val  = max(1, int(len(unique) * val_ratio))
    val_s  = rng.choice(unique, size=n_val, replace=False)
    val_m  = np.isin(subjects, val_s)
    return X[~val_m], y[~val_m], X[val_m], y[val_m]


# ── Linear probing ────────────────────────────────────────────────────────────

def linear_probe(X_tr, y_tr, X_val, y_val):
    le = LabelEncoder()
    y_tr_enc  = le.fit_transform(y_tr)
    y_val_enc = le.transform(y_val)

    clf = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_tr, y_tr_enc)
        preds = clf.predict(X_val)
        bal_acc = balanced_accuracy_score(y_val_enc, preds)

    return bal_acc


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, run_names: list, datasets: list, out_path: str):
    """
    results[run_name][dataset] = balanced_accuracy (float or None)
    """
    x      = np.arange(len(datasets))
    width  = 0.8 / max(len(run_names), 1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(datasets)), 5))
    for ri, run in enumerate(run_names):
        vals = [results[run].get(ds) for ds in datasets]
        vals_plot = [v if v is not None else 0.0 for v in vals]
        label = os.path.basename(run)
        bars = ax.bar(x + ri * width, vals_plot, width * 0.9,
                      label=label, color=colors[ri % len(colors)], alpha=0.85)
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * (len(run_names) - 1) / 2)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_title("Linear Probing: Pretrained Encoder Comparison\n(frozen encoder → LogisticRegression)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot → {out_path}")
    plt.close(fig)


def print_table(results: dict, run_names: list, datasets: list):
    short = [os.path.basename(r) for r in run_names]
    col_w = max(len(s) for s in short) + 2
    ds_w  = max(len(d) for d in datasets) + 2

    header = f"{'Dataset':<{ds_w}}" + "".join(f"{s:<{col_w}}" for s in short)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for ds in datasets:
        row = f"{ds:<{ds_w}}"
        for run in run_names:
            v = results[run].get(ds)
            row += f"{'N/A' if v is None else f'{v:.4f}':<{col_w}}"
        print(row)
    print("=" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["vit1d_mae_realdata_ocdetect_flu_v3",
                            "vit1d_mae_realdata_ocdetect_flu_v3_MoE"])
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--datasets",  nargs="+",
                   default=["wisdm", "pamap2", "utd_mhad", "samosa", "recofit"])
    p.add_argument("--parquet",   default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--out-dir",   default="data_vis/model_compare")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    rng    = np.random.default_rng(42)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load all encoders
    encoders = {}
    cfgs     = {}
    for run in args.runs:
        run_dir = os.path.join(args.runs_root, run)
        print(f"\nLoading {run} ...")
        enc, cfg, step = load_encoder(run_dir, device)
        encoders[run] = enc
        cfgs[run]     = cfg

    # Use first model's cfg for data loading (same preprocessing)
    data_cfg = cfgs[args.runs[0]]

    results = {run: {} for run in args.runs}

    for ds in args.datasets:
        print(f"\n── Dataset: {ds} ──")
        X, y, subjects = load_dataset_windows(args.parquet, ds, data_cfg, rng)
        if X is None:
            print(f"  Skipping {ds} (no data found)")
            for run in args.runs:
                results[run][ds] = None
            continue

        X_tr, y_tr, X_val, y_val = subject_split(X, y, subjects,
                                                  val_ratio=args.val_ratio)
        print(f"    Train: {len(X_tr)}  Val: {len(X_val)}")

        if len(np.unique(y_val)) < 2:
            print(f"  Skipping {ds} (val set has < 2 classes)")
            for run in args.runs:
                results[run][ds] = None
            continue

        for run in args.runs:
            enc = encoders[run]
            t0  = time.time()

            print(f"  [{run}] extracting features ...", end=" ", flush=True)
            F_tr  = extract_features(X_tr,  enc, device)
            F_val = extract_features(X_val, enc, device)

            bal_acc = linear_probe(F_tr, y_tr, F_val, y_val)
            elapsed = time.time() - t0
            results[run][ds] = bal_acc
            print(f"bal_acc={bal_acc:.4f}  ({elapsed:.1f}s)")

    # Print table
    print_table(results, args.runs, args.datasets)

    # Save CSV
    rows = []
    for ds in args.datasets:
        row = {"dataset": ds}
        for run in args.runs:
            row[os.path.basename(run)] = results[run].get(ds)
        rows.append(row)
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "linear_probe_results.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"Saved CSV  → {csv_path}")

    # Plot
    plot_results(results, args.runs, args.datasets,
                 os.path.join(args.out_dir, "linear_probe_comparison.png"))


if __name__ == "__main__":
    main()
