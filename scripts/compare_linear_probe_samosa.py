#!/usr/bin/env python3
"""
Linear probing comparison on the SAMOSA dataset for two pretrained models.

Frozen encoder → extract mean-pooled features → LogisticRegression → balanced accuracy.
Uses a subject-based train/val split; filters to classes present in both splits.

Usage:
    python scripts/compare_linear_probe_samosa.py \
        --runs  vit1d_mae_realdata_ocdetect_flu_v3 \
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.utils.helpers import cfg_get

FS       = 50.0
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
DATASET  = "samosa"


# ── Model loading ─────────────────────────────────────────────────────────────

def available_steps(run_dir: str) -> list[int]:
    """Return sorted list of step numbers that have saved checkpoints."""
    import glob, re
    paths = glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pt"))
    steps = []
    for p in paths:
        m = re.search(r"step_(\d+)\.pt$", p)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def load_encoder(run_dir: str, device: torch.device, step: int = None):
    if step is not None:
        ckpt_path = os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt")
    else:
        ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]
    enc  = ViT1DEncoder(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    actual_step = ckpt.get("step", "?")
    print(f"  {os.path.basename(run_dir)}  step={actual_step}  freq_moe={enc._use_freq_moe}")
    return enc, cfg


# ── Data loading ──────────────────────────────────────────────────────────────

def load_samosa(parquet: str, cfg: dict):
    needed = ["dataset", "subject_id", "session_id", "timestamp_ns",
              "dataset_activity_id", "dataset_activity_label"] + IMU_COLS
    df = pd.read_parquet(parquet, columns=needed,
                         filters=[("dataset", "==", DATASET)])
    df = df.sort_values(["subject_id", "session_id", "timestamp_ns"]).reset_index(drop=True)

    T   = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    hop = max(1, T // 2)

    wins, labels, subjs, label_names = [], [], [], []
    for (subj, sess), grp in df.groupby(["subject_id", "session_id"]):
        vals  = grp[IMU_COLS].values.astype(np.float32)
        acts  = grp["dataset_activity_id"].values
        names = grp["dataset_activity_label"].values
        for start in range(0, len(vals) - T + 1, hop):
            proc = preprocess_window(vals[start:start + T], cfg)
            if proc is not None:
                wins.append(proc.astype(np.float32))
                center = start + T // 2
                labels.append(int(acts[center]))
                subjs.append(subj)
                label_names.append(str(names[center]))

    X = np.stack(wins)
    y = np.array(labels)
    s = np.array(subjs)
    n = np.array(label_names)
    print(f"  Loaded {len(X)} windows | {len(np.unique(y))} classes | {len(np.unique(s))} subjects")
    return X, y, s, n


def subject_split(X, y, s, n, val_subjects):
    val_m  = np.isin(s, val_subjects)
    X_tr, y_tr, n_tr = X[~val_m], y[~val_m], n[~val_m]
    X_val, y_val, n_val = X[val_m],  y[val_m],  n[val_m]

    # Keep only classes present in BOTH splits
    common = set(np.unique(y_tr)) & set(np.unique(y_val))
    tr_m  = np.isin(y_tr,  list(common))
    val_m2 = np.isin(y_val, list(common))
    removed = len(np.unique(y_tr)) + len(np.unique(y_val)) - 2 * len(common)
    if removed > 0:
        print(f"  Filtered {removed} class(es) not in both splits → {len(common)} common classes")

    return (X_tr[tr_m], y_tr[tr_m], n_tr[tr_m],
            X_val[val_m2], y_val[val_m2], n_val[val_m2],
            sorted(common))


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(X: np.ndarray, enc, device, batch_size=256) -> np.ndarray:
    feats = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i + batch_size]).to(device)
        feats.append(enc(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_per_class(results: dict, label_map: dict, run_names: list, out_path: str):
    """Bar chart of per-class accuracy for each model."""
    classes  = sorted(label_map.keys())
    names    = [label_map[c] for c in classes]
    colors   = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    short    = [r.replace("vit1d_mae_realdata_ocdetect_flu_", "") for r in run_names]

    x     = np.arange(len(classes))
    width = 0.8 / len(run_names)

    fig, ax = plt.subplots(figsize=(max(14, len(classes) * 0.6), 5))
    for ri, run in enumerate(run_names):
        per_cls = results[run]["per_class"]
        vals    = [per_cls.get(c, 0.0) for c in classes]
        ax.bar(x + ri * width, vals, width * 0.9,
               label=short[ri], color=colors[ri % len(colors)], alpha=0.85)

    ax.set_xticks(x + width * (len(run_names) - 1) / 2)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Per-class Accuracy")
    ax.set_title(f"SAMOSA — Per-class Accuracy (Linear Probe)\n" +
                 "  ".join(f"{short[ri]}: bal_acc={results[run]['bal_acc']:.4f}"
                           for ri, run in enumerate(run_names)),
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def plot_summary(results: dict, run_names: list, out_path: str):
    short  = [r.replace("vit1d_mae_realdata_ocdetect_flu_", "") for r in run_names]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    vals   = [results[run]["bal_acc"] for run in run_names]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(short, vals, color=colors[:len(run_names)], alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("SAMOSA Linear Probing — Balanced Accuracy", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["vit1d_mae_realdata_ocdetect_flu_v3",
                            "vit1d_mae_realdata_ocdetect_flu_v3_MoE"])
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--parquet",   default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--step",         type=int, default=None,
                   help="Compare at this step (default: auto = max common step)")
    p.add_argument("--val-subjects", nargs="+", default=None,
                   help="Subject IDs to use as val (default: last 4 subjects)")
    p.add_argument("--out-dir",   default="data_vis/samosa_probe")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Determine comparison step
    step = args.step
    if step is None:
        all_steps = [set(available_steps(os.path.join(args.runs_root, r))) for r in args.runs]
        common = set.intersection(*all_steps)
        if common:
            step = max(common)
            print(f"\nAuto-selected step={step} (max common step across all runs)")
        else:
            print("\nNo common step checkpoints found — using best.pt for each model")

    # Load encoders
    encoders = {}
    cfgs     = {}
    for run in args.runs:
        print(f"\nLoading {run} ...")
        enc, cfg = load_encoder(os.path.join(args.runs_root, run), device, step=step)
        encoders[run] = enc
        cfgs[run]     = cfg

    # Load data once using first model's config
    print(f"\nLoading {DATASET} ...")
    X, y, subjects, label_names = load_samosa(args.parquet, cfgs[args.runs[0]])

    # Subject split
    all_subjs  = sorted(np.unique(subjects))
    val_subjs  = args.val_subjects or all_subjs[-4:]   # last 4 subjects as val
    print(f"  Val subjects: {val_subjs}  (Train: {len(all_subjs) - len(val_subjs)} subjects)")

    X_tr, y_tr, n_tr, X_val, y_val, n_val, common_classes = subject_split(
        X, y, subjects, label_names, val_subjs
    )
    print(f"  Train: {len(X_tr)} windows  Val: {len(X_val)} windows")

    # Label name map
    label_map = {}
    for cls, name in zip(y_val, n_val):
        label_map[cls] = name

    # Probe each model
    results = {}
    for run in args.runs:
        enc = encoders[run]
        short = run.replace("vit1d_mae_realdata_ocdetect_flu_", "")
        print(f"\n── {short} ── extracting features ...")

        F_tr  = extract_features(X_tr,  enc, device)
        F_val = extract_features(X_val, enc, device)

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(F_tr, y_tr)
            preds = clf.predict(F_val)

        bal_acc = balanced_accuracy_score(y_val, preds)

        # Per-class accuracy
        cm = confusion_matrix(y_val, preds, labels=common_classes)
        per_class = {c: (cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0)
                     for i, c in enumerate(common_classes)}

        results[run] = {"bal_acc": bal_acc, "per_class": per_class}
        print(f"  Balanced accuracy: {bal_acc:.4f}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<45} {'Bal Acc':>8}")
    print("-" * 60)
    for run in args.runs:
        short = run.replace("vit1d_mae_realdata_ocdetect_flu_", "")
        print(f"{short:<45} {results[run]['bal_acc']:>8.4f}")
    print("=" * 60)

    # Print per-class comparison
    print(f"\n{'Class':<28}" + "".join(
        f"{r.replace('vit1d_mae_realdata_ocdetect_flu_',''):<14}" for r in args.runs))
    print("-" * (28 + 14 * len(args.runs)))
    for c in sorted(common_classes):
        name = label_map.get(c, str(c))
        row = f"{name:<28}"
        for run in args.runs:
            row += f"{results[run]['per_class'].get(c, 0.0):<14.4f}"
        print(row)

    # Plots
    plot_summary(results, args.runs,
                 os.path.join(args.out_dir, "samosa_bal_acc.png"))
    plot_per_class(results, label_map, args.runs,
                   os.path.join(args.out_dir, "samosa_per_class.png"))

    # Save CSV
    rows = [{"class_id": c, "class_name": label_map.get(c, str(c)),
             **{r.replace("vit1d_mae_realdata_ocdetect_flu_", ""): results[r]["per_class"].get(c)
                for r in args.runs}}
            for c in sorted(common_classes)]
    pd.DataFrame(rows).to_csv(
        os.path.join(args.out_dir, "samosa_per_class.csv"), index=False)


if __name__ == "__main__":
    main()
