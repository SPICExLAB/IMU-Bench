#!/usr/bin/env python3
"""
Fully-supervised, full-parameter training from scratch on a HAR dataset,
comparing the v6 3D freq-token encoder vs the v0 standard ViT1D encoder.

Both encoders are randomly initialised (NO pretraining) and trained end-to-end
(encoder + linear classification head, all parameters trainable) with
cross-entropy on the activity labels.

This isolates the **architectural** contribution of the 3D freq-token design
from any pretraining effect.

Usage:
    python scripts/supervised_train_samosa.py \
        --configs configs/vit1d_mae_realdata_ocdetect_flu_v0.yaml \
                  configs/vit1d_mae_realdata_ocdetect_flu_v6.yaml \
        --labels  v0 v6 \
        --dataset samosa \
        --out-dir data_vis/supervised_samosa/
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoder3D
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.utils.helpers import cfg_get

FS       = 50.0
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ── Encoder builder ───────────────────────────────────────────────────────────

def build_encoder(cfg: dict):
    """Build the appropriate encoder (3D freq-token or standard) from config."""
    # Ensure 6 IMU channels are registered so encoder in_channels = 6
    cfg.setdefault("data", {})
    cfg["data"]["sensor_columns"] = IMU_COLS

    use_3d = bool(cfg_get(cfg, ["vit1d", "encoder", "use_3d_freq"], False))
    if use_3d:
        enc = ViT1DEncoder3D(cfg)
        kind = "3D-freq-token"
    else:
        enc = ViT1DEncoder(cfg)
        kind = "standard-2D"
    return enc, kind


class SupervisedModel(nn.Module):
    """Encoder + linear classification head, all trainable."""

    def __init__(self, encoder, embed_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)            # [B, D] mean-pooled (no mask)
        feat = self.norm(feat)
        return self.head(self.dropout(feat))


# ── Data ──────────────────────────────────────────────────────────────────────

def load_dataset(parquet: str, cfg: dict, dataset: str):
    needed = ["dataset", "subject_id", "session_id", "timestamp_ns",
              "dataset_activity_id", "dataset_activity_label"] + IMU_COLS
    df = pd.read_parquet(parquet, columns=needed,
                         filters=[("dataset", "==", dataset)])
    df = df.sort_values(["subject_id", "session_id", "timestamp_ns"]).reset_index(drop=True)
    T   = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    hop = max(1, T // 2)
    wins, labels, subjs, names = [], [], [], []
    for (subj, sess), grp in df.groupby(["subject_id", "session_id"]):
        vals   = grp[IMU_COLS].values.astype(np.float32)
        acts   = grp["dataset_activity_id"].values
        lnames = grp["dataset_activity_label"].values
        for start in range(0, len(vals) - T + 1, hop):
            proc = preprocess_window(vals[start:start + T], cfg)
            if proc is not None:
                wins.append(proc.astype(np.float32))
                center = start + T // 2
                labels.append(int(acts[center]))
                subjs.append(subj)
                names.append(str(lnames[center]))
    X = np.stack(wins)
    y = np.array(labels)
    s = np.array(subjs)
    n = np.array(names)
    print(f"  {len(X)} windows | {len(np.unique(y))} classes | {len(np.unique(s))} subjects")
    return X, y, s, n


def subject_split(X, y, s, val_subjects):
    val_m = np.isin(s, val_subjects)
    X_tr, y_tr = X[~val_m], y[~val_m]
    X_val, y_val = X[val_m], y[val_m]
    common = set(np.unique(y_tr)) & set(np.unique(y_val))
    tr_m   = np.isin(y_tr,  list(common))
    val_m2 = np.isin(y_val, list(common))
    if len(common) < len(np.unique(y_tr)):
        print(f"  Filtered to {len(common)} common classes")
    return X_tr[tr_m], y_tr[tr_m], X_val[val_m2], y_val[val_m2], sorted(common)


# ── Training ──────────────────────────────────────────────────────────────────

def train_supervised(model, X_tr, y_tr_enc, X_val, y_val_enc, device,
                     lr=3e-4, weight_decay=0.05, max_epochs=100, batch_size=128,
                     warmup_epochs=5, patience=20):
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_enc))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val_enc)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.95))
    scaler    = GradScaler()

    # Warmup + cosine schedule
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        prog = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_bal_acc = 0.0
    best_state   = None
    no_improve   = 0
    history      = []

    for epoch in range(max_epochs):
        model.train()
        total_loss, n_b = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(xb)
                loss   = F.cross_entropy(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_b += 1
        scheduler.step()

        # Validation
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_val_t), 256):
                xb = X_val_t[i:i + 256].to(device)
                preds.append(model(xb).argmax(dim=-1).cpu().numpy())
        preds   = np.concatenate(preds)
        bal_acc = balanced_accuracy_score(y_val_t.numpy(), preds)
        history.append(bal_acc)

        if bal_acc > best_bal_acc + 1e-4:
            best_bal_acc = bal_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}  loss={total_loss/n_b:.4f}"
                  f"  val_bal_acc={bal_acc:.4f}  best={best_bal_acc:.4f}")

        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return best_bal_acc, history


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, labels, histories, ds_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Bar chart
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 2), 4.5))
    vals = [results[l] for l in labels]
    bars = ax.bar(labels, vals,
                  color=[colors[i % len(colors)] for i in range(len(labels))],
                  alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_title(f"{ds_name.upper()} — Supervised Training from Scratch\n"
                 f"(full-parameter, no pretraining)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "supervised_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/supervised_comparison.png")

    # Learning curves
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for li, label in enumerate(labels):
        ax.plot(range(1, len(histories[label]) + 1), histories[label],
                label=f"{label} (best={results[label]:.4f})",
                color=colors[li % len(colors)], lw=1.8)
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Val Balanced Accuracy", fontsize=10)
    ax.set_title(f"{ds_name.upper()} — Validation Learning Curves", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/learning_curves.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+",
                   default=["configs/vit1d_mae_realdata_ocdetect_flu_v0.yaml",
                            "configs/vit1d_mae_realdata_ocdetect_flu_v6.yaml"])
    p.add_argument("--labels",  nargs="+", default=["v0", "v6"])
    p.add_argument("--dataset", default="samosa")
    p.add_argument("--parquet", default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--val-subjects", nargs="+", default=None)
    p.add_argument("--n-val-subjects", type=int, default=None)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--patience", type=int,   default=20)
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--out-dir",  default="data_vis/supervised_samosa")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    assert len(args.configs) == len(args.labels), "configs and labels must match length"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data once using the first config's windowing settings
    base_cfg = yaml.safe_load(open(args.configs[0]))
    base_cfg.setdefault("data", {})["sensor_columns"] = IMU_COLS

    print(f"\nLoading {args.dataset} ...")
    X, y, subjects, names = load_dataset(args.parquet, base_cfg, args.dataset)
    all_subjs = sorted(np.unique(subjects))
    n_val = args.n_val_subjects or max(1, round(len(all_subjs) * 0.2))
    val_subjs = args.val_subjects or all_subjs[-n_val:]
    print(f"  Val subjects: {val_subjs}")

    X_tr, y_tr, X_val, y_val, common = subject_split(X, y, subjects, val_subjs)
    le = LabelEncoder().fit(common)
    y_tr_enc  = le.transform(y_tr).astype(np.int64)
    y_val_enc = le.transform(y_val).astype(np.int64)
    n_classes = len(common)
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Classes: {n_classes}")

    results, histories = {}, {}
    for cfg_path, label in zip(args.configs, args.labels):
        print(f"\n{'='*55}")
        print(f"Training: {label}  ({cfg_path})")
        print("="*55)

        cfg = yaml.safe_load(open(cfg_path))
        encoder, kind = build_encoder(cfg)
        model = SupervisedModel(encoder, encoder.embed_dim, n_classes).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Encoder type: {kind}")
        print(f"  Total trainable params: {n_params/1e6:.1f}M")

        best_acc, hist = train_supervised(
            model, X_tr, y_tr_enc, X_val, y_val_enc, device,
            lr=args.lr, max_epochs=args.epochs,
            batch_size=args.batch_size, patience=args.patience,
        )
        results[label]   = best_acc
        histories[label] = hist
        print(f"  Best val balanced accuracy: {best_acc:.4f}")

    # Summary
    print(f"\n{'='*50}")
    print(f"{args.dataset.upper()} — Supervised from Scratch")
    print(f"{'Model':<12} {'Bal Acc':>10}  {'Params':>8}")
    print("-"*50)
    for label in args.labels:
        print(f"{label:<12} {results[label]:>10.4f}")
    print("="*50)

    # Save CSV
    pd.DataFrame([{"model": l, "bal_acc": results[l]} for l in args.labels]).to_csv(
        os.path.join(args.out_dir, f"supervised_{args.dataset}.csv"), index=False)

    plot_results(results, args.labels, histories, args.dataset, args.out_dir)


if __name__ == "__main__":
    main()
