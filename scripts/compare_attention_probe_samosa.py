#!/usr/bin/env python3
"""
Compare v3 vs v5_MoE on SAMOSA using an attention-based probing head.

Instead of mean-pooling → linear layer, the probe uses a learnable query
that cross-attends over all 48 encoder tokens, allowing it to:
  - Weight different time/channel patches by relevance
  - Capture cross-band interactions (not possible with linear probing)
  - Attend to activity-specific frequency/temporal patterns

Architecture:
    Frozen encoder [B, 48, 384]
        ↓
    Cross-attention probe (learnable query → attends all 48 tokens)
        ↓
    LayerNorm → Linear → [B, n_classes]

Also reports linear probe (LogisticRegression) as baseline for comparison.

Usage:
    python scripts/compare_attention_probe_samosa.py \
        --runs vit1d_mae_realdata_ocdetect_flu_v3 \
               vit1d_mae_realdata_ocdetect_flu_v5_MoE \
        --step 80000 \
        --out-dir data_vis/attn_probe/
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.utils.helpers import cfg_get

FS       = 50.0
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ── Attention probe ───────────────────────────────────────────────────────────

class AttentionProbe(nn.Module):
    """Cross-attention probe over all encoder tokens.

    A single learnable query attends to all N encoder tokens.
    The attended representation is passed through LayerNorm + linear head.

    This allows the probe to:
      - Weight patches by activity relevance (not uniform mean pool)
      - Combine information across frequency bands and time positions
      - Learn which tokens carry discriminative activity information
    """

    def __init__(self, embed_dim: int, num_classes: int,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query     = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.attn      = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm      = nn.LayerNorm(embed_dim)
        self.dropout   = nn.Dropout(dropout)
        self.head      = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D]  frozen encoder output
        Returns:
            logits: [B, num_classes]
        """
        B = tokens.shape[0]
        q   = self.query.expand(B, -1, -1)            # [B, 1, D]
        out, attn_w = self.attn(q, tokens, tokens)    # [B, 1, D], [B, 1, N]
        out = self.norm(out.squeeze(1))                # [B, D]
        return self.head(self.dropout(out))            # [B, C]

    def get_attention_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return attention weights [B, N] for visualisation."""
        B = tokens.shape[0]
        q = self.query.expand(B, -1, -1)
        _, attn_w = self.attn(q, tokens, tokens)      # [B, 1, N]
        return attn_w.squeeze(1)                      # [B, N]


# ── Model / data loading ──────────────────────────────────────────────────────

def load_encoder(run_dir: str, step: int, device: torch.device):
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
    actual = ckpt.get("step", "?")
    short  = os.path.basename(run_dir).replace("vit1d_mae_realdata_ocdetect_flu_", "")
    print(f"  {short}  step={actual}  bert_mask={enc._use_bert_mask}  freq_moe={enc._use_freq_moe}")
    return enc, cfg, short


def load_dataset(parquet: str, cfg: dict, dataset: str = "samosa"):
    needed = ["dataset", "subject_id", "session_id", "timestamp_ns",
              "dataset_activity_id", "dataset_activity_label"] + IMU_COLS
    df = pd.read_parquet(parquet, columns=needed,
                         filters=[("dataset", "==", dataset)])
    df = df.sort_values(["subject_id", "session_id", "timestamp_ns"]).reset_index(drop=True)
    T   = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    hop = max(1, T // 2)
    wins, labels, subjs, names = [], [], [], []
    for (subj, sess), grp in df.groupby(["subject_id", "session_id"]):
        vals = grp[IMU_COLS].values.astype(np.float32)
        acts = grp["dataset_activity_id"].values
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


def subject_split(X, y, s, n, val_subjects):
    val_m  = np.isin(s, val_subjects)
    X_tr, y_tr = X[~val_m], y[~val_m]
    X_val, y_val = X[val_m], y[val_m]
    names_val = n[val_m]
    common = set(np.unique(y_tr)) & set(np.unique(y_val))
    tr_m   = np.isin(y_tr,  list(common))
    val_m2 = np.isin(y_val, list(common))
    if len(common) < len(np.unique(y_tr)):
        print(f"  Filtered to {len(common)} common classes")
    return (X_tr[tr_m], y_tr[tr_m],
            X_val[val_m2], y_val[val_m2], names_val[val_m2],
            sorted(common))


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_tokens(X: np.ndarray, enc, device, batch_size=256) -> np.ndarray:
    """Extract per-token encoder output [N_wins, N_tokens, D] — no pooling."""
    out = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i + batch_size]).to(device)
        toks, _ = enc.forward_tokens(x)   # [B, 48, D]
        out.append(toks.cpu())
    return torch.cat(out, dim=0)  # [N_wins, 48, D]


@torch.no_grad()
def extract_pooled(X: np.ndarray, enc, device, batch_size=256) -> np.ndarray:
    """Extract mean-pooled embeddings [N_wins, D] for linear probe baseline."""
    feats = []
    for i in range(0, len(X), batch_size):
        x = torch.from_numpy(X[i:i + batch_size]).to(device)
        feats.append(enc(x).cpu().numpy())
    return np.concatenate(feats, axis=0)


# ── Attention probe training ──────────────────────────────────────────────────

def train_attention_probe(
    T_tr: torch.Tensor,   # [N, 48, D]
    y_tr: np.ndarray,
    T_val: torch.Tensor,  # [N, 48, D]
    y_val: np.ndarray,
    classes: list,
    embed_dim: int,
    device: torch.device,
    num_heads: int = 4,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    batch_size: int = 256,
    patience: int = 15,
):
    le = LabelEncoder()
    le.fit(classes)
    y_tr_enc  = torch.from_numpy(le.transform(y_tr)).long()
    y_val_enc = torch.from_numpy(le.transform(y_val)).long()
    n_classes = len(classes)

    probe     = AttentionProbe(embed_dim, n_classes, num_heads, dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    train_ds = TensorDataset(T_tr, y_tr_enc)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    best_bal_acc   = 0.0
    best_state     = None
    no_improve     = 0

    for epoch in range(max_epochs):
        probe.train()
        total_loss = 0.0
        for toks_b, y_b in train_dl:
            toks_b, y_b = toks_b.to(device), y_b.to(device)
            logits = probe(toks_b)
            loss   = F.cross_entropy(logits, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Validation
        probe.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(T_val), batch_size):
                toks_b = T_val[i:i + batch_size].to(device)
                preds.append(probe(toks_b).argmax(dim=-1).cpu().numpy())
            preds = np.concatenate(preds)

        bal_acc = balanced_accuracy_score(y_val_enc.numpy(), preds)

        if bal_acc > best_bal_acc + 1e-4:
            best_bal_acc = bal_acc
            best_state   = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  loss={total_loss/len(train_dl):.4f}"
                  f"  val_bal_acc={bal_acc:.4f}  best={best_bal_acc:.4f}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Load best weights
    probe.load_state_dict(best_state)
    probe.eval()
    return probe, best_bal_acc, le


def per_class_accuracy(probe, T_val, y_val_raw, le, classes, device, batch_size=256):
    preds = []
    with torch.no_grad():
        for i in range(0, len(T_val), batch_size):
            toks_b = T_val[i:i + batch_size].to(device)
            preds.append(probe(toks_b).argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(preds)
    y_enc = le.transform(y_val_raw)
    per_cls = {}
    for i, c in enumerate(le.classes_):
        m = y_enc == i
        if m.sum() > 0:
            per_cls[classes[i]] = float((preds[m] == i).mean())
    return per_cls


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(results, shorts, out_dir, ds_name="SAMOSA"):
    os.makedirs(out_dir, exist_ok=True)

    # ── Figure 1: overall balanced accuracy ───────────────────────────────────
    methods = ["linear_probe", "attn_probe"]
    method_labels = ["Linear Probe\n(LogReg)", "Attention Probe\n(cross-attn)"]
    x     = np.arange(len(shorts))
    width = 0.35
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(max(7, len(shorts) * 2), 5))
    for mi, (mkey, mlbl) in enumerate(zip(methods, method_labels)):
        vals = [results[s][mkey]["bal_acc"] for s in shorts]
        bars = ax.bar(x + mi * width, vals, width * 0.9,
                      label=mlbl, color=colors[mi], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(shorts, fontsize=10)
    ax.set_ylim(0, max(results[s][m]["bal_acc"] for s in shorts for m in methods) * 1.2)
    ax.set_ylabel("Balanced Accuracy", fontsize=11)
    ax.set_title(f"{ds_name}: Linear Probe vs Attention Probe\n(frozen encoder)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probe_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/probe_comparison.png")

    # ── Figure 2: per-class comparison ────────────────────────────────────────
    all_classes = sorted(set().union(*[
        set(results[s]["attn_probe"]["per_class"].keys()) for s in shorts
    ]))
    n_cls = len(all_classes)
    n_cols = 2
    width2 = 0.8 / (len(shorts) * n_cols)

    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, max(5, n_cls * 0.35)),
                             sharey=True)
    for mi, (mkey, mlbl) in enumerate(zip(methods, method_labels)):
        ax  = axes[mi]
        y   = np.arange(n_cls)
        for ri, short in enumerate(shorts):
            per_cls = results[short][mkey]["per_class"]
            vals    = [per_cls.get(c, 0.0) for c in all_classes]
            ax.barh(y + ri * (0.8 / len(shorts)), vals,
                    height=0.8 / len(shorts) * 0.9,
                    label=short, color=colors[ri], alpha=0.85)

        ax.set_yticks(y + 0.8 / len(shorts) * (len(shorts) - 1) / 2)
        ax.set_yticklabels(all_classes, fontsize=8)
        ax.set_xlabel("Per-class Accuracy")
        ax.set_title(mlbl, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"{ds_name} Per-class Accuracy", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/per_class_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["vit1d_mae_realdata_ocdetect_flu_v3",
                            "vit1d_mae_realdata_ocdetect_flu_v5_MoE"])
    p.add_argument("--runs-root",  default="runs")
    p.add_argument("--step",       type=int, default=None,
                   help="Checkpoint step (default: auto = max common step)")
    p.add_argument("--dataset",    default="samosa",
                   help="Parquet dataset name (e.g. samosa, utd_mhad, wisdm, pamap2)")
    p.add_argument("--parquet",    default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--n-val-subjects", type=int, default=None,
                   help="Number of subjects to hold out for val (default: 20%% of total)")
    p.add_argument("--val-subjects", nargs="+", default=None)
    p.add_argument("--num-heads",  type=int, default=4)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--patience",   type=int, default=15)
    p.add_argument("--out-dir",    default="data_vis/attn_probe")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
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
            print(f"Auto-selected step={step}")

    # Load encoders
    encoders, cfgs, shorts = {}, {}, []
    for run in args.runs:
        print(f"\nLoading {run} ...")
        enc, cfg, short = load_encoder(os.path.join(args.runs_root, run), step, device)
        encoders[short] = enc
        cfgs[short]     = cfg
        shorts.append(short)

    base_cfg = cfgs[shorts[0]]

    # Load dataset
    ds_name = args.dataset.upper().replace("_", "-")
    print(f"\nLoading {ds_name} ...")
    X, y, subjects, label_names = load_dataset(args.parquet, base_cfg, args.dataset)
    all_subjs = sorted(np.unique(subjects))
    n_val = args.n_val_subjects or max(1, round(len(all_subjs) * 0.2))
    val_subjs = args.val_subjects or all_subjs[-n_val:]
    print(f"  Val subjects: {val_subjs}")
    X_tr, y_tr, X_val, y_val, names_val, common_classes = subject_split(
        X, y, subjects, label_names, val_subjs
    )
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}  Classes: {len(common_classes)}")

    # Label name map
    label_map = {c: n for c, n in zip(y_val, names_val)}
    class_names = [label_map[c] for c in common_classes]

    results = {}
    for short, enc in encoders.items():
        print(f"\n{'='*55}")
        print(f"Model: {short}")
        print("="*55)
        results[short] = {}

        # ── Linear probe baseline ──────────────────────────────────────────────
        print("  [1/2] Linear probe (LogisticRegression) ...")
        F_tr  = extract_pooled(X_tr,  enc, device)
        F_val = extract_pooled(X_val, enc, device)
        clf   = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(F_tr, y_tr)
            preds_lin = clf.predict(F_val)
        bal_lin = balanced_accuracy_score(y_val, preds_lin)
        per_cls_lin = {label_map[c]: float(balanced_accuracy_score(
            y_val[y_val == c], preds_lin[y_val == c]
        )) for c in common_classes if (y_val == c).sum() > 0}
        print(f"  Linear probe bal_acc = {bal_lin:.4f}")
        results[short]["linear_probe"] = {"bal_acc": bal_lin, "per_class": per_cls_lin}

        # ── Attention probe ────────────────────────────────────────────────────
        print("  [2/2] Attention probe (training) ...")
        T_tr  = extract_tokens(X_tr,  enc, device)  # [N, 48, D]
        T_val = extract_tokens(X_val, enc, device)

        probe, best_bal, le = train_attention_probe(
            T_tr, y_tr, T_val, y_val, common_classes, enc.embed_dim, device,
            num_heads=args.num_heads, dropout=args.dropout,
            lr=args.lr, max_epochs=args.epochs, patience=args.patience,
        )
        per_cls_attn = per_class_accuracy(probe, T_val, y_val, le, class_names,
                                          device)
        print(f"  Attention probe bal_acc = {best_bal:.4f}")
        results[short]["attn_probe"] = {"bal_acc": best_bal, "per_class": per_cls_attn}

    # ── Summary ────────────────────────────────────────────────────────────────
    step_label = f"step_{step}" if step else "best"
    print(f"\n{'='*60}")
    print(f"{ds_name} Probing Comparison — {step_label}")
    print(f"{'Model':<20} {'Linear Probe':>14} {'Attention Probe':>16}")
    print("-"*60)
    for short in shorts:
        lp = results[short]["linear_probe"]["bal_acc"]
        ap = results[short]["attn_probe"]["bal_acc"]
        print(f"{short:<20} {lp:>14.4f} {ap:>16.4f}  (+{ap-lp:+.4f})")
    print("="*60)

    # Per-class table
    print(f"\n{'Class':<26}" +
          "".join(f"  {s[:10]+' lin':>14}{s[:10]+' attn':>14}" for s in shorts))
    print("-" * (26 + 28 * len(shorts)))
    for cn in class_names:
        row = f"{cn:<26}"
        for short in shorts:
            lp = results[short]["linear_probe"]["per_class"].get(cn, 0.0)
            ap = results[short]["attn_probe"]["per_class"].get(cn, 0.0)
            row += f"  {lp:>14.4f}{ap:>14.4f}"
        print(row)

    # Save CSV
    rows = []
    for cn in class_names:
        row = {"class": cn}
        for short in shorts:
            row[f"{short}_linear"] = results[short]["linear_probe"]["per_class"].get(cn, 0.0)
            row[f"{short}_attn"]   = results[short]["attn_probe"]["per_class"].get(cn, 0.0)
        rows.append(row)
    csv_path = os.path.join(args.out_dir, f"attn_probe_{step_label}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    plot_comparison(results, shorts, args.out_dir, ds_name=ds_name)


if __name__ == "__main__":
    main()
