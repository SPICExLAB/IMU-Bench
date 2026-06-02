#!/usr/bin/env python3
"""
Compare v3 vs v5_MoE on IMUPoser 23-joint full-body pose estimation
using a single attention-layer probe (frozen encoder).

Probe architecture:
    Frozen encoder  [B, 48, 384]  (6 channels × 8 time patches)
         ↓
    8 learnable time-patch queries cross-attend to all 48 tokens  (1 layer)
         ↓
    LayerNorm → Linear head → [B, 8, 23×6 = 138]  (6D rotations per patch)

Metric: mean geodesic rotation error per joint (degrees), averaged over
        all joints and time patches.  Lower = better.

Usage:
    python scripts/compare_pose_probe_imuposer.py \
        --runs vit1d_mae_realdata_ocdetect_flu_v3 \
               vit1d_mae_realdata_ocdetect_flu_v5_MoE \
        --step 80000 \
        --out-dir data_vis/pose_probe/
"""
from __future__ import annotations

import argparse
import os
import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.data.imuposer_wrist_full_body_loader_FLU import (
    make_imuposer_wrist_full_body_23_flu_loaders,
)
from imu_lm.utils.helpers import cfg_get

NUM_JOINTS  = 23
ROT6D_DIM   = 6
OUT_DIM     = NUM_JOINTS * ROT6D_DIM   # 138


# ── Rotation helpers ──────────────────────────────────────────────────────────

def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """[..., 6] → [..., 3, 3] via Gram-Schmidt orthogonalisation."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)   # [..., 3, 3]


def geodesic_error_deg(pred_6d: torch.Tensor, gt_6d: torch.Tensor) -> torch.Tensor:
    """Mean geodesic rotation error in degrees.

    Args:
        pred_6d : [..., J, 6]
        gt_6d   : [..., J, 6]
    Returns:
        scalar mean error in degrees
    """
    R_pred = rot6d_to_matrix(pred_6d)    # [..., J, 3, 3]
    R_gt   = rot6d_to_matrix(gt_6d)      # [..., J, 3, 3]
    R_err  = R_pred @ R_gt.transpose(-1, -2)
    trace  = R_err.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1 + 1e-6, 3 - 1e-6)
    angle  = torch.acos((trace - 1.0) / 2.0)   # [..., J] radians
    return angle.mean() * (180.0 / torch.pi)


# ── Attention pose probe ──────────────────────────────────────────────────────

class AttentionPoseProbe(nn.Module):
    """Single cross-attention layer probe for per-patch joint rotation prediction.

    8 learnable time-patch queries attend to all 48 encoder tokens.
    One attention layer, one linear head.  Encoder is frozen externally.

    Args:
        embed_dim  : encoder hidden size (384)
        num_patches: time patches = T // patch_size  (8 for 128 samples, ps=16)
        out_dim    : 23 * 6 = 138
        num_heads  : attention heads (default 4)
        dropout    : dropout probability
    """

    def __init__(self, embed_dim: int, num_patches: int, out_dim: int = OUT_DIM,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_patches = num_patches

        # Learnable time-patch queries — one per time step
        self.queries = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Single cross-attention layer: queries attend to encoder tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(embed_dim, out_dim)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens : [B, N, D]  frozen encoder output  (N = C * Pt = 48)
        Returns:
            [B, Pt, 138]  predicted 6D rotations for 23 joints per time patch
        """
        B  = tokens.shape[0]
        q  = self.queries.expand(B, -1, -1)          # [B, Pt, D]
        out, _ = self.cross_attn(q, tokens, tokens)  # [B, Pt, D]
        out = self.norm(out)                          # [B, Pt, D]
        return self.head(self.dropout(out))           # [B, Pt, 138]


# ── Model loading ─────────────────────────────────────────────────────────────

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


# ── Training & evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_tokens_batch(x: torch.Tensor, enc) -> torch.Tensor:
    """Forward tokens without masking → [B, 48, D]."""
    toks, _ = enc.forward_tokens(x.float())
    return toks


def train_probe(
    enc, probe, train_loader, val_loader, device,
    lr=3e-4, weight_decay=1e-4, max_epochs=60, patience=12,
):
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler    = GradScaler()

    best_val_err  = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(max_epochs):
        probe.train()
        total_loss = 0.0
        n_batches  = 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            # y: [B, Pt, 138]
            with torch.no_grad():
                toks = extract_tokens_batch(x, enc)  # [B, 48, D]

            optimizer.zero_grad()
            with autocast():
                pred = probe(toks)                   # [B, Pt, 138]
                loss = F.mse_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        val_err = evaluate_probe(enc, probe, val_loader, device)

        if val_err < best_val_err - 1e-4:
            best_val_err = val_err
            best_state   = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            no_improve   = 0
        else:
            no_improve  += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}  train_mse={total_loss/n_batches:.5f}"
                  f"  val_geo_err={val_err:.3f}°  best={best_val_err:.3f}°")

        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch + 1}")
            break

    probe.load_state_dict(best_state)
    probe.eval()
    return best_val_err


@torch.no_grad()
def evaluate_probe(enc, probe, loader, device) -> float:
    """Returns mean geodesic error in degrees."""
    probe.eval()
    errs = []
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        toks = extract_tokens_batch(x, enc)       # [B, 48, D]
        pred = probe(toks)                         # [B, Pt, 138]
        pred_j = pred.reshape(*pred.shape[:2], NUM_JOINTS, ROT6D_DIM)
        gt_j   = y.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
        errs.append(geodesic_error_deg(pred_j, gt_j).item())
    return float(np.mean(errs))


@torch.no_grad()
def per_joint_error(enc, probe, loader, device) -> np.ndarray:
    """Returns per-joint mean geodesic error [23] in degrees."""
    probe.eval()
    joint_errs = [[] for _ in range(NUM_JOINTS)]
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        toks = extract_tokens_batch(x, enc)
        pred = probe(toks)
        pred_j = pred.reshape(*pred.shape[:2], NUM_JOINTS, ROT6D_DIM)
        gt_j   = y.reshape(*y.shape[:2],   NUM_JOINTS, ROT6D_DIM)
        R_pred = rot6d_to_matrix(pred_j)
        R_gt   = rot6d_to_matrix(gt_j)
        R_err  = R_pred @ R_gt.transpose(-1, -2)
        trace  = R_err.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1 + 1e-6, 3 - 1e-6)
        angle  = torch.acos((trace - 1.0) / 2.0) * (180.0 / torch.pi)  # [B, Pt, J]
        for j in range(NUM_JOINTS):
            joint_errs[j].append(angle[:, :, j].mean().item())
    return np.array([float(np.mean(joint_errs[j])) for j in range(NUM_JOINTS)])


# ── Plotting ──────────────────────────────────────────────────────────────────

SMPL_23_JOINT_NAMES = [
    "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]


def plot_results(results: dict, shorts: list, step_label: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ── Figure 1: overall geodesic error bar chart ────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(shorts) * 2), 4.5))
    vals = [results[s]["geo_err"] for s in shorts]
    bars = ax.bar(shorts, vals,
                  color=[colors[i % len(colors)] for i in range(len(shorts))],
                  alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                f"{v:.3f}°", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Geodesic Error (degrees)", fontsize=11)
    ax.set_title(f"IMUPoser 23-Joint Pose Probe — {step_label}\n"
                 f"Frozen encoder + 1 cross-attention layer",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pose_probe_overall.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/pose_probe_overall.png")

    # ── Figure 2: per-joint error comparison ─────────────────────────────────
    n_joints = NUM_JOINTS
    y        = np.arange(n_joints)
    width    = 0.8 / len(shorts)

    fig, ax = plt.subplots(figsize=(9, max(6, n_joints * 0.4)))
    for ri, short in enumerate(shorts):
        per_j = results[short]["per_joint_err"]
        ax.barh(y + ri * width, per_j, height=width * 0.9,
                label=short, color=colors[ri % len(colors)], alpha=0.85)

    ax.set_yticks(y + width * (len(shorts) - 1) / 2)
    ax.set_yticklabels(SMPL_23_JOINT_NAMES, fontsize=8)
    ax.set_xlabel("Geodesic Error (degrees)", fontsize=10)
    ax.set_title(f"Per-Joint Pose Error — {step_label}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pose_probe_per_joint.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/pose_probe_per_joint.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["vit1d_mae_realdata_ocdetect_flu_v3",
                            "vit1d_mae_realdata_ocdetect_flu_v5_MoE"])
    p.add_argument("--runs-root",   default="runs")
    p.add_argument("--step",        type=int, default=None,
                   help="Checkpoint step (default: auto = max common step)")
    p.add_argument("--train-dir",   default="/mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/train")
    p.add_argument("--val-dir",     default="/mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test")
    p.add_argument("--num-heads",   type=int,   default=4)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--epochs",      type=int,   default=60)
    p.add_argument("--patience",    type=int,   default=12)
    p.add_argument("--out-dir",     default="data_vis/pose_probe")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Auto-detect max common step
    step = args.step
    if step is None:
        import re
        common = None
        for run in args.runs:
            paths  = glob.glob(os.path.join(args.runs_root, run, "checkpoints", "step_*.pt"))
            steps  = {int(re.search(r"step_(\d+)\.pt", p).group(1)) for p in paths}
            common = steps if common is None else common & steps
        if common:
            step = max(common)
            print(f"Auto-selected step={step}")

    step_label = f"step_{step}" if step else "best"

    # Load encoders
    encoders, cfgs, shorts = {}, {}, []
    for run in args.runs:
        print(f"\nLoading {run} ...")
        enc, cfg, short = load_encoder(os.path.join(args.runs_root, run), step, device)
        encoders[short] = enc
        cfgs[short]     = cfg
        shorts.append(short)

    base_cfg = cfgs[shorts[0]]

    # Load IMUPoser data
    print("\nLoading IMUPoser 23-joint FLU data ...")
    train_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))
    val_files   = sorted(glob.glob(os.path.join(args.val_dir,   "*.pkl")))
    print(f"  Train files: {len(train_files)}  Val files: {len(val_files)}")

    loaders = make_imuposer_wrist_full_body_23_flu_loaders(
        cfg=base_cfg,
        train_files=train_files,
        val_files=val_files,
        add_gravity=False,
    )
    train_loader = loaders["train_loader"]
    val_loader   = loaders["val_loader"]

    # Infer patch config from first encoder
    enc0   = encoders[shorts[0]]
    T      = int(round(float(cfg_get(base_cfg, ["windowing", "window_seconds"], 2.56)) * 50))
    Pt     = T // enc0.patch_size   # number of time patches (8)
    D      = enc0.embed_dim         # 384

    print(f"  T={T}  patch_size={enc0.patch_size}  Pt={Pt}  embed_dim={D}")

    results = {}
    for short, enc in encoders.items():
        print(f"\n{'='*55}")
        print(f"Model: {short}")
        print("="*55)

        probe = AttentionPoseProbe(
            embed_dim=D, num_patches=Pt,
            num_heads=args.num_heads, dropout=args.dropout,
        ).to(device)

        n_params = sum(p.numel() for p in probe.parameters())
        print(f"  Probe parameters: {n_params:,}")
        print(f"  Training ...")

        best_err = train_probe(
            enc, probe, train_loader, val_loader, device,
            lr=args.lr, max_epochs=args.epochs, patience=args.patience,
        )

        print(f"  Computing per-joint errors ...")
        per_j = per_joint_error(enc, probe, val_loader, device)

        results[short] = {
            "geo_err":       best_err,
            "per_joint_err": per_j,
        }
        print(f"  Final: mean geodesic error = {best_err:.3f}°")

    # Summary
    print(f"\n{'='*50}")
    print(f"IMUPoser 23-Joint Pose Probe — {step_label}")
    print(f"{'Model':<22} {'Geo Error (°)':>14}")
    print("-"*50)
    for short in shorts:
        print(f"{short:<22} {results[short]['geo_err']:>14.3f}")
    print("="*50)

    # Per-joint table
    print(f"\n{'Joint':<16}" + "".join(f"{s:>14}" for s in shorts))
    print("-" * (16 + 14 * len(shorts)))
    for j, jname in enumerate(SMPL_23_JOINT_NAMES):
        row = f"{jname:<16}"
        for short in shorts:
            row += f"{results[short]['per_joint_err'][j]:>14.3f}"
        print(row)

    # Save CSV
    import pandas as pd
    rows = [{"joint": SMPL_23_JOINT_NAMES[j],
             **{s: results[s]["per_joint_err"][j] for s in shorts}}
            for j in range(NUM_JOINTS)]
    csv_path = os.path.join(args.out_dir, f"pose_probe_{step_label}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved CSV → {csv_path}")

    plot_results(results, shorts, step_label, args.out_dir)


if __name__ == "__main__":
    main()
