#!/usr/bin/env python3
"""
Visualize DINO/iBOT masked patch prediction quality.

For each sequence, computes per-patch cosine similarity between:
  Teacher (EMA encoder, full signal, no mask) → ground-truth patch embeddings
  Student (trained encoder, 40% mask applied) → predicted patch embeddings

This directly measures the iBOT objective quality:
  • Visible patches  : student sees real data    → similarity should be HIGH
  • Masked patches   : student sees [MASK] token → similarity shows prediction quality

Output figures (saved to --out dir):
  dino_patch_pred_panels.png  — per-sequence panel: signal + per-patch similarity heatmap
  dino_patch_pred_summary.png — aggregate box-plot + masked vs visible separation

Usage:
    python scripts/vis_dino_patch_prediction.py \\
        --ckpt  runs/vit1d_dino_ocdetect_anyrot_v4/checkpoints/latest.pt \\
        --data  /mnt/nas2/yiquan_data/preprocessed \\
        --out   data_vis/dino_v4_features \\
        --n     8
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.models.ViT1D.model            import ViT1DEncoder

# ── constants ──────────────────────────────────────────────────────────────
WINDOW_LEN    = 128
N_CHANNELS    = 6
N_PATCHES     = WINDOW_LEN // 16   # 8 per channel
N_TOKENS      = N_CHANNELS * N_PATCHES   # 48
MASK_RATIO    = 0.40
OCDETECT_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
CH_NAMES      = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CH_COLORS     = ["#e74c3c", "#e67e22", "#f1c40f",
                 "#2980b9", "#27ae60", "#8e44ad"]


# ── loading ────────────────────────────────────────────────────────────────

def _remap(sd: dict) -> dict:
    return {k.replace(".self_attn.", ".attn.").replace(".linear1.", ".ff.0.")
             .replace(".linear2.", ".ff.3."): v for k, v in sd.items()}


def load_encoders(ckpt_path: str):
    """Return (student_enc, teacher_enc, cfg)."""
    state   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg     = state["cfg"]
    student = ViT1DEncoder(cfg)
    teacher = ViT1DEncoder(cfg)
    student.load_state_dict(_remap(state["model"]),            strict=True)
    teacher.load_state_dict(_remap(state["teacher_encoder"]),  strict=True)
    student.eval(); teacher.eval()
    return student, teacher, cfg


def sample_windows(data_dir: str, n: int, cfg: dict, seed: int = 42):
    import pandas as pd
    csvs = sorted(f for f in os.listdir(data_dir)
                  if f.endswith(".csv") and "metadata" not in f)
    rng = random.Random(seed)
    chosen = rng.sample(csvs, min(n * 3, len(csvs)))   # sample extra in case some fail

    windows, labels = [], []
    for fname in chosen:
        path = os.path.join(data_dir, fname)
        try:
            df    = pd.read_csv(path, usecols=OCDETECT_COLS).dropna()
            offset = 20 * (WINDOW_LEN // 2)
            if len(df) < offset + WINDOW_LEN:
                continue
            chunk = df.iloc[offset: offset + WINDOW_LEN][OCDETECT_COLS].values.astype(np.float32)
            w = preprocess_window(chunk, cfg)
            if w is None or w.shape != (N_CHANNELS, WINDOW_LEN):
                continue
            windows.append(w)
            labels.append(fname[:28])
        except Exception:
            continue
        if len(windows) == n:
            break
    return windows, labels


def make_mask(N: int, seed: int = 42) -> torch.BoolTensor:
    """[N_TOKENS] bool mask, True = masked — same mask reused across all samples."""
    torch.manual_seed(seed)
    n_mask = int(N_TOKENS * MASK_RATIO)
    idx    = torch.randperm(N_TOKENS)[:n_mask]
    m      = torch.zeros(N_TOKENS, dtype=torch.bool)
    m[idx] = True
    return m.unsqueeze(0).expand(N, -1)   # [N, N_TOKENS]


# ── per-patch similarity ───────────────────────────────────────────────────

@torch.no_grad()
def compute_patch_sim(
    student: ViT1DEncoder,
    teacher: ViT1DEncoder,
    windows: list[np.ndarray],
    mask: torch.BoolTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        sim_map  : [N, N_CH, N_PT]  per-patch cosine similarity (student-masked vs teacher-full)
        pat_s    : [N, N_TOKENS, D] student patch embeddings (masked input)
        pat_t    : [N, N_TOKENS, D] teacher patch embeddings (full input)
    """
    x = torch.from_numpy(np.stack(windows))          # [N, C, T]
    _, pat_s = student.forward_dino(x, patch_mask=mask)   # student masked  → [N, N_TOKENS, D]
    _, pat_t = teacher.forward_dino(x, patch_mask=None)   # teacher full    → [N, N_TOKENS, D]

    pat_s_n = F.normalize(pat_s, dim=-1)
    pat_t_n = F.normalize(pat_t, dim=-1)
    sim      = (pat_s_n * pat_t_n).sum(dim=-1)            # [N, N_TOKENS]  element-wise dot
    sim_map  = sim.reshape(len(windows), N_CHANNELS, N_PATCHES)
    return sim_map, pat_s, pat_t


# ── figure 1: per-sequence panels ─────────────────────────────────────────

def plot_panels(windows: list[np.ndarray], sim_map: torch.Tensor,
                mask: torch.BoolTensor, labels: list[str], out_path: str):
    N      = len(windows)
    mask_g = mask[0].reshape(N_CHANNELS, N_PATCHES).numpy()  # [CH, PT]

    fig = plt.figure(figsize=(14, N * 2.6))
    fig.suptitle(
        "iBOT patch prediction quality\n"
        "Student (masked input) vs Teacher (full input) — per-patch cosine similarity\n"
        "red border = masked patch  |  green = high similarity  |  purple = low/negative",
        fontsize=10, y=1.01,
    )
    outer = gridspec.GridSpec(N, 1, figure=fig, hspace=0.55)

    for i in range(N):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[i], wspace=0.4,
            width_ratios=[3, 2, 2],
        )
        ax_sig  = fig.add_subplot(inner[0, 0])
        ax_sim  = fig.add_subplot(inner[0, 1])
        ax_diff = fig.add_subplot(inner[0, 2])

        t = np.arange(WINDOW_LEN) / 50.0
        for c in range(N_CHANNELS):
            ax_sig.plot(t, windows[i][c], color=CH_COLORS[c],
                        lw=0.8, alpha=0.85, label=CH_NAMES[c])
        for p in range(1, N_PATCHES):
            ax_sig.axvline(p * 16 / 50.0, color="k", lw=0.4, alpha=0.3)
        # shade masked patches across full signal height
        for c_idx in range(N_CHANNELS):
            for p_idx in range(N_PATCHES):
                if mask_g[c_idx, p_idx]:
                    ax_sig.axvspan(p_idx * 16 / 50.0, (p_idx + 1) * 16 / 50.0,
                                   alpha=0.12, color="red")
        ax_sig.set_title(labels[i], fontsize=7, pad=2)
        ax_sig.set_ylabel("a.u.", fontsize=7)
        ax_sig.tick_params(labelsize=6)
        ax_sig.margins(x=0)

        # ── similarity heatmap ─────────────────────────────────────────────
        s = sim_map[i].cpu().numpy()              # [CH, PT]
        im = ax_sim.imshow(s, cmap="RdYlGn", vmin=-1, vmax=1,
                           aspect="auto", interpolation="nearest")
        ax_sim.set_yticks(range(N_CHANNELS))
        ax_sim.set_yticklabels(CH_NAMES, fontsize=6)
        ax_sim.set_xticks(range(N_PATCHES))
        ax_sim.set_xticklabels([f"p{p}" for p in range(N_PATCHES)], fontsize=6)
        ax_sim.set_title("Student–Teacher patch cos-sim\n(red border = masked)", fontsize=8, pad=2)
        plt.colorbar(im, ax=ax_sim, fraction=0.05, pad=0.02)
        # overlay masked positions with red border
        for c_idx in range(N_CHANNELS):
            for p_idx in range(N_PATCHES):
                if mask_g[c_idx, p_idx]:
                    ax_sim.add_patch(plt.Rectangle(
                        (p_idx - 0.5, c_idx - 0.5), 1, 1,
                        fill=False, edgecolor="red", lw=1.5))

        # ── bar chart: visible vs masked similarity for this sequence ──────
        s_flat   = sim_map[i].cpu().numpy().flatten()   # [N_TOKENS]
        m_flat   = mask[0].numpy()                       # [N_TOKENS]
        vis_sim  = s_flat[~m_flat]
        mask_sim = s_flat[m_flat]
        positions = [0, 1]
        bp = ax_diff.boxplot(
            [vis_sim, mask_sim],
            positions=positions,
            widths=0.5,
            patch_artist=True,
            medianprops=dict(color="black", lw=1.5),
        )
        bp["boxes"][0].set_facecolor("#2ecc71")   # visible
        bp["boxes"][1].set_facecolor("#e74c3c")   # masked
        ax_diff.set_xticks([0, 1])
        ax_diff.set_xticklabels(
            [f"Visible\n({(~m_flat).sum()})", f"Masked\n({m_flat.sum()})"],
            fontsize=8,
        )
        ax_diff.set_ylabel("Cos sim", fontsize=8)
        ax_diff.set_ylim(-1.05, 1.05)
        ax_diff.axhline(0, color="gray", lw=0.8, ls="--")
        ax_diff.set_title("Visible vs masked\nsimilarity", fontsize=8, pad=2)
        ax_diff.tick_params(labelsize=7)
        # annotate medians
        for pos, vals, col in zip([0, 1], [vis_sim, mask_sim], ["#27ae60", "#c0392b"]):
            ax_diff.text(pos, np.median(vals) + 0.06, f"{np.median(vals):.2f}",
                         ha="center", fontsize=7, color=col, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── figure 2: aggregate summary ───────────────────────────────────────────

def plot_summary(sim_map: torch.Tensor, mask: torch.BoolTensor,
                 labels: list[str], out_path: str):
    N       = sim_map.shape[0]
    m_flat  = mask[0].numpy()                          # [N_TOKENS]
    all_vis, all_msk = [], []
    per_seq_vis, per_seq_msk = [], []

    for i in range(N):
        s = sim_map[i].cpu().numpy().flatten()
        all_vis.extend(s[~m_flat].tolist())
        all_msk.extend(s[ m_flat].tolist())
        per_seq_vis.append(s[~m_flat].mean())
        per_seq_msk.append(s[ m_flat].mean())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── left: overall distribution ─────────────────────────────────────────
    bins = np.linspace(-1, 1, 50)
    axes[0].hist(all_vis, bins=bins, alpha=0.6, color="#2ecc71", density=True,
                 label=f"Visible  (n={len(all_vis)})\nμ={np.mean(all_vis):.3f}")
    axes[0].hist(all_msk, bins=bins, alpha=0.6, color="#e74c3c", density=True,
                 label=f"Masked   (n={len(all_msk)})\nμ={np.mean(all_msk):.3f}")
    axes[0].axvline(np.mean(all_vis), color="#27ae60", lw=1.5, ls="--")
    axes[0].axvline(np.mean(all_msk), color="#c0392b", lw=1.5, ls="--")
    axes[0].axvline(0, color="gray", lw=0.8, ls=":")
    axes[0].set_xlabel("Student–Teacher cosine similarity", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].set_title("All patches: visible vs masked\ndistribution", fontsize=10)
    axes[0].legend(fontsize=8)
    gap = np.mean(all_vis) - np.mean(all_msk)
    axes[0].text(0.04, 0.97,
                 f"Visible mean  = {np.mean(all_vis):+.3f}\n"
                 f"Masked  mean  = {np.mean(all_msk):+.3f}\n"
                 f"Gap           = {gap:+.3f}",
                 transform=axes[0].transAxes, va="top", fontsize=8,
                 bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8, pad=4))

    # ── middle: per-sequence average similarity ────────────────────────────
    x = np.arange(N)
    w = 0.35
    axes[1].bar(x - w/2, per_seq_vis, width=w, color="#2ecc71",
                label="Visible", alpha=0.85)
    axes[1].bar(x + w/2, per_seq_msk, width=w, color="#e74c3c",
                label="Masked",  alpha=0.85)
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=55, ha="right", fontsize=5)
    axes[1].set_ylabel("Mean cosine similarity", fontsize=10)
    axes[1].set_title("Per-sequence mean patch similarity", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-1, 1)

    # ── right: scatter — per-seq visible vs masked ─────────────────────────
    axes[2].scatter(per_seq_vis, per_seq_msk, s=50, alpha=0.8,
                    c=np.arange(N), cmap="tab20")
    for i, lbl in enumerate(labels):
        axes[2].annotate(lbl[:14], (per_seq_vis[i], per_seq_msk[i]),
                         fontsize=5, alpha=0.7)
    lim = [-0.1, 1.05]
    axes[2].plot(lim, lim, "k--", lw=0.8, label="y=x")
    axes[2].set_xlim(*lim); axes[2].set_ylim(-0.6, 1.05)
    axes[2].set_xlabel("Mean visible patch similarity", fontsize=10)
    axes[2].set_ylabel("Mean masked patch similarity", fontsize=10)
    axes[2].set_title("Visible vs masked per sequence\n(above diagonal = masked > visible)", fontsize=9)
    axes[2].legend(fontsize=8)
    axes[2].axhline(0, color="gray", lw=0.8, ls=":")

    fig.suptitle(
        f"iBOT masked patch prediction — student (masked) vs teacher (full)\n"
        f"N={N} sequences, mask_ratio={MASK_RATIO:.0%}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out",  default="data_vis/dino_v4_features")
    ap.add_argument("--n",    type=int, default=8,
                    help="Number of sequences to visualize in panel plot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading student + teacher encoders from {args.ckpt} ...")
    student, teacher, cfg = load_encoders(args.ckpt)

    print(f"Sampling {args.n} windows ...")
    windows, labels = sample_windows(args.data, args.n, cfg, seed=args.seed)
    N = len(windows)
    if N < 1:
        sys.exit("No valid windows loaded.")
    print(f"  loaded {N} windows")

    mask = make_mask(N, seed=args.seed)

    print("Computing per-patch student–teacher similarity ...")
    sim_map, _, _ = compute_patch_sim(student, teacher, windows, mask)

    out1 = os.path.join(args.out, "dino_patch_pred_panels.png")
    out2 = os.path.join(args.out, "dino_patch_pred_summary.png")
    plot_panels(windows, sim_map, mask, labels, out1)
    plot_summary(sim_map, mask, labels, out2)

    # ── console summary ────────────────────────────────────────────────────
    m_flat = mask[0].numpy()
    all_s  = sim_map.cpu().numpy().reshape(N, -1)
    vis_mean  = all_s[:, ~m_flat].mean()
    mask_mean = all_s[:,  m_flat].mean()
    print(f"\nMasked patch prediction quality (N={N}):")
    print(f"  Visible  patches mean cos-sim : {vis_mean:+.4f}")
    print(f"  Masked   patches mean cos-sim : {mask_mean:+.4f}")
    print(f"  Gap (visible − masked)        : {vis_mean - mask_mean:+.4f}")
    print(f"\n  Interpretation:")
    print(f"    masked sim > 0.5 → model predicts masked patches well")
    print(f"    masked sim ~ 0.0 → predictions are uninformative")
    print(f"    masked sim < 0.0 → predictions are worse than random")


if __name__ == "__main__":
    main()
