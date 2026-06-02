#!/usr/bin/env python3
"""
Visualize DINO invariance: 2N cosine-similarity matrix.

Loads N sequences from OCDetect and creates two views per sequence:
  View A (rows 0..N-1)    : full signal, no augmentation, no masking
  View B (rows N..2N-1)   : same signal + DINO augmentation + 40% random mask

The resulting 2N×2N cosine-similarity matrix has four quadrants:
  ┌──────────────┬──────────────┐
  │  A vs A      │  A vs B      │  ← top-right diagonal = invariance score
  │  (full–full) │  (full–aug)  │    same-sequence pairs should be bright
  ├──────────────┼──────────────┤
  │  B vs A      │  B vs B      │
  │  (aug–full)  │  (aug–aug)   │
  └──────────────┴──────────────┘

Good DINO representations show:
  • A–B diagonal (same sequence, different views) → HIGH (bright)
  • Off-diagonal A–B entries (different sequences) → LOW (dark)
  • The contrast between these two is the "invariance gap"

Also plots:
  • Histogram: diagonal vs off-diagonal A–B similarities
  • Bar chart: per-sequence invariance score (diag of top-right block)

Usage:
    python scripts/vis_dino_cosim_2n.py \\
        --ckpt  runs/vit1d_dino_ocdetect_anyrot_v4/checkpoints/latest.pt \\
        --data  /mnt/nas2/yiquan_data/preprocessed \\
        --out   data_vis/dino_v4_features \\
        --n     32
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.data.augmentations.transform  import apply_augment
from imu_lm.models.ViT1D.model            import ViT1DEncoder

# ── constants ──────────────────────────────────────────────────────────────
WINDOW_LEN  = 128
N_CHANNELS  = 6
N_PATCHES   = WINDOW_LEN // 16    # 8
N_TOKENS    = N_CHANNELS * N_PATCHES   # 48
MASK_RATIO  = 0.40
OCDETECT_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]


# ── helpers ────────────────────────────────────────────────────────────────

def load_encoder(ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg   = state["cfg"]
    enc   = ViT1DEncoder(cfg)
    sd    = {k.replace(".self_attn.", ".attn.").replace(".linear1.", ".ff.0.")
              .replace(".linear2.", ".ff.3."): v
             for k, v in state["model"].items()}
    enc.load_state_dict(sd, strict=True)
    enc.eval()
    return enc, cfg


def sample_windows(data_dir: str, n: int, cfg: dict,
                   seed: int = 42) -> tuple[list[np.ndarray], list[str]]:
    """Return (windows, labels) — one window per sampled file."""
    import pandas as pd
    csvs = sorted(f for f in os.listdir(data_dir)
                  if f.endswith(".csv") and "metadata" not in f)
    rng = random.Random(seed)
    chosen = rng.sample(csvs, min(n, len(csvs)))

    windows, labels = [], []
    for fname in chosen:
        path = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(path, usecols=OCDETECT_COLS).dropna()
            if len(df) < WINDOW_LEN + 20 * WINDOW_LEN // 2:
                continue
            offset = 20 * (WINDOW_LEN // 2)
            chunk  = df.iloc[offset: offset + WINDOW_LEN][OCDETECT_COLS].values.astype(np.float32)
            w = preprocess_window(chunk, cfg)   # [T,C] → [C,T]
            if w is None or w.shape != (N_CHANNELS, WINDOW_LEN):
                continue
            windows.append(w)
            labels.append(fname[:22])
        except Exception:
            continue
        if len(windows) == n:
            break
    return windows, labels


def make_masked_aug(windows: list[np.ndarray], cfg: dict,
                    seed: int = 0) -> tuple[list[np.ndarray], torch.BoolTensor]:
    """Augment each window and generate a random 40% mask."""
    rng_np = np.random.RandomState(seed)
    aug_windows = []
    for w in windows:
        # apply_augment expects [T, C]
        aug = apply_augment(w.T, cfg).T        # [C, T]
        aug_windows.append(aug)

    # fixed mask per sample
    torch.manual_seed(seed)
    n_mask = int(N_TOKENS * MASK_RATIO)
    masks  = []
    for _ in range(len(windows)):
        idx = torch.randperm(N_TOKENS)[:n_mask]
        m   = torch.zeros(N_TOKENS, dtype=torch.bool)
        m[idx] = True
        masks.append(m)
    return aug_windows, torch.stack(masks)


@torch.no_grad()
def get_cls(enc: ViT1DEncoder, windows: list[np.ndarray],
            mask: torch.BoolTensor | None = None) -> torch.Tensor:
    x = torch.from_numpy(np.stack(windows))    # [B, C, T]
    cls, _ = enc.forward_dino(x, patch_mask=mask)
    return F.normalize(cls, dim=-1)            # [B, D]


# ── figures ────────────────────────────────────────────────────────────────

def plot_2n_cosim(sim: np.ndarray, labels: list[str], N: int, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7),
                             gridspec_kw={"width_ratios": [4, 1.6, 1.6]})

    # ── left: 2N×2N similarity matrix ─────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(sim, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto",
                   interpolation="nearest")

    # quadrant borders
    ax.axhline(N - 0.5, color="white", lw=2)
    ax.axvline(N - 0.5, color="white", lw=2)

    # tick labels (every other to avoid clutter)
    ticks = list(range(0, N, max(1, N // 16)))
    ax.set_xticks(ticks + [t + N for t in ticks])
    ax.set_yticks(ticks + [t + N for t in ticks])
    xlabels = [labels[t] for t in ticks] + [labels[t] + "*" for t in ticks]
    ax.set_xticklabels(xlabels, rotation=60, ha="right", fontsize=5)
    ax.set_yticklabels(xlabels, fontsize=5)

    # quadrant annotations
    for (ry, cx, txt) in [(N/4, N/4,     "A vs A\n(full–full)"),
                           (N/4, N+N/4,   "A vs B\n(full–aug+mask)"),
                           (N+N/4, N/4,   "B vs A\n(aug+mask–full)"),
                           (N+N/4, N+N/4, "B vs B\n(aug+mask–aug+mask)")]:
        ax.text(cx, ry, txt, ha="center", va="center",
                fontsize=7, color="black",
                bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=1))

    plt.colorbar(im, ax=ax, fraction=0.03, label="Cosine similarity")
    ax.set_title(f"2N×2N cosine similarity  (N={N})\n"
                 "* = augmented + 40% masked view",
                 fontsize=10)

    # ── middle: diagonal vs off-diagonal histogram (A–B quadrant) ──────────
    ax2 = axes[1]
    ab_block = sim[:N, N:]                      # top-right N×N block
    diag_vals    = np.diag(ab_block)            # same sequence  (N,)
    off_mask     = ~np.eye(N, dtype=bool)
    offdiag_vals = ab_block[off_mask]           # different sequences

    bins = np.linspace(-1, 1, 40)
    ax2.hist(offdiag_vals, bins=bins, alpha=0.6, color="#e74c3c",
             label=f"Different seq\n(n={len(offdiag_vals)})", density=True)
    ax2.hist(diag_vals, bins=bins, alpha=0.8, color="#2ecc71",
             label=f"Same seq\n(n={len(diag_vals)})", density=True)
    ax2.axvline(diag_vals.mean(),    color="#27ae60", lw=1.5, ls="--",
                label=f"same mean={diag_vals.mean():.3f}")
    ax2.axvline(offdiag_vals.mean(), color="#c0392b", lw=1.5, ls="--",
                label=f"diff mean={offdiag_vals.mean():.3f}")
    ax2.set_xlabel("Cosine similarity", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("A–B similarity\ndistribution", fontsize=10)
    ax2.legend(fontsize=7)
    gap = diag_vals.mean() - offdiag_vals.mean()
    ax2.text(0.05, 0.97, f"Invariance gap\n{gap:+.3f}",
             transform=ax2.transAxes, va="top", fontsize=8,
             color="black",
             bbox=dict(facecolor="#f0f0f0", alpha=0.8, edgecolor="gray", pad=3))

    # ── right: per-sequence invariance score ───────────────────────────────
    ax3 = axes[2]
    order  = np.argsort(diag_vals)[::-1]
    colors = ["#2ecc71" if v > offdiag_vals.mean() + 0.05 else
              "#e67e22" if v > offdiag_vals.mean() else
              "#e74c3c" for v in diag_vals[order]]
    ax3.barh(range(N), diag_vals[order], color=colors, edgecolor="none", height=0.8)
    ax3.axvline(offdiag_vals.mean(), color="#c0392b", lw=1, ls="--",
                label="off-diag mean")
    ax3.set_yticks(range(N))
    ax3.set_yticklabels([labels[i] for i in order], fontsize=5)
    ax3.set_xlabel("A–B cosine similarity", fontsize=9)
    ax3.set_title("Per-sequence\ninvariance score", fontsize=10)
    ax3.legend(fontsize=7)
    ax3.set_xlim(-1, 1)

    fig.suptitle("DINO invariance: full signal vs augmented+masked view",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="DINO checkpoint .pt")
    ap.add_argument("--data", required=True, help="OCDetect preprocessed CSV dir")
    ap.add_argument("--out",  default="data_vis/dino_v4_features")
    ap.add_argument("--n",    type=int, default=32,
                    help="Number of sequences N (matrix will be 2N×2N)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading encoder from {args.ckpt} ...")
    enc, cfg = load_encoder(args.ckpt)

    print(f"Sampling {args.n} windows ...")
    windows, labels = sample_windows(args.data, args.n, cfg, seed=args.seed)
    N = len(windows)
    if N < 2:
        sys.exit("Not enough valid windows loaded.")
    print(f"  loaded {N} windows")

    print("Creating augmented + masked views ...")
    aug_windows, masks = make_masked_aug(windows, cfg, seed=args.seed)

    print("Encoding ...")
    cls_full = get_cls(enc, windows)                   # [N, D]  view A
    cls_aug  = get_cls(enc, aug_windows, mask=masks)   # [N, D]  view B

    # ── 2N×2N similarity matrix ────────────────────────────────────────────
    all_cls = torch.cat([cls_full, cls_aug], dim=0)    # [2N, D]
    sim = (all_cls @ all_cls.T).cpu().numpy()          # [2N, 2N]

    out_path = os.path.join(args.out, "dino_2n_cosim.png")
    plot_2n_cosim(sim, labels, N, out_path)

    # ── print summary ──────────────────────────────────────────────────────
    ab = sim[:N, N:]
    diag_mean    = float(np.diag(ab).mean())
    offdiag_mean = float(ab[~np.eye(N, dtype=bool)].mean())
    print(f"\nInvariance summary (N={N}):")
    print(f"  Same-sequence  A–B mean cosine sim : {diag_mean:+.4f}")
    print(f"  Cross-sequence A–B mean cosine sim : {offdiag_mean:+.4f}")
    print(f"  Invariance gap (same − cross)      : {diag_mean - offdiag_mean:+.4f}")


if __name__ == "__main__":
    main()
