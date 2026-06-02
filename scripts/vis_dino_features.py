#!/usr/bin/env python3
"""
Visualize DINO encoder representations: CLS token, patch features, and masked patches.

For each input window the script produces:
  Panel A — Raw 6-channel IMU signal with patch-boundary grid
  Panel B — Patch feature activation map  (6 ch × 8 time patches, colour = L2 norm)
  Panel C — CLS-to-patch cosine similarity (shows which patches the global repr attends to)
  Panel D — Same activation map but with 40% patches masked (BERT-style mask token)
  Panel E — Difference in activation norm: full – masked  (masked patches highlighted)

Plus two aggregate figures:
  Figure 2 — CLS cosine-similarity matrix across all loaded sequences
  Figure 3 — t-SNE of CLS embeddings (one point per window)

Usage:
    python scripts/vis_dino_features.py \\
        --ckpt  runs/vit1d_dino_ocdetect_anyrot_v4/checkpoints/best.pt \\
        --data  /mnt/nas2/yiquan_data/preprocessed \\
        --out   data_vis/dino_v4_features \\
        --n-seq 8         # number of OCDetect sequences to compare
        --tsne-n 200      # windows to include in t-SNE
"""

from __future__ import annotations

import argparse
import os
import sys
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.models.ViT1D.model import ViT1DEncoder

# ── constants ──────────────────────────────────────────────────────────────
WINDOW_LEN  = 128          # 2.56 s @ 50 Hz
PATCH_SIZE  = 16           # matches vit1d_dino_ocdetect_anyrot config
N_CHANNELS  = 6
N_PATCHES   = WINDOW_LEN // PATCH_SIZE  # 8 per channel
N_TOKENS    = N_CHANNELS * N_PATCHES    # 48 total patch tokens
MASK_RATIO  = 0.40

CH_NAMES    = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CH_COLORS   = ["#e74c3c", "#e67e22", "#f1c40f",   # acc: red / orange / yellow
               "#2980b9", "#27ae60", "#8e44ad"]    # gyro: blue / green / purple
OCDetect_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]


# ── data loading ───────────────────────────────────────────────────────────

def _load_csv_windows(csv_path: str, cfg: dict, n_windows: int = 1,
                      start_offset: int = 30) -> list[np.ndarray]:
    """Return a list of (6, WINDOW_LEN) float32 arrays from an OCDetect CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=OCDetect_COLS)
    df = df.dropna().reset_index(drop=True)
    windows = []
    hop = WINDOW_LEN // 2
    for i in range(n_windows):
        idx = (start_offset + i) * hop
        if idx + WINDOW_LEN > len(df):
            break
        chunk = df.iloc[idx: idx + WINDOW_LEN][OCDetect_COLS].values.astype(np.float32)
        w = preprocess_window(chunk, cfg)     # input [T,C] → returns [C,T]
        windows.append(w)
    return windows


def _sample_csvs(data_dir: str, n: int, seed: int = 0) -> list[str]:
    """Return n randomly sampled CSV paths from data_dir."""
    csvs = sorted(f for f in os.listdir(data_dir)
                  if f.endswith(".csv") and "metadata" not in f)
    rng = random.Random(seed)
    return [os.path.join(data_dir, f) for f in rng.sample(csvs, min(n, len(csvs)))]


# ── encoder helpers ────────────────────────────────────────────────────────

def load_encoder(ckpt_path: str) -> tuple[ViT1DEncoder, dict]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg   = state["cfg"]
    enc   = ViT1DEncoder(cfg)
    # remap old key names if needed
    sd = {k.replace(".self_attn.", ".attn.").replace(".linear1.", ".ff.0.")
           .replace(".linear2.", ".ff.3."): v
          for k, v in state["model"].items()}
    enc.load_state_dict(sd, strict=True)
    enc.eval()
    return enc, cfg


def random_mask(B: int, N: int, ratio: float, seed: int | None = None) -> torch.BoolTensor:
    """Return [B, N] bool mask, True = masked."""
    rng  = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    n_mask = int(N * ratio)
    masks  = []
    for _ in range(B):
        idx = torch.randperm(N, generator=rng)[:n_mask]
        m   = torch.zeros(N, dtype=torch.bool)
        m[idx] = True
        masks.append(m)
    return torch.stack(masks)


@torch.no_grad()
def encode(enc: ViT1DEncoder, x: torch.Tensor,
           mask: torch.BoolTensor | None = None):
    """Run forward_dino and return (cls [B,D], patches [B,N,D])."""
    return enc.forward_dino(x, patch_mask=mask)


# ── per-sequence figure ────────────────────────────────────────────────────

def _patch_activation(patches: torch.Tensor) -> np.ndarray:
    """[N, D] → (N_CH, N_PT) float32 activation (L2 norm)."""
    norms = patches.norm(dim=-1).cpu().numpy()          # (N,)
    return norms.reshape(N_CHANNELS, N_PATCHES)


def _cls_patch_sim(cls: torch.Tensor, patches: torch.Tensor) -> np.ndarray:
    """Cosine similarity of CLS [D] to each patch [N, D] → (N_CH, N_PT)."""
    cls_n = F.normalize(cls.unsqueeze(0), dim=-1)       # [1, D]
    pat_n = F.normalize(patches, dim=-1)                 # [N, D]
    sim   = (pat_n @ cls_n.T).squeeze(-1)                # [N]
    return sim.cpu().numpy().reshape(N_CHANNELS, N_PATCHES)


def plot_sequence_panel(ax_signal, ax_full, ax_sim, ax_mask, ax_diff,
                        signal: np.ndarray,
                        cls_full: torch.Tensor, pat_full: torch.Tensor,
                        cls_mask: torch.Tensor, pat_mask: torch.Tensor,
                        mask: torch.BoolTensor,
                        seq_label: str):
    """Fill one row of the per-sequence figure."""
    t = np.arange(WINDOW_LEN) / 50.0   # seconds

    # ── A: raw signal ──────────────────────────────────────────────────────
    for c in range(N_CHANNELS):
        ax_signal.plot(t, signal[c], color=CH_COLORS[c],
                       lw=0.8, label=CH_NAMES[c], alpha=0.85)
    for p in range(1, N_PATCHES):
        ax_signal.axvline(p * PATCH_SIZE / 50.0, color="k", lw=0.4, alpha=0.3)
    ax_signal.set_title(seq_label, fontsize=8, pad=2)
    ax_signal.set_ylabel("a.u.", fontsize=7)
    ax_signal.tick_params(labelsize=6)
    ax_signal.margins(x=0)

    kw = dict(aspect="auto", interpolation="nearest")

    # ── B: full patch activation ───────────────────────────────────────────
    act_full = _patch_activation(pat_full[0])
    im = ax_full.imshow(act_full, cmap="viridis", **kw)
    ax_full.set_yticks(range(N_CHANNELS)); ax_full.set_yticklabels(CH_NAMES, fontsize=6)
    ax_full.set_xticks(range(N_PATCHES))
    ax_full.set_xticklabels([f"p{i}" for i in range(N_PATCHES)], fontsize=6)
    ax_full.set_title("Patch activation (full)", fontsize=8, pad=2)
    plt.colorbar(im, ax=ax_full, fraction=0.04, pad=0.02)

    # Overlay masked patches with hatching
    mask_grid = mask[0].cpu().numpy().reshape(N_CHANNELS, N_PATCHES)
    for r in range(N_CHANNELS):
        for c in range(N_PATCHES):
            if mask_grid[r, c]:
                ax_full.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="red", lw=0.5))

    # ── C: CLS–patch cosine similarity ────────────────────────────────────
    sim = _cls_patch_sim(cls_full[0], pat_full[0])
    im2 = ax_sim.imshow(sim, cmap="RdYlGn", vmin=-1, vmax=1, **kw)
    ax_sim.set_yticks(range(N_CHANNELS)); ax_sim.set_yticklabels(CH_NAMES, fontsize=6)
    ax_sim.set_xticks(range(N_PATCHES))
    ax_sim.set_xticklabels([f"p{i}" for i in range(N_PATCHES)], fontsize=6)
    ax_sim.set_title("CLS↔patch cosine sim", fontsize=8, pad=2)
    plt.colorbar(im2, ax=ax_sim, fraction=0.04, pad=0.02)

    # ── D: masked patch activation ────────────────────────────────────────
    act_mask = _patch_activation(pat_mask[0])
    im3 = ax_mask.imshow(act_mask, cmap="viridis", **kw)
    ax_mask.set_yticks(range(N_CHANNELS)); ax_mask.set_yticklabels(CH_NAMES, fontsize=6)
    ax_mask.set_xticks(range(N_PATCHES))
    ax_mask.set_xticklabels([f"p{i}" for i in range(N_PATCHES)], fontsize=6)
    ax_mask.set_title("Patch activation (40% masked)", fontsize=8, pad=2)
    # highlight masked positions in red border
    for r in range(N_CHANNELS):
        for c in range(N_PATCHES):
            if mask_grid[r, c]:
                ax_mask.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    fill=True, facecolor="red", alpha=0.35, lw=0))
    plt.colorbar(im3, ax=ax_mask, fraction=0.04, pad=0.02)

    # ── E: activation difference full − masked ────────────────────────────
    diff = act_full - act_mask
    lim  = np.abs(diff).max() or 1e-6
    im4  = ax_diff.imshow(diff, cmap="bwr", vmin=-lim, vmax=lim, **kw)
    ax_diff.set_yticks(range(N_CHANNELS)); ax_diff.set_yticklabels(CH_NAMES, fontsize=6)
    ax_diff.set_xticks(range(N_PATCHES))
    ax_diff.set_xticklabels([f"p{i}" for i in range(N_PATCHES)], fontsize=6)
    ax_diff.set_title("Δ activation (full − masked)", fontsize=8, pad=2)
    plt.colorbar(im4, ax=ax_diff, fraction=0.04, pad=0.02)


# ── aggregate figures ──────────────────────────────────────────────────────

def plot_cls_similarity_matrix(cls_list: list[torch.Tensor],
                               labels: list[str], out_path: str):
    n   = len(cls_list)
    mat = torch.stack(cls_list)                 # [n, D]
    mat = F.normalize(mat, dim=-1)
    sim = (mat @ mat.T).cpu().numpy()           # [n, n]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.65)))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    fontsize=5, color="black" if abs(sim[i,j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax, fraction=0.04, label="Cosine similarity")
    ax.set_title("CLS token cosine similarity across sequences", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


def plot_tsne(cls_list: list[torch.Tensor], labels: list[str], out_path: str):
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  [skip t-SNE] sklearn not available")
        return
    mat  = F.normalize(torch.stack(cls_list), dim=-1).numpy()
    perp = min(30, max(5, len(mat) // 3))
    emb  = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(mat)

    # colour by unique label (sequence file)
    unique = sorted(set(labels))
    cmap   = plt.cm.get_cmap("tab20", len(unique))
    c_idx  = {l: i for i, l in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (xy, lab) in enumerate(zip(emb, labels)):
        ax.scatter(xy[0], xy[1], color=cmap(c_idx[lab]), s=20, alpha=0.7)
    # legend (one entry per unique label)
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=cmap(c_idx[l]), markersize=6, label=l)
               for l in unique]
    ax.legend(handles=handles, fontsize=6, loc="best",
              title="Sequence", title_fontsize=7,
              ncol=max(1, len(unique) // 12))
    ax.set_title("t-SNE of DINO CLS embeddings", fontsize=11)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt",   required=True, help="Path to DINO checkpoint .pt")
    ap.add_argument("--data",   required=True, help="OCDetect preprocessed CSV dir")
    ap.add_argument("--out",    default="data_vis/dino_features",
                    help="Output directory")
    ap.add_argument("--n-seq",  type=int, default=6,
                    help="Number of OCDetect sequences to compare")
    ap.add_argument("--tsne-n", type=int, default=200,
                    help="Total windows to include in t-SNE plot")
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── load model ─────────────────────────────────────────────────────────
    print(f"Loading encoder from {args.ckpt} ...")
    enc, cfg = load_encoder(args.ckpt)

    # ── sample CSV files ───────────────────────────────────────────────────
    csvs = _sample_csvs(args.data, args.n_seq, seed=args.seed)
    if not csvs:
        sys.exit(f"No CSV files found in {args.data}")
    print(f"Using {len(csvs)} CSV sequences")

    # ── load one window per sequence ───────────────────────────────────────
    windows, seq_labels = [], []
    for csv_path in csvs:
        wins = _load_csv_windows(csv_path, cfg, n_windows=1, start_offset=20)
        if wins:
            windows.append(wins[0])
            seq_labels.append(os.path.basename(csv_path)[:30])

    if not windows:
        sys.exit("Could not load any windows from the provided CSV files.")

    # ── build fixed mask (same mask applied to all sequences for comparability)
    N = N_TOKENS
    mask = random_mask(len(windows), N, MASK_RATIO, seed=args.seed)  # [B, N]

    # ── encode all sequences at once ───────────────────────────────────────
    x = torch.from_numpy(np.stack(windows))        # [B, 6, 128]
    with torch.no_grad():
        cls_full, pat_full = encode(enc, x)         # [B,D], [B,N,D]
        cls_mask, pat_mask = encode(enc, x, mask)   # masked

    # ── Figure 1: per-sequence panels ──────────────────────────────────────
    n_rows = len(windows)
    n_cols = 5   # signal | full act | cls sim | masked act | diff

    col_ratios = [3, 2, 2, 2, 2]
    fig = plt.figure(figsize=(sum(col_ratios) * 2.5, n_rows * 2.8))
    fig.suptitle("DINO encoder features: CLS, patch activations, masked comparison",
                 fontsize=11, y=1.01)

    outer = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.55)

    for i in range(n_rows):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=outer[i], wspace=0.4,
            width_ratios=col_ratios,
        )
        axes = [fig.add_subplot(inner[0, j]) for j in range(n_cols)]

        # single-sample tensors
        c_f = cls_full[i: i+1]
        p_f = pat_full[i: i+1]
        c_m = cls_mask[i: i+1]
        p_m = pat_mask[i: i+1]
        mk  = mask[i: i+1]

        plot_sequence_panel(
            ax_signal=axes[0], ax_full=axes[1], ax_sim=axes[2],
            ax_mask=axes[3],   ax_diff=axes[4],
            signal=windows[i], cls_full=c_f, pat_full=p_f,
            cls_mask=c_m,      pat_mask=p_m, mask=mk,
            seq_label=seq_labels[i],
        )

    # shared legend for signal channels
    patches = [matplotlib.patches.Patch(color=CH_COLORS[c], label=CH_NAMES[c])
               for c in range(N_CHANNELS)]
    fig.legend(handles=patches, loc="lower center", ncol=N_CHANNELS,
               fontsize=7, title="IMU channel", title_fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    out1 = os.path.join(args.out, "dino_sequence_panels.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out1}")

    # ── Figure 2: CLS similarity matrix ────────────────────────────────────
    out2 = os.path.join(args.out, "dino_cls_similarity_matrix.png")
    plot_cls_similarity_matrix(
        [cls_full[i] for i in range(len(windows))],
        seq_labels, out2,
    )

    # ── Figure 3: t-SNE (load more windows for density) ───────────────────
    print(f"Loading {args.tsne_n} windows for t-SNE ...")
    all_wins, all_labels = [], []
    wins_per_file = max(1, args.tsne_n // max(1, len(csvs)))

    # also load extra files if needed
    extra_csvs = _sample_csvs(args.data,
                               min(args.tsne_n, 30),
                               seed=args.seed + 1)
    for csv_path in extra_csvs:
        wins = _load_csv_windows(csv_path, cfg,
                                 n_windows=wins_per_file,
                                 start_offset=5)
        lbl  = os.path.basename(csv_path)[:20]
        for w in wins:
            all_wins.append(w)
            all_labels.append(lbl)
        if len(all_wins) >= args.tsne_n:
            break

    if all_wins:
        x_all = torch.from_numpy(np.stack(all_wins[:args.tsne_n]))
        with torch.no_grad():
            cls_all, _ = encode(enc, x_all)
        out3 = os.path.join(args.out, "dino_cls_tsne.png")
        plot_tsne([cls_all[i] for i in range(len(cls_all))],
                  all_labels[:args.tsne_n], out3)

    print(f"\nAll figures saved to {args.out}/")


if __name__ == "__main__":
    import matplotlib
    main()
