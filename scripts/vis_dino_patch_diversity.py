#!/usr/bin/env python3
"""
Visualize patch-level feature diversity in a DINO/iBOT encoder.

Checks whether patch embeddings are meaningfully different per position/channel
(healthy) or all collapse to nearly the same vector (degenerate).

Four panels saved to --out:
  dino_patch_diversity_sim.png   — intra-sequence patch×patch cosine-similarity matrices
  dino_patch_diversity_pca.png   — PCA of patch embeddings coloured by channel / position
  dino_patch_diversity_var.png   — per-dimension feature variance + norm heatmap
  dino_patch_diversity_inter.png — inter-sequence patch similarity (same position, diff seq)

Usage:
    python scripts/vis_dino_patch_diversity.py \\
        --ckpt  runs/vit1d_dino_ocdetect_anyrot_v4/checkpoints/latest.pt \\
        --data  /mnt/nas2/yiquan_data/preprocessed \\
        --out   data_vis/dino_v4_features \\
        --n     16
"""

from __future__ import annotations
import argparse, os, random, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.models.ViT1D.model            import ViT1DEncoder

WINDOW_LEN  = 128
N_CHANNELS  = 6
N_PATCHES   = WINDOW_LEN // 16   # 8 per channel
N_TOKENS    = N_CHANNELS * N_PATCHES   # 48
OCDETECT_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
CH_NAMES      = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def _remap(sd):
    return {k.replace(".self_attn.", ".attn.").replace(".linear1.", ".ff.0.")
             .replace(".linear2.", ".ff.3."): v for k, v in sd.items()}


def load_encoder(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg   = state["cfg"]
    # prefer teacher (EMA, no dropout noise)
    key   = "teacher_encoder" if "teacher_encoder" in state else "model"
    enc   = ViT1DEncoder(cfg)
    enc.load_state_dict(_remap(state[key]), strict=True)
    enc.eval()
    return enc, cfg, key


def sample_windows(data_dir, n, cfg, seed=42):
    import pandas as pd
    csvs = sorted(f for f in os.listdir(data_dir)
                  if f.endswith(".csv") and "metadata" not in f)
    rng  = random.Random(seed)
    chosen = rng.sample(csvs, min(n * 4, len(csvs)))
    windows, labels = [], []
    for fname in chosen:
        try:
            df    = pd.read_csv(os.path.join(data_dir, fname),
                                usecols=OCDETECT_COLS).dropna()
            offset = 20 * (WINDOW_LEN // 2)
            if len(df) < offset + WINDOW_LEN:
                continue
            chunk = df.iloc[offset: offset + WINDOW_LEN][OCDETECT_COLS].values.astype(np.float32)
            w = preprocess_window(chunk, cfg)
            if w is None or w.shape != (N_CHANNELS, WINDOW_LEN):
                continue
            windows.append(w); labels.append(fname[:22])
        except Exception:
            continue
        if len(windows) == n:
            break
    return windows, labels


@torch.no_grad()
def get_patches(enc, windows):
    x = torch.from_numpy(np.stack(windows))   # [N, C, T]
    _, patches = enc.forward_dino(x, patch_mask=None)   # [N, N_TOKENS, D]
    return patches.cpu()


# ── Figure 1: intra-sequence patch×patch cosine-similarity ─────────────────

def plot_intra_sim(patches, labels, out_path, n_show=8):
    """patches: [N, N_TOKENS, D]"""
    N = min(n_show, patches.shape[0])
    cols = min(4, N)
    rows = (N + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = np.array(axes).flatten()

    for i in range(N):
        p   = F.normalize(patches[i], dim=-1)     # [N_TOKENS, D]
        sim = (p @ p.T).numpy()                   # [N_TOKENS, N_TOKENS]
        im  = axes[i].imshow(sim, cmap="RdYlGn", vmin=-1, vmax=1,
                             aspect="auto", interpolation="nearest")
        # channel separators
        for k in range(1, N_CHANNELS):
            axes[i].axhline(k * N_PATCHES - 0.5, color="white", lw=0.8)
            axes[i].axvline(k * N_PATCHES - 0.5, color="white", lw=0.8)
        axes[i].set_title(labels[i], fontsize=6, pad=2)
        tick_pos = [k * N_PATCHES + N_PATCHES // 2 for k in range(N_CHANNELS)]
        axes[i].set_xticks(tick_pos); axes[i].set_yticks(tick_pos)
        axes[i].set_xticklabels(CH_NAMES, fontsize=5, rotation=45, ha="right")
        axes[i].set_yticklabels(CH_NAMES, fontsize=5)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        # report mean off-diagonal similarity
        off = sim[~np.eye(N_TOKENS, dtype=bool)]
        axes[i].text(0.02, 0.98,
                     f"mean off-diag\n{off.mean():.3f}",
                     transform=axes[i].transAxes,
                     va="top", fontsize=7, color="white",
                     bbox=dict(facecolor="black", alpha=0.55, pad=2))

    for j in range(N, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Intra-sequence patch×patch cosine similarity  (teacher encoder, no mask)\n"
        "Healthy: diagonal bright, off-diagonal mixed   |   Collapsed: entire matrix bright",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")

    # return mean off-diag per sequence for summary
    means = []
    for i in range(patches.shape[0]):
        p   = F.normalize(patches[i], dim=-1)
        sim = (p @ p.T).numpy()
        means.append(sim[~np.eye(N_TOKENS, dtype=bool)].mean())
    return np.array(means)


# ── Figure 2: PCA of all patch embeddings ──────────────────────────────────

def plot_pca(patches, labels, out_path):
    """patches: [N, N_TOKENS, D]"""
    N, T, D = patches.shape
    flat = patches.reshape(N * T, D).numpy()        # [N*T, D]
    flat_n = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat_n)              # [N*T, 2]

    ch_idx  = np.tile(np.repeat(np.arange(N_CHANNELS), N_PATCHES), N)  # [N*T]
    pt_idx  = np.tile(np.tile(np.arange(N_PATCHES), N_CHANNELS), N)    # [N*T]
    seq_idx = np.repeat(np.arange(N), N_TOKENS)                         # [N*T]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    var_exp = pca.explained_variance_ratio_

    # colour by channel
    cmap_ch = plt.cm.get_cmap("tab10", N_CHANNELS)
    for c in range(N_CHANNELS):
        m = ch_idx == c
        axes[0].scatter(coords[m, 0], coords[m, 1],
                        color=cmap_ch(c), s=6, alpha=0.4, label=CH_NAMES[c])
    axes[0].set_title(f"PCA coloured by channel\n"
                      f"PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}", fontsize=9)
    axes[0].legend(fontsize=6, markerscale=2)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    # colour by patch position within channel
    cmap_pt = plt.cm.get_cmap("viridis", N_PATCHES)
    sc = axes[1].scatter(coords[:, 0], coords[:, 1],
                         c=pt_idx, cmap="viridis", vmin=0, vmax=N_PATCHES - 1,
                         s=6, alpha=0.4)
    plt.colorbar(sc, ax=axes[1], label="patch index within channel")
    axes[1].set_title("PCA coloured by patch position\n"
                      "(temporal order within channel)", fontsize=9)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    # colour by sequence identity
    cmap_seq = plt.cm.get_cmap("tab20", N)
    sc2 = axes[2].scatter(coords[:, 0], coords[:, 1],
                          c=seq_idx, cmap="tab20", vmin=0, vmax=N - 1,
                          s=6, alpha=0.4)
    plt.colorbar(sc2, ax=axes[2], label="sequence index")
    axes[2].set_title("PCA coloured by sequence identity\n"
                      "(good: sequences mix; bad: clusters by sequence)", fontsize=9)
    axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")

    # collapse score: std of PC1 coords — higher = more spread = less collapsed
    pc1_std = coords[:, 0].std()
    fig.suptitle(
        f"PCA of ALL patch embeddings  (N={N} seqs × {T} tokens, D={D})\n"
        f"PC1 std = {pc1_std:.4f}   "
        f"Collapsed if std ≈ 0 or all tokens cluster at single point",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")
    return pc1_std


# ── Figure 3: per-dimension variance + norm heatmap ─────────────────────────

def plot_variance(patches, out_path):
    """patches: [N, N_TOKENS, D]"""
    N, T, D = patches.shape
    flat = patches.reshape(N * T, D).numpy()        # [N*T, D]

    # std across all tokens for each dim
    dim_std = flat.std(axis=0)                      # [D]

    # norm per token (averaged across sequences)
    norms = np.linalg.norm(patches.numpy(), axis=-1)  # [N, T]
    norm_by_ch_pt = norms.mean(axis=0).reshape(N_CHANNELS, N_PATCHES)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # per-dim std histogram
    axes[0].hist(dim_std, bins=50, color="#3498db", edgecolor="white", lw=0.5)
    axes[0].axvline(dim_std.mean(), color="red", lw=1.5, ls="--",
                    label=f"mean std = {dim_std.mean():.4f}")
    axes[0].set_xlabel("Std dev across all tokens", fontsize=10)
    axes[0].set_ylabel("# dimensions", fontsize=10)
    axes[0].set_title(f"Per-embedding-dimension std\nD={D} dims  "
                      f"(collapse if most dims near 0)", fontsize=9)
    axes[0].legend(fontsize=8)
    frac_dead = (dim_std < 0.01).mean()
    axes[0].text(0.97, 0.97, f"{frac_dead:.1%} dims std<0.01",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=8, color="red")

    # norm heatmap [CH, PT]
    im = axes[1].imshow(norm_by_ch_pt, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=axes[1], label="mean L2 norm")
    axes[1].set_yticks(range(N_CHANNELS)); axes[1].set_yticklabels(CH_NAMES, fontsize=8)
    axes[1].set_xticks(range(N_PATCHES))
    axes[1].set_xticklabels([f"p{p}" for p in range(N_PATCHES)], fontsize=8)
    axes[1].set_title("Mean L2 norm per patch position\n"
                      "(very uniform = possible collapse)", fontsize=9)

    # cosine similarity to mean vector (dead-feature test)
    mean_vec = flat.mean(axis=0, keepdims=True)         # [1, D]
    mean_vec_n = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
    flat_n   = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
    sim_to_mean = (flat_n @ mean_vec_n.T).squeeze()     # [N*T]
    axes[2].hist(sim_to_mean, bins=50, color="#e74c3c", edgecolor="white", lw=0.5)
    axes[2].axvline(sim_to_mean.mean(), color="black", lw=1.5, ls="--",
                    label=f"mean = {sim_to_mean.mean():.4f}")
    axes[2].set_xlabel("Cosine similarity to mean embedding", fontsize=10)
    axes[2].set_ylabel("Count", fontsize=10)
    axes[2].set_title("Each patch's similarity to the mean patch vector\n"
                      "(collapse if all sim ≈ 1.0)", fontsize=9)
    axes[2].legend(fontsize=8)
    frac_high = (sim_to_mean > 0.95).mean()
    axes[2].text(0.03, 0.97, f"{frac_high:.1%} patches have\nsim>0.95 to mean",
                 transform=axes[2].transAxes, ha="left", va="top",
                 fontsize=8, color="red" if frac_high > 0.5 else "green")

    fig.suptitle(
        "Patch embedding diversity diagnostics  (teacher encoder, full signal)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")

    return dim_std.mean(), sim_to_mean.mean(), frac_high


# ── Figure 4: inter-sequence similarity for same patch position ─────────────

def plot_inter_sim(patches, out_path):
    """
    For each patch position p in [0..N_TOKENS),
    compute mean cosine similarity across different sequences.

    If inter-seq sim is HIGH for all positions → content-invariant (bad)
    If inter-seq sim is LOW → patches encode sequence-specific content (good)
    """
    N, T, D = patches.shape
    p_norm = F.normalize(patches, dim=-1)  # [N, T, D]

    # for each token position: mean pairwise sim across sequences
    inter_sim = np.zeros(T)
    for t in range(T):
        vecs = p_norm[:, t, :]                    # [N, D]
        sim  = (vecs @ vecs.T).numpy()            # [N, N]
        off  = sim[~np.eye(N, dtype=bool)]
        inter_sim[t] = off.mean()

    inter_map = inter_sim.reshape(N_CHANNELS, N_PATCHES)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    im = axes[0].imshow(inter_map, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_yticks(range(N_CHANNELS)); axes[0].set_yticklabels(CH_NAMES, fontsize=8)
    axes[0].set_xticks(range(N_PATCHES))
    axes[0].set_xticklabels([f"p{p}" for p in range(N_PATCHES)], fontsize=8)
    plt.colorbar(im, ax=axes[0], label="mean inter-seq cos sim")
    axes[0].set_title(
        "Inter-sequence cosine similarity at each patch position\n"
        "Green (high) = all sequences look the same here  |  Red = content-discriminative",
        fontsize=9,
    )

    # bar chart of inter_sim sorted
    order = np.argsort(inter_sim)[::-1]
    colors = ["#e74c3c" if v > 0.8 else "#e67e22" if v > 0.5 else "#2ecc71"
              for v in inter_sim[order]]
    axes[1].bar(range(T), inter_sim[order], color=colors, edgecolor="none", width=0.8)
    axes[1].axhline(inter_sim.mean(), color="black", lw=1.5, ls="--",
                    label=f"mean = {inter_sim.mean():.3f}")
    axes[1].set_xlabel("Patch token (sorted by inter-seq sim)", fontsize=9)
    axes[1].set_ylabel("Mean inter-sequence cosine sim", fontsize=9)
    axes[1].set_title(
        f"Per-token inter-sequence similarity distribution\n"
        f"Red = nearly sequence-invariant token  (collapse signal)",
        fontsize=9,
    )
    axes[1].legend(fontsize=8)
    frac_collapsed = (inter_sim > 0.8).mean()
    axes[1].text(0.97, 0.97,
                 f"{frac_collapsed:.1%} tokens\ninter-sim > 0.8",
                 transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=9,
                 color="red" if frac_collapsed > 0.3 else "green")

    fig.suptitle(
        f"Inter-sequence patch similarity  (N={N} sequences)\n"
        "Healthy: most positions are low (each sequence encodes unique content)  |  "
        "Collapsed: all positions high",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")

    return inter_sim.mean(), frac_collapsed


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out",  default="data_vis/dino_v4_features")
    ap.add_argument("--n",    type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading encoder from {args.ckpt} ...")
    enc, cfg, key = load_encoder(args.ckpt)
    print(f"  using key: '{key}'")

    print(f"Sampling {args.n} windows ...")
    windows, labels = sample_windows(args.data, args.n, cfg, seed=args.seed)
    N = len(windows)
    print(f"  loaded {N} windows")

    print("Extracting patch embeddings (teacher, no mask) ...")
    patches = get_patches(enc, windows)   # [N, N_TOKENS, D]
    print(f"  patch tensor: {patches.shape}")

    print("\nFigure 1: intra-sequence patch similarity ...")
    mean_off = plot_intra_sim(patches, labels,
                              os.path.join(args.out, "dino_patch_diversity_sim.png"))

    print("Figure 2: PCA of patch embeddings ...")
    pc1_std = plot_pca(patches, labels,
                       os.path.join(args.out, "dino_patch_diversity_pca.png"))

    print("Figure 3: per-dimension variance ...")
    dim_std_mean, sim_to_mean_avg, frac_high = plot_variance(
        patches, os.path.join(args.out, "dino_patch_diversity_var.png"))

    print("Figure 4: inter-sequence patch similarity ...")
    inter_mean, frac_collapsed = plot_inter_sim(
        patches, os.path.join(args.out, "dino_patch_diversity_inter.png"))

    # ── collapse diagnosis ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PATCH FEATURE COLLAPSE DIAGNOSIS  (N={N})")
    print(f"{'='*60}")
    print(f"  Intra-seq off-diag cosine sim  : {mean_off.mean():+.4f}  "
          f"(healthy <0.3, collapsed >0.8)")
    print(f"  Per-dim std mean               : {dim_std_mean:.4f}  "
          f"(healthy >0.1, collapsed <0.01)")
    print(f"  Sim to mean embedding (avg)    : {sim_to_mean_avg:.4f}  "
          f"(healthy <0.5, collapsed >0.95)")
    print(f"  Inter-seq sim mean             : {inter_mean:.4f}  "
          f"(healthy <0.3, collapsed >0.8)")
    print(f"  Fraction collapsed tokens      : {frac_collapsed:.1%}  "
          f"(healthy <10%)")
    score = np.mean([
        mean_off.mean() > 0.7,
        sim_to_mean_avg > 0.8,
        inter_mean > 0.7,
        frac_collapsed > 0.4,
    ])
    verdict = ("🔴 STRONG COLLAPSE" if score >= 0.75 else
               "🟠 PARTIAL COLLAPSE" if score >= 0.5 else
               "🟡 MILD CONCERN" if score >= 0.25 else
               "🟢 HEALTHY")
    print(f"\n  Verdict: {verdict}")


if __name__ == "__main__":
    main()
