#!/usr/bin/env python3
"""Analyze RVQ tokenizer quality: do different activities get different tokens?

Checks:
  1. TSNE of mean-pooled cumulative RVQ embeddings, colored by activity (per level)
  2. Token usage heatmap: activity x top-N tokens (do activities share tokens?)
  3. 1-NN classification accuracy using RVQ embeddings (no training needed)
  4. Per-token purity: for the most-used tokens, what fraction comes from one activity?

Usage:
    MPLBACKEND=Agg python scripts/vis_rvq_token_quality.py \
        [--rvq-ckpt  runs/vit1d_rvq_tokenizer_realdata_nopooling_8lvl/checkpoints/latest.pt] \
        [--parquet   /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet] \
        [--dataset   utd_mhad]      # dataset with clear activity labels
        [--n-per-act 50]            # windows per activity class
        [--out       data_vis/rvq_token_quality.png]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives.rvq_tokenizer import ResidualVectorQuantizer, PatchDecoder

WIN      = 128
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ---------------------------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------------------------

def load_rvq(ckpt_path: str):
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg     = ckpt["cfg"]
    cfg_key = cfg["objective"]["type"]
    obj_cfg = cfg["objective"][cfg_key]

    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval()

    rvq = ResidualVectorQuantizer(
        num_levels=obj_cfg["num_levels"],
        num_codes=obj_cfg["num_codes"],
        code_dim=obj_cfg["code_dim"],
        ema_decay=obj_cfg["ema_decay"],
        commitment_weight=obj_cfg["commitment_weight"],
    )
    rvq.load_state_dict(ckpt["rvq"], strict=True)
    rvq.eval()

    return encoder, rvq, cfg


# ---------------------------------------------------------------------------
# Encode windows → token indices + cumulative embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode(x_np: np.ndarray, encoder, rvq):
    """x_np [B, C, T] -> indices_list (K x [B, N]), quantized_list (K x [B, N, D])
       and mean-pooled per-window embeddings (K x [B, D]).
    """
    x = torch.from_numpy(x_np).float()
    tokens, _ = encoder.forward_tokens(x)          # [B, N, D]
    z = F.normalize(tokens, dim=-1)
    quantized_list, indices_list, _ = rvq(z)
    # quantized_list[k]: [B, N, code_dim]  cumulative up to level k
    # indices_list[k]:   [B, N]  token index at level k

    # Mean-pool patch tokens → per-window embedding at each level
    pooled = [q.mean(dim=1).cpu().numpy() for q in quantized_list]   # list of [B, D]
    indices = [idx.cpu().numpy() for idx in indices_list]             # list of [B, N]
    return pooled, indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rvq-ckpt",
                   default="runs/vit1d_rvq_tokenizer_realdata_nopooling_8lvl/checkpoints/latest.pt")
    p.add_argument("--parquet",
                   default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--dataset",    default="utd_mhad",
                   help="Dataset with activity labels in the parquet")
    p.add_argument("--n-per-act",  type=int, default=50,
                   help="Windows to sample per activity class")
    p.add_argument("--top-tokens", type=int, default=30,
                   help="Top-N most frequent tokens to show in heatmap")
    p.add_argument("--out", default="data_vis/rvq_token_quality.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(42)

    print(f"Loading RVQ tokenizer: {args.rvq_ckpt}")
    encoder, rvq, cfg = load_rvq(args.rvq_ckpt)
    num_levels = rvq.num_levels
    num_codes  = rvq.quantizers[0].num_codes if hasattr(rvq.quantizers[0], 'num_codes') \
                 else rvq.quantizers[0].codebook.shape[0]
    print(f"  {num_levels} levels x {num_codes} codes")

    # ── Load data ────────────────────────────────────────────────────────────
    needed = ["dataset", "dataset_activity_label", "session_id", "timestamp_ns"] + IMU_COLS
    print(f"Loading {args.dataset} from parquet ...")
    df = pd.read_parquet(args.parquet, columns=needed,
                         filters=[("dataset", "==", args.dataset)])
    if df.empty:
        sys.exit(f"Dataset '{args.dataset}' not found in parquet.")
    df = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)

    activities = sorted(df["dataset_activity_label"].dropna().unique())
    print(f"  {len(activities)} activity classes: {activities[:10]}{'...' if len(activities)>10 else ''}")

    # ── Sample windows per activity ──────────────────────────────────────────
    all_windows: list[np.ndarray] = []
    all_labels:  list[str]        = []

    for act in activities:
        rows = df[df["dataset_activity_label"] == act]
        vals = rows[IMU_COLS].values.T.astype(np.float32)   # [C, L]
        L    = vals.shape[1]
        if L < WIN:
            continue
        n = min(args.n_per_act, L - WIN + 1)
        starts = rng.choice(L - WIN + 1, size=n, replace=False)
        for s in starts:
            all_windows.append(vals[:, s : s + WIN])
            all_labels.append(act)

    if not all_windows:
        sys.exit("No windows sampled — check dataset and parquet.")

    X = np.stack(all_windows, axis=0)   # [N_total, C, T]
    Y = np.array(all_labels)
    print(f"  Sampled {len(X)} windows across {len(activities)} activities")

    # ── Encode all windows ───────────────────────────────────────────────────
    print("Encoding ...")
    BS = 128
    pooled_all   = [[] for _ in range(num_levels)]
    indices_all  = [[] for _ in range(num_levels)]

    for start in range(0, len(X), BS):
        batch = X[start : start + BS]
        pooled, indices = encode(batch, encoder, rvq)
        for k in range(num_levels):
            pooled_all[k].append(pooled[k])     # [B, D]
            indices_all[k].append(indices[k])   # [B, N]

    pooled_all  = [np.concatenate(p, axis=0) for p in pooled_all]    # [num_levels, N_total, D]
    indices_all = [np.concatenate(i, axis=0) for i in indices_all]   # [num_levels, N_total, N_patches]

    # ── 1-NN classification accuracy (level 0, 3, 7) ────────────────────────
    print("\n── 1-NN classification accuracy (using RVQ mean-pool embedding) ──")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import cross_val_score

    le = LabelEncoder()
    y_enc = le.fit_transform(Y)
    for lvl in [0, num_levels // 2, num_levels - 1]:
        emb = pooled_all[lvl]
        # Normalize for cosine-like distance
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
        scores = cross_val_score(knn, emb_n, y_enc, cv=5, scoring="balanced_accuracy")
        print(f"  Level {lvl+1:2d}: 5-NN bal_acc = {scores.mean():.3f} ± {scores.std():.3f}")

    # ── Token usage heatmap at level 0 ───────────────────────────────────────
    # For each activity, which tokens appear most?
    print("\n── Token usage analysis (level 1) ──")
    act_list   = sorted(set(Y))
    n_acts     = len(act_list)
    act_to_idx = {a: i for i, a in enumerate(act_list)}

    # Count token usage per activity (level 0, raw indices per patch)
    token_counts = np.zeros((n_acts, num_codes), dtype=np.float32)
    idx_0 = indices_all[0]   # [N_total, N_patches]
    for i, act in enumerate(Y):
        ai = act_to_idx[act]
        for tok in idx_0[i]:
            token_counts[ai, tok] += 1

    # Normalise per activity (row-wise)
    row_sums = token_counts.sum(axis=1, keepdims=True) + 1e-8
    token_freq = token_counts / row_sums    # [n_acts, num_codes]

    # Token purity: for each token, how concentrated is it in one activity?
    token_total = token_counts.sum(axis=0)  # [num_codes]
    used_tokens  = np.where(token_total > 0)[0]
    token_max_frac = (token_counts / (token_total[None, :] + 1e-8)).max(axis=0)
    print(f"  Tokens used: {len(used_tokens)} / {num_codes}")
    print(f"  Mean token purity (dominant activity fraction): {token_max_frac[used_tokens].mean():.3f}")
    print(f"  Tokens with >80% purity (mostly one activity): "
          f"{(token_max_frac[used_tokens] > 0.8).sum()}")
    print(f"  Tokens with <20% purity (shared by many activities): "
          f"{(token_max_frac[used_tokens] < 0.2).sum()}")

    # Top-N most used tokens for heatmap
    top_tok_idx = np.argsort(token_total)[::-1][:args.top_tokens]
    heat_data   = token_freq[:, top_tok_idx]    # [n_acts, top_tokens]

    # ── TSNE ─────────────────────────────────────────────────────────────────
    from sklearn.manifold import TSNE
    tsne_levels = [0, num_levels - 1]   # level 1 and level 8
    tsne_embeds = {}
    for lvl in tsne_levels:
        emb = pooled_all[lvl]
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
        print(f"Computing TSNE for level {lvl+1} ...")
        tsne = TSNE(n_components=2, perplexity=min(30, len(X) // 5),
                    random_state=42, max_iter=1000)
        tsne_embeds[lvl] = tsne.fit_transform(emb_n)

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_tsne = len(tsne_levels)
    fig = plt.figure(figsize=(6 * (n_tsne + 1) + 2, max(8, n_acts * 0.35 + 4)))
    fig.suptitle(
        f"RVQ Token Quality — {args.rvq_ckpt.split('/')[1]}\n"
        f"Dataset: {args.dataset}  |  {len(X)} windows  |  {len(act_list)} activities",
        fontsize=12, fontweight="bold",
    )

    gs = gridspec.GridSpec(1, n_tsne + 1, figure=fig, wspace=0.35,
                           width_ratios=[1.2] * n_tsne + [1.5])

    # Consistent colors per activity
    cmap  = plt.colormaps["tab20"].resampled(min(len(act_list), 20))
    if len(act_list) > 20:
        cmap2 = plt.colormaps["tab20b"].resampled(len(act_list) - 20)
        colors = [cmap(i) for i in range(20)] + [cmap2(i) for i in range(len(act_list) - 20)]
    else:
        colors = [cmap(i) for i in range(len(act_list))]
    act_colors = {a: colors[i] for i, a in enumerate(act_list)}

    # TSNE panels
    for col, lvl in enumerate(tsne_levels):
        ax = fig.add_subplot(gs[0, col])
        xy = tsne_embeds[lvl]
        for act in act_list:
            mask = Y == act
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=[act_colors[act]], s=12, alpha=0.7, linewidths=0,
                       label=act)
        ax.set_title(f"TSNE — RVQ level {lvl+1}\n(cumulative, mean-pool)",
                     fontsize=10, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if col == n_tsne - 1:
            ax.legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left",
                      markerscale=1.5, framealpha=0.7,
                      ncol=max(1, len(act_list) // 25))

    # Token heatmap panel
    ax_h = fig.add_subplot(gs[0, -1])
    im = ax_h.imshow(heat_data, aspect="auto", cmap="YlOrRd", origin="upper",
                     interpolation="nearest")
    ax_h.set_xticks(range(len(top_tok_idx)))
    ax_h.set_xticklabels([f"T{t}" for t in top_tok_idx],
                          rotation=90, fontsize=7)
    ax_h.set_yticks(range(n_acts))
    ax_h.set_yticklabels(act_list, fontsize=7)
    ax_h.set_xlabel(f"Top-{args.top_tokens} most-used tokens (level 1)", fontsize=9)
    ax_h.set_title("Token usage frequency per activity\n(normalized per row)",
                   fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")

    # ── Text summary ─────────────────────────────────────────────────────────
    print("\n── Token purity distribution (level 1) ──")
    print("  (purity = fraction of a token's uses belonging to its dominant activity)")
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels_b = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    hist, _ = np.histogram(token_max_frac[used_tokens], bins=bins)
    for label, count in zip(labels_b, hist):
        bar = "█" * int(count / max(hist) * 30)
        print(f"  {label:>8}: {bar} {count}")


if __name__ == "__main__":
    main()
