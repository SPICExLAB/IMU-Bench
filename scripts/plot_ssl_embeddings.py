#!/usr/bin/env python3
"""UMAP + t-SNE visualization of the MAE SSL encoder's embedding space
across real HAR datasets (2026-07-29, user request).

Pipeline: for each of 5 real datasets (wisdm / utd_mhad / pamap2 from the
master parquet; hhar / mhealth from their own parquets), slide
non-overlapping 10.24s (512-sample @50Hz) windows over every session,
keep windows whose activity label is >=80% pure, preprocess exactly as
training does (impute -> 24Hz lowpass -> per-window per-channel z-score,
via the repo's own preprocess_window), embed with the FROZEN Stage-1 MAE
encoder (runs/vit1d_ssl_mae_varlen_altattn_multidata, mean-pooled 384-d),
then project to 2-D with UMAP and t-SNE (cosine metric).

Outputs (analysis/embedding_viz/):
    ssl_umap_by_dataset.png    ssl_tsne_by_dataset.png
    ssl_umap_by_activity.png   ssl_tsne_by_activity.png
    ssl_embedding_projections.png   (2x2 combined)
    ssl_embeddings.npz         (embeddings + 2-D coords + labels, for reuse)

Color assignment follows the dataviz procedure: fixed-order categorical
palette (validated), datasets in sorted order; activities colored for the
8 most frequent (fixed order by frequency), all others gray "other".
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENSORS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
T_WIN = 512          # 10.24 s @ 50 Hz -- the SSL model's max training window
T_MIN_SHORT = 128    # sessions shorter than T_WIN but >= 2.56s are wrap-padded
PER_DATASET_CAP = 800
SEED = 42

# Cross-dataset activity-name normalization: merge trivially-synonymous label
# strings so the "colored by activity" plots genuinely group the same
# activity ACROSS datasets (hhar says "walk"/"sit"/"bike", wisdm/pamap2 say
# "walking"/"sitting"/"cycling", mhealth says "standing still" etc.).
ACT_NORM = {
    "walk": "walking",
    "sit": "sitting", "sitting relaxing": "sitting",
    "stand": "standing", "standing still": "standing",
    "bike": "cycling",
    "jog": "jogging",
    "run": "running",
    "lying down": "lying", "laying": "lying",
    "stairsup": "stairs up", "ascending stairs": "stairs up", "climbing stairs": "stairs up",
    "stairsdown": "stairs down", "descending stairs": "stairs down",
}

DATASETS = {
    "wisdm":    "/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet",
    "utd_mhad": "/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet",
    "pamap2":   "/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet",
    "hhar":     "/projects/b53017/vbc1591/hhar_dataset.parquet",
    "mhealth":  "/projects/b53017/vbc1591/mhealth_dataset.parquet",
}

# dataviz reference categorical palette (validated: ALL CHECKS PASS, contrast
# WARN relieved by visible legends). Fixed order, never cycled.
PALETTE8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
            "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_GRAY = "#9b9a92"


def extract_windows():
    import pyarrow.dataset as pa_ds
    from imu_lm.data.augmentations.preprocess import preprocess_window

    prep_cfg = {
        "preprocess": {
            "impute":   {"enabled": True, "method": "linear", "max_missing_frac": 0.2},
            "filter":   {"enabled": True, "type": "lowpass", "high_hz": 24, "low_hz": 0.25, "order": 4},
            "normalize": {"enabled": True, "method": "zscore", "eps": 1e-6},
        }
    }

    rng = np.random.default_rng(SEED)
    X_all, ds_labels, act_labels = [], [], []
    for ds_name, path in DATASETS.items():
        d = pa_ds.dataset(path, format="parquet")
        cols = SENSORS + ["dataset", "subject_id", "session_id", "dataset_activity_label"]
        tbl = d.to_table(columns=cols, filter=(pa_ds.field("dataset") == ds_name))
        df = tbl.to_pandas()
        if df.empty:
            print(f"[{ds_name}] EMPTY -- skipped", flush=True)
            continue

        wins = []
        for (_, _), g in df.groupby(["subject_id", "session_id"], sort=False):
            arr = g[SENSORS].to_numpy(dtype=np.float32)
            labs = g["dataset_activity_label"].astype(str).str.strip().str.lower().to_numpy()
            spans = ([(s0, s0 + T_WIN) for s0 in range(0, len(arr) - T_WIN + 1, T_WIN)]
                     if len(arr) >= T_WIN else
                     ([(0, len(arr))] if len(arr) >= T_MIN_SHORT else []))
            for a0, a1 in spans:
                w = arr[a0:a1]
                wl = labs[a0:a1]
                vals, cnts = np.unique(wl, return_counts=True)
                top = int(np.argmax(cnts))
                if cnts[top] / len(wl) < 0.8:
                    continue
                lab = vals[top]
                if lab in ("nan", "none", "", "unknown", "-1"):
                    continue
                lab = ACT_NORM.get(lab, lab)
                wins.append((w, lab))
        rng.shuffle(wins)
        wins = wins[:PER_DATASET_CAP]
        kept = 0
        for w, lab in wins:
            xp = preprocess_window(w, prep_cfg, None)   # [T,C] -> [C,T] or None
            if xp is None:
                continue
            if xp.shape[1] < T_WIN:
                # wrap-pad short sessions (e.g. UTD-MHAD ~3.5s trials) to the
                # model window -- same cyclic convention as training/eval
                xp = np.pad(xp, ((0, 0), (0, T_WIN - xp.shape[1])), mode="wrap")
            X_all.append(xp.astype(np.float32))
            ds_labels.append(ds_name)
            act_labels.append(lab)
            kept += 1
        print(f"[{ds_name}] windows kept: {kept}", flush=True)
    return np.stack(X_all), np.array(ds_labels), np.array(act_labels)


def embed(X):
    import torch
    from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn

    ckpt = torch.load("runs/vit1d_ssl_mae_varlen_altattn_multidata/checkpoints/best.pt",
                      map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    enc = ViT1DEncoderAltAttn(cfg)
    missing, unexpected = enc.load_state_dict(ckpt["model"], strict=False)
    assert not missing, f"encoder missing keys: {missing[:5]}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc.eval().to(device)

    Z = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256]).to(device)   # [B, 6, 512]
            Z.append(enc(xb).cpu().numpy())
    return np.concatenate(Z, 0)                              # [N, 384]


def project(Z):
    from sklearn.manifold import TSNE
    import umap

    print("computing t-SNE ...", flush=True)
    tsne = TSNE(n_components=2, perplexity=30, metric="cosine",
                init="pca", random_state=SEED).fit_transform(Z)
    print("computing UMAP ...", flush=True)
    um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="cosine", random_state=SEED).fit_transform(Z)
    return um, tsne


def scatter(ax, coords, labels, order, colors, title):
    ax.set_facecolor("#ffffff")
    for lab, col in zip(order, colors):
        m = labels == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=7, c=col, alpha=0.75,
                   linewidths=0, label=f"{lab} (n={int(m.sum())})")
    ax.set_title(title, fontsize=12, color="#1a1a19")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d7d6cc"); sp.set_linewidth(0.8)
    ax.legend(loc="best", fontsize=7, framealpha=0.9, markerscale=1.8,
              handletextpad=0.3, borderpad=0.4)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = "analysis/embedding_viz"
    os.makedirs(out, exist_ok=True)

    X, ds_lab, act_lab = extract_windows()
    print(f"total windows: {len(X)}", flush=True)
    Z = embed(X)
    um, ts = project(Z)

    np.savez_compressed(f"{out}/ssl_embeddings.npz", Z=Z, umap=um, tsne=ts,
                        dataset=ds_lab, activity=act_lab)

    # dataset coloring: sorted order, fixed palette slots
    ds_order = sorted(set(ds_lab))
    ds_colors = PALETTE8[:len(ds_order)]

    # activity coloring: top-8 by frequency (fixed order), rest -> "other" gray
    vals, cnts = np.unique(act_lab, return_counts=True)
    top8 = list(vals[np.argsort(-cnts)][:8])
    act_plot_lab = np.array([a if a in top8 else "other" for a in act_lab])
    act_order = top8 + (["other"] if (act_plot_lab == "other").any() else [])
    act_colors = PALETTE8[:len(top8)] + ([OTHER_GRAY] if "other" in act_order else [])

    panels = [
        (um, ds_lab,       ds_order,  ds_colors,  "UMAP — colored by dataset",   "ssl_umap_by_dataset.png"),
        (ts, ds_lab,       ds_order,  ds_colors,  "t-SNE — colored by dataset",  "ssl_tsne_by_dataset.png"),
        (um, act_plot_lab, act_order, act_colors, "UMAP — colored by activity",  "ssl_umap_by_activity.png"),
        (ts, act_plot_lab, act_order, act_colors, "t-SNE — colored by activity", "ssl_tsne_by_activity.png"),
    ]
    for coords, labels, order, colors, title, fname in panels:
        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
        scatter(ax, coords, labels, order, colors,
                f"MAE SSL encoder — {title}")
        fig.tight_layout()
        fig.savefig(f"{out}/{fname}", facecolor="#ffffff")
        plt.close(fig)
        print("wrote", f"{out}/{fname}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=150)
    for ax, (coords, labels, order, colors, title, _) in zip(axes.ravel(), panels):
        scatter(ax, coords, labels, order, colors, title)
    fig.suptitle("MAE SSL encoder embeddings — 5 real HAR datasets, "
                 f"{len(X)} windows of 10.24s", fontsize=14, color="#1a1a19")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{out}/ssl_embedding_projections.png", facecolor="#ffffff")
    print("wrote", f"{out}/ssl_embedding_projections.png", flush=True)


if __name__ == "__main__":
    main()
