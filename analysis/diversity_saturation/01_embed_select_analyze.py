#!/usr/bin/env python
"""01_embed_select_analyze.py -- diversity-saturation experiment, stage 1.

Tests the hypothesis that the flat 1.3%->100% recipe scaling curve
(EXPERIMENTS.md 11i) is explained by intrinsic motion-diversity saturation:
transfer quality tracks the number of independent motion modes covered, not
the raw window count. Causal design ("same size, different diversity"):
three equal-size (N=46,116, matching the session-1pct point) window subsets
of the full cache-order train pool --

  random-1%     uniform draw (seed 42)            -> medium diversity
  diverse-1%    equal windows per k-means cluster -> max mode coverage
  redundant-1%  fill largest clusters first       -> min mode coverage

plus the diversity diagnostics the causal runs will be read against:
cluster-coverage curve vs fraction, nearest-neighbor novelty of the
unsampled 99% relative to the 1%, and embedding effective rank.

Pipeline (single SLURM job, GPU then CPU):
  1. Embed every train window of the v3 cache pool with the FROZEN
     recipe_100pct encoder (CLS output, [B,384]); augmentation force-disabled
     (same convention as 12_recipe_val_rescore.py); sequential order == the
     ConcatDataset order that scaling_mode=index_file rebuilds, so indices
     correspond by construction.
  2. PCA-64 -> MiniBatchKMeans K=1000 on all embeddings.
  3. Write the three index files + a manifest with per-source composition.
  4. Coverage curve / NN novelty / effective rank -> JSON.

NOTE the pool here is the CACHE pool (3,555,159 windows, humanml3d included)
-- the recipe sweep's raw-rebuild path dropped humanml3d (missing joints/
dir, see EXPERIMENTS.md 11i), so the three index-file pretrains see the 11-
source pool like the original recipe_100pct run did. The session-1pct subset
cannot be mapped into this index space (raw path, different ordering) --
random-1% is the bridge point instead.

Embedding-model circularity caveat: subsets are selected in the
recipe_100pct encoder's own embedding space. Fine for the causal contrast
(all three subsets selected in the SAME space); flag it if diverse-1% is
ever promoted to a "data pruning method" claim.

Outputs (big arrays under /projects, small results in-repo):
  /projects/b53017/vbc1591/diversity_saturation/embeddings_f16.npy
  /projects/b53017/vbc1591/diversity_saturation/pca64_f32.npy
  /projects/b53017/vbc1591/diversity_saturation/idx_{random,diverse,redundant}_1pct.npy
  analysis/diversity_saturation/out/diversity_analysis.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "base.yaml")
MODEL_CONFIG = os.path.join(
    REPO_ROOT, "configs", "vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe100pct.yaml")
CKPT = os.path.join(REPO_ROOT, "runs", "vit1d_ssl_mae_recipe_100pct", "checkpoints", "best.pt")

N_TARGET = 46116          # == session-1pct kept-window count (job 8836196)
K_CLUSTERS = 1000
PCA_DIM = 64
PCA_FIT_SAMPLE = 500_000
NOVELTY_QUERY_SAMPLE = 200_000
SEED = 42

_AUGMENTATION_KEYS = (
    "random_rotation", "gaussian_noise", "jitter", "bias_drift", "sensor_dropout",
    "time_crop", "frequency_mask", "gain_jitter", "axis_bias", "time_warp",
)


def load_cfg_no_aug():
    from imu_lm.utils.helpers import deep_update, load_yaml
    cfg = deep_update(load_yaml(BASE_CONFIG), load_yaml(MODEL_CONFIG))
    for key in _AUGMENTATION_KEYS:
        blk = cfg.get(key)
        if isinstance(blk, dict):
            blk["enabled"] = False
    return cfg


def stage1_embed():
    from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders
    from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn

    os.makedirs(WORK_DIR, exist_ok=True)
    emb_path = os.path.join(WORK_DIR, "embeddings_f16.npy")
    manifest_path = os.path.join(WORK_DIR, "pool_manifest.json")

    cfg = load_cfg_no_aug()
    loaders = make_combined_mae_multidata_loaders(cfg)
    train_ds = loaders["train_loader"].dataset
    n = len(train_ds)
    print(f"[embed] train pool = {n} windows", flush=True)

    # per-source composition of the pool, for the manifest
    names, cum = [], []
    if hasattr(train_ds, "datasets"):
        for d in train_ds.datasets:
            names.append(getattr(d, "name", type(d).__name__))
        cum = list(np.cumsum([len(d) for d in train_ds.datasets]).tolist())

    if os.path.exists(emb_path) and os.path.getsize(emb_path) > 0:
        arr = np.load(emb_path, mmap_mode="r")
        if arr.shape[0] == n:
            print("[embed] embeddings already on disk, skipping forward pass", flush=True)
            return emb_path, n, names, cum

    device = torch.device("cuda")
    encoder = ViT1DEncoderAltAttn(cfg).to(device)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    encoder.load_state_dict(state["model"])
    encoder.eval()
    print(f"[embed] loaded {CKPT} (step={state.get('step')})", flush=True)

    # sequential loader; reuse the val collate (deterministic, no random crop)
    val_collate = loaders["val_loader"].collate_fn
    seq_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=512, shuffle=False, num_workers=8,
        pin_memory=True, collate_fn=val_collate, drop_last=False,
    )

    out = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(n, 384))
    i, t0 = 0, time.time()
    with torch.no_grad():
        for batch in seq_loader:
            x = batch[0].to(device, non_blocking=True)
            cls = encoder(x)                       # [B, 384] pooled CLS (probe convention)
            b = cls.shape[0]
            out[i:i + b] = cls.half().cpu().numpy()
            i += b
            if (i // 512) % 200 == 0:
                rate = i / max(1e-9, time.time() - t0)
                print(f"[embed] {i}/{n} ({rate:.0f} win/s)", flush=True)
    out.flush()
    assert i == n, f"embedded {i} != pool {n}"
    with open(manifest_path, "w") as f:
        json.dump({"n": n, "sources": names, "cumulative_sizes": cum,
                   "ckpt": CKPT, "ckpt_step": int(state.get("step", -1))}, f, indent=2)
    print(f"[embed] wrote {emb_path} + manifest", flush=True)
    return emb_path, n, names, cum


def source_of(idx, names, cum):
    j = int(np.searchsorted(cum, idx, side="right"))
    return names[j] if j < len(names) else "?"


def stage2_select_and_analyze(emb_path, n, names, cum):
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.neighbors import NearestNeighbors

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    emb = np.load(emb_path, mmap_mode="r")

    # ---- PCA-64 ----
    pca_path = os.path.join(WORK_DIR, "pca64_f32.npy")
    if os.path.exists(pca_path) and np.load(pca_path, mmap_mode="r").shape[0] == n:
        print("[pca] reusing pca64 on disk", flush=True)
        z = np.load(pca_path, mmap_mode="r")
        pca_var = None
    else:
        fit_idx = rng.choice(n, size=min(PCA_FIT_SAMPLE, n), replace=False)
        fit_idx.sort()
        pca = PCA(n_components=PCA_DIM, random_state=SEED)
        pca.fit(np.asarray(emb[fit_idx], dtype=np.float32))
        pca_var = float(pca.explained_variance_ratio_.sum())
        print(f"[pca] {PCA_DIM}D keeps {100*pca_var:.1f}% variance", flush=True)
        z = np.lib.format.open_memmap(pca_path, mode="w+", dtype=np.float32, shape=(n, PCA_DIM))
        for s in range(0, n, 200_000):
            e = min(n, s + 200_000)
            z[s:e] = pca.transform(np.asarray(emb[s:e], dtype=np.float32))
        z.flush()

    # ---- k-means ----
    km = MiniBatchKMeans(n_clusters=K_CLUSTERS, batch_size=10_000, n_init=3,
                         random_state=SEED, max_no_improvement=50)
    km.fit(np.asarray(z))
    labels = km.predict(np.asarray(z))
    sizes = np.bincount(labels, minlength=K_CLUSTERS)
    print(f"[kmeans] K={K_CLUSTERS} sizes: min={sizes.min()} median={int(np.median(sizes))} "
          f"max={sizes.max()}", flush=True)

    # ---- three equal-size subsets ----
    by_cluster = [np.where(labels == c)[0] for c in range(K_CLUSTERS)]

    idx_random = rng.choice(n, size=N_TARGET, replace=False)

    # diverse: waterfill equally across clusters, smallest first
    order = np.argsort(sizes)
    remaining, chosen = N_TARGET, []
    clusters_left = (sizes > 0).sum()
    for c in order:
        if sizes[c] == 0:
            continue
        take = min(len(by_cluster[c]), int(np.ceil(remaining / clusters_left)))
        chosen.append(rng.choice(by_cluster[c], size=take, replace=False))
        remaining -= take
        clusters_left -= 1
        if remaining <= 0:
            break
    idx_diverse = np.concatenate(chosen)[:N_TARGET]

    # redundant: whole clusters, largest first (fewest modes covered)
    remaining, chosen = N_TARGET, []
    for c in np.argsort(sizes)[::-1]:
        take = min(len(by_cluster[c]), remaining)
        chosen.append(rng.choice(by_cluster[c], size=take, replace=False) if take < len(by_cluster[c])
                      else by_cluster[c])
        remaining -= take
        if remaining <= 0:
            break
    idx_redundant = np.concatenate(chosen)[:N_TARGET]

    subsets = {"random": idx_random, "diverse": idx_diverse, "redundant": idx_redundant}
    for name, idx in subsets.items():
        assert len(idx) == N_TARGET and len(np.unique(idx)) == N_TARGET, name
        path = os.path.join(WORK_DIR, f"idx_{name}_1pct.npy")
        np.save(path, np.sort(idx).astype(np.int64))
        print(f"[select] wrote {path}", flush=True)

    # ---- diagnostics ----
    def coverage(idx):
        return float(len(np.unique(labels[idx])) / (sizes > 0).sum())

    cov_curve = {}
    for frac in (0.0001, 0.001, 0.005, 0.013, 0.05, 0.106, 0.25, 1.0):
        k = max(1, int(round(n * frac)))
        draw = rng.choice(n, size=k, replace=False) if frac < 1.0 else np.arange(n)
        cov_curve[frac] = coverage(draw)
    cov_subsets = {name: coverage(idx) for name, idx in subsets.items()}

    # NN novelty of the unsampled pool relative to each 1% subset
    novelty = {}
    zq_idx = rng.choice(n, size=NOVELTY_QUERY_SAMPLE, replace=False)
    for name, idx in subsets.items():
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto", n_jobs=-1)
        nn.fit(np.asarray(z[np.sort(idx)]))
        mask = ~np.isin(zq_idx, idx)
        q = np.asarray(z[np.sort(zq_idx[mask])])
        d, _ = nn.kneighbors(q)
        d = d.ravel()
        # scale reference: NN distance INSIDE the subset itself (self-redundancy)
        sub = np.asarray(z[np.sort(idx)])
        nn2 = NearestNeighbors(n_neighbors=2, n_jobs=-1).fit(sub)
        d_self, _ = nn2.kneighbors(sub[rng.choice(N_TARGET, 20_000, replace=False)])
        d_self = d_self[:, 1]
        novelty[name] = {
            "pool_to_subset_pcts": {p: float(np.percentile(d, p)) for p in (5, 25, 50, 75, 95)},
            "subset_self_nn_median": float(np.median(d_self)),
            "frac_pool_below_self_median": float((d < np.median(d_self)).mean()),
        }
        print(f"[novelty] {name}: median pool->subset d={np.median(d):.3f}, "
              f"self-NN median={np.median(d_self):.3f}, "
              f"{100*novelty[name]['frac_pool_below_self_median']:.1f}% of pool is 'not novel'", flush=True)

    # effective rank of embedding covariance
    samp = np.asarray(emb[np.sort(rng.choice(n, 300_000, replace=False))], dtype=np.float32)
    samp -= samp.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(np.cov(samp, rowvar=False))
    ev = np.clip(ev, 1e-12, None); p = ev / ev.sum()
    erank = float(np.exp(-(p * np.log(p)).sum()))

    comp = {name: {} for name in subsets}
    for name, idx in subsets.items():
        for i in idx:
            s = source_of(int(i), names, cum)
            comp[name][s] = comp[name].get(s, 0) + 1

    results = {
        "n_pool": int(n), "n_target": N_TARGET, "k_clusters": K_CLUSTERS,
        "pca_var_kept": pca_var, "coverage_curve": cov_curve,
        "coverage_subsets": cov_subsets, "novelty": novelty,
        "embedding_effective_rank_384d": erank,
        "cluster_sizes_pcts": {p: int(np.percentile(sizes, p)) for p in (0, 25, 50, 75, 100)},
        "subset_source_composition": comp,
    }
    with open(os.path.join(OUT_DIR, "diversity_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("coverage_curve", "coverage_subsets",
                               "embedding_effective_rank_384d")}, indent=2), flush=True)


def main():
    emb_path, n, names, cum = stage1_embed()
    stage2_select_and_analyze(emb_path, n, names, cum)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
