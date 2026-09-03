#!/usr/bin/env python
"""13_motion_type_diversity.py -- what a "motion type" is, per dataset.

The motion-types axis of the 11l grid is defined operationally: embed every
window with the frozen recipe_100pct encoder, PCA-64, MiniBatchKMeans K=1000.
A "motion type" is one of those 1000 centroids -- a region of the encoder's
representation space, i.e. a recurring 10.24 s wrist-IMU signal shape. It is
NOT a semantic activity label; nobody named these.

This script answers "how diverse is each dataset in that space", for the full
11-source cache pool (3,555,159 windows), and separates the two things that
get conflated:

  richness   how many of the 1000 types a source touches at all
  evenness   how concentrated its mass is inside the types it touches

Raw richness is almost pure dataset size, so every richness number is also
reported RAREFIED -- expected type count in a random draw of m windows,
analytically via the standard rarefaction formula -- which is what makes
capture24 (1.4M windows) and openwatch (1,842) comparable.

Also computed:
  - source x source Jensen-Shannon distance over type histograms (redundancy)
  - exclusive types (a source is the only occupant / the majority occupant)
  - leave-one-out richness drop (marginal contribution to pool coverage)
  - AMI(source, cluster): how much the partition is just dataset identity
  - per-source effective rank in PCA-64 (partition-free spread cross-check)

Reuses the cached PCA-64 from 01_embed_select_analyze.py. K-means labels for
the 11-source pool are cached to labels11src_k1000.npy on first run.

Outputs analysis/diversity_saturation/out/motion_type_diversity.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.special import gammaln

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")

K_CLUSTERS = 1000
SEED = 42
# rarefaction depths: smallest source, then the sizes that let progressively
# more sources into the comparison
RAREFY_M = [137, 461, 1842, 2825, 4683, 32662, 93510]


def log_choose(n, k):
    """log C(n, k), vectorised, -inf when k > n."""
    n = np.asarray(n, dtype=np.float64)
    bad = k > n
    out = gammaln(n + 1) - gammaln(k + 1) - gammaln(np.maximum(n - k, 0) + 1)
    return np.where(bad, -np.inf, out)


def rarefied_richness(counts, m):
    """E[# non-empty types] in a draw of m windows without replacement.

    E[S_m] = sum_c 1 - C(n - n_c, m) / C(n, m).  Log-space so the 1.4M-window
    sources do not overflow.
    """
    counts = counts[counts > 0].astype(np.float64)
    n = counts.sum()
    if m > n:
        return float("nan")
    log_p_absent = log_choose(n - counts, m) - log_choose(n, m)
    return float(np.sum(1.0 - np.exp(log_p_absent)))


def shannon(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def simpson_effective(counts):
    """1 / sum p^2 with the unbiased (hypergeometric) estimator of sum p^2."""
    c = counts[counts > 0].astype(np.float64)
    n = c.sum()
    if n < 2:
        return float("nan")
    lam = float(((c * (c - 1)).sum()) / (n * (n - 1)))
    return float(1.0 / lam) if lam > 0 else float("nan")


def effective_rank(x):
    """exp(entropy of the eigenvalue spectrum of cov(x)) -- partition-free."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return float("nan")
    x = x - x.mean(0, keepdims=True)
    ev = np.linalg.eigvalsh(np.cov(x, rowvar=False))
    ev = np.clip(ev, 1e-12, None)
    p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def js_distance(p, q):
    """sqrt of Jensen-Shannon divergence (a metric), base 2, in [0, 1]."""
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return float(np.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m))))


def get_labels(z, n):
    lab_path = os.path.join(WORK_DIR, "labels11src_k1000.npy")
    if os.path.exists(lab_path):
        lab = np.load(lab_path)
        if len(lab) == n:
            print(f"[kmeans] reusing {lab_path}", flush=True)
            return lab.astype(np.int32)

    from sklearn.cluster import MiniBatchKMeans
    t0 = time.time()
    # identical hyper-parameters to 01_embed_select_analyze.py, so the type
    # definition is the same object the grid experiment used
    km = MiniBatchKMeans(n_clusters=K_CLUSTERS, batch_size=10_000, n_init=3,
                         random_state=SEED, max_no_improvement=50)
    km.fit(z)
    print(f"[kmeans] fit in {time.time() - t0:.0f}s, predicting...", flush=True)
    labels = np.empty(n, dtype=np.int32)
    for s in range(0, n, 500_000):
        e = min(n, s + 500_000)
        labels[s:e] = km.predict(z[s:e])
    np.save(lab_path, labels)
    print(f"[kmeans] wrote {lab_path} ({time.time() - t0:.0f}s total)", flush=True)
    return labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = json.load(open(os.path.join(WORK_DIR, "pool_manifest.json")))
    names = manifest["sources"]
    cum = np.array(manifest["cumulative_sizes"])
    n = manifest["n"]
    starts = np.concatenate([[0], cum[:-1]])
    sizes_src = cum - starts

    z = np.load(os.path.join(WORK_DIR, "pca64_f32.npy"), mmap_mode="r")
    assert z.shape[0] == n, (z.shape, n)
    z = np.asarray(z)                      # 910 MB, fits comfortably
    labels = get_labels(z, n)

    pool_counts = np.bincount(labels, minlength=K_CLUSTERS)
    n_types_pool = int((pool_counts > 0).sum())
    print(f"[pool] {n:,} windows, {n_types_pool} non-empty types", flush=True)

    # source x type contingency table
    src_of = np.searchsorted(cum, np.arange(n), side="right")
    table = np.zeros((len(names), K_CLUSTERS), dtype=np.int64)
    np.add.at(table, (src_of, labels), 1)
    assert table.sum() == n

    # ---- per-source metrics ----
    per_source = {}
    for i, name in enumerate(names):
        c = table[i]
        nz = c > 0
        rich = int(nz.sum())
        h = shannon(c)
        rec = {
            "n_windows": int(sizes_src[i]),
            "types_touched": rich,
            "coverage_pct": round(100.0 * rich / n_types_pool, 2),
            "shannon_nats": round(h, 4),
            "effective_types_shannon": round(float(np.exp(h)), 1),
            "effective_types_simpson": round(simpson_effective(c), 1),
            "evenness": round(h / np.log(rich), 4) if rich > 1 else float("nan"),
            "top1_type_share_pct": round(100.0 * c.max() / c.sum(), 2),
            "rarefied_richness": {},
        }
        for m in RAREFY_M:
            rec["rarefied_richness"][str(m)] = (
                round(rarefied_richness(c, m), 1) if m <= sizes_src[i] else None)
        # partition-free spread: effective rank of this source's PCA-64 block
        sl = z[starts[i]:cum[i]]
        if len(sl) > 50_000:
            rng = np.random.default_rng(SEED)
            sl = sl[rng.choice(len(sl), 50_000, replace=False)]
        rec["effective_rank_pca64"] = round(effective_rank(sl), 2)
        per_source[name] = rec
        print(f"  {name:14s} n={sizes_src[i]:>9,}  types={rich:>4d}  "
              f"eff(shannon)={rec['effective_types_shannon']:>7.1f}  "
              f"rare@137={rec['rarefied_richness']['137']}  "
              f"erank={rec['effective_rank_pca64']}", flush=True)

    # ---- exclusivity / marginal contribution ----
    occupants = (table > 0).sum(0)                       # sources per type
    share = table / np.maximum(table.sum(0, keepdims=True), 1)
    for i, name in enumerate(names):
        c = table[i]
        sole = int(((occupants == 1) & (c > 0)).sum())
        major = int(((share[i] > 0.5) & (c > 0)).sum())
        # leave-one-out: types that vanish from the pool without this source
        loo = int(((pool_counts - c) == 0).sum() - (pool_counts == 0).sum())
        per_source[name].update({
            "types_sole_occupant": sole,
            "types_majority_occupant": major,
            "types_lost_if_removed": loo,
        })

    # ---- pairwise redundancy ----
    js = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = js_distance(table[i].astype(float), table[j].astype(float))
            js[i, j] = js[j, i] = d
    js_matrix = {names[i]: {names[j]: round(float(js[i, j]), 3)
                            for j in range(len(names))} for i in range(len(names))}
    # nearest neighbour source for each source
    for i, name in enumerate(names):
        d = js[i].copy()
        d[i] = np.inf
        j = int(np.argmin(d))
        per_source[name]["most_similar_source"] = names[j]
        per_source[name]["js_to_most_similar"] = round(float(d[j]), 3)

    # ---- is the partition just dataset identity? ----
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
    rng = np.random.default_rng(SEED)
    sub = rng.choice(n, 400_000, replace=False)
    ami = float(adjusted_mutual_info_score(src_of[sub], labels[sub]))
    nmi = float(normalized_mutual_info_score(src_of[sub], labels[sub]))

    results = {
        "definition": ("motion type = one of K=1000 MiniBatchKMeans centroids over "
                       "PCA-64 of the frozen recipe_100pct encoder's CLS embedding of "
                       "each 10.24 s window; unnamed, purely geometric"),
        "pool_n": int(n), "k_clusters": K_CLUSTERS,
        "n_types_nonempty": n_types_pool,
        "encoder_ckpt": manifest["ckpt"], "ckpt_step": manifest["ckpt_step"],
        "pool_effective_rank_pca64": round(effective_rank(z[sub[:100_000]]), 2),
        "ami_source_vs_cluster": round(ami, 4),
        "nmi_source_vs_cluster": round(nmi, 4),
        "rarefy_depths": RAREFY_M,
        "per_source": per_source,
        "js_distance_matrix": js_matrix,
        "types_by_n_occupant_sources": {str(k): int((occupants == k).sum())
                                        for k in range(0, len(names) + 1)
                                        if (occupants == k).sum() > 0},
    }
    p = os.path.join(OUT_DIR, "motion_type_diversity.json")
    with open(p, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAMI(source, type) = {ami:.4f}   NMI = {nmi:.4f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
