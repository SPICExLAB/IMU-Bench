#!/usr/bin/env python
"""18_cluster_validity.py -- are the 1000 "motion types" real modes, or slices of a continuum?

Every diversity number in this analysis is defined relative to a k-means
partition, so it matters whether that partition is discovering structure or
imposing it. Three measurements, each against a matched null:

  stability   refit MiniBatchKMeans with 3 seeds, compare with adjusted Rand.
              A partition of genuinely separated modes is seed-stable; a
              partition of a smooth cloud is not.
  silhouette  mean silhouette on a subsample. >0.5 is "reasonable structure",
              0.2-0.5 weak, <0.2 essentially none.
  null        the same two measurements on a Gaussian sample with the pool's
              own mean and covariance -- i.e. the same shape and effective
              rank but no cluster structure whatsoever. Whatever the null
              scores is the floor that any k-means run would produce.

Swept over K in {10, 50, 200, 1000} so the answer is not specific to K=1000.

Outputs analysis/diversity_saturation/out/cluster_validity.json
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")

KS = [10, 50, 200, 1000]
SEEDS = [42, 7, 2024]
CMP_N = 200_000        # subsample the ARI comparison is scored on
SIL_N = 20_000         # silhouette is O(n^2) in memory -- keep it modest


def fit(x, K, seed):
    return MiniBatchKMeans(n_clusters=K, batch_size=10_000, n_init=3,
                           random_state=seed, max_no_improvement=50).fit(x)


def measure(x, K, cmp_idx, sil_idx, tag):
    labs = [fit(x, K, s).predict(x[cmp_idx]) for s in SEEDS]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    km = fit(x, K, SEEDS[0])
    sil = float(silhouette_score(x[sil_idx], km.predict(x[sil_idx])))
    print(f"    {tag:5s} K={K:<5d} ARI={np.mean(aris):.3f} sil={sil:.3f}", flush=True)
    return {"ari_mean": float(np.mean(aris)), "ari_pairs": [float(a) for a in aris],
            "silhouette": sil}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    z = np.asarray(np.load(os.path.join(WORK_DIR, "pca64_f32.npy"), mmap_mode="r"))
    n = len(z)
    rng = np.random.default_rng(0)
    cmp_idx = np.sort(rng.choice(n, CMP_N, replace=False))
    sil_idx = np.sort(rng.choice(n, SIL_N, replace=False))

    # matched null: same mean and covariance, no modes
    cov_idx = np.sort(rng.choice(n, 300_000, replace=False))
    null = rng.multivariate_normal(z.mean(0), np.cov(z[cov_idx], rowvar=False),
                                   size=CMP_N).astype(np.float32)
    ncmp = np.arange(CMP_N)
    nsil = np.sort(rng.choice(CMP_N, SIL_N, replace=False))

    res, t0 = {}, time.time()
    for K in KS:
        print(f"  K={K} ...", flush=True)
        real = measure(z, K, cmp_idx, sil_idx, "real")
        nul = measure(null, K, ncmp, nsil, "null")
        res[str(K)] = {"real": real, "null": nul,
                       "ari_excess": round(real["ari_mean"] - nul["ari_mean"], 4),
                       "sil_excess": round(real["silhouette"] - nul["silhouette"], 4)}
        print(f"  K={K} done ({time.time() - t0:.0f}s)", flush=True)

    out = {"ks": KS, "seeds": SEEDS, "cmp_n": CMP_N, "silhouette_n": SIL_N,
           "null": "Gaussian with the pool's mean and covariance (same effective rank, no modes)",
           "results": res}
    p = os.path.join(OUT_DIR, "cluster_validity.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{'K':>6s}{'ARI real':>10s}{'ARI null':>10s}{'sil real':>10s}{'sil null':>10s}")
    for K in KS:
        r = res[str(K)]
        print(f"{K:>6d}{r['real']['ari_mean']:>10.3f}{r['null']['ari_mean']:>10.3f}"
              f"{r['real']['silhouette']:>10.3f}{r['null']['silhouette']:>10.3f}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
