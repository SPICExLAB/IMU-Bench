#!/usr/bin/env python
"""Stage 11: at which K does the PHYSICAL feature space actually have modes?

Every diversity number on the motion-type axis is defined relative to a k-means
partition at K=1000, and 03 reported only that one K. The encoder-space sweep in
analysis/diversity_saturation/18_cluster_validity.py swept K in {10,50,200,1000}
and found structure is STRONGEST AT SMALL K -- silhouette excess over a matched
Gaussian null falls 0.170 (K=10) -> 0.067 (50) -> 0.044 (200) -> ~0.02 (1000).
K=1000 is the weakest point on that curve, i.e. the type axis was built where
there is least structure to find.

This runs the identical measurement on the 32-d physical PCA that actually
DEFINES the type ladder, so the paper can state where the modes are instead of
asserting that K=1000 clusters are "motion types".

Null: a Gaussian with the pool's own mean and covariance -- same shape and
effective rank, no modes. Whatever it scores is the floor any k-means produces,
so only the EXCESS over it is evidence of structure.

Outputs analysis/volume_diversity/out/k_sweep_phys.json
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import OUT_DIR, WORK_DIR

KS = [10, 25, 50, 100, 200, 500, 1000]
SEEDS = [42, 7, 2024]
CMP_N = 200_000
SIL_N = 20_000


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
    # `--space enc` runs the same sweep on the mae_4src_z CLS PCA-64 (T-enc), so
    # the two type definitions can be compared at matched K instead of only at 1000.
    space = "enc" if "--space" in sys.argv and sys.argv[sys.argv.index("--space") + 1] == "enc" else "phys"
    fname = {"phys": "phys_pca32_f32.npy", "enc": "enc_pca64_f32.npy"}[space]
    z = np.asarray(np.load(os.path.join(WORK_DIR, fname), mmap_mode="r"))
    n = len(z)
    rng = np.random.default_rng(0)
    cmp_idx = rng.choice(n, size=min(CMP_N, n), replace=False)
    sil_idx = rng.choice(n, size=min(SIL_N, n), replace=False)

    print(f"[ksweep] {space} PCA {z.shape}", flush=True)
    mu, cov = z.mean(0), np.cov(z[rng.choice(n, size=min(200_000, n), replace=False)].T)
    null = rng.multivariate_normal(mu, cov, size=min(400_000, n)).astype(np.float32)
    null_cmp = rng.choice(len(null), size=min(CMP_N, len(null)), replace=False)
    null_sil = rng.choice(len(null), size=min(SIL_N, len(null)), replace=False)

    out = {"ks": KS, "seeds": SEEDS, "space": fname[:-8], "pool_n": int(n),
           "null": "Gaussian with the pool's mean and covariance (same effective rank, no modes)",
           "results": {}}
    for K in KS:
        t = time.time()
        real = measure(z, K, cmp_idx, sil_idx, "real")
        nul = measure(null, K, null_cmp, null_sil, "null")
        out["results"][str(K)] = {
            "real": real, "null": nul,
            "ari_excess": round(real["ari_mean"] - nul["ari_mean"], 4),
            "sil_excess": round(real["silhouette"] - nul["silhouette"], 4)}
        print(f"  K={K} done in {time.time()-t:.0f}s "
              f"ari_excess={out['results'][str(K)]['ari_excess']} "
              f"sil_excess={out['results'][str(K)]['sil_excess']}", flush=True)

    p = os.path.join(OUT_DIR, f"k_sweep_{space}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[ksweep] wrote {p}")
    best_s = max(KS, key=lambda k: out["results"][str(k)]["sil_excess"])
    best_a = max(KS, key=lambda k: out["results"][str(k)]["ari_excess"])
    print(f"[ksweep] silhouette excess peaks at K={best_s}, ARI excess at K={best_a}")


if __name__ == "__main__":
    main()
