#!/usr/bin/env python
"""04_verify_coverage.py -- independent verification of the coverage-saturation claim.

01_embed_select_analyze.py reported that a random draw already covers 99.9% of
K=1000 motion clusters at 0.5% of the pool and 100% at 1.3%. That number
carried three weaknesses, each of which this script addresses:

  1. ONE random draw per fraction (single seed) -> here: 5 draws, mean +/- sd.
  2. ONE clustering (K=1000, seed 42) -> here: refit at seed 7 to check the
     result is not an artifact of one k-means solution.
  3. ONE mode granularity -> here: K=100 / 1000 / 4000. If the saturation
     point moves right as K grows, the claim is granularity-dependent and
     must be stated as "saturated at the granularity this encoder resolves",
     not as "diversity is saturated" full stop.

It also cross-checks every empirical number against the closed form
    E[coverage] = 1 - mean_i exp(-m_i * k / N)
(probability that a cluster of size m_i survives a k-of-N draw), which
depends only on the cluster-size distribution. Agreement means the empirical
number is a property of the data, not a bug in the sampling code.

Reads the PCA-64 embeddings 01_... already wrote; no GPU needed.

Usage (SLURM):  sbatch launch_slurm_verify_coverage.sh
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")

FRACTIONS = [0.0001, 0.001, 0.005, 0.013, 0.05, 0.106, 0.25]
CONFIGS = [(1000, 42), (1000, 7), (100, 42), (4000, 42)]
N_DRAWS = 5


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    z = np.load(os.path.join(WORK_DIR, "pca64_f32.npy"), mmap_mode="r")
    n = z.shape[0]
    print(f"[verify] pool n={n} dim={z.shape[1]}", flush=True)
    Z = np.asarray(z)

    results = {"n_pool": int(n), "n_draws": N_DRAWS, "configs": {}}

    for K, seed in CONFIGS:
        tag = f"K{K}_seed{seed}"
        km = MiniBatchKMeans(n_clusters=K, batch_size=10_000, n_init=3,
                             random_state=seed, max_no_improvement=50).fit(Z)
        lab = km.predict(Z)
        sizes = np.bincount(lab, minlength=K)
        nz_sizes = sizes[sizes > 0].astype(float)
        nz = len(nz_sizes)
        print(f"\n--- {tag}: non-empty={nz}  min={int(nz_sizes.min())} "
              f"median={int(np.median(nz_sizes))} max={int(nz_sizes.max())}", flush=True)
        print(f"{'fraction':>10}{'windows':>10}{'empirical':>22}{'closed form':>14}")

        entry = {"n_clusters_nonempty": int(nz),
                 "cluster_size_min": int(nz_sizes.min()),
                 "cluster_size_median": float(np.median(nz_sizes)),
                 "cluster_size_max": int(nz_sizes.max()),
                 "coverage": {}}
        for f in FRACTIONS:
            k = max(1, int(round(n * f)))
            emp = []
            for s in range(N_DRAWS):
                rng = np.random.default_rng(1000 + s)
                draw = rng.choice(n, size=k, replace=False)
                emp.append(len(np.unique(lab[draw])) / nz)
            m, sd = float(np.mean(emp)), float(np.std(emp))
            ana = float(1.0 - np.mean(np.exp(-nz_sizes * k / n)))
            entry["coverage"][f"{f}"] = {"empirical_mean": m, "empirical_sd": sd,
                                         "closed_form": ana, "k_windows": k}
            print(f"{f*100:>9.2f}%{k:>10}{m*100:>16.2f}% (±{sd*100:.2f}){ana*100:>13.2f}%",
                  flush=True)
        results["configs"][tag] = entry

    out = os.path.join(OUT_DIR, "coverage_verification.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[verify] wrote {out}", flush=True)

    # headline: does the saturation point move with K?
    print("\n=== granularity dependence: fraction needed for >=99% coverage ===", flush=True)
    for tag, e in results["configs"].items():
        hit = next((f for f in FRACTIONS
                    if e["coverage"][f"{f}"]["empirical_mean"] >= 0.99), None)
        print(f"  {tag:14s} {'%.2f%%' % (hit*100) if hit else '>25%'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
