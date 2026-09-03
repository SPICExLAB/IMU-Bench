"""Sub-1% volume sweep: build the index files for 0.01 / 0.05 / 0.1 / 0.5 %.

Why this sweep exists
---------------------
EXPERIMENTS.md 11i swept 1.3% -> 100% and found a flat/inverted curve; 11j
showed the mechanism is that k-means motion-mode coverage is already 100% at
1.3%. Every point in 11i therefore sits ON the plateau. The rising limb -- if
there is one -- has to live BELOW 1%, because that is the only place coverage
is not saturated:

    0.01%  ->  28.8% coverage
    0.1%   ->  92.7%
    0.5%   ->  99.94%
    1.3%   ->  100%      (= 11j's random-1% arm, already trained)

If downstream performance rises across 0.01 -> 0.5% and then goes flat from
1.3% -> 100%, the knee coincides with coverage saturation and the two halves
of the argument join into one curve.

Design
------
* NESTED subsets. The four subsets are prefixes of a single shuffle of 11j's
  own `idx_random_1pct.npy`, so 356 c 1,778 c 3,555 c 17,776 c 46,116. Each
  arm literally has all the data of the smaller ones plus more, which is what
  "data amount" on the x-axis should mean, and it removes draw-to-draw
  variance from the curve. A uniform subsample of a uniform sample is still a
  uniform sample, so nesting costs nothing statistically.
* Same index space as 11j (v3 cache concat order, 11 sources, NHANES off), so
  random-1% is a free fifth point at the identical 105K-step budget.
* Coverage is measured in the SAME PCA-64 / k-means(K=1000) space as 11j, by
  refitting the identical estimator (seed 42, deterministic) on the cached
  `pca64_f32.npy`. No re-embedding.

Outputs:
  /projects/b53017/vbc1591/diversity_saturation/idx_sub_{0p01,0p05,0p1,0p5}pct.npy
  analysis/diversity_saturation/out/subpct_selection.json
"""
from __future__ import annotations

import json
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")

K_CLUSTERS = 1000
SEED = 42

# (config tag, nominal pool fraction) -- N is derived from the pool size so the
# numbers match 11j's coverage_verification k_windows exactly.
FRACTIONS = [("0p01", 0.0001), ("0p05", 0.0005), ("0p1", 0.001), ("0p5", 0.005)]


def main():
    from sklearn.cluster import MiniBatchKMeans

    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = json.load(open(os.path.join(WORK_DIR, "pool_manifest.json")))
    n_pool = int(manifest["n"]) if "n" in manifest else int(manifest["n_pool"])

    base = np.load(os.path.join(WORK_DIR, "idx_random_1pct.npy"))
    print(f"[sub] pool={n_pool}  random-1% anchor={len(base)}", flush=True)

    # one shuffle -> prefixes are nested by construction
    rng = np.random.default_rng(SEED)
    shuffled = base.copy()
    rng.shuffle(shuffled)

    subsets = {}
    for tag, frac in FRACTIONS:
        k = max(1, int(round(n_pool * frac)))
        assert k <= len(base), f"{tag}: {k} > anchor {len(base)}"
        subsets[tag] = np.sort(shuffled[:k].astype(np.int64))

    # nesting check (the whole point of the single shuffle)
    ordered = [t for t, _ in FRACTIONS]
    for small, large in zip(ordered, ordered[1:]):
        assert np.isin(subsets[small], subsets[large]).all(), f"{small} not nested in {large}"
    assert np.isin(subsets[ordered[-1]], base).all(), "0p5 not nested in random-1%"
    print("[sub] nesting verified: " + " c ".join(f"{t}({len(subsets[t])})" for t in ordered)
          + f" c random-1%({len(base)})", flush=True)

    # ---- coverage in 11j's own cluster space -------------------------------
    z = np.load(os.path.join(WORK_DIR, "pca64_f32.npy"), mmap_mode="r")
    assert z.shape[0] == n_pool, (z.shape, n_pool)
    km = MiniBatchKMeans(n_clusters=K_CLUSTERS, batch_size=10_000, n_init=3,
                         random_state=SEED, max_no_improvement=50)
    zz = np.asarray(z)
    km.fit(zz)
    labels = km.predict(zz)
    n_nonempty = int((np.bincount(labels, minlength=K_CLUSTERS) > 0).sum())
    print(f"[sub] kmeans refit: {n_nonempty} non-empty clusters", flush=True)

    def coverage(idx):
        return float(len(np.unique(labels[idx])) / n_nonempty)

    rows = {}
    for tag, frac in FRACTIONS:
        idx = subsets[tag]
        path = os.path.join(WORK_DIR, f"idx_sub_{tag}pct.npy")
        np.save(path, idx)
        rows[tag] = {"nominal_fraction": frac,
                     "n_windows": int(len(idx)),
                     "effective_fraction": float(len(idx) / n_pool),
                     "coverage": coverage(idx),
                     "index_file": path}
        print(f"[sub] {tag:5s} n={len(idx):6d}  frac={len(idx)/n_pool:.5%}  "
              f"coverage={rows[tag]['coverage']:.3%}  -> {path}", flush=True)

    rows["random_1pct"] = {"nominal_fraction": 0.013,
                           "n_windows": int(len(base)),
                           "effective_fraction": float(len(base) / n_pool),
                           "coverage": coverage(base),
                           "index_file": os.path.join(WORK_DIR, "idx_random_1pct.npy")}
    print(f"[sub] anchor n={len(base)} coverage={rows['random_1pct']['coverage']:.3%}", flush=True)

    out = {"n_pool": n_pool, "k_clusters": K_CLUSTERS, "seed": SEED,
           "nested": True, "arms": rows}
    out_path = os.path.join(OUT_DIR, "subpct_selection.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[sub] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
