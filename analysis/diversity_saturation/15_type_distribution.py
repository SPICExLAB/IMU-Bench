#!/usr/bin/env python
"""15_type_distribution.py -- how each dataset's mass is distributed over the 1000 types.

13_motion_type_diversity.py counted how MANY types a source touches. This asks
what the distribution over them looks like, three ways:

  1. Shape      rank-abundance (Whittaker) curve per source + Gini, and the
                mass held by the top 1/10/50/100 types. Distinguishes "broad and
                flat" from "broad but living in a handful of types".
  2. Location   the 1000 types are Ward-clustered into 10 macro-families over
                the k-means centroids, each family characterised by the physical
                signal statistics sampled in 14_characterise_motion_types.py.
                Each source then has a readable 10-bin distribution -- WHERE in
                motion space it sits, not just how much of it it covers.
  3. Overlap    per-type source composition, so a type can be read as
                single-source or shared.

Centroids are recomputed as the PCA-64 mean of each label (the k-means object
itself was not persisted); that is the same quantity k-means converged to.

Outputs analysis/diversity_saturation/out/motion_type_distribution.json
"""
from __future__ import annotations

import json
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")

K = 1000
N_FAMILIES = 10
SEED = 42
STATS = ["acc_dyn", "gyro_mag", "dom_freq", "spec_ent", "still_frac"]


def gini(x):
    """Gini of a non-negative vector; 0 = perfectly even, 1 = all mass in one bin."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    man = json.load(open(os.path.join(WORK_DIR, "pool_manifest.json")))
    names, cum, n = man["sources"], np.array(man["cumulative_sizes"]), man["n"]
    labels = np.load(os.path.join(WORK_DIR, "labels11src_k1000.npy")).astype(np.int32)
    z = np.load(os.path.join(WORK_DIR, "pca64_f32.npy"), mmap_mode="r")

    src_of = np.searchsorted(cum, np.arange(n), side="right")
    table = np.zeros((len(names), K), dtype=np.int64)
    np.add.at(table, (src_of, labels), 1)
    pool = table.sum(0)

    # ---------- 1. shape of each source's distribution ----------
    shape = {}
    for i, nm in enumerate(names):
        c = table[i].astype(np.float64)
        tot = c.sum()
        srt = np.sort(c)[::-1]
        cs = np.cumsum(srt) / tot
        top = {}
        for kk in (1, 10, 50, 100, 250):
            top[str(kk)] = round(float(cs[min(kk, len(cs)) - 1] * 100), 2)
        # types needed to cover 50% / 90% of this source's windows
        shape[nm] = {
            "n_windows": int(tot),
            "types_touched": int((c > 0).sum()),
            "gini_over_touched": round(gini(c[c > 0]), 4),
            "gini_over_all_1000": round(gini(c), 4),
            "mass_in_top_k_types_pct": top,
            "types_for_50pct_mass": int(np.searchsorted(cs, 0.50) + 1),
            "types_for_90pct_mass": int(np.searchsorted(cs, 0.90) + 1),
            # rank-abundance curve, subsampled for plotting
            "rank_abundance_pct": [round(float(srt[r - 1] / tot * 100), 5)
                                   for r in _ranks(int((c > 0).sum()))],
            "rank_abundance_ranks": _ranks(int((c > 0).sum())),
        }

    # ---------- 2. macro-families ----------
    cent = np.zeros((K, z.shape[1]), dtype=np.float64)
    for c in range(K):
        m = labels == c
        cent[c] = np.asarray(z[m]).mean(0)

    from scipy.cluster.hierarchy import linkage, fcluster
    Zl = linkage(cent, method="ward")
    fam = fcluster(Zl, t=N_FAMILIES, criterion="maxclust") - 1     # 0-indexed

    # physical signature of each family, from the 12k sampled windows
    raw = np.load(os.path.join(WORK_DIR, "type_semantics_raw.npz"), allow_pickle=True)
    S, pick_type = raw["S"], raw["pick_type"]
    fam_of_window = fam[pick_type]
    fam_stats = {}
    for f in range(N_FAMILIES):
        m = fam_of_window == f
        fam_stats[f] = {k: round(float(S[m, j].mean()), 4) for j, k in enumerate(STATS)}

    fam_types = {f: int((fam == f).sum()) for f in range(N_FAMILIES)}
    fam_windows = np.zeros(N_FAMILIES, dtype=np.int64)
    for f in range(N_FAMILIES):
        fam_windows[f] = pool[fam == f].sum()

    # per-source distribution over families
    fam_table = np.zeros((len(names), N_FAMILIES), dtype=np.int64)
    for f in range(N_FAMILIES):
        fam_table[:, f] = table[:, fam == f].sum(1)
    fam_share = fam_table / fam_table.sum(1, keepdims=True) * 100

    # order families by tempo so the axis reads left-to-right as "slow -> fast"
    order = sorted(range(N_FAMILIES), key=lambda f: fam_stats[f]["dom_freq"])

    families = []
    for rank, f in enumerate(order):
        families.append({
            "id": int(f), "slot": rank,
            "n_types": fam_types[f],
            "n_windows": int(fam_windows[f]),
            "pool_share_pct": round(float(fam_windows[f] / pool.sum() * 100), 2),
            **fam_stats[f],
            "dominant_source": names[int(np.argmax(fam_table[:, f]))],
            "dominant_source_share_pct": round(
                float(fam_table[:, f].max() / max(fam_table[:, f].sum(), 1) * 100), 1),
        })

    per_source_families = {
        nm: {str(rank): round(float(fam_share[i, f]), 2) for rank, f in enumerate(order)}
        for i, nm in enumerate(names)}

    # ---------- 3. per-type composition ----------
    occ = (table > 0).sum(0)
    dom_share = table.max(0) / np.maximum(pool, 1)
    composition = {
        "types_by_occupant_count": {str(k): int((occ == k).sum())
                                    for k in range(1, len(names) + 1) if (occ == k).sum()},
        "types_with_dominant_source_over_90pct": int((dom_share > 0.90).sum()),
        "types_with_dominant_source_over_50pct": int((dom_share > 0.50).sum()),
        "median_dominant_share_pct": round(float(np.median(dom_share) * 100), 1),
        "pool_type_size_pcts": {p: int(np.percentile(pool, p)) for p in (0, 10, 50, 90, 100)},
        "pool_gini": round(gini(pool.astype(float)), 4),
    }

    out = {"k": K, "n_families": N_FAMILIES, "family_order_by": "dom_freq",
           "stat_units_caveat": ("signal statistics are measured on per-window z-scored "
                                 "channels -- shape descriptors, not g or rad/s"),
           "shape": shape, "families": families,
           "per_source_family_share_pct": per_source_families,
           "composition": composition}
    p = os.path.join(OUT_DIR, "motion_type_distribution.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'source':14s}{'types':>6s}{'gini':>7s}{'top1%':>7s}{'top10%':>8s}"
          f"{'top50%':>8s}{'n@50%':>7s}{'n@90%':>7s}")
    for nm in names:
        s = shape[nm]
        print(f"{nm:14s}{s['types_touched']:>6d}{s['gini_over_touched']:>7.3f}"
              f"{s['mass_in_top_k_types_pct']['1']:>7.1f}{s['mass_in_top_k_types_pct']['10']:>8.1f}"
              f"{s['mass_in_top_k_types_pct']['50']:>8.1f}"
              f"{s['types_for_50pct_mass']:>7d}{s['types_for_90pct_mass']:>7d}")

    print(f"\n{'fam':>4s}{'types':>7s}{'pool%':>7s}{'accdyn':>8s}{'gyro':>7s}"
          f"{'domHz':>7s}{'spent':>7s}{'still':>7s}  dominant")
    for fd in families:
        print(f"{fd['slot']:>4d}{fd['n_types']:>7d}{fd['pool_share_pct']:>7.2f}"
              f"{fd['acc_dyn']:>8.3f}{fd['gyro_mag']:>7.3f}{fd['dom_freq']:>7.2f}"
              f"{fd['spec_ent']:>7.3f}{fd['still_frac']:>7.3f}  "
              f"{fd['dominant_source']} ({fd['dominant_source_share_pct']}%)")

    print(f"\n{'source':14s}" + "".join(f"{'F'+str(r):>7s}" for r in range(N_FAMILIES)))
    for nm in names:
        print(f"{nm:14s}" + "".join(f"{per_source_families[nm][str(r)]:>7.1f}"
                                    for r in range(N_FAMILIES)))
    print(f"\nwrote {p}")


def _ranks(n_touched):
    """log-spaced ranks for a plottable rank-abundance curve."""
    if n_touched <= 1:
        return [1]
    r = np.unique(np.round(np.logspace(0, np.log10(n_touched), 40)).astype(int))
    return [int(x) for x in r if 1 <= x <= n_touched]


if __name__ == "__main__":
    main()
