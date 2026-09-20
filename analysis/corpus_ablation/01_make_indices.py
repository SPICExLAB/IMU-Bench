#!/usr/bin/env python3
"""Corpus-construction ablation (2026-09-18): three pretraining subsets of the 5-source pool, in the
same 11-source cache index space as the motion-type grid, so every arm shares one cache, one
validation set and one training recipe with grid cell k1000_f10 (arm C, equal-per-cluster over
1,000 k-means motion modes). Only the index file differs between arms.

  ref      per-source counts identical to the reference corpus (HAR, SONIC, Motion-X in full,
           EatingSpeed 57,536, OCDetect 8,703 windows; random windows within a source)
  uniform  195,929 windows drawn uniformly from the pool
  temp03   195,929 windows, source share p_s ~ n_s^0.3 (temperature sampling), sources that run
           out (Motion-X) are taken in full and the remainder re-split over the others;
           uniform within a source

Writes /projects/b53017/vbc1591/diversity_saturation/corpus_ablation/idx_{ref,uniform,temp03}.npy
and analysis/corpus_ablation/out/indices.json
"""
import json
import os

import numpy as np

WORK = "/projects/b53017/vbc1591/diversity_saturation"
OUT = os.path.join(WORK, "corpus_ablation")
HERE = os.path.dirname(os.path.abspath(__file__))
SEED, N, ALPHA = 42, 195_929, 0.3
REF_COUNTS = {"realdata": 93_510, "ocdetect": 8_703, "eatingspeed": 57_536, "motionx": 4_683, "sonic": 32_662}

man = json.load(open(os.path.join(WORK, "pool_manifest.json")))
cum = np.array(man["cumulative_sizes"])
starts = np.concatenate([[0], cum[:-1]])
rng_of = {s: (int(starts[i]), int(cum[i])) for i, s in enumerate(man["sources"])}
idx5 = np.load(os.path.join(WORK, "idx5src.npy"))
pool = {s: np.arange(*rng_of[s]) for s in REF_COUNTS}
assert sum(len(v) for v in pool.values()) == len(idx5) == 1_959_288
assert np.array_equal(np.sort(np.concatenate(list(pool.values()))), np.sort(idx5))


def draw(counts, seed):
    rng = np.random.default_rng(seed)
    parts = [rng.choice(pool[s], size=c, replace=False) if c < len(pool[s]) else pool[s] for s, c in counts.items()]
    return np.sort(np.concatenate(parts)).astype(np.int64)


def temperature_counts(n, alpha):
    sizes = {s: len(v) for s, v in pool.items()}
    counts, left, active = {}, n, dict(sizes)
    while True:                                   # waterfill: cap exhausted sources, re-split the rest
        w = {s: sizes[s] ** alpha for s in active}
        tot = sum(w.values())
        want = {s: left * w[s] / tot for s in active}
        full = [s for s in active if want[s] >= sizes[s]]
        if not full:
            break
        for s in full:
            counts[s] = sizes[s]; left -= sizes[s]; active.pop(s)
    fl = {s: int(np.floor(want[s])) for s in active}
    rem = left - sum(fl.values())
    for s in sorted(active, key=lambda s: want[s] - fl[s], reverse=True)[:rem]:
        fl[s] += 1
    counts.update(fl)
    return {s: counts[s] for s in REF_COUNTS}


arms = {"ref": REF_COUNTS, "temp03": temperature_counts(N, ALPHA)}
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
report = {}
for name, counts in arms.items():
    idx = draw(counts, SEED)
    np.save(os.path.join(OUT, f"idx_{name}.npy"), idx)
    report[name] = {"n": int(len(idx)), "counts": {s: int(c) for s, c in counts.items()}}
rng = np.random.default_rng(SEED)
uni = np.sort(rng.choice(idx5, size=N, replace=False)).astype(np.int64)
np.save(os.path.join(OUT, "idx_uniform.npy"), uni)
src = np.searchsorted(cum, uni, side="right")
report["uniform"] = {"n": int(len(uni)), "counts": {s: int((src == man["sources"].index(s)).sum()) for s in REF_COUNTS}}
for name, r in report.items():
    r["share_pct"] = {s: round(100 * c / r["n"], 1) for s, c in r["counts"].items()}
    all_idx = np.load(os.path.join(OUT, f"idx_{name}.npy"))
    assert np.isin(all_idx, idx5).all(), name      # never outside the 5 sources
    assert len(np.unique(all_idx)) == len(all_idx), name
json.dump(report, open(os.path.join(HERE, "out", "indices.json"), "w"), indent=1)
for name, r in report.items():
    print(f"{name:8s} n={r['n']:>7,}  " + "  ".join(f"{s}={r['share_pct'][s]:>5.1f}%" for s in REF_COUNTS))
