#!/usr/bin/env python
"""Stage 4: build the index file for every cell of the volume x diversity
factorial, and MEASURE the diversity of each one instead of asserting it.

The factorial (EXPERIMENTS.md 11m) is the design the reviewer asked for:

                        N fixed, diversity varied  ->  read a column
    +----------+--------------------------------------------------------+
    | volume N | nar        one          nat            bal              |
    +----------+--------------------------------------------------------+
    | 12,500   | 1 source   1 source     4 sources      4 sources        |
    | 50,000   | narrow     natural      natural        source- AND      |
    | 200,000  | type band  type mix     proportions    type-balanced    |
    | 800,000  |                                                         |
    +----------+--------------------------------------------------------+
        diversity fixed, N varied  ->  read a row

`nar` vs `one` varies motion-type diversity with the source count pinned at 1;
`one` vs `nat`/`bal` varies source diversity. That is what stops the two
notions of diversity from being confounded with each other, and 03 reports
AMI(type, source) so the reader can check they are not collinear to begin with.

Selection is done in the ENCODER-FREE T-phys space (see 02/03). Within a cell:
  nar  the fewest largest T-phys clusters (restricted to ocdetect) whose total
       size reaches N, then a largest-first fill inside them
  one  uniform draw from ocdetect
  nat  uniform draw from the whole pool  (== what the paper's own scaling
       curve did, so this column reproduces it inside the controlled design)
  bal  per-source budget by equal water-filling (equal share, deficits from
       small sources redistributed to sources that still have windows), then
       equal-per-cluster water-filling inside each source, smallest cluster
       first so nothing is starved

Writes $WORK/idx/vd_*.npy and out/arms.json (realised composition + every
diversity statistic, under BOTH type definitions).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diversity_stats as ds
from arms import (IDX_DIR, K_TYPES, OUT_DIR, POOL_N, SEED, SOURCES, WORK_DIR,
                  all_arms, idx_path, source_ranges)

QUERY_N = 20_000       # pool sample used for the fill-distance coverage stat


def waterfill_clusters(cluster_members, order, n_target, rng):
    """Take `n_target` windows spread as evenly as possible over the clusters in
    `order`: repeatedly give every still-non-empty cluster an equal share,
    smallest first, so a small cluster is exhausted rather than starving a
    large one. Returns the chosen global indices."""
    chosen, remaining = [], int(n_target)
    pool = [np.asarray(cluster_members[int(c)]) for c in order]
    pool = [p for p in pool if len(p)]
    pool.sort(key=len)                      # smallest first
    for i, members in enumerate(pool):
        left = len(pool) - i
        share = int(np.ceil(remaining / left))
        take = min(len(members), share)
        pick = members if take == len(members) else rng.choice(members, take, replace=False)
        chosen.append(pick)
        remaining -= take
        if remaining <= 0:
            break
    out = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)
    return np.sort(out[:int(n_target)])


def narrow_band(cluster_members, sizes, allowed_clusters, n_target, rng):
    """Fewest (largest) clusters that can supply n_target, filled largest-first.
    The realised type count is whatever this minimum turns out to be, and it is
    recorded per cell -- at large N even the narrowest feasible band is not that
    narrow, which is a real property of the design and is reported, not hidden."""
    order = allowed_clusters[np.argsort(sizes[allowed_clusters])[::-1]]
    cum = np.cumsum(sizes[order])
    k_need = int(np.searchsorted(cum, n_target) + 1)
    assert k_need <= len(order), f"cannot reach {n_target} windows from the allowed clusters"
    chosen, remaining = [], int(n_target)
    for c in order[:k_need]:
        members = np.asarray(cluster_members[int(c)])
        take = min(len(members), remaining)
        chosen.append(members if take == len(members) else rng.choice(members, take, replace=False))
        remaining -= take
        if remaining <= 0:
            break
    return np.sort(np.concatenate(chosen))


def source_waterfill(n_target, avail):
    """Equal per-source budget with deficits redistributed. `avail` is an
    ordered dict name -> pool size."""
    names = list(avail)
    budget = {n: 0 for n in names}
    remaining, active = int(n_target), list(names)
    while remaining > 0 and active:
        share = remaining // len(active)
        if share == 0:
            for n in active[:remaining]:
                budget[n] += 1
            remaining = 0
            break
        progressed = False
        for n in list(active):
            take = min(share, avail[n] - budget[n])
            budget[n] += take
            remaining -= take
            if take:
                progressed = True
            if budget[n] >= avail[n]:
                active.remove(n)
        if not progressed:
            break
    return budget


def main():
    os.makedirs(IDX_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    labels_phys = np.load(os.path.join(WORK_DIR, "labels_phys_k1000.npy"))
    labels_enc = np.load(os.path.join(WORK_DIR, "labels_enc_k1000.npy"))
    z_enc = np.load(os.path.join(WORK_DIR, "enc_pca64_f32.npy"), mmap_mode="r")
    assert len(labels_phys) == POOL_N == len(labels_enc)

    pool_counts_phys = np.bincount(labels_phys, minlength=K_TYPES)
    pool_counts_enc = np.bincount(labels_enc, minlength=K_TYPES)
    ranges = source_ranges()
    oc_lo, oc_hi = ranges["ocdetect"]

    def group_by_cluster(labels, offset):
        """{cluster: global indices} in one sort instead of K passes."""
        order = np.argsort(labels, kind="stable")
        counts = np.bincount(labels, minlength=K_TYPES)
        bounds = np.concatenate([[0], np.cumsum(counts)])
        return {c: order[bounds[c]:bounds[c + 1]] + offset for c in range(K_TYPES)}

    members_src = {name: group_by_cluster(labels_phys[lo:hi], lo)
                   for name, (lo, hi) in ranges.items()}
    members_oc = members_src["ocdetect"]
    sizes_oc = np.array([len(members_oc[c]) for c in range(K_TYPES)])
    allowed_oc = np.where(sizes_oc > 0)[0]

    query_idx = np.sort(rng.choice(POOL_N, min(QUERY_N, POOL_N), replace=False))
    manifest = {"pool_n": POOL_N, "k": K_TYPES, "seed": SEED,
                "selection_space": "T-phys (encoder-free physical features)",
                "arms": {}}

    for arm, div, vol_tag, n in all_arms():
        # 100 % of the pool under the balanced sampler: the windows cannot be
        # re-selected, only re-WEIGHTED, so this arm gets no index file (05
        # emits the sampler config instead) but is still measured on the full
        # pool so it has a row in every table.
        sampler_arm = (div == "bal" and vol_tag == "nfull")

        if sampler_arm:
            idx = np.arange(POOL_N)
        elif div == "nar":
            idx = narrow_band(members_oc, sizes_oc, allowed_oc, n, rng)
        elif div == "one":
            idx = np.sort(rng.choice(np.arange(oc_lo, oc_hi), n, replace=False))
        elif div == "nat":
            idx = (np.arange(POOL_N) if n >= POOL_N
                   else np.sort(rng.choice(POOL_N, n, replace=False)))
        elif div == "bal":
            budget = source_waterfill(n, {s: (ranges[s][1] - ranges[s][0]) for s in SOURCES})
            parts = []
            for s in SOURCES:
                if budget[s] == 0:
                    continue
                mem = members_src[s]
                present = np.array([c for c in range(K_TYPES) if len(mem[c])])
                parts.append(waterfill_clusters(mem, present, budget[s], rng))
            idx = np.sort(np.concatenate(parts))
        else:
            raise ValueError(div)

        assert len(idx) == n, f"{arm}: selected {len(idx)} != {n}"
        assert len(np.unique(idx)) == len(idx), f"{arm}: duplicate indices"
        if not sampler_arm:
            np.save(idx_path(arm), idx.astype(np.int64))

        stats = ds.summarise(idx, labels_phys, labels_enc, pool_counts_phys,
                             pool_counts_enc, ranges, K_TYPES,
                             z_enc=z_enc, z_query_idx=query_idx, seed=SEED)
        stats.update({"diversity": div, "volume_tag": vol_tag,
                      "index_file": None if sampler_arm else idx_path(arm),
                      "sampler": "balanced (source_weights)" if sampler_arm else "natural"})
        if sampler_arm:
            stats["note"] = ("same 100 % of the pool as vd_nat_nfull, different sampling "
                             "law -- the type/source statistics below are of the DATA, the "
                             "sampler re-weights how often each window is drawn")
        manifest["arms"][arm] = stats
        print(f"[{arm:16s}] N={len(idx):>9,}  sources={stats['n_sources']}  "
              f"phys types={stats['phys']['types_present']:>4} "
              f"eff={stats['phys']['effective_types_shannon']:>7.1f}  "
              f"enc eff={stats['enc']['effective_types_shannon']:>7.1f}  "
              f"vendi={stats.get('vendi_enc', float('nan')):.1f}  "
              f"fill_med={stats.get('pool_coverage', {}).get('fill_median', float('nan')):.3f}",
              flush=True)

    with open(os.path.join(OUT_DIR, "arms.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[arms] wrote {os.path.join(OUT_DIR, 'arms.json')}", flush=True)


if __name__ == "__main__":
    main()
