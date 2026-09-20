#!/usr/bin/env python
"""Stage 24: the physical-type x volume grid with RANDOM, source-stratified motion types.

Design (agreed 2026-09-13):
  types   K = 1 / 10 / 50 / 100 % of the 1,000 clusters of the gyro-unit-fixed T-phys
          partition (23_refit_tphys_gyrofix.py) = 10 / 100 / 500 / 1,000 clusters.
          Clusters are taken as prefixes of ONE nested random order (seed 42), stratified
          so every prefix holds each source's dominant clusters in proportion to the pool
          (largest-remainder interleave). Nested: the 10 clusters are inside the 100, etc.
  volume  N = 0.1 / 1 / 10 / 100 % of the 1,738,094-window pool, drawn uniformly at random
          inside the chosen clusters, as prefixes of one per-K permutation (nested volumes).
  exists  only where the chosen clusters hold >= N windows; when they hold 85-100 % of N
          (the diagonal), the cell takes every window of those clusters.
  reuse   K 100 % x N 100 % is the whole pool = the completed arm vd_ek1000_f100; not rebuilt.

This fixes both confounds of the largest-first grids: the K axis no longer changes the
source mix (every prefix is stratified), and no cell is pinned to the dominant mode.

Writes idx/vd_pk{K%}_n{N%}.npy, updates out/arms.json, writes out/phys_random_grid.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diversity_stats as ds
from arms import IDX_DIR, K_TYPES, OUT_DIR, POOL_N, SEED, SOURCES, WORK_DIR, source_ranges

_spec = importlib.util.spec_from_file_location("axis_arms", os.path.join(HERE, "10_axis_arms.py"))
ax = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ax)

K_LEVELS = [(1, 10), (10, 100), (50, 500), (100, 1000)]
N_LEVELS = [("0p1", 1e-3), ("1", 1e-2), ("10", 1e-1), ("100", 1.0)]
DIAGONAL_MIN = 0.85
REUSE = {(100, "100"): "vd_ek1000_f100"}


def stratified_order(dom, rng):
    pools = {j: list(rng.permutation(np.flatnonzero(dom == j))) for j in range(len(SOURCES))}
    tot = {j: len(v) for j, v in pools.items()}
    taken = {j: 0 for j in pools}
    order = []
    for k in range(1, K_TYPES + 1):
        j = max((j for j in pools if taken[j] < tot[j]), key=lambda j: k * tot[j] / K_TYPES - taken[j])
        order.append(int(pools[j][taken[j]])); taken[j] += 1
    return order


def main():
    labels = np.load(os.path.join(WORK_DIR, "labels_phys_gyrofix_k1000.npy"))
    labels_enc = np.load(os.path.join(WORK_DIR, "labels_enc_k1000.npy"))
    z_enc = np.load(os.path.join(WORK_DIR, "enc_pca64_f32.npy"), mmap_mode="r")
    ranges = source_ranges()
    pc_phys = np.bincount(labels, minlength=K_TYPES)
    pc_enc = np.bincount(labels_enc, minlength=K_TYPES)
    rng = np.random.default_rng(SEED)
    query_idx = np.sort(rng.choice(POOL_N, 20_000, replace=False))
    subj = np.load(os.path.join(WORK_DIR, "prov_subject.npy"))
    sess = np.load(os.path.join(WORK_DIR, "prov_session.npy"))

    M = np.zeros((K_TYPES, len(SOURCES)), np.int64)
    for j, s in enumerate(SOURCES):
        M[:, j] = np.bincount(labels[ranges[s][0]:ranges[s][1]], minlength=K_TYPES)
    dom = (M / np.maximum(M.sum(1, keepdims=True), 1)).argmax(1)
    order = stratified_order(dom, np.random.default_rng(SEED))

    arms_out, skipped, cells = {}, [], []
    for kpct, K in K_LEVELS:
        chosen = np.array(order[:K])
        members = np.flatnonzero(np.isin(labels, chosen))
        perm = np.random.default_rng([SEED, K]).permutation(members)
        for tag, f in N_LEVELS:
            target = int(round(f * POOL_N))
            if (kpct, tag) in REUSE:
                cells.append({"k_pct": kpct, "n_tag": tag, "arm": REUSE[(kpct, tag)], "reused": True,
                              "n_windows": POOL_N, "target": target})
                continue
            if len(members) < DIAGONAL_MIN * target:
                skipped.append({"k_pct": kpct, "n_tag": tag, "target": target, "available": int(len(members))})
                continue
            n = min(target, len(members))
            idx = np.sort(perm[:n]).astype(np.int64)
            arm = f"vd_pk{kpct}_n{tag}"
            np.save(os.path.join(IDX_DIR, f"{arm}.npy"), idx)
            st = ds.summarise(idx, labels, labels_enc, pc_phys, pc_enc, ranges, K_TYPES,
                              z_enc=z_enc, z_query_idx=query_idx, seed=SEED)
            st["nn_self_median"] = ax.self_nn_median(idx, z_enc, np.random.default_rng(SEED))
            s_, g_ = subj[idx], sess[idx]
            st.update({"diversity": f"pk{kpct}", "volume_tag": f"n{tag}",
                       "index_file": os.path.join(IDX_DIR, f"{arm}.npy"), "sampler": "natural",
                       "n_types_allowed": K, "type_space": "phys_gyrofix", "grid_k_pct": kpct,
                       "grid_fraction": f, "target_windows": target,
                       "short_of_target_pct": round(100 * (1 - n / target), 2),
                       "cluster_order": "nested source-stratified random, seed 42",
                       "n_subjects_known": int(len(np.unique(s_[s_ >= 0]))),
                       "n_sessions_known": int(len(np.unique(g_[g_ >= 0])))})
            arms_out[arm] = st
            cells.append({"k_pct": kpct, "n_tag": tag, "arm": arm, "reused": False, "n_windows": n, "target": target})
            print(f"[{arm:15s}] n={n:>9,} (target {target:>9,})  phys eff={st['phys']['effective_types_shannon']:6.1f}  "
                  f"types seen={st['phys'].get('types_present', '-')}  sources={st.get('source_counts')}", flush=True)

    apath = os.path.join(OUT_DIR, "arms.json")
    manifest = json.load(open(apath))
    manifest["arms"].update(arms_out)
    json.dump(manifest, open(apath, "w"), indent=2)
    json.dump({"k_levels": K_LEVELS, "n_levels": N_LEVELS, "pool_n": POOL_N, "cluster_order": order,
               "dominant_source_of_cluster": dom.tolist(), "cells": cells, "skipped": skipped,
               "arms": arms_out}, open(os.path.join(OUT_DIR, "phys_random_grid.json"), "w"), indent=2)
    print(f"[grid] {len(arms_out)} new cells, {len(cells) - len(arms_out)} reused, {len(skipped)} infeasible")


if __name__ == "__main__":
    main()
