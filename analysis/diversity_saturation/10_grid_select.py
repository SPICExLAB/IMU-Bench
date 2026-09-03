"""Build the index files for the motion-types x data-volume grid (11l).

Cell (K, F) = "pretrain on F% of the 5-source pool, drawn from only K motion
types". Two independent feasibility constraints, BOTH enforced here:

  (a) enough data exists   sum(sizes of the K largest clusters) >= F x N
  (b) K types fit in the draw            F x N >= K

(b) is easy to miss and is what kills (K=1000, F=0.01%): 196 windows cannot
contain 1000 distinct motion types. Ignoring it would silently produce a cell
labelled "1000 types" that actually holds ~196 of them, i.e. the x-axis would
be a lie at exactly the point the reader cares most about.

Selection within a cell is an equal-per-cluster waterfill over the K chosen
clusters (smallest first, so nothing is starved), NOT a uniform draw from their
union. A uniform draw would miss clusters whenever F x N is close to K -- e.g.
K=100 at 0.01% (196 windows) would land in ~85 distinct clusters, not 100. The
waterfill guarantees the realised type count equals K, which is verified and
recorded for every cell.

Clusters are taken LARGEST-FIRST, so K reads as "the K most common motion
types" and the feasible region is as large as it can be.

Pool: the 5 SSL sources (realdata, ocdetect, eatingspeed, motionx, sonic).
Indices are written in the 11-source cache concat space (via idx5src.npy), so
the pretrain configs keep all 11 sources enabled and the index file alone does
the restricting -- same mechanism as 11j/11k.

Writes /projects/b53017/vbc1591/diversity_saturation/grid/idx_k{K}_f{tag}.npy
       analysis/diversity_saturation/out/grid_selection.json
"""
from __future__ import annotations

import json
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
GRID_DIR = os.path.join(WORK_DIR, "grid")
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")

K_TYPES = [1, 5, 10, 25, 50, 100, 1000]
FRACTIONS = [("0p01", 0.0001), ("0p1", 0.001), ("1", 0.01), ("10", 0.10), ("100", 1.00)]
SEED = 42


def main():
    os.makedirs(GRID_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    idx5 = np.load(os.path.join(WORK_DIR, "idx5src.npy"))          # local -> global
    labels = np.load(os.path.join(WORK_DIR, "labels5src_k1000.npy")).astype(np.int32)
    n5 = len(idx5)
    assert len(labels) == n5
    sizes = np.bincount(labels, minlength=1000)
    order = np.argsort(sizes)[::-1]                                 # largest first
    by_cluster = {int(c): np.where(labels == c)[0] for c in range(1000)}
    print(f"[grid] 5-source pool {n5:,} windows, {int((sizes > 0).sum())} clusters")

    rng = np.random.default_rng(SEED)
    cells, skipped = {}, []

    for K in K_TYPES:
        chosen = order[:K]
        avail = int(sizes[chosen].sum())
        for tag, F in FRACTIONS:
            need = int(round(n5 * F))
            name = f"k{K}_f{tag}"
            if need > avail:
                skipped.append({"cell": name, "reason": "not enough data in K clusters",
                                "need": need, "available": avail})
                continue
            if need < K:
                skipped.append({"cell": name, "reason": "fewer windows than types",
                                "need": need, "K": K})
                continue

            # equal-per-cluster waterfill, smallest of the chosen first
            picked, remaining = [], need
            csorted = sorted(chosen, key=lambda c: sizes[c])
            left = len(csorted)
            for c in csorted:
                take = min(len(by_cluster[c]), int(np.ceil(remaining / left)))
                take = max(take, 1)                      # never starve a type
                take = min(take, len(by_cluster[c]), remaining - (left - 1))
                picked.append(rng.choice(by_cluster[c], size=take, replace=False))
                remaining -= take
                left -= 1
                if remaining <= 0:
                    break
            local = np.concatenate(picked)
            assert len(local) == need, (name, len(local), need)
            realised = int(len(np.unique(labels[local])))
            assert realised == K, f"{name}: realised {realised} types, wanted {K}"

            gidx = np.sort(idx5[local]).astype(np.int64)
            path = os.path.join(GRID_DIR, f"idx_{name}.npy")
            np.save(path, gidx)
            cells[name] = {"K": K, "fraction": F, "frac_tag": tag,
                           "n_windows": int(need), "types_realised": realised,
                           "windows_per_type": round(need / K, 1), "index_file": path}
            print(f"  {name:14s} K={K:<5d} n={need:>8,}  types={realised:<5d} "
                  f"({need / K:>8.1f} win/type)")

    print(f"\n[grid] {len(cells)} cells built, {len(skipped)} skipped")
    for s in skipped:
        print(f"  skip {s['cell']:14s} {s['reason']}")

    out = {"pool_n": int(n5), "seed": SEED, "cluster_choice": "largest_first",
           "selection": "equal_per_cluster_waterfill",
           "cells": cells, "skipped": skipped}
    p = os.path.join(OUT_DIR, "grid_selection.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
