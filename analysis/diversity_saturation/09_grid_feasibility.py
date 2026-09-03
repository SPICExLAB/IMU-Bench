"""Feasibility of the motion-types x data-volume grid on the 5-source pool.

The requested figure is x = number of motion types (1,5,10,25,50,100,1000),
y = performance, one line per data fraction (0.01/0.1/1/10/100%). That grid is
NOT a rectangle: picking K motion types caps how much data exists to draw from.
A cell (K types, F% of pool) is only realisable if

    sum(sizes of the K chosen clusters)  >=  F% x pool_size

so the bottom-right of the grid -- lots of data, few motion types -- is empty by
construction. This script computes exactly which cells survive, for both ways of
choosing the K clusters.

Pool: the 5 SSL sources (realdata, ocdetect, eatingspeed, motionx, sonic), i.e.
the same set behind `vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic`.
Note this pool excludes capture24, which is what made 11j's `redundant` arm
accidentally accelerometer-only -- so the grid is free of that confound.

Reuses the cached 11-source PCA-64 by slicing it to the 5 sources' index ranges
(pool_manifest.json cumulative_sizes) and re-clustering. No re-embedding.

Writes analysis/diversity_saturation/out/grid_feasibility.json
"""
from __future__ import annotations

import json
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")

FIVE_SOURCES = ["realdata", "ocdetect", "eatingspeed", "motionx", "sonic"]
K_TYPES = [1, 5, 10, 25, 50, 100, 1000]
FRACTIONS = [0.0001, 0.001, 0.01, 0.10, 1.00]
K_CLUSTERS = 1000
SEED = 42


def source_ranges(manifest):
    names, cum = manifest["sources"], manifest["cumulative_sizes"]
    lo = 0
    out = {}
    for name, hi in zip(names, cum):
        out[name] = (lo, hi)
        lo = hi
    return out


def main():
    from sklearn.cluster import MiniBatchKMeans

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = json.load(open(os.path.join(WORK_DIR, "pool_manifest.json")))
    ranges = source_ranges(manifest)

    idx = np.concatenate([np.arange(*ranges[s]) for s in FIVE_SOURCES])
    idx.sort()
    n5 = len(idx)
    print(f"[grid] 5-source pool = {n5:,} windows "
          f"({n5 / manifest['cumulative_sizes'][-1]:.1%} of the 11-source pool)")
    for s in FIVE_SOURCES:
        lo, hi = ranges[s]
        print(f"         {s:12s} {hi - lo:>9,}")

    # labels are expensive (~10 min) and reused by the grid builder, so cache them
    lab_path = os.path.join(WORK_DIR, "labels5src_k1000.npy")
    if os.path.exists(lab_path) and len(np.load(lab_path, mmap_mode="r")) == n5:
        print("[grid] reusing cached 5-source labels", flush=True)
        labels = np.load(lab_path)
    else:
        z_all = np.load(os.path.join(WORK_DIR, "pca64_f32.npy"), mmap_mode="r")
        z = np.asarray(z_all[idx])
        print(f"[grid] clustering {z.shape} ...", flush=True)
        km = MiniBatchKMeans(n_clusters=K_CLUSTERS, batch_size=10_000, n_init=3,
                             random_state=SEED, max_no_improvement=50)
        km.fit(z)
        labels = km.predict(z)
        np.save(lab_path, labels.astype(np.int16))
        np.save(os.path.join(WORK_DIR, "idx5src.npy"), idx.astype(np.int64))
        print(f"[grid] cached labels -> {lab_path}", flush=True)
    sizes = np.bincount(labels, minlength=K_CLUSTERS)
    nonempty = int((sizes > 0).sum())
    print(f"[grid] {nonempty} non-empty clusters  "
          f"min={sizes[sizes > 0].min()} median={int(np.median(sizes[sizes > 0]))} "
          f"max={sizes.max()}")

    rng = np.random.default_rng(SEED)
    order_large = np.argsort(sizes)[::-1]
    order_rand = rng.permutation(np.where(sizes > 0)[0])

    report = {"pool": {"n_windows": int(n5), "sources": FIVE_SOURCES},
              "k_clusters": K_CLUSTERS, "n_nonempty": nonempty,
              "cluster_sizes": {"min": int(sizes[sizes > 0].min()),
                                "median": int(np.median(sizes[sizes > 0])),
                                "max": int(sizes.max())},
              "grid": {}}

    for pick, order in (("largest_first", order_large), ("random", order_rand)):
        rows = {}
        print(f"\n[grid] cluster choice = {pick}")
        header = "  K types |" + "".join(f"{f:>10.2%}" for f in FRACTIONS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for K in K_TYPES:
            avail = int(sizes[order[:K]].sum())
            cells = {}
            line = f"  {K:>7d} |"
            for f in FRACTIONS:
                need = int(round(n5 * f))
                ok = need <= avail
                cells[f"{f}"] = {"need": need, "available": avail, "feasible": bool(ok)}
                line += f"{('OK' if ok else '--'):>10s}"
            rows[str(K)] = {"available_windows": avail,
                            "available_fraction": avail / n5, "cells": cells}
            print(line + f"   (max {avail:>9,} = {avail / n5:6.2%} of pool)")
        report["grid"][pick] = rows

    for pick in list(report["grid"]):
        n_ok = sum(c["feasible"] for r in report["grid"][pick].values()
                   for c in r["cells"].values())
        print(f"\n[grid] {pick}: {n_ok} of {len(K_TYPES) * len(FRACTIONS)} cells realisable")
        report["n_feasible_" + pick] = n_ok

    out = os.path.join(OUT_DIR, "grid_feasibility.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
