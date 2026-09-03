#!/usr/bin/env python
"""04_source_weighting_audit.py -- TODO §4 (audit source weighting).

Reports each source's exact contribution to the combined train set at every
scaling fraction, after per-source `sample_fraction` (fixed, baked into every
data{X}pct config identically) and `overall_fraction` (pooled mode, the mode
every non-`_stratified` config actually uses). Confirms/quantifies whether the
5%->100% growth is dominated by large sources (ocdetect/capture24) and
whether sampling_mode='natural' (fixed across the whole sweep, confirmed
during research) ever oversamples small sources.

Cheap: reuses 01_composition_audit.py's window_counts.json if present (same
one-time raw build), otherwise does its own lightweight raw build.

Usage:
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/04_source_weighting_audit.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import (  # noqa: E402
    FRACTION_CONFIGS, OUT_DIR, build_all_sources_once, concat_offsets,
    fraction_indices_for, load_merged_cfg,
)


def main():
    comp_path = os.path.join(OUT_DIR, "01_composition", "window_counts.json")
    if os.path.exists(comp_path):
        print(f"[04] reusing full-resolution build info from {comp_path} (no rebuild)")
        with open(comp_path) as f:
            data = json.load(f)
        names = list(data["full_source_sizes"].keys())
        sizes = list(data["full_source_sizes"].values())
        cfg = load_merged_cfg(FRACTION_CONFIGS["100pct"][1])
        seed = int(cfg["trainer"]["seed"])
    else:
        print("[04] no cached build from 01_composition_audit.py -- building sources once now "
              "(expensive; consider running 01 first so 04 can reuse it)")
        cfg = load_merged_cfg(FRACTION_CONFIGS["100pct"][1])
        seed = int(cfg["trainer"]["seed"])
        train_parts, _val_parts, names = build_all_sources_once(cfg)
        sizes = [len(p) for p in train_parts]

    total_n = sum(sizes)
    offsets = concat_offsets_from_sizes(sizes)

    rows = []
    print(f"\n{'fraction':8s} " + " ".join(f"{n:>14s}" for n in names) + f" {'total':>10s}")
    for frac_name, (frac, _cfg_path) in FRACTION_CONFIGS.items():
        global_idx = fraction_indices_for(total_n, frac, seed=seed)
        per_source = {}
        for i, name in enumerate(names):
            lo, hi = offsets[i], offsets[i] + sizes[i]
            n_kept = int(((global_idx >= lo) & (global_idx < hi)).sum())
            per_source[name] = n_kept
        total_kept = sum(per_source.values())
        row = {"fraction": frac_name, "overall_fraction": frac, "total": total_kept, **per_source}
        rows.append(row)
        print(f"{frac_name:8s} " + " ".join(f"{per_source[n]:14d}" for n in names) + f" {total_kept:10d}")

    print("\n[04] per-source % of the fraction's TOTAL train windows (which sources dominate at each fraction):")
    for row in rows:
        total = row["total"]
        pct_str = "  ".join(f"{n}={100.0*row[n]/max(1,total):5.1f}%" for n in names)
        print(f"  {row['fraction']:8s}: {pct_str}")

    print("\n[04] per-source share of the 100%->fraction DELTA "
          "(what's actually added going from a smaller to a larger fraction):")
    deltas = []
    frac_names = list(FRACTION_CONFIGS.keys())
    for a, b in zip(frac_names, frac_names[1:]):
        ra = next(r for r in rows if r["fraction"] == a)
        rb = next(r for r in rows if r["fraction"] == b)
        delta_total = rb["total"] - ra["total"]
        delta_per_source = {n: rb[n] - ra[n] for n in names}
        deltas.append({"from": a, "to": b, "delta_total": delta_total, "delta_per_source": delta_per_source})
        pct_str = "  ".join(f"{n}={100.0*delta_per_source[n]/max(1,delta_total):5.1f}%" for n in names)
        print(f"  {a}->{b} (+{delta_total} windows): {pct_str}")

    out_dir = os.path.join(OUT_DIR, "04_source_weighting")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "source_weighting.json"), "w") as f:
        json.dump({"per_fraction": rows, "deltas": deltas}, f, indent=2)
    print(f"\n[04] wrote {out_dir}/source_weighting.json")

    # Headline: which single source explains the largest share of every delta?
    print("\n[04] headline: dominant source of each fraction-to-fraction delta")
    for d in deltas:
        dom = max(d["delta_per_source"].items(), key=lambda kv: kv[1])
        print(f"  {d['from']}->{d['to']}: {dom[0]} contributes {dom[1]}/{d['delta_total']} "
              f"({100.0*dom[1]/max(1,d['delta_total']):.1f}%) of the added windows")


def concat_offsets_from_sizes(sizes):
    return np.concatenate([[0], np.cumsum(sizes)])[:-1]


if __name__ == "__main__":
    main()
