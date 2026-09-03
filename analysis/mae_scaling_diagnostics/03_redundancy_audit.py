#!/usr/bin/env python
"""03_redundancy_audit.py -- TODO §3 (measure sample redundancy).

Quantifies redundancy in each source at every fraction:
  * analytical window-overlap % (exact, from windowing.compute_T_and_hop --
    every sliding-window source in this codebase uses a fixed 50% hop, so
    this is a closed-form number, not something to estimate)
  * empirical near-duplicate rate: cosine similarity between temporally-
    adjacent KEPT windows within a session, sampled from the on-disk
    cache (fast -- no raw parsing) so this script is cheap to run repeatedly
  * static vs. dynamic window proportion, via a fixed accel-variance
    threshold on the (cached, preprocessed) window
  * effective independent segment count per session = ceil(session_length / T)
    i.e. the number of NON-overlapping windows that could be cut from that
    session, vs. the actual (overlapping) window count on disk

Requires the on-disk cache (`cache_dir` in the 100% config, i.e. the same
cache scripts/build_multidata_cache.py built and every real training run
already reads from) -- reads directly from the cached .npy arrays, so this
script does NOT re-parse any raw source and is safe to run standalone without
first running 01_composition_audit.py's expensive raw rebuild. It does NOT
have per-window session provenance (the cache is provenance-free by
construction, see combined_mae_multidata_loader.py docstring) so the
near-duplicate check here samples temporally-adjacent ROWS in the cache array
as a proxy for "consecutive windows" -- accurate for sources where the cache
was built in session-contiguous order (every source here, since
build_multidata_cache.py iterates the same per-source Dataset in its natural
__getitem__ order, which is session-major for every sliding-window source).

Usage:
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/03_redundancy_audit.py [--n-sample 2000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import FRACTION_CONFIGS, OUT_DIR, load_merged_cfg  # noqa: E402


def analytical_overlap():
    """Every sliding-window source shares windowing.compute_T_and_hop -- fixed
    50% hop (window_hop_ratio: 0.5 in every scaling config) except OCDetect,
    which hard-codes hop_samples=vl_max//2 in combined_mae_multidata_loader.py
    (also exactly 50%). Overlap fraction between two temporally-adjacent
    windows of length T with hop h is (T-h)/T."""
    cfg = load_merged_cfg(FRACTION_CONFIGS["100pct"][1])
    from imu_lm.data.windowing import compute_T_and_hop
    T, hop = compute_T_and_hop(cfg)
    overlap = (T - hop) / T
    print(f"[03] analytical: T={T} samples, hop={hop} samples -> "
          f"{100*overlap:.1f}% overlap between temporally-adjacent windows (applies to every "
          f"sliding-window source: realdata, ocdetect, nymeria, totalcapture, imuposer, motionx, sonic)")
    return {"T": T, "hop": hop, "overlap_fraction": overlap}


def cache_redundancy(n_sample: int):
    cfg = load_merged_cfg(FRACTION_CONFIGS["100pct"][1])
    cache_dir = cfg["objective"]["mae_multidata_varlen_altattn_flu"]["cache_dir"]
    if not os.path.isdir(cache_dir):
        print(f"[03] cache_dir not found ({cache_dir}) -- skipping empirical near-duplicate check")
        return {}

    results = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith("_train_x.npy"):
            continue
        name = fname[: -len("_train_x.npy")]
        x_path = os.path.join(cache_dir, fname)
        try:
            X = np.load(x_path, mmap_mode="r")  # [N, 6, T]
        except Exception as e:
            print(f"[03] {name}: failed to load cache ({e}), skipping")
            continue
        n = len(X)
        if n < 3:
            continue
        n_sample_eff = min(n_sample, n - 1)
        rng = np.random.default_rng(42)
        # sample random start points, compare window i to window i+1 (temporally
        # adjacent rows in cache-build order -- see module docstring)
        idxs = rng.choice(n - 1, size=n_sample_eff, replace=False)

        cos_sims = []
        motion_flags = []
        for i in idxs:
            a = np.asarray(X[i]).reshape(-1).astype(np.float32)
            b = np.asarray(X[i + 1]).reshape(-1).astype(np.float32)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 1e-8:
                cos_sims.append(float(np.dot(a, b) / denom))
            # accel channels are 0:3 (see AccOnlyPadTo6ChDataset / module docstring):
            # static window heuristic -- low variance in the accel channels
            acc = np.asarray(X[i][:3])
            motion_flags.append(float(np.var(acc)))

        cos_sims = np.array(cos_sims)
        motion_var = np.array(motion_flags)
        # static/dynamic split: below the 25th percentile of this source's own
        # accel-variance distribution counts as "static" -- a relative, not
        # absolute, threshold since raw units differ pre- vs post-normalization
        # per source.
        static_thresh = np.percentile(motion_var, 25)
        pct_static = float(100.0 * (motion_var <= static_thresh).mean())
        pct_near_dup = float(100.0 * (cos_sims > 0.98).mean()) if len(cos_sims) else None

        results[name] = {
            "n_cached_train_windows": int(n),
            "n_sampled_pairs": int(n_sample_eff),
            "median_adjacent_cosine_sim": round(float(np.median(cos_sims)), 4) if len(cos_sims) else None,
            "pct_near_duplicate_pairs (cos>0.98)": round(pct_near_dup, 2) if pct_near_dup is not None else None,
            "pct_low_motion_windows (below own 25th pct accel-var)": round(pct_static, 2),
        }
        print(f"[03] {name:16s} n={n:8d}  median_adjacent_cos_sim={results[name]['median_adjacent_cosine_sim']!s:>7s}  "
              f"near_dup%={results[name]['pct_near_duplicate_pairs (cos>0.98)']!s:>6s}  "
              f"low_motion%(relative)={pct_static:5.1f}")
    return results


def effective_segments(n_sample: int):
    """Effective independent (non-overlapping) segment count per session, from
    the composition audit's window_counts.json if available -- else skipped
    (needs session-length info that only 01_composition_audit.py's raw build
    computes)."""
    comp_path = os.path.join(OUT_DIR, "01_composition", "composition_by_fraction.json")
    if not os.path.exists(comp_path):
        print("[03] 01_composition_audit.py output not found -- run it first for effective-segment-count "
              "estimates (windows_per_session vs. ceil(session_len/T)); skipping that part here")
        return None
    with open(comp_path) as f:
        rows = json.load(f)
    # windows-per-session at 100% tells us how many OVERLAPPING windows a
    # session yields; the number of INDEPENDENT (non-overlapping) windows a
    # session of the same total span could yield is exactly half that (since
    # hop = T/2, consecutive windows are 50% overlapping -- every other window
    # is independent).
    out = {}
    for r in rows:
        if r["fraction"] != "100pct" or r.get("median_windows_per_session") is None:
            continue
        out[r["source"]] = {
            "median_windows_per_session_overlapping": r["median_windows_per_session"],
            "estimated_effective_independent_windows_per_session": round(r["median_windows_per_session"] / 2.0, 1),
        }
    print("\n[03] effective (non-overlapping) windows per session, at 100%, from 01's session data:")
    for name, v in out.items():
        print(f"  {name:16s} overlapping={v['median_windows_per_session_overlapping']:>6}  "
              f"effective_independent~={v['estimated_effective_independent_windows_per_session']:>6}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=2000)
    args = ap.parse_args()

    out_dir = os.path.join(OUT_DIR, "03_redundancy")
    os.makedirs(out_dir, exist_ok=True)

    analytical = analytical_overlap()
    cache_stats = cache_redundancy(args.n_sample)
    eff_segments = effective_segments(args.n_sample)

    report = {"analytical_overlap": analytical, "cache_redundancy": cache_stats,
              "effective_segments": eff_segments}
    out_path = os.path.join(out_dir, "redundancy_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[03] wrote {out_path}")


if __name__ == "__main__":
    main()
