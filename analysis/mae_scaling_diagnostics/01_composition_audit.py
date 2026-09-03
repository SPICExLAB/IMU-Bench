#!/usr/bin/env python
"""01_composition_audit.py -- TODO §1 (audit the current subsampling behavior).

For each fraction {0.1%, 1%, 5%, 25%, 50%, 100%}, reports per source:
  * windows sampled, unique subjects, unique sessions
  * mean/median sampled windows per session
  * % of all subjects/sessions (of the 100% pool) retained
  * median temporal distance between kept windows from the same session
  * median window-overlap fraction between temporally-adjacent kept windows

Expensive step (raw parsing / SMPL FK / windowing for every source) happens
ONCE at full resolution; all 6 fractions are then produced by replaying the
exact same seeded `_fraction_subset` draw combined_mae_multidata_loader.py
itself uses (pooled mode, seed=trainer.seed=42) against that single build, so
this does not redo I/O per fraction.

capture24/eatingspeed have no per-window provenance in this codebase (see
lib/common.py and combined_mae_multidata_loader._group_windows_by_session) --
reported as window-count-only, with a note. All other 9 sources get full
session/subject-level detail.

Usage:
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/01_composition_audit.py [--quick]

--quick caps every source to a small number of files/sessions for a fast
smoke test of the script itself (NOT a real audit -- use full run for the
actual diagnosis).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import (  # noqa: E402
    FRACTION_CONFIGS, OUT_DIR, build_all_sources_once, concat_offsets,
    fraction_indices_for, load_merged_cfg,
)

SAMPLE_RATE_HZ = 50.0  # matches windowing.CANONICAL_SAMPLE_RATE_HZ, used by every source here


def _quick_cap(cfg: dict) -> dict:
    """Shrink every source to a handful of files/clips for a fast smoke test."""
    mm = cfg["objective"]["mae_multidata_varlen_altattn_flu"]
    src = mm["sources"]
    src["ocdetect"]["max_files"] = 3
    src["ocdetect"]["sample_fraction"] = None
    src["nymeria"]["max_train_files"] = 3
    src["nymeria"]["max_val_files"] = 2
    src["imuposer"]["max_train_files"] = 3
    src["eatingspeed"]["max_participants_per_subset"] = 2
    src["motionx"]["max_clips_per_category"] = 3
    src["humanml3d"]["max_train_clips"] = 50
    src["humanml3d"]["max_val_clips"] = 20
    src["sonic"]["max_train_sessions"] = 3
    src["sonic"]["max_val_sessions"] = 2
    src["capture24"]["max_files"] = 3
    src["capture24"]["sample_fraction"] = None
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    from imu_lm.data import combined_mae_multidata_loader as cm

    _, cfg_path_100 = FRACTION_CONFIGS["100pct"]
    cfg = load_merged_cfg(cfg_path_100)
    if args.quick:
        cfg = _quick_cap(cfg)
    seed = int(cfg["trainer"]["seed"])

    print("[01] building every source once at full resolution (this is the expensive step)...")
    train_parts, _val_parts, names = build_all_sources_once(cfg)
    sizes = [len(p) for p in train_parts]
    total_n = sum(sizes)
    offsets = concat_offsets(train_parts)
    print(f"[01] built {len(names)} sources, total_train={total_n} windows: "
          + ", ".join(f"{n}={s}" for n, s in zip(names, sizes)))

    # Full-pool (100%) groupings, used as the denominator for "% retained".
    full_groups = {}
    full_subject_keys = {}
    for name, part in zip(names, train_parts):
        g = cm._group_windows_by_session(name, part)
        full_groups[name] = g
        if g is not None:
            full_subject_keys[name] = {k: cm._subject_key_from_session_key(name, k) for k in g}

    out_dir = os.path.join(OUT_DIR, "01_composition")
    os.makedirs(out_dir, exist_ok=True)

    per_fraction_rows = []
    total_train_by_fraction = {}

    for frac_name, (frac, _cfg_path) in FRACTION_CONFIGS.items():
        global_idx = fraction_indices_for(total_n, frac, seed=seed)
        total_train_by_fraction[frac_name] = int(len(global_idx))
        print(f"\n[01] === fraction={frac_name} (overall_fraction={frac}) -> {len(global_idx)} windows ===")

        # Split global indices back out per source (mirrors ConcatDataset's own bookkeeping).
        for si, name in enumerate(names):
            lo, hi = offsets[si], offsets[si] + sizes[si]
            in_range = global_idx[(global_idx >= lo) & (global_idx < hi)]
            local_idx = in_range - lo

            row = {
                "fraction": frac_name, "source": name,
                "windows": int(len(local_idx)),
                "full_windows": int(sizes[si]),
            }

            groups = full_groups[name]
            if groups is None:
                row.update({
                    "sessions": None, "subjects": None,
                    "pct_sessions_retained": None, "pct_subjects_retained": None,
                    "mean_windows_per_session": None, "median_windows_per_session": None,
                    "median_temporal_gap_s": None, "median_overlap_frac": None,
                    "note": "no per-window provenance for this source (see lib/common.py)",
                })
                per_fraction_rows.append(row)
                continue

            local_idx_set = set(local_idx.tolist())
            sess_counts = {}
            for sess_key, idxs in groups.items():
                kept = [i for i in idxs if i in local_idx_set]
                if kept:
                    sess_counts[sess_key] = kept
            n_sessions_kept = len(sess_counts)
            subj_keys_kept = {full_subject_keys[name][k] for k in sess_counts}
            n_subjects_kept = len(subj_keys_kept)
            n_sessions_full = len(groups)
            n_subjects_full = len(set(full_subject_keys[name].values()))
            win_per_sess = [len(v) for v in sess_counts.values()]

            raw = cm._unwrap_raw(train_parts[si])
            starts = getattr(raw, "_starts", None)
            # temporal gap / overlap: only meaningful for sliding-window sources
            # that expose per-window start offsets (_starts). OCDetect/humanml3d/
            # openwatch don't (no _starts attribute / not a sliding window at all).
            gaps_s, overlaps = [], []
            if starts is not None and win_per_sess:
                T = int(raw._T)
                for sess_key, idxs in sess_counts.items():
                    s = sorted(int(starts[i]) for i in idxs)
                    for a, b in zip(s, s[1:]):
                        d = b - a
                        gaps_s.append(d / SAMPLE_RATE_HZ)
                        overlaps.append(max(0.0, (T - d) / T))

            row.update({
                "sessions": n_sessions_kept, "subjects": n_subjects_kept,
                "pct_sessions_retained": round(100.0 * n_sessions_kept / max(1, n_sessions_full), 2),
                "pct_subjects_retained": round(100.0 * n_subjects_kept / max(1, n_subjects_full), 2),
                "mean_windows_per_session": round(float(np.mean(win_per_sess)), 2) if win_per_sess else None,
                "median_windows_per_session": float(np.median(win_per_sess)) if win_per_sess else None,
                "median_temporal_gap_s": round(float(np.median(gaps_s)), 3) if gaps_s else None,
                "median_overlap_frac": round(float(np.median(overlaps)), 3) if overlaps else None,
                "note": None,
            })
            per_fraction_rows.append(row)
            print(f"  {name:16s} windows={row['windows']:7d}  sessions={row['sessions']!s:>6s} "
                  f"({row['pct_sessions_retained']}% of full)  subjects={row['subjects']!s:>5s} "
                  f"({row['pct_subjects_retained']}% of full)  "
                  f"med_windows/session={row['median_windows_per_session']!s:>6s}  "
                  f"med_gap_s={row['median_temporal_gap_s']!s:>7s}  med_overlap={row['median_overlap_frac']!s:>6s}")

    with open(os.path.join(out_dir, "composition_by_fraction.json"), "w") as f:
        json.dump(per_fraction_rows, f, indent=2)
    with open(os.path.join(out_dir, "window_counts.json"), "w") as f:
        json.dump({"total_train_by_fraction": total_train_by_fraction,
                   "full_source_sizes": dict(zip(names, sizes))}, f, indent=2)

    # CSV for quick spreadsheet inspection
    import csv
    csv_path = os.path.join(out_dir, "composition_by_fraction.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_fraction_rows[0].keys()))
        w.writeheader()
        w.writerows(per_fraction_rows)

    print(f"\n[01] wrote {out_dir}/composition_by_fraction.{{json,csv}} and window_counts.json")

    # Headline check for the core hypothesis: does 1-5% already cover most
    # subjects/sessions, with 25-100% mostly adding redundant windows from the
    # same recordings?
    print("\n[01] headline: % of full (100%) subjects/sessions retained, session-tracked sources only")
    for frac_name in FRACTION_CONFIGS:
        rows = [r for r in per_fraction_rows if r["fraction"] == frac_name and r["sessions"] is not None]
        if not rows:
            continue
        avg_sess_pct = np.mean([r["pct_sessions_retained"] for r in rows])
        avg_subj_pct = np.mean([r["pct_subjects_retained"] for r in rows])
        print(f"  {frac_name:8s}: avg sessions retained={avg_sess_pct:6.2f}%  avg subjects retained={avg_subj_pct:6.2f}%")


if __name__ == "__main__":
    main()
