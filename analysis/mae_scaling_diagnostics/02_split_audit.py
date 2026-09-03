#!/usr/bin/env python
"""02_split_audit.py -- TODO §2 (verify MAE train/val split independence).

For every source, reports the split unit actually used (window / file /
session / sequence / subject) and flags any source where val is only
window-disjoint rather than session-/subject-disjoint.

Two sources get focused, evidence-based checks here (the rest were already
confirmed correct-by-construction during the initial research pass and are
just summarized from the loader code, since re-deriving them numerically adds
no information):

  * OCDetect: `_build_ocdetect` in combined_mae_multidata_loader.py does a
    positional slice through file-concatenated windows (not shuffled, not
    session-disjoint) -- this script loads the real 100%-baseline OCDetect
    source (its `sample_fraction: 0.01` is fixed across every data{X}pct
    config, so this is a one-time, not per-fraction, check) and quantifies
    exactly how many windows straddle the boundary and from which file.

  * HumanML3D: train/val is the official HumanML3D `train.txt`/`val.txt` clip
    split, not controlled by this repo. Checks whether any clip's parent AMASS
    sequence (via `_get_amass_index`) appears on both sides.

Usage:
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/02_split_audit.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import FRACTION_CONFIGS, OUT_DIR, load_merged_cfg  # noqa: E402

SPLIT_UNIT_SUMMARY = {
    "realdata":     ("subject",   True,  "splits.group_key=subject_id (imu_lm/data/splits.py::make_splits)"),
    "ocdetect":     ("window",    False, "positional slice through file-ordered windows -- SEE DETAILED CHECK BELOW"),
    "nymeria":      ("file/session", True, "train_root vs. val_root are physically separate directories"),
    "totalcapture": ("subject",   True,  "train_subjects=[s1..s4] vs val_subjects=[s5], disjoint by construction"),
    "imuposer":     ("subject",   True,  "val_participants={P9,P10} held out entirely, rest is train"),
    "eatingspeed":  ("participant", True, "make_eatingspeed_loaders_random splits at participant level (rng.permutation)"),
    "motionx":      ("clip",      True,  "val_frac holdout of clips, no session-sliding-window reuse (synthetic, 1 window/clip-ish)"),
    "humanml3d":    ("clip",      "unverified", "official HumanML3D train.txt/val.txt -- SEE DETAILED CHECK BELOW"),
    "sonic":        ("session",   True,  "max_train_sessions/max_val_sessions from disjoint file lists"),
    "capture24":    ("participant", True, "make_capture24_loaders_random splits at participant level explicitly (docstring-verified)"),
    "openwatch":    ("subject",   True,  "train_subjects/val_subjects, subject-level split"),
}


def check_ocdetect():
    from imu_lm.data.combined_mae_multidata_loader import _build_ocdetect

    _, cfg_path = FRACTION_CONFIGS["100pct"]
    cfg = load_merged_cfg(cfg_path)
    src_cfg = cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["ocdetect"]
    vl_max = int(cfg["data"]["variable_length"]["max_samples"])

    print("[02] building OCDetect (fixed sample_fraction=0.01 baseline) with provenance...")
    train_ds, val_ds = _build_ocdetect(cfg, src_cfg, vl_max, collect_provenance=True)
    train_raw = train_ds._ds  # OCDetectDataset with .provenance attached
    val_raw = val_ds._ds

    train_files = train_raw.provenance["files"]
    train_file_ids = train_raw.provenance["file_ids"]
    val_file_ids = val_raw.provenance["file_ids"]
    n_train, n_val = len(train_file_ids), len(val_file_ids)

    boundary_file_id = int(train_file_ids[-1])
    # windows in the train tail that belong to the same file as the last train window
    tail = 0
    for fid in train_file_ids[::-1]:
        if int(fid) != boundary_file_id:
            break
        tail += 1
    # windows in the val head that belong to the same file as the first val window
    # (only relevant if it's the SAME file as the boundary -- i.e. the file straddles the cut)
    straddles = int(val_file_ids[0]) == boundary_file_id if n_val else False
    head = 0
    if straddles:
        for fid in val_file_ids:
            if int(fid) != boundary_file_id:
                break
            head += 1

    result = {
        "n_train": n_train, "n_val": n_val,
        "boundary_file": train_files[boundary_file_id] if straddles or tail else None,
        "straddles_boundary": straddles,
        "train_windows_from_boundary_file": tail,
        "val_windows_from_boundary_file": head if straddles else 0,
        "note": (
            "The train/val cut is a plain positional slice through file-concatenated, "
            "NOT shuffled windows (combined_mae_multidata_loader.py _build_ocdetect). "
            f"At this build, {'the boundary file DOES straddle train/val' if straddles else 'the boundary file happens to end exactly at a file boundary this time (no straddle)'}"
            + (f": {tail} trailing train windows and {head} leading val windows both come from "
               f"{train_files[boundary_file_id]!r}. Since windows use a fixed 50% hop, each pair of "
               "adjacent same-file windows overlaps by ~50% of their samples by construction -- so the "
               f"{tail} tail windows and {head} head windows are NOT independent of each other, even "
               "though nominally in different splits." if straddles else
               ". This is incidental to the current file selection/ordering, NOT a property of the "
               "split logic itself -- a different `sample_fraction`/`max_files`/file count would very "
               "likely reintroduce the straddle. The split logic is unsafe in general even when this "
               "particular build got lucky."),
        ),
    }
    print(f"[02] OCDetect: n_train={n_train} n_val={n_val} straddles_boundary={straddles} "
          f"train_tail={tail} val_head={result['val_windows_from_boundary_file']}")
    return result


def check_humanml3d():
    from imu_lm.data.humanml3d_dataset_FLU import _get_amass_index

    _, cfg_path = FRACTION_CONFIGS["100pct"]
    cfg = load_merged_cfg(cfg_path)
    root = cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["humanml3d"]["root"]

    def _read_split(name):
        path = os.path.join(root, "HumanML3D", f"{name}.txt")
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]

    train_ids = _read_split("train")
    val_ids = _read_split("val")
    print(f"[02] HumanML3D: official split train={len(train_ids)} val={len(val_ids)} clips")

    amass_index = _get_amass_index(root)

    def _parents(ids):
        out = {}
        for cid in ids:
            entry = amass_index.get(f"{cid}.npy")
            if entry is not None:
                out[cid] = entry[0]  # amass_path_clip
        return out

    train_parents = _parents(train_ids)
    val_parents = _parents(val_ids)
    train_seqs = set(train_parents.values())
    val_seqs = set(val_parents.values())
    shared_seqs = train_seqs & val_seqs
    affected_train_clips = [c for c, p in train_parents.items() if p in shared_seqs]
    affected_val_clips = [c for c, p in val_parents.items() if p in shared_seqs]

    result = {
        "train_clips": len(train_ids), "val_clips": len(val_ids),
        "train_clips_with_amass_parent": len(train_parents), "val_clips_with_amass_parent": len(val_parents),
        "shared_parent_amass_sequences": len(shared_seqs),
        "train_clips_from_shared_sequences": len(affected_train_clips),
        "val_clips_from_shared_sequences": len(affected_val_clips),
        "example_shared_sequences": list(shared_seqs)[:5],
    }
    print(f"[02] HumanML3D: {len(shared_seqs)} AMASS parent sequences appear on BOTH sides "
          f"({len(affected_train_clips)} train clips / {len(affected_val_clips)} val clips affected)")
    return result


def main():
    out_dir = os.path.join(OUT_DIR, "02_split_audit")
    os.makedirs(out_dir, exist_ok=True)

    print("[02] split unit summary (from loader code, not re-derived numerically):")
    for name, (unit, disjoint, note) in SPLIT_UNIT_SUMMARY.items():
        flag = "OK" if disjoint is True else ("UNVERIFIED" if disjoint == "unverified" else "LEAK-FLAGGED")
        print(f"  {name:16s} split_unit={unit:12s} [{flag:12s}]  {note}")

    ocdetect_result = check_ocdetect()
    humanml3d_result = check_humanml3d()

    report = {
        "split_unit_summary": {
            k: {"unit": v[0], "disjoint": v[1], "note": v[2]} for k, v in SPLIT_UNIT_SUMMARY.items()
        },
        "ocdetect_boundary_check": ocdetect_result,
        "humanml3d_amass_parent_check": humanml3d_result,
    }
    out_path = os.path.join(out_dir, "split_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[02] wrote {out_path}")


if __name__ == "__main__":
    main()
