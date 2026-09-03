#!/usr/bin/env python3
"""Derive the two OpenWatch contamination maps hardcoded in
`imu_lm/data/openwatch_loader.py`:

1. EXACT-DUPLICATE FILES in the "real" release: the 50 "real participants"
   contain byte-identical recordings under different subject ids (id_41 and
   id_42 are complete copies of id_39; id_43 of id_40; id_48 of id_46; id_49
   is half-duplicated; assorted 1-6 file overlaps entangle id_39/46/47/48/49).
   Any subject-level split that separates two ids sharing identical files
   leaks train data into val/test — which every split used before 2026-08-15
   (both `_split_subjects` and `_stratified_split_subjects_by_position`) did.
   Output: duplicate file groups -> the loader drops every copy except the
   lowest-(subject, position, name) instance when `dedup_exact` is on.

2. AUG id -> SOURCE SUBJECT map: `open_watch_augmentations` ids 51-85 are
   "motion-augmented participants" (per the release README) derived from real
   subjects via wrist-mirroring (+ locomotion overlay in walking_* dirs).
   Mirroring preserves the accelerometer MAGNITUDE exactly, so a mirrored
   copy is identified by max-lag-corr(|acc|) > 0.999 against its source
   recording in a non-walking position dir. Any split treating ids 51-85 as
   independent subjects (as every pre-2026-08-15 run on the aug release did,
   incl. EXPERIMENTS.md §5a and the first NormWear-aug eval, job 9382934)
   can put a mirrored copy of a held-out subject into train.
   Output: {aug_id: source_id}; ambiguous ids (mirror matches >1 real
   identity even after grouping duplicate sources) are reported separately
   and must be EXCLUDED from training rather than guessed.

Usage:
    python analysis/openwatch_forensics/derive_contamination_maps.py
Writes openwatch_contamination_maps.json next to this script and prints a
paste-ready constants block.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

REAL = Path("/projects/b53017/vbc1591/openwatch_raw/open_watch_without_augmentations")
AUG = Path("/projects/b53017/vbc1591/openwatch_raw/open_watch_augmentations")
OUT = Path(__file__).parent / "openwatch_contamination_maps.json"

# Gestures used for source matching: the 5 probe targets + 3 common basics.
MATCH_GESTURES = [
    "double_clench.csv", "double_pinch.csv", "pinch_down.csv", "pinch_up.csv",
    "slide.csv", "clench.csv", "extend.csv", "spread.csv",
]
# walking_* dirs carry the locomotion overlay, which breaks magnitude
# equality; only pure-mirror positions identify the source exactly.
NON_WALKING = {"sitting", "standing", "standing_arm_down"}
MATCH_THRESHOLD = 0.999


def read_acc(p: Path) -> np.ndarray:
    with open(p) as fh:
        r = csv.DictReader(fh)
        return np.array(
            [[float(row[c]) for c in ["AccX", "AccY", "AccZ"]] for row in r],
            dtype=np.float64,
        )


def mag(x: np.ndarray) -> np.ndarray:
    m = np.linalg.norm(x, axis=1)
    m = m - m.mean()
    s = m.std()
    return m / s if s > 1e-9 else m


def maxlag_corr(a: np.ndarray, b: np.ndarray, maxlag: int = 60) -> float:
    best = -2.0
    for lag in range(-maxlag, maxlag + 1):
        aa = a[lag:] if lag >= 0 else a[:lag]
        bb = b[: len(aa)] if lag >= 0 else b[-lag:]
        n = min(len(aa), len(bb))
        if n < 100:
            continue
        best = max(best, float(np.dot(aa[:n], bb[:n]) / n))
    return best


def subject_of(p: Path) -> int | None:
    m = re.match(r"id_(\d+)$", p.parts[-3])
    return int(m.group(1)) if m else None


def find_duplicate_groups() -> tuple[list[list[int]], dict[str, list[str]]]:
    by_hash: dict[str, list[Path]] = collections.defaultdict(list)
    for p in sorted(REAL.glob("id_*/*/*.csv")):
        if subject_of(p) is None:
            continue
        by_hash[hashlib.md5(p.read_bytes()).hexdigest()].append(p)

    dup_file_groups = {
        h: [str(p.relative_to(REAL)) for p in ps]
        for h, ps in by_hash.items()
        if len(ps) > 1
    }

    # connected components over subjects sharing >=1 identical file
    adj: dict[int, set[int]] = collections.defaultdict(set)
    for ps in by_hash.values():
        sids = sorted({subject_of(p) for p in ps})
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                adj[sids[i]].add(sids[j])
                adj[sids[j]].add(sids[i])
    seen, groups = set(), []
    for s in sorted(adj):
        if s in seen:
            continue
        comp, stack = [], [s]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack.extend(adj[u] - seen)
        groups.append(sorted(comp))
    return groups, dup_file_groups


def derive_aug_source_map(dup_groups: list[list[int]]):
    canon = {}
    for g in dup_groups:
        for s in g:
            canon[s] = min(g)

    real_sigs = {g: [] for g in MATCH_GESTURES}
    for g in MATCH_GESTURES:
        for p in sorted(REAL.glob(f"id_*/*/{g}")):
            sid = subject_of(p)
            if sid is not None:
                real_sigs[g].append((sid, mag(read_acc(p))))

    mapping, ambiguous, report = {}, [], {}
    aug_ids = sorted(
        int(m.group(1))
        for d in AUG.glob("id_*")
        if (m := re.match(r"id_(\d+)$", d.name)) and int(m.group(1)) > 50
    )
    for aug_id in aug_ids:
        votes = collections.Counter()
        n_files = 0
        for g in MATCH_GESTURES:
            for ap in sorted(AUG.glob(f"id_{aug_id}/*/{g}")):
                if ap.parts[-2] not in NON_WALKING:
                    continue
                a = mag(read_acc(ap))
                hits = {
                    canon.get(sid, sid)
                    for sid, m_ in real_sigs[g]
                    if maxlag_corr(a, m_) > MATCH_THRESHOLD
                }
                n_files += 1
                for h in hits:
                    votes[h] += 1
        report[aug_id] = {"votes": dict(votes), "files_checked": n_files}
        ranked = votes.most_common()
        if not ranked:
            ambiguous.append(aug_id)
        elif len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            ambiguous.append(aug_id)
        else:
            mapping[aug_id] = ranked[0][0]
        print(f"aug id_{aug_id}: votes={dict(votes)} files={n_files}"
              + ("  <-- AMBIGUOUS" if aug_id in ambiguous else ""))
    return mapping, ambiguous, report


def main():
    dup_groups, dup_files = find_duplicate_groups()
    print("duplicate subject groups (>=1 shared identical file):", dup_groups)
    mapping, ambiguous, report = derive_aug_source_map(dup_groups)

    out = {
        "duplicate_subject_groups": dup_groups,
        "duplicate_file_groups": dup_files,
        "aug_source_map": mapping,
        "aug_ambiguous_ids": ambiguous,
        "aug_vote_report": report,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT}")

    print("\n# ---- paste-ready constants for imu_lm/data/openwatch_loader.py ----")
    print("OPENWATCH_DUPLICATE_SUBJECT_GROUPS =", dup_groups)
    print("OPENWATCH_AUG_SOURCE = {")
    for k in sorted(mapping):
        print(f"    {k}: {mapping[k]},")
    print("}")
    print("OPENWATCH_AUG_AMBIGUOUS = ", sorted(ambiguous))


if __name__ == "__main__":
    main()
