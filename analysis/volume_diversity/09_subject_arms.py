#!/usr/bin/env python
"""Stage 9 (phase 3): the SUBJECT-diversity ladder -- the last confound the
scaling curve mixed in.

Fixed N (50,000 windows), fixed source (ocdetect), and the motion-type
distribution held as close to the pool's as the selection allows; only the
NUMBER OF WEARERS changes: 1, 3, 10, all. Without the type-matching step this
would just be another diversity contrast, because fewer subjects means fewer
motion types too -- so within each arm the windows are drawn by equal-per-type
water-fill over the chosen subjects' union, and the realised effective type
count is reported for every arm so the reader can verify the type axis really
did stay flat while the subject axis moved.

Subjects are chosen greedily to maximise type coverage (so the small-k arms are
the *best* few subjects, not an unlucky draw -- a conservative choice: it works
against the hypothesis that more subjects help).

Also augments out/arms.json with the subject/session diversity of every cell
built in 04, so subject diversity is measured there too rather than assumed away.

Requires 08_provenance.py to have succeeded for ocdetect.

Writes $WORK/idx/vd_sub{1,3,10,all}_n50k.npy, updates out/arms.json,
       out/subject_arms.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diversity_stats as ds
from arms import IDX_DIR, K_TYPES, OUT_DIR, POOL_N, SEED, WORK_DIR, source_ranges

N_TARGET = 50_000
LEVELS = [1, 3, 10, None]          # None = every subject with data


def main():
    rng = np.random.default_rng(SEED)
    subj = np.load(os.path.join(WORK_DIR, "prov_subject.npy"))
    labels_phys = np.load(os.path.join(WORK_DIR, "labels_phys_k1000.npy"))
    labels_enc = np.load(os.path.join(WORK_DIR, "labels_enc_k1000.npy"))
    z_enc = np.load(os.path.join(WORK_DIR, "enc_pca64_f32.npy"), mmap_mode="r")
    pool_counts_phys = np.bincount(labels_phys, minlength=K_TYPES)
    pool_counts_enc = np.bincount(labels_enc, minlength=K_TYPES)
    ranges = source_ranges()
    lo, hi = ranges["ocdetect"]

    have = subj[lo:hi] >= 0
    assert have.any(), "no ocdetect subject provenance -- run 08_provenance.py first"
    gidx = lo + np.where(have)[0]
    s_of = subj[gidx]
    subjects = np.unique(s_of)

    # greedy max-coverage ordering over subjects, by T-phys types added
    covered, order = set(), []
    pool_types = {int(s): set(np.unique(labels_phys[gidx[s_of == s]])) for s in subjects}
    remaining = set(int(s) for s in subjects)
    while remaining:
        best = max(remaining, key=lambda s: (len(pool_types[s] - covered),
                                             int((s_of == s).sum())))
        order.append(best)
        covered |= pool_types[best]
        remaining.discard(best)

    query_idx = np.sort(rng.choice(POOL_N, min(20_000, POOL_N), replace=False))
    out, arms_out = {}, {}
    for k in LEVELS:
        chosen = order if k is None else order[:k]
        tag = "all" if k is None else str(k)
        sel = gidx[np.isin(s_of, chosen)]
        if len(sel) < N_TARGET:
            out[f"vd_sub{tag}_n50k"] = {"status": f"only {len(sel)} windows from {len(chosen)} "
                                                  f"subject(s), need {N_TARGET}"}
            print(f"[sub{tag}] infeasible: {len(sel):,} windows < {N_TARGET:,}", flush=True)
            continue
        lab = labels_phys[sel]
        by_type = {}
        o = np.argsort(lab, kind="stable")
        counts = np.bincount(lab, minlength=K_TYPES)
        b = np.concatenate([[0], np.cumsum(counts)])
        for c in range(K_TYPES):
            m = o[b[c]:b[c + 1]]
            if len(m):
                by_type[c] = sel[m]
        # equal-per-type water-fill, smallest first: holds the type axis as flat
        # as the subject restriction allows
        idx, remaining_n = [], N_TARGET
        for i, c in enumerate(sorted(by_type, key=lambda c: len(by_type[c]))):
            left = len(by_type) - i
            take = min(len(by_type[c]), int(np.ceil(remaining_n / left)))
            m = by_type[c]
            idx.append(m if take == len(m) else rng.choice(m, take, replace=False))
            remaining_n -= take
            if remaining_n <= 0:
                break
        arm_idx = np.sort(np.concatenate(idx))[:N_TARGET]
        arm = f"vd_sub{tag}_n50k"
        np.save(os.path.join(IDX_DIR, f"{arm}.npy"), arm_idx.astype(np.int64))
        st = ds.summarise(arm_idx, labels_phys, labels_enc, pool_counts_phys,
                          pool_counts_enc, ranges, K_TYPES, z_enc=z_enc,
                          z_query_idx=query_idx, seed=SEED)
        st.update({"diversity": f"sub{tag}", "volume_tag": "n50k", "n_subjects": len(chosen),
                   "index_file": os.path.join(IDX_DIR, f"{arm}.npy"), "sampler": "natural"})
        st.update(subject_stats(arm_idx, subj, np.load(os.path.join(WORK_DIR, "prov_session.npy"))))
        out[arm] = st
        arms_out[arm] = st
        print(f"[{arm:16s}] subjects={len(chosen):>3d}  phys types={st['phys']['types_present']:>4d} "
              f"eff={st['phys']['effective_types_shannon']:>7.1f}  "
              f"eff_subjects={st['effective_subjects']:.2f}", flush=True)

    # measure subject/session diversity of the 04 cells too
    apath = os.path.join(OUT_DIR, "arms.json")
    with open(apath) as f:
        manifest = json.load(f)
    sessions = np.load(os.path.join(WORK_DIR, "prov_session.npy"))
    for arm, info in manifest["arms"].items():
        p = info.get("index_file")
        idx = np.load(p) if p else np.arange(POOL_N)
        info.update(subject_stats(idx, subj, sessions))
    manifest["arms"].update(arms_out)
    with open(apath, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(OUT_DIR, "subject_arms.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[sub] updated {apath}")


def subject_stats(idx, subj, sessions):
    s = subj[idx]
    g = sessions[idx]
    s, g = s[s >= 0], g[g >= 0]
    return {
        "n_subjects_known": int(len(np.unique(s))),
        "n_sessions_known": int(len(np.unique(g))),
        "effective_subjects": ds.effective_count_shannon(np.bincount(s)) if len(s) else 0.0,
        "effective_sessions": ds.effective_count_shannon(np.bincount(g)) if len(g) else 0.0,
        "provenance_coverage": float(len(s) / max(1, len(idx))),
    }


if __name__ == "__main__":
    main()
