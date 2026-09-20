#!/usr/bin/env python
"""Stage 10 (phase 3): the remaining one-at-a-time diversity axes.

Stage 04 varied volume and a bundled notion of "diversity"; stage 09 isolates
the WEARER axis. This stage isolates the other two, plus the control that
decides how to read them:

  src{1,2,3,4}_n50k   SOURCE axis. N=50,000, motion-type distribution held to
                      the pool's by equal-per-type water-fill inside whatever
                      sources the arm may use. Sources are added in ascending
                      size (motionx -> sonic -> eatingspeed -> ocdetect) so the
                      1-source arm is NOT simply "ocdetect again" -- src1 is
                      the smallest source, which is the conservative choice.

  typ{3,30,300,all}_n50k
                      MOTION-TYPE axis. N=50,000, single source (ocdetect),
                      only the number of distinct T-phys types changes. Types
                      are picked largest-first so every arm can reach N without
                      resampling, and windows inside a type are drawn UNIFORMLY
                      AT RANDOM -- this is the distinction the dup arm tests.

  dup3_n50k           NEAR-DUPLICATE control, and the reason this stage exists.
                      Stage 03 of the earlier diversity-saturation study built
                      a "redundant" arm by filling whole clusters largest-first
                      and it collapsed (WISDM 61.7 -> 35.3). The factorial's
                      narrowest arm (vd_nar_*, effective types 2.8) did NOT
                      collapse. Both are "low diversity", so one of them is
                      measuring something else. dup3 holds the type count equal
                      to typ3 but selects windows by nearest-neighbour packing
                      in encoder space instead of uniformly, i.e. maximal
                      near-duplication at matched type count. If dup3 collapses
                      and typ3 does not, what kills a corpus is REDUNDANT
                      WINDOWS, not FEW MOTION TYPES -- and the two prior results
                      stop contradicting each other.

Every arm here is N=50,000 at the same STEPS budget as the factorial, so all of
them are directly comparable to vd_one_n50k / vd_nat_n50k / vd_bal_n50k.

Writes $WORK/idx/vd_*.npy, updates out/arms.json, writes out/axis_arms.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diversity_stats as ds
from arms import (IDX_DIR, K_TYPES, OUT_DIR, POOL_N, SEED, SOURCES, WORK_DIR,
                  source_ranges)

N_TARGET = 50_000
# The type ladder runs at a smaller N than the source ladder: ocdetect's three
# largest T-phys types hold only 15,065 windows between them, so typ3 -- and
# with it the dup3 control that the whole stage exists for -- is infeasible at
# 50k. 12,500 is the factorial's own smallest volume level, so the type arms
# anchor against vd_nar_n12k / vd_one_n12k rather than against nothing.
N_TYPE = 12_500
# ascending size: the 1-source arm is the SMALLEST source, not the largest
SRC_ORDER = ["motionx", "sonic", "eatingspeed", "ocdetect"]
TYPE_LEVELS = [3, 30, 300, None]        # None = every type ocdetect has


def types_of(sel, labels):
    """{type -> member indices} for the given global indices."""
    lab = labels[sel]
    o = np.argsort(lab, kind="stable")
    counts = np.bincount(lab, minlength=K_TYPES)
    b = np.concatenate([[0], np.cumsum(counts)])
    return {c: sel[o[b[c]:b[c + 1]]] for c in range(K_TYPES) if counts[c]}


def waterfill(by_type, n_target, rng):
    """Equal-per-type draw, smallest type first so a small type is exhausted
    rather than starving a large one. Uniform inside each type."""
    idx, remaining = [], int(n_target)
    order = sorted(by_type, key=lambda c: len(by_type[c]))
    for i, c in enumerate(order):
        left = len(order) - i
        take = min(len(by_type[c]), int(np.ceil(remaining / left)))
        m = by_type[c]
        idx.append(m if take >= len(m) else rng.choice(m, take, replace=False))
        remaining -= take
        if remaining <= 0:
            break
    return np.sort(np.concatenate(idx))[:n_target]


def dup_ball(candidates, n_target, z, rng, chunk=200_000):
    """The maximally-redundant arm at this N: the n_target windows closest to a
    random seed window in encoder space, drawn from the WHOLE single-source pool
    rather than from a restricted type set.

    Selecting inside a 3-type set does not work -- those types hold only 15,065
    windows, so taking 12,500 of them takes 83 % of everything available and
    there is no room left to be selective (measured: nn_self_median 2.41 vs
    2.47 for the uniform arm, i.e. no redundancy created). Drawing from all of
    ocdetect puts the ratio at ~1 % and leaves room for a genuinely tight ball.
    The realised type count is measured and reported, not assumed."""
    seed = candidates[rng.integers(len(candidates))]
    zseed = np.asarray(z[seed], dtype=np.float32)
    d = np.empty(len(candidates), dtype=np.float32)
    for lo in range(0, len(candidates), chunk):
        blk = candidates[lo:lo + chunk]
        zc = np.asarray(z[blk], dtype=np.float32)
        d[lo:lo + len(blk)] = np.linalg.norm(zc - zseed[None, :], axis=1)
    return np.sort(candidates[np.argpartition(d, n_target)[:n_target]])


def self_nn_median(idx, z, rng, sample=4000):
    """Median distance from a window to its nearest OTHER window in the arm, in
    encoder space. Low = the arm is full of near-duplicates. Reported for every
    arm so the dup3/typ3 contrast is measured rather than asserted."""
    sub = np.sort(idx if len(idx) <= sample else rng.choice(idx, sample, replace=False))
    zs = np.asarray(z[sub], dtype=np.float32)
    # Gram trick: the broadcast form allocates n*n*dim floats (4 GB at n=4000,
    # dim=64) and is killed on a login node; this is n*n.
    sq = (zs * zs).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (zs @ zs.T)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(np.maximum(d2.min(axis=1), 0.0))))


def main():
    rng = np.random.default_rng(SEED)
    labels_phys = np.load(os.path.join(WORK_DIR, "labels_phys_k1000.npy"))
    labels_enc = np.load(os.path.join(WORK_DIR, "labels_enc_k1000.npy"))
    z_enc = np.load(os.path.join(WORK_DIR, "enc_pca64_f32.npy"), mmap_mode="r")
    pc_phys = np.bincount(labels_phys, minlength=K_TYPES)
    pc_enc = np.bincount(labels_enc, minlength=K_TYPES)
    ranges = source_ranges()
    query_idx = np.sort(rng.choice(POOL_N, min(20_000, POOL_N), replace=False))

    subj = sess = None
    for f, name in (("prov_subject.npy", "subj"), ("prov_session.npy", "sess")):
        p = os.path.join(WORK_DIR, f)
        if os.path.exists(p):
            v = np.load(p)
            if name == "subj":
                subj = v
            else:
                sess = v

    out, arms_out = {}, {}

    def emit(arm, arm_idx, diversity, extra, vol_tag="n50k"):
        np.save(os.path.join(IDX_DIR, f"{arm}.npy"), arm_idx.astype(np.int64))
        st = ds.summarise(arm_idx, labels_phys, labels_enc, pc_phys, pc_enc, ranges,
                          K_TYPES, z_enc=z_enc, z_query_idx=query_idx, seed=SEED)
        # within-arm nearest-neighbour distance: the statistic that separates
        # "few motion types" from "near-duplicate windows". summarise() does not
        # report it, and it is the whole point of the dup3 control.
        st["nn_self_median"] = self_nn_median(arm_idx, z_enc, rng)
        st.update({"diversity": diversity, "volume_tag": vol_tag,
                   "index_file": os.path.join(IDX_DIR, f"{arm}.npy"),
                   "sampler": "natural", **extra})
        if subj is not None and sess is not None:
            s, g = subj[arm_idx], sess[arm_idx]
            s, g = s[s >= 0], g[g >= 0]
            st.update({"n_subjects_known": int(len(np.unique(s))),
                       "n_sessions_known": int(len(np.unique(g)))})
        out[arm] = arms_out[arm] = st
        print(f"[{arm:16s}] n={len(arm_idx):>6,}  phys types={st['phys']['types_present']:>4d} "
              f"eff={st['phys']['effective_types_shannon']:>7.1f}  "
              f"vendi={st.get('vendi_enc', float('nan')):.2f}  "
              f"nn_self_median={st['nn_self_median']:.4f}", flush=True)

    # ---- SOURCE axis --------------------------------------------------------
    for k in (1, 2, 3, 4):
        use = SRC_ORDER[:k]
        sel = np.concatenate([np.arange(*ranges[s]) for s in use])
        if len(sel) < N_TARGET:
            print(f"[src{k}] infeasible: {len(sel):,} < {N_TARGET:,}"); continue
        emit(f"vd_src{k}_n50k", waterfill(types_of(sel, labels_phys), N_TARGET, rng),
             f"src{k}", {"n_sources": k, "sources_used": use})

    # ---- MOTION-TYPE axis (single source, uniform inside each type) ---------
    lo, hi = ranges["ocdetect"]
    oc = np.arange(lo, hi)
    oc_types = types_of(oc, labels_phys)
    by_size = sorted(oc_types, key=lambda c: -len(oc_types[c]))
    typ_sets = {}
    for k in TYPE_LEVELS:
        tag = "all" if k is None else str(k)
        keep = by_size if k is None else by_size[:k]
        sub = {c: oc_types[c] for c in keep}
        tot = sum(len(v) for v in sub.values())
        if tot < N_TYPE:
            print(f"[typ{tag}] infeasible: {tot:,} < {N_TYPE:,}"); continue
        typ_sets[tag] = sub
        emit(f"vd_typ{tag}_n12k", waterfill(sub, N_TYPE, rng),
             f"typ{tag}", {"n_types_allowed": len(keep)}, vol_tag="n12k")

    # ---- NEAR-DUPLICATE control at the typ3 type count ----------------------
    emit("vd_dup_n12k", dup_ball(oc, N_TYPE, z_enc, rng),
         "dup", {"note": "maximally redundant at N=12,500: the nearest 12,500 windows "
                         "to a random seed in encoder space, drawn from all of ocdetect. "
                         "Compare against vd_typ3_n12k (same order of type count, uniform "
                         "within type) to separate near-duplication from type scarcity."},
         vol_tag="n12k")

    apath = os.path.join(OUT_DIR, "arms.json")
    with open(apath) as f:
        manifest = json.load(f)
    manifest["arms"].update(arms_out)
    with open(apath, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(OUT_DIR, "axis_arms.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[axis] wrote {len(arms_out)} arms -> {apath}")


if __name__ == "__main__":
    main()
