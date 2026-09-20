#!/usr/bin/env python
"""Stage 14: cross the motion-type ladder with volume, so the 4-source corpus has a
true types x volume grid instead of a ladder at one volume.

Why this exists: the printed figure's panel (c) is a genuine 2-D grid on the
5-source pool -- read down a column and volume is fixed, so the spread is a pure
type effect; read across a row and the type count is fixed, so it is a pure volume
effect. The 4-source study has no such grid. Its type ladders live at N=12,500 only,
and the 4x4 factorial crosses a *diversity rule* with volume, not a type count. So
the two figures cannot be compared cell for cell, and the one disagreement between
them -- the 5-source grid still shows 9.5 WISDM points between 5 and 1000 clusters
at ~20k windows, while the 4-source `nar` vs `one` contrast at 12,500 is inside
noise -- cannot be resolved without matched cells.

Construction is identical to 10_axis_arms.py's type ladder, only N moves:
ocdetect only, the k largest T-phys clusters, equal-per-type water-fill, uniform
inside each type. The grid is TRIANGULAR because k caps how much data exists:
ocdetect's largest 3 clusters hold 15,065 windows, its largest 30 hold 129,964,
its largest 300 hold 913,555. A cell is emitted only where the chosen clusters can
actually supply N without resampling, which is the same feasibility rule
09_grid_feasibility.py applies on the 5-source pool.

Existing cells are not rebuilt: the four N=12,500 arms are already trained and
scored, and rewriting their index files would invalidate them.

Writes $WORK/idx/vd_typ{30,300,all}_n{50k,200k,800k}.npy (feasible ones only),
       updates out/arms.json, writes out/type_volume_grid.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diversity_stats as ds
from arms import IDX_DIR, K_TYPES, OUT_DIR, POOL_N, SEED, WORK_DIR, source_ranges

_spec = importlib.util.spec_from_file_location(
    "axis_arms", os.path.join(os.path.dirname(os.path.abspath(__file__)), "10_axis_arms.py"))
ax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ax)

# (tag, k) -- k = how many of the largest T-phys clusters the arm may draw from.
LEVELS = [("30", 30), ("300", 300), ("all", None)]
# (volume tag, N). 12,500 is deliberately absent: those four arms already exist.
VOLUMES = [("n50k", 50_000), ("n200k", 200_000), ("n800k", 800_000)]


def main():
    rng = np.random.default_rng(SEED)
    labels_phys = np.load(os.path.join(WORK_DIR, "labels_phys_k1000.npy"))
    labels_enc = np.load(os.path.join(WORK_DIR, "labels_enc_k1000.npy"))
    z_enc = np.load(os.path.join(WORK_DIR, "enc_pca64_f32.npy"), mmap_mode="r")
    pc_phys = np.bincount(labels_phys, minlength=K_TYPES)
    pc_enc = np.bincount(labels_enc, minlength=K_TYPES)
    ranges = source_ranges()
    query_idx = np.sort(rng.choice(POOL_N, min(20_000, POOL_N), replace=False))
    subj = np.load(os.path.join(WORK_DIR, "prov_subject.npy"))
    sess = np.load(os.path.join(WORK_DIR, "prov_session.npy"))

    lo, hi = ranges["ocdetect"]
    oc = np.arange(lo, hi)
    by_type = ax.types_of(oc, labels_phys)
    by_size = sorted(by_type, key=lambda c: -len(by_type[c]))

    arms_out, skipped = {}, []
    for tag, k in LEVELS:
        keep = by_size if k is None else by_size[:k]
        sub = {c: by_type[c] for c in keep}
        cap = sum(len(v) for v in sub.values())
        for vt, n in VOLUMES:
            arm = f"vd_typ{tag}_{vt}"
            if cap < n:
                skipped.append({"arm": arm, "need": n, "available": cap})
                print(f"[{arm:18s}] infeasible: {cap:,} < {n:,}", flush=True)
                continue
            idx = ax.waterfill(sub, n, rng)
            np.save(os.path.join(IDX_DIR, f"{arm}.npy"), idx.astype(np.int64))
            st = ds.summarise(idx, labels_phys, labels_enc, pc_phys, pc_enc, ranges,
                              K_TYPES, z_enc=z_enc, z_query_idx=query_idx, seed=SEED)
            st["nn_self_median"] = ax.self_nn_median(idx, z_enc, rng)
            s, g = subj[idx], sess[idx]
            st.update({"diversity": f"typ{tag}", "volume_tag": vt,
                       "index_file": os.path.join(IDX_DIR, f"{arm}.npy"),
                       "sampler": "natural", "n_types_allowed": len(keep),
                       "type_space": "phys",
                       "n_subjects_known": int(len(np.unique(s[s >= 0]))),
                       "n_sessions_known": int(len(np.unique(g[g >= 0])))})
            arms_out[arm] = st
            print(f"[{arm:18s}] n={len(idx):>7,}  phys eff={st['phys']['effective_types_shannon']:>6.1f}  "
                  f"enc eff={st['enc']['effective_types_shannon']:>6.1f}  "
                  f"vendi={st.get('vendi_enc', float('nan')):.2f}", flush=True)

    apath = os.path.join(OUT_DIR, "arms.json")
    manifest = json.load(open(apath))
    manifest["arms"].update(arms_out)
    json.dump(manifest, open(apath, "w"), indent=2)
    json.dump({"arms": arms_out, "skipped": skipped},
              open(os.path.join(OUT_DIR, "type_volume_grid.json"), "w"), indent=2)
    print(f"[grid] wrote {len(arms_out)} arms, skipped {len(skipped)} infeasible -> {apath}")


if __name__ == "__main__":
    main()
