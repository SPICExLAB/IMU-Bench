#!/usr/bin/env python
"""Stage 15: the paper's motion-type x volume grid, replicated cell-for-cell on the
leak-free 4-source pool with MAE-feature motion types.

The printed panel (c) (Sections/6_Scaling.tex, fig:scaling_coverage) is a triangular
grid on the 5-source pool, which contains the HAR evaluation datasets. This is the
same design with exactly one thing changed -- the corpus:

  same   K levels 1 / 5 / 10 / 25 / 50 / 100 / 1000, meaning "draw only from the K
         LARGEST clusters of a K=1000 partition" (not a re-clustering at K)
  same   volume levels 0.01 / 0.1 / 1 / 10 / 100 % of the pool
  same   equal-per-cluster water-fill inside the chosen clusters, smallest first, so
         the realised type count equals K instead of collapsing onto the head
  same   largest-first cluster choice
  same   triangular feasibility rule: a cell exists only where the K chosen clusters
         hold at least N windows, so no window is drawn twice
  same   50,000-step budget
  same   encoder-space type definition -- the printed grid clustered the 11-source
         encoder's PCA-64, this one clusters mae_4src_z's CLS PCA-64 (T-enc)

  differs   the pool: 4 leak-free sources, 1,738,094 windows, versus 5 sources and
            1,959,288 windows with the evaluation datasets inside

24 of the 35 cells are feasible, against 22 in the printed grid, because the two
pools have similar cluster-size distributions (largest T-enc cluster here holds
6,358 windows against 7,114 there).

Note the intent: this grid is deliberately built in the MODEL's representation
space, not the encoder-free physical one. For a null result that is the setting
most favourable to the diversity hypothesis, since it varies exactly what the model
can perceive. The encoder-free counterpart is 14_type_volume_grid.py.

Writes idx/vd_ek{K}_f{tag}.npy under WORK_DIR, updates out/arms.json,
       writes out/enc_type_volume_grid.json
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

KS = [1, 5, 10, 25, 50, 100, 1000]
FRACS = [("0p01", 1e-4), ("0p1", 1e-3), ("1", 1e-2), ("10", 1e-1), ("100", 1.0)]


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

    # whole pool, T-enc partition, largest clusters first -- as the printed grid does
    allidx = np.arange(POOL_N)
    by_type = ax.types_of(allidx, labels_enc)
    by_size = sorted(by_type, key=lambda c: -len(by_type[c]))

    arms_out, skipped = {}, []
    for K in KS:
        keep = by_size[:K]
        sub = {c: by_type[c] for c in keep}
        cap = sum(len(v) for v in sub.values())
        for tag, f in FRACS:
            n = int(round(f * POOL_N))
            arm = f"vd_ek{K}_f{tag}"
            if cap < n:
                skipped.append({"arm": arm, "K": K, "need": n, "available": cap})
                continue
            idx = ax.waterfill(sub, n, rng)
            np.save(os.path.join(IDX_DIR, f"{arm}.npy"), idx.astype(np.int64))
            st = ds.summarise(idx, labels_phys, labels_enc, pc_phys, pc_enc, ranges,
                              K_TYPES, z_enc=z_enc, z_query_idx=query_idx, seed=SEED)
            st["nn_self_median"] = ax.self_nn_median(idx, z_enc, rng)
            s, g = subj[idx], sess[idx]
            st.update({"diversity": f"ek{K}", "volume_tag": f"f{tag}",
                       "index_file": os.path.join(IDX_DIR, f"{arm}.npy"),
                       "sampler": "natural", "n_types_allowed": K,
                       "type_space": "enc", "grid_K": K, "grid_fraction": f,
                       "n_subjects_known": int(len(np.unique(s[s >= 0]))),
                       "n_sessions_known": int(len(np.unique(g[g >= 0])))})
            arms_out[arm] = st
            print(f"[{arm:16s}] n={len(idx):>9,}  enc eff={st['enc']['effective_types_shannon']:>6.1f}  "
                  f"phys eff={st['phys']['effective_types_shannon']:>6.1f}  "
                  f"vendi={st.get('vendi_enc', float('nan')):.2f}", flush=True)

    apath = os.path.join(OUT_DIR, "arms.json")
    manifest = json.load(open(apath))
    manifest["arms"].update(arms_out)
    json.dump(manifest, open(apath, "w"), indent=2)
    json.dump({"ks": KS, "fracs": [f for _, f in FRACS], "pool_n": POOL_N,
               "arms": arms_out, "skipped": skipped},
              open(os.path.join(OUT_DIR, "enc_type_volume_grid.json"), "w"), indent=2)
    print(f"[grid] {len(arms_out)} feasible cells, {len(skipped)} infeasible -> {apath}")


if __name__ == "__main__":
    main()
