#!/usr/bin/env python
"""Stage 12: the motion-type ladder rebuilt in the ENCODER's own space (T-enc).

Why: the existing ladder vd_typ{3,30,300,all}_n12k selects by T-phys clusters,
and re-measured in T-enc it barely moves -- typ3 already covers 225 effective
T-enc types and the whole ladder spans only 225 -> 609 (2.7x), against 3 -> 673
(224x) in T-phys. The two definitions are nearly orthogonal (AMI 0.209). So the
flat result so far says physical-feature diversity does not matter; it says
nothing about diversity the model can actually perceive.

This ladder is identical in every respect except the space used to SELECT:
ocdetect only, N=12,500, the k largest T-enc clusters (k = 3, 30, 300, all),
equal-per-type water-fill, uniform inside each type. Simulated realised
diversity: 3 -> 851 effective T-enc types (284x), 181 -> 496 in T-phys.

On circularity: selecting in the model's representation space is fatal for a
POSITIVE result (it is data pruning tuned to the model) but is the setting most
favourable to "diversity helps" for a NULL one -- it varies exactly what the
model can see. If this ladder is flat too, the null is stronger; if it is not,
confirm with a third-party encoder (HarNet) before claiming it.

Does not touch the existing arms: rerunning 10_axis_arms.py would rewrite the
index files of arms that are already trained, so this stage only appends.

Writes $WORK/idx/vd_etyp{3,30,300,all}_n12k.npy, updates out/arms.json,
       out/enc_type_arms.json
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

N_TYPE = ax.N_TYPE                      # 12,500 -- same volume as the T-phys ladder
LEVELS = [3, 30, 300, None]


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
    by_type = ax.types_of(oc, labels_enc)
    by_size = sorted(by_type, key=lambda c: -len(by_type[c]))

    arms_out = {}
    for k in LEVELS:
        tag = "all" if k is None else str(k)
        keep = by_size if k is None else by_size[:k]
        sub = {c: by_type[c] for c in keep}
        tot = sum(len(v) for v in sub.values())
        if tot < N_TYPE:
            print(f"[etyp{tag}] infeasible: {tot:,} < {N_TYPE:,}"); continue
        arm = f"vd_etyp{tag}_n12k"
        idx = ax.waterfill(sub, N_TYPE, rng)
        np.save(os.path.join(IDX_DIR, f"{arm}.npy"), idx.astype(np.int64))
        st = ds.summarise(idx, labels_phys, labels_enc, pc_phys, pc_enc, ranges,
                          K_TYPES, z_enc=z_enc, z_query_idx=query_idx, seed=SEED)
        st["nn_self_median"] = ax.self_nn_median(idx, z_enc, rng)
        s, g = subj[idx], sess[idx]
        st.update({"diversity": f"etyp{tag}", "volume_tag": "n12k",
                   "index_file": os.path.join(IDX_DIR, f"{arm}.npy"),
                   "sampler": "natural", "n_types_allowed": len(keep),
                   "type_space": "enc",
                   "n_subjects_known": int(len(np.unique(s[s >= 0]))),
                   "n_sessions_known": int(len(np.unique(g[g >= 0])))})
        arms_out[arm] = st
        print(f"[{arm:17s}] n={len(idx):>6,}  enc eff={st['enc']['effective_types_shannon']:>6.1f}  "
              f"phys eff={st['phys']['effective_types_shannon']:>6.1f}  "
              f"vendi={st.get('vendi_enc', float('nan')):.2f}  nn={st['nn_self_median']:.4f}",
              flush=True)

    apath = os.path.join(OUT_DIR, "arms.json")
    manifest = json.load(open(apath))
    manifest["arms"].update(arms_out)
    json.dump(manifest, open(apath, "w"), indent=2)
    json.dump(arms_out, open(os.path.join(OUT_DIR, "enc_type_arms.json"), "w"), indent=2)
    print(f"[etyp] wrote {len(arms_out)} arms -> {apath}")


if __name__ == "__main__":
    main()
