#!/usr/bin/env python3
"""Window index file for the 'nozscore' design-ablation arm (physical-unit input) on the un-normalised cache twin
mae_multidata_cache_v3raw.

The reference arm (corpus_ref) draws 197,094 windows from the z-scored cache v3: realdata, Motion-X and SONIC in
full, 8,703 OCDetect and 57,536 EatingSpeed windows (idx_ref.npy over the 11-source concatenation). The raw twin was
rebuilt on 2026-09-21 with the current loaders, and only OCDetect is row-for-row identical to v3 (100 % of sampled rows
match after z-scoring); the HAR pool (master parquet rewindowed since July) and the mocap sources (spike sanitiser,
2026-08-22, drops 399 Motion-X and 36 SONIC windows) are not. So this arm keeps the reference COMPOSITION, not the
identical windows:  realdata / motionx / sonic = every raw window (as in the reference), ocdetect = the reference's own
8,703 rows (aligned), eatingspeed = the reference rows if the raw slice has the same length as v3, else an evenly
strided subset of the same size.  Output: idx_ref_v3raw.npy in the 5-source order realdata, ocdetect, eatingspeed,
motionx, sonic (the order of the enabled sources in configs/design_ablation/vit1d_ssl_mae_abl_nozscore.yaml).
"""
import json, os, sys
import numpy as np

V3 = "/projects/b53017/vbc1591/mae_multidata_cache_v3"
RAW = "/projects/b53017/vbc1591/mae_multidata_cache_v3raw"
ORDER11 = ["realdata", "ocdetect", "nymeria", "totalcapture", "imuposer", "eatingspeed", "motionx", "humanml3d", "sonic",
           "capture24", "openwatch", "nhanes"]          # enabled-source order of corpus_ref's index space
KEEP = ["realdata", "ocdetect", "eatingspeed", "motionx", "sonic"]
OUT = "/projects/b53017/vbc1591/diversity_saturation/corpus_ablation/idx_ref_v3raw.npy"

n3 = {s: np.load(f"{V3}/{s}_train_x.npy", mmap_mode="r").shape[0] for s in ORDER11 if os.path.exists(f"{V3}/{s}_train_x.npy")}
order = [s for s in ORDER11 if s in n3]
off3, t = {}, 0
for s in order:
    off3[s] = t; t += n3[s]
idx = np.sort(np.load("/projects/b53017/vbc1591/diversity_saturation/corpus_ablation/idx_ref.npy"))
src_of = np.searchsorted(np.array([off3[s] for s in order] + [t]), idx, side="right") - 1
ref = {s: idx[src_of == order.index(s)] - off3[s] for s in order}
nraw = {s: np.load(f"{RAW}/{s}_train_x.npy", mmap_mode="r").shape[0] for s in KEEP}
parts, offr, tr, report = [], {}, 0, {}
for s in KEEP:
    offr[s] = tr
    if s in ("realdata", "motionx", "sonic"):
        sel = np.arange(nraw[s]); rule = "all raw windows"
    elif nraw[s] == n3[s]:
        sel = ref[s]; rule = "reference rows (raw slice has the v3 length)"
    else:
        sel = np.linspace(0, nraw[s] - 1, len(ref[s])).round().astype(np.int64); rule = "evenly strided, reference count"
    assert sel.max() < nraw[s]
    parts.append(sel + tr); tr += nraw[s]
    report[s] = {"ref_count": int(len(ref[s])), "raw_pool": int(nraw[s]), "v3_pool": int(n3[s]), "selected": int(len(sel)), "rule": rule}
new = np.sort(np.concatenate(parts)).astype(np.int64)
np.save(OUT, new)
json.dump({"total": int(len(new)), "reference_total": int(len(idx)), "pool": int(tr), "per_source": report},
          open(OUT.replace(".npy", ".json"), "w"), indent=1)
print(json.dumps({"total": int(len(new)), "reference_total": int(len(idx)), "pool": int(tr), "per_source": report}, indent=1))
