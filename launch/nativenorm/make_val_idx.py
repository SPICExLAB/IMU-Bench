#!/usr/bin/env python3
"""Per-source VALIDATION index files for the native-normalisation arms.

The un-normalised cache twin (mae_multidata_cache_v3raw) was built without the
per-source val caps of the production cache, so its val pool is 290,638 windows,
64 % of it OCDetect -- against the 27,463 windows (OCDetect 4 %) the z-score rows
this family is compared against validated on. val_loss is the early-stopping
monitor AND the best.pt selector, so an arm validated on a different mixture is
not selecting a comparable checkpoint, and validating 10x more windows every
1,000 steps is pure wall-clock.

This writes one evenly-strided index per source that restores the production
val COUNT (hence composition). Where the raw twin already has at most as many
windows as production (realdata, motionx, sonic) every window is kept.
Deliberately evenly strided, not the first N, so all participants/sessions stay
represented -- the same rule _one()'s max_hours_val path uses.
"""
import json, os
import numpy as np

PROD = "/projects/b53017/vbc1591/mae_multidata_cache"
RAW = "/projects/b53017/vbc1591/mae_multidata_cache_v3raw"
OUT = "/projects/b53017/vbc1591/diversity_saturation/nativenorm"
SOURCES = ["realdata", "ocdetect", "eatingspeed", "motionx", "sonic"]

os.makedirs(OUT, exist_ok=True)
report = {"prod_cache": PROD, "raw_cache": RAW, "per_source": {}}
for s in SOURCES:
    n_prod = np.load(f"{PROD}/{s}_val_x.npy", mmap_mode="r").shape[0]
    n_raw = np.load(f"{RAW}/{s}_val_x.npy", mmap_mode="r").shape[0]
    if n_raw <= n_prod:
        keep = np.arange(n_raw)
        rule = "all raw val windows (raw twin has no more than production)"
    else:
        keep = np.linspace(0, n_raw - 1, n_prod).astype(int)
        keep = np.unique(keep)
        rule = "evenly strided to the production count"
    np.save(f"{OUT}/val_idx_{s}.npy", keep)
    report["per_source"][s] = {"raw_pool": int(n_raw), "prod_count": int(n_prod),
                               "selected": int(len(keep)), "rule": rule}
    print(f"{s:12s} raw {n_raw:>7d} -> {len(keep):>6d} (production {n_prod:>6d})  {rule}")

report["total_selected"] = sum(v["selected"] for v in report["per_source"].values())
report["prod_total"] = sum(v["prod_count"] for v in report["per_source"].values())
json.dump(report, open(f"{OUT}/val_idx.json", "w"), indent=1)
print(f"total {report['total_selected']} (production {report['prod_total']})")
