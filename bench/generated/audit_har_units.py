#!/usr/bin/env python3
"""Audit accelerometer units for every HAR / downstream evaluation dataset.

`normalize.method: physical` and StatsConcatEncoder both assume m/s^2 (divide by 9.81). Any dataset
stored in g would be scaled 9.81x wrong. Reads a few raw windows per dataset straight from the
loader (impute+filter only, normalisation off) and reports the per-window |mean acc|.
"""
import sys, yaml, numpy as np
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from imu_lm.utils.helpers import deep_update
from imu_lm.data.loaders import make_loaders

TASKS = {
    "har_wisdm": "wisdm", "har_utd_mhad": "utd_mhad", "har_pamap2": "pamap2", "har_recofit": "recofit",
    "har_samosa": "samosa", "har_shoaib": "shoaib", "har_ut_watch": "ut_watch", "har_prism": "prism",
}
EXTRA = {"har_hhar": "/projects/b53017/vbc1591/hhar_dataset.parquet",
         "har_mhealth": "/projects/b53017/vbc1591/mhealth_dataset.parquet",
         "pd_pads": "/projects/b53017/vbc1591/pads_dataset.parquet"}
base = yaml.safe_load(open("configs/base.yaml"))
tpl = yaml.safe_load(open("configs/vit1d_har_probe_hhar_mae_multidata_flu.yaml"))
print(f"{'dataset':14s}{'|mean acc| per window':>24s}{'acc std':>10s}   unit")
def probe(cfg, name):
    cfg = deep_update(deep_update({}, base), cfg)
    cfg["preprocess"]["normalize"]["enabled"] = False
    cfg["data"]["num_workers"] = 2; cfg["data"]["batch_size"] = 64
    L = make_loaders(cfg)
    xs = []
    for b in L["val_loader"]:
        if b is None: continue
        xs.append(b[0].numpy()); break
    if not xs: print(f"{name:14s}   (no val batch)"); return
    x = xs[0]
    pm = float(np.linalg.norm(x[:, :3].mean(-1), axis=1).mean())
    print(f"{name:14s}{pm:24.3f}{x[:, :3].std():10.3f}   {'g' if pm < 3 else 'm/s^2'}")
for t, dsname in TASKS.items():
    c = dict(tpl); c = yaml.safe_load(yaml.safe_dump(c))
    c.setdefault("data", {})["train_on_datasets"] = [dsname]
    c["paths"] = {"dataset_path": "/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet",
                  "runs_root": "/gpfs/home/vbc1591/IMU-SSL/runs"}
    try: probe(c, dsname)
    except Exception as e: print(f"{dsname:14s}   ERR {str(e)[:60]}")
for t, path in EXTRA.items():
    c = yaml.safe_load(yaml.safe_dump(dict(tpl)))
    c.setdefault("data", {})["train_on_datasets"] = [t.replace("har_", "").replace("pd_", "")]
    c["paths"] = {"dataset_path": path, "runs_root": "/gpfs/home/vbc1591/IMU-SSL/runs"}
    try: probe(c, t)
    except Exception as e: print(f"{t:14s}   ERR {str(e)[:60]}")
