#!/usr/bin/env python3
"""After the 5src Inertia-1 pretrains: register both encoders in the bench and submit the
identical protocol sweep (9 HAR x probe/finetune x 3 LRs, weighted CE + bal-acc selection)."""
import os, sys, copy, subprocess, yaml
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL"); os.chdir("/gpfs/home/vbc1591/IMU-SSL")
e = yaml.safe_load(open("bench/registry/encoders.yaml"))
for var, run, order in [("patchtst", "inertia1_patchtst_small_5src", 118), ("art", "inertia1_art_small_5src", 119)]:
    base = copy.deepcopy(e["encoders"][f"inertia1_{var}"])
    base.update({"name": base["name"].replace("(NHANES repro)", "(5src pool, acc-only)"),
                 "short": base["short"] + "-5src", "order": order, "run": run, "group": "ablation",
                 "pretrain_data": "SAME five sources as mae_5src (realdata 7-set HAR pool incl. eval leak, ocdetect, eatingspeed both wrists, motionx++, sonic), 30 s @ 20 Hz acc-only in g (~occupied hours reported by build_5src_corpus)"})
    e["encoders"][f"inertia1_{var}_5src"] = base
yaml.safe_dump(e, open("bench/registry/encoders.yaml", "w"), sort_keys=False); print("encoders registered")
from bench.make_config import build_config
TASKS = ["har_wisdm","har_utd_mhad","har_pamap2","har_recofit","har_hhar","har_mhealth","har_samosa","har_ut_watch","har_prism"]
GRID = {"probe": ["3e-4","1e-3","3e-3"], "finetune": ["3e-5","1e-4","3e-4"]}
cells = []
for enc in ["inertia1_patchtst_5src","inertia1_art_5src"]:
    for t in TASKS:
        for proto, lrs in GRID.items():
            for lr in lrs:
                name, cfg = build_config(t, enc, proto); cfg = copy.deepcopy(cfg)
                ot = cfg["objective"]["type"]
                cfg["trainer"]["optim"]["lr"] = float(lr)
                cfg["objective"][ot]["class_weighted_ce"] = True
                cfg["trainer"]["early_stopping"] = {"enabled": True, "monitor": "bal_acc", "patience": 20}
                cfg["trainer"].pop("extra_best", None)
                run = f"lrt_{t}__{enc}__{proto}__lr{lr}"
                yaml.safe_dump(cfg, open(f"bench/generated/configs/{run}.yaml", "w"), sort_keys=False)
                cells.append((run, t))
print(len(cells), "cells")
jobs = []
for i in range(0, len(cells), 3):
    pk = cells[i:i+3]; k = i // 3
    body = """#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --job-name=lrt5s%d
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%%x_%%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%%x_%%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
""" % k
    for run, t in pk:
        body += """
( if [ ! -f bench/generated/results_lrtune/%s.json ]; then
    python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/%s.yaml --run-name %s &&
    python bench/generated/lrtune_score.py %s %s
  fi ) > logs/lrtcell_%s.log 2>&1 &""" % (run, run, run, run, t, run)
    body += "\nwait\necho LRT5S_PACK_DONE\n"
    p = f"bench/generated/slurm/lrtune/lrt5s{k}.sh"; open(p, "w").write(body); os.chmod(p, 0o755)
    jobs.append(subprocess.check_output(["sbatch", "--parsable", p], text=True).strip())
print("submitted", len(jobs), "packs:", jobs[0], "...", jobs[-1])
