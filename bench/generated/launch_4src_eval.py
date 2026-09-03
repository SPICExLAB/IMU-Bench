#!/usr/bin/env python3
"""Evaluate the two 4-source MAE encoders on the full downstream set once their pretrains finish.

Tasks: 9 HAR (probe) + full_body_pose, odometry, microgesture, language (probe + finetune).
HAR uses the Inertia-1-paper protocol (inverse-frequency weighted CE, val bal-acc selection) so the
numbers line up with results_lrtune/; the other tasks use their bench task-card recipes.
Both encoders carry normalize_method + preprocess_from, so make_config gives each cell the SAME
preprocessing its encoder was pretrained with (raw/DC for _dc, physical for _phys).
"""
import copy, os, subprocess, sys, yaml
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL"); os.chdir("/gpfs/home/vbc1591/IMU-SSL")
from bench.make_config import build_config

ENCS = ["mae_4src_dc", "mae_4src_phys"]
HAR = ["har_wisdm","har_utd_mhad","har_pamap2","har_recofit","har_hhar","har_mhealth","har_samosa","har_ut_watch","har_prism"]
OTHER = ["full_body_pose_nymeria", "odometry_nymeria_worldframe", "microgesture_openwatch", "language_humanml3d"]
HDR = """#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=%s
#SBATCH --mem=%s
#SBATCH --cpus-per-task=12
#SBATCH --job-name=%s
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%%x_%%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%%x_%%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
"""
CELL = """
( if [ ! -f bench/results/%s/%s__%s__s42.json ]; then
    python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/configs/%s.yaml --run-name %s &&
    python -m bench.score --task %s --encoder %s --protocol %s --seed 42
  fi ) > logs/e4cell_%s.log 2>&1 &"""

def write_cfg(task, enc, proto):
    name, cfg = build_config(task, enc, proto); cfg = copy.deepcopy(cfg)
    if task.startswith("har_"):
        ot = cfg["objective"]["type"]
        cfg["objective"][ot]["class_weighted_ce"] = True
        cfg["trainer"]["early_stopping"] = {"enabled": True, "monitor": "bal_acc", "patience": 20}
        cfg["trainer"]["optim"]["lr"] = 1e-3          # best single probe LR from the 216-cell sweep
    yaml.safe_dump(cfg, open(f"bench/generated/configs/{name}.yaml", "w"), sort_keys=False)
    return name

def submit(body, tag):
    p = f"bench/generated/slurm/e4/{tag}.sh"; os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write(body + "\nwait\npython -m bench.leaderboard && python -m bench.leaderboard_html || true\necho E4_DONE\n")
    os.chmod(p, 0o755)
    return subprocess.check_output(["sbatch", "--parsable", p], text=True).strip()

jobs = []
for enc in ENCS:
    har_cells = [(write_cfg(t, enc, "probe"), t, "probe") for t in HAR]
    for i in range(0, len(har_cells), 3):
        body = HDR % ("05:00:00", "64G", f"e4{enc[-4:]}h{i//3}")
        for name, t, pr in har_cells[i:i+3]:
            body += CELL % (t, enc, pr, name, name, t, enc, pr, name)
        jobs.append(submit(body, f"e4_{enc}_h{i//3}"))
    for t in OTHER:
        for pr in ("probe", "finetune"):
            name = write_cfg(t, enc, pr)
            mem = "80G" if t == "language_humanml3d" else "64G"
            body = HDR % ("12:00:00", mem, f"e4{name[6:20]}") + CELL % (t, enc, pr, name, name, t, enc, pr, name)
            jobs.append(submit(body, f"e4_{name}"))
print(f"submitted {len(jobs)} jobs for {ENCS}")
open("bench/generated/.e4_jobs", "w").write(":".join(jobs))
