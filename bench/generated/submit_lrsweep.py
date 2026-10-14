#!/usr/bin/env python3
"""Shared-protocol LR sweep over the 4-source family.

    4 encoders x 5 tasks x 3 learning rates = 60 frozen-probe cells

The point is fairness, not a better headline. Tuning the LR for `mae_4src_z`
alone and leaving the baselines on their default is the classic rigged
comparison, so the same grid runs for every 4-source encoder and the winning LR
per task is written into the task card for ALL of them.

Tasks: arm pose, odometry, microgesture, language, PD.
  - full-body pose is EXCLUDED: the 2026-08-15 sweep already showed its frozen
    probe is not LR-limited (between-cell spread < within-run noise), so 12 more
    cells would buy nothing.
  - HAR is excluded here because `mae_4src_z` already has a 27-cell sweep in
    results_lrtune/; extending that to the other three encoders is a separate,
    larger batch (81 cells).

Caveat carried into the analysis: with one seed, picking the best of three LRs
is picking the max of three draws, so the winner is biased high by roughly the
run-to-run noise (~0.05-0.1 cm on arm pose). Applying the same grid to every
encoder cancels the bias in the RANKING but not in the absolute numbers.

    python bench/generated/submit_lrsweep.py [--dry]
"""
import copy, os, subprocess, sys, yaml

sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
os.chdir("/gpfs/home/vbc1591/IMU-SSL")
from bench.make_config import build_config

ENCODERS = ["mae_4src_z", "simmtm_4src", "relcon_4src", "focal_4src"]
TASKS = ["arm_pose_nymeria", "odometry_nymeria_worldframe",
         "microgesture_openwatch", "language_humanml3d", "pd_pads"]
LRS = [("3e-4", 3e-4), ("1e-3", 1e-3), ("3e-3", 3e-3)]
LONG = {"arm_pose_nymeria", "odometry_nymeria_worldframe", "language_humanml3d"}
BAD_NODES = "qgpu3006,qgpu3013,qgpu2014,qgpu2006"

dry = "--dry" in sys.argv
os.makedirs("bench/generated/results_lrsweep", exist_ok=True)
os.makedirs("bench/generated/slurm/lrsweep", exist_ok=True)

jobs = []
for enc in ENCODERS:
    for task in TASKS:
        for tag, lr in LRS:
            name = f"lrs_{task}__{enc}__probe__lr{tag}"
            res = f"bench/generated/results_lrsweep/{name}.json"
            if os.path.exists(res):
                continue
            try:
                _, cfg = build_config(task, enc, "probe")
            except Exception as exc:                       # noqa: BLE001
                print(f"[skip] {name}: {exc}"); continue
            cfg = copy.deepcopy(cfg)
            cfg["trainer"]["optim"]["lr"] = lr
            yaml.safe_dump(cfg, open(f"bench/generated/configs/{name}.yaml", "w"), sort_keys=False)

            tl = "14:00:00" if task in LONG else "04:00:00"
            mem = "80G" if task == "language_humanml3d" else "48G"
            sh = f"""#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --exclude={BAD_NODES}
#SBATCH --gres=gpu:a100:1
#SBATCH --time={tl}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=8
#SBATCH --job-name=L{enc[:6]}{task[:6]}{tag}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
if [ -f {res} ]; then echo already-scored; exit 0; fi
rm -rf runs/{name}
python scripts/run_pretrain.py --config configs/base.yaml \\
  --model-config bench/generated/configs/{name}.yaml --run-name {name}
python bench/generated/lrsweep_score.py {name} {task}
"""
            p = f"bench/generated/slurm/lrsweep/{name}.sh"
            open(p, "w").write(sh); os.chmod(p, 0o755)
            if dry:
                print(f"[dry] {name}"); continue
            jobs.append(subprocess.check_output(["sbatch", "--parsable", p], text=True).strip())

print(f"submitted {len(jobs)} cells" if not dry else "dry run complete")
if jobs:
    open("bench/generated/.lrsweep_jobs", "w").write(":".join(jobs))
