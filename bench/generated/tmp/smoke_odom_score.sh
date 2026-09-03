#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --job-name=smoke_odom_score
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python - <<'PY'
import yaml, json
from bench.registry_io import load_tasks
from bench.score.odometry import score
task = load_tasks()["tasks"]["odometry_nymeria_worldframe"]
# score the smoke run on ONE val session (limit via a shallow copy of the task card pointing at a tmp dir)
import os, glob, shutil
vd = "bench/generated/tmp/odom_val_1"; os.makedirs(vd, exist_ok=True)
src = sorted(glob.glob(task["dataset"]["val"]+"/*.pkl"))[0]
dst = os.path.join(vd, os.path.basename(src))
if not os.path.exists(dst): os.symlink(src, dst)
task = dict(task); task["dataset"] = dict(task["dataset"]); task["dataset"]["val"] = os.path.abspath(vd)
m, info = score("bench/generated/tmp/runs/smoke_odom_probe", "bench/generated/tmp/runs/smoke_odom_probe/checkpoints/best.pt", task)
print("METRICS", json.dumps(m), json.dumps(info))
PY
echo SMOKE_SCORE_OK
