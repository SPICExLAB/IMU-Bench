#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=smoke_scorers
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export BENCH_TMP=/gpfs/home/vbc1591/IMU-SSL/bench/generated/tmp
python - <<'PY'
import os, glob, json, traceback
from bench.registry_io import load_tasks
T = load_tasks()["tasks"]
def run(name, fn):
    print(f"\n===== {name}"); 
    try:
        m, info = fn(); print("METRICS", name, json.dumps(m)); print("INFO", json.dumps({k:v for k,v in info.items() if k!='subject_confusion_mean_prob'}))
    except Exception:
        traceback.print_exc(); print("FAILED", name)
# arm pose on 2 val sessions
from bench.score import arm_pose
task = dict(T["arm_pose_nymeria"]); task["dataset"]=dict(task["dataset"])
vd="bench/generated/tmp/arm_val_2"; os.makedirs(vd, exist_ok=True)
for src in sorted(glob.glob(T["arm_pose_nymeria"]["dataset"]["val"]+"/*.pkl"))[:2]:
    d=os.path.join(vd, os.path.basename(src)); os.path.exists(d) or os.symlink(src, d)
task["dataset"]["val"]=os.path.abspath(vd)
run("arm_pose/mae_11src_probe", lambda: arm_pose.score("runs/vit1d_arm_pose_probe_nymeria_mae_multidata","runs/vit1d_arm_pose_probe_nymeria_mae_multidata/checkpoints/best.pt", task))
run("arm_pose/mae_acconly_probe", lambda: arm_pose.score("runs/vit1d_arm_pose_probe_nymeria_mae_acconly_multidata","runs/vit1d_arm_pose_probe_nymeria_mae_acconly_multidata/checkpoints/best.pt", task))
from bench.score import microgesture
run("microgesture/focal_probe", lambda: microgesture.score("/gpfs/home/vbc1591/IMU-SSL/runs/vit1d_gesture_probe_openwatch_6class_focal_windowed_strat","/gpfs/home/vbc1591/IMU-SSL/runs/vit1d_gesture_probe_openwatch_6class_focal_windowed_strat/checkpoints/best.pt", T["microgesture_openwatch"]))
from bench.score import pd
run("pd/mae_probe", lambda: pd.score("/gpfs/home/vbc1591/IMU-SSL/runs/vit1d_pd_probe_pads_mae","/gpfs/home/vbc1591/IMU-SSL/runs/vit1d_pd_probe_pads_mae/checkpoints/best.pt", T["pd_pads"]))
PY
echo SMOKE_SCORERS_DONE
