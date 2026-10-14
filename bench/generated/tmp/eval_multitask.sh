#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --qos=buyin
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --job-name=eval_multitask
#SBATCH --output=bench/generated/tmp/%x_%j.out
#SBATCH --error=bench/generated/tmp/%x_%j.err
# Score the two joint runs' own task heads into benchmark cells, so they are comparable with the single-task
# fine-tune rows. The joint trainer logs per-task LOSSES to the slurm log and never writes the benchmark metric into
# runs/<r>/logs/metrics.txt, so the scorers that read metrics.txt (full_body_pose, arm_pose) cannot harvest it and we
# recompute MPJPE from the checkpoint instead; odometry_local and microgesture recompute from the checkpoint anyway.
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate

for pair in "mt_body_odom odometry_nymeria_local" "mt_body_gesture microgesture_openwatch"; do
  set -- $pair
  echo "===== bench.score $2 <- $1"
  python -m bench.score --task "$2" --encoder "$1" --protocol finetune --run "$1__$2" \
      --note "joint multi-task run; this cell is the joint model's own $2 head" 2>&1 | tail -4
done

echo "===== full-body MPJPE recomputed from the joint checkpoints"
python - <<'PY'
import sys, os, json
sys.path.insert(0, ".")
from bench.rescore_full_body_extra import evaluate
for run in ("mt_body_odom__full_body_pose_nymeria", "mt_body_gesture__full_body_pose_nymeria"):
    ck = os.path.join("runs", run, "checkpoints", "best.pt")
    if not os.path.exists(ck):
        print(f"[skip] {run}: no checkpoint"); continue
    m, info = evaluate(os.path.join("runs", run), ck)
    print(f"[{run}] mpjpe {m['mpjpe_mm_recomputed']:.2f} mm | MobilePoser joint err {m['mp_joint_err_cm']:.2f} cm | "
          f"global ang {m['mp_global_ang_err_deg']:.2f} deg  ({info['n_windows']} windows)", flush=True)
print("single-task reference (finetune): MAE 88.11 mm, LIMU-BERT 89.74, SimMTM 92.91, SSL-Wearables 90.61, scratch 92.7")
PY
echo "[eval_multitask] done $(date -Is)"
