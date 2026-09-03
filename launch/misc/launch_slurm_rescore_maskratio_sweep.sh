#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --job-name=rescore_mr_sweep
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# Reconstruction quality of the six 30k-step mask-ratio arms, all scored at ONE fixed eval
# mask ratio with an identical per-batch mask sequence (the arms' own val losses are measured
# at their own ratio and cannot be ranked). Cheap: ~2 min/eval-ratio, no downstream training.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then echo "[rescore] twin RUNNING -> skip"; exit 0; fi
for i in $(seq 1 90); do
  n=0; for r in 40 50 60 85 90; do [ -f "runs/mae_mr${r}/.done" ] && n=$((n+1)); done
  [ "$n" -ge 5 ] && break; sleep 60
done
for r in 40 50 60 85 90; do
  [ -f "runs/mae_mr${r}/.done" ] || { echo "[rescore] runs/mae_mr${r}/.done missing -> abort"; exit 1; }
done
for R in 0.50 0.75 0.90; do
  python -u analysis/mae_scaling_diagnostics/checks/rescore_mask_ratio.py \
      --stage maskratio --eval-mask-ratio $R --batches 60
done
