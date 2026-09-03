#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --job-name=rescore_mask_ratio
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# Fair comparison of the 300k-step Stage-1 mask-ratio arms (0.50 / 0.75 / 0.90).
# Each arm's logged val loss is measured at its OWN mask ratio, so the arms are not
# comparable as logged; this rescoes all three checkpoints at ONE fixed eval ratio with an
# identical per-batch mask sequence. Run at three eval ratios so the ranking is not an
# artifact of the ratio we happen to score at.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then echo "[rescore] twin RUNNING -> skip"; exit 0; fi
for R in 0.50 0.75 0.90; do
  python -u analysis/mae_scaling_diagnostics/checks/rescore_mask_ratio.py \
      --stage stage1 --eval-mask-ratio $R --batches 60
done
