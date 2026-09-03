#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# 2x2 NHANES-scale x normalisation MAE pretrain (2026-08-18). ARM=A|B|C|D (env).
#   A 400 h NHANES + z-score   B 20k h NHANES + z-score
#   C 400 h NHANES + physical  D 20k h NHANES + physical
# All arms: same 11-source pool, balanced sampling with NHANES at 25 % of mass
# (acc-only total ~32 %), modality loss weights acc 1.0 / gyro 2.0, 100k steps.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
ARM="${ARM:?set ARM=A|B|C|D|E}"
RUN="mae_nhanes2x2_arm${ARM}"
if [ -f "runs/${RUN}/.done" ]; then echo "[arm] done -> skip"; exit 0; fi
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then echo "[arm] twin RUNNING -> skip"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \
    --model-config configs/nhanes_scale/mae_arm${ARM}.yaml --run-name "$RUN" --resume latest \
  && touch "runs/${RUN}/.done"
