#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# Recipe ablation (2026-08-19), one change added at a time on the 8,692 h diversity-balanced cache:
#   R0 baseline | R1 +yaw rotation +physical scaling +lr 3e-4 | R2 +band-weighted loss | R3 +modality masking
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
ARM="${ARM:?set ARM=R0|R1|R2|R3|R4|Rraw}"
RUN="mae_recipe_${ARM}"
if [ -f "runs/${RUN}/.done" ]; then echo "[recipe] done -> skip"; exit 0; fi
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then echo "[recipe] twin RUNNING -> skip"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \
    --model-config configs/nhanes_scale/mae_recipe_${ARM}.yaml --run-name "$RUN" --resume latest \
  && touch "runs/${RUN}/.done"
