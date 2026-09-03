#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# Mask-ratio ablation (2026-08-20). One arm per mask_ratio on the leak-free 514 h raw cache,
# 30k steps, baseline recipe (= configs/nhanes_scale/mae_recipe_R0.yaml) held fixed in every
# other key. mask_ratio=0.75 is NOT relaunched here -- it is runs/mae_recipe_R0.
#   ARM=40|50|60|85|90   ->  runs/mae_mr<ARM>
# The AltAttn encoder uses BERT-style masking (all tokens go through the encoder, masked ones
# replaced by a learned mask token), so per-step FLOPs are identical across arms: equal steps
# really is equal compute here.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
ARM="${ARM:?set ARM=30|40|50|60|85|90}"
RUN="mae_mr${ARM}"
if [ -f "runs/${RUN}/.done" ]; then echo "[maskratio] done -> skip"; exit 0; fi
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then echo "[maskratio] twin RUNNING -> skip"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \
    --model-config configs/maskratio/mae_mr${ARM}.yaml --run-name "$RUN" --resume latest \
  && touch "runs/${RUN}/.done"
