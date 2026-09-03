#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
#
# FRAME-MATCH ABLATION, step 1: one pretraining arm.
#   ARM=canon|oct|so3   ->  runs/mae_frame_<ARM>
#
# All three arms share the 200 h OCDetect-only cache, 30k steps, seed 42 and the R0 recipe.
# The ONLY key that differs is `random_rotation` (verified by the generator):
#   canon  enabled: false                      -- one fixed wrist frame, no orientation entropy
#   oct    enabled: true, octahedral           -- 24 discrete orientations (the R0 setting)
#   so3    enabled: true, so3                  -- continuous uniform SO(3), arbitrary frame
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
ARM="${ARM:?set ARM=canon|oct|so3}"
RUN="mae_frame_${ARM}"
CACHE=/projects/b53017/vbc1591/mae_cache_frameabl_ocd200
if [ -f "runs/${RUN}/.done" ]; then echo "[framematch] $RUN done -> skip"; exit 0; fi
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then
  echo "[framematch] twin RUNNING elsewhere -> skip"; exit 0; fi
if [ ! -f "$CACHE/.done" ]; then echo "[framematch] cache not built ($CACHE/.done missing) -> abort"; exit 1; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \
    --model-config "configs/framematch/${RUN}.yaml" --run-name "$RUN" --resume latest \
  && touch "runs/${RUN}/.done"
