#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:a100:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_stage1_acconly_mr50
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Moved to the b53017 BUYIN partition (2026-08-07): the gengpu h100 queue put
# these ~15h out (est. start Aug 8 13:15), and requesting h100 specifically was
# an over-constraint -- nothing here needs one. b53017 is a dedicated node
# (qgpu2015, 4x a100, 449 GB free, one other job) with an infinite time limit,
# so both arms start immediately. a100 is slower than the baseline's h100, so
# --time is raised to 48h; the arms are still compared to each other and to the
# 0.75 baseline by STEP COUNT, not wall-clock, so GPU type does not confound.
# Mask-ratio ablation arm for Stage 1. Only objective.mask_ratio differs from
# the baseline (0.75) arm -- verified by config diff. 300k steps at the
# measured 4.05 steps/s is ~20.6h; 32h leaves margin since a different mask
# ratio changes the encoder's token count and therefore the step time
# (0.50 masking = 2x the visible tokens of 0.75 = slower per step).
# If it still times out, resume with --resume latest (NOT best).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_stage1_acconly_mr50_flu.yaml \
    --run-name vit1d_ssl_mae_stage1_acconly_mr50 \
    --resume latest
