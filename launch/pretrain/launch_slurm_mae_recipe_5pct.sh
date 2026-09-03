#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_recipe_5pct
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 3-point "does more data help" experiment, approved fixed-step-budget
# protocol (300,000 steps identical across 5%/25%/100%), session_fraction
# scaling (natural sampling -- balanced was rejected, see config header
# comment), NHANES excluded. 5pct/25pct rebuild all 11 sources from raw with
# provenance at job start (scaling_mode requires it, bypasses cache_dir) --
# ~1.5-2h overhead observed in analysis/mae_scaling_diagnostics/10_recipe_budget.py
# (job 7888102); 100pct loads straight from mae_multidata_cache_v3 (fast).
#
# RESUME (2026-07-26): the first submission (job 7919072) TIMEOUT'd at
# --time=16:00:00 / step=197,000 -- true observed throughput was ~3.86
# steps/sec, not the ~9 steps/sec estimate that budget was based on. Resuming
# from checkpoints/latest.pt (step=190,000, saved by ckpt_every_steps=10000;
# best.pt was actually a few thousand steps further along but resuming from
# it would desync the Trainer's early-stopping bookkeeping restore, which
# always prefers latest.pt -- see imu_lm/runtime_consistency/trainer.py:118).
# Remaining budget: 110,000 steps @ ~3.86/s (~7.9h) + ~1.5-2h raw rebuild
# (unavoidable on resume -- the dataset composition isn't checkpointed, only
# model/optimizer/step) + margin -> 14h.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe5pct.yaml"

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "$BASE_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --run-name vit1d_ssl_mae_recipe_5pct \
    --resume latest
