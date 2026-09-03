#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_har_probe_wisdm_mae_div1pct_random
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# WISDM linear-probe scaling check for the 3-point recipe experiment
# (session_fraction scaling, fixed-step 300K budget). pretrained_ckpt: best
# in the probe config picks up whatever runs/vit1d_ssl_mae_div1pct_random/checkpoints/best.pt
# is AT THE TIME THIS JOB ACTUALLY RUNS -- pretraining job for this fraction
# may still be in progress (2026-07-26), so this is a preliminary read, not
# the final number; rerun once all 3 pretraining jobs finish their full
# 300,000-step budget. Same methodology as EXPERIMENTS.md section 9b
# (vit1d_har_probe_wisdm_mae_data{X}pct family), ~20-35 min historically.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_har_probe_wisdm_mae_div1pct_random_multidata_flu.yaml \
    --run-name vit1d_har_probe_wisdm_mae_div1pct_random
