#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_stage1_acconly
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Two-stage MAE. Run order: build stage1 cache -> stage1 train ->
# build stage2 cache -> stage2 train (stage 2 warm-starts from stage 1's best.pt).
# Time sized off the MEASURED ~3.85-3.9 steps/s from the Section 11 runs
# (the earlier ~9 steps/s estimate TIMEOUT'd all three of those). If it still
# times out, resume with --resume latest (NOT best -- that desyncs the
# early-stopping bookkeeping restore, trainer.py:118).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_stage1_acconly_flu.yaml \
    --run-name vit1d_ssl_mae_stage1_acconly \
    --resume latest
