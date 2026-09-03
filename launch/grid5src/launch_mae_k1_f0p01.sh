#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_grid_k1_f0p01
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Sub-1% volume sweep arm 0.5% (N=17,776), fresh 105K-step run from
# the v3 cache via scaling_mode=index_file -- no raw rebuild. 12h covers 105K
# steps at the observed ~3.86 steps/s (~7.6h) with margin. --resume latest lets a
# TIMEOUT resubmit as-is. See EXPERIMENTS.md 11k.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/grid5src/vit1d_ssl_mae_grid_k1_f0p01.yaml \
    --run-name vit1d_ssl_mae_grid_k1_f0p01 \
    --resume latest
