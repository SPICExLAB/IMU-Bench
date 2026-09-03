#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_pd_probe_pads_mae
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_pd_probe_pads_mae_multidata_flu.yaml \
    --run-name vit1d_pd_probe_pads_mae
