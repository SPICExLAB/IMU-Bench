#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu-long
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_ssl_mae_varlen_acconly_multidata_nhanes
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/vit1d_ssl_mae_varlen_acconly_multidata_nhanes_flu.yaml"

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "$BASE_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --run-name vit1d_ssl_mae_varlen_acconly_multidata_nhanes
