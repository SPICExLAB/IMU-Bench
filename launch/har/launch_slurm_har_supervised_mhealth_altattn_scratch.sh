#!/bin/bash
#SBATCH --account=b1232
#SBATCH --partition=b1232
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_har_supervised_mhealth_scratch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL

if [ ! -f TransfHAR_env/bin/activate ]; then
    python -m venv TransfHAR_env
    source TransfHAR_env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m pip install -e .
else
    source TransfHAR_env/bin/activate
fi

# wandb is already authenticated on this machine via ~/.netrc (from a prior
# `wandb login`); no API key needs to be embedded in this script.

BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/vit1d_har_supervised_mhealth_altattn_scratch.yaml"

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "$BASE_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --run-name vit1d_har_supervised_mhealth_altattn_scratch
