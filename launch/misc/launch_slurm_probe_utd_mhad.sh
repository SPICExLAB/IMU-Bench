#!/bin/bash
#SBATCH --account=b1232
#SBATCH --partition=b1232
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_har_probe_utd_mhad
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL

# Set up the venv once; reused (not rebuilt) on every subsequent job submission.
# Delete TransfHAR_env/ to force a full reinstall (e.g. after changing requirements.txt).
if [ ! -f TransfHAR_env/bin/activate ]; then
    python -m venv TransfHAR_env
    source TransfHAR_env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m pip install -e .
else
    source TransfHAR_env/bin/activate
fi

# wandb: authenticate once with `wandb login` (never commit an API key)


BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/vit1d_har_probe_utd_mhad_mae_multidata_flu.yaml"

PRETRAINED_CKPT="runs/vit1d_ssl_mae_varlen_altattn_multidata/checkpoints/best.pt"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT (run launch_slurm_MAE.sh first)" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "$BASE_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --run-name vit1d_har_probe_utd_mhad_mae_multidata
