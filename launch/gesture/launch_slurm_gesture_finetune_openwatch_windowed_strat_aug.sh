#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_gesture_finetune_openwatch_6class_mae_windowed_strat_aug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

# OpenWatch paper-protocol (arXiv:2605.04791) run: windowed 1s/stride-20
# training, posture-stratified contamination-safe split. See EXPERIMENTS.md §5f.
BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/vit1d_gesture_finetune_openwatch_6class_mae_multidata_flu_windowed_stratified_aug.yaml"

PRETRAINED_CKPT="runs/vit1d_ssl_mae_varlen_altattn_multidata/checkpoints/best.pt"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "$BASE_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --run-name vit1d_gesture_finetune_openwatch_6class_mae_windowed_strat_aug
