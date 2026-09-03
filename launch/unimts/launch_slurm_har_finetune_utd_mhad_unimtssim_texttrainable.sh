#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_har_finetune_utd_mhad_unimtssim_texttrainable
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

PRETRAINED_CKPT="/gpfs/home/vbc1591/IMU-SSL/runs/vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5/checkpoints/best_pinned.pt"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "configs/base.yaml" \
    --model-config "configs/vit1d_har_finetune_utd_mhad_unimtssim_texttrainable_multidata_flu.yaml" \
    --run-name vit1d_har_finetune_utd_mhad_unimtssim_texttrainable
