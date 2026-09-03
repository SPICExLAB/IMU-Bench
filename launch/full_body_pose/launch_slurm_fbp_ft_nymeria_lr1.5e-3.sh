#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=fbp_ft_nymeria_lr1.5e-3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_mae_multidata_lr1.5e-3_flu.yaml \
    --run-name vit1d_full_body_pose_finetune_nymeria_mae_multidata_lr1.5e-3
