#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --job-name=vis_pose_recovery
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python analysis/mae_scaling_diagnostics/checks/vis_pose_recovery.py \
    --run vit1d_full_body_pose_finetune_nymeria_mae_multidata_lr1e-3 \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_mae_multidata_flu.yaml \
    --ckpt best_mpjpe --batches 0 --frames 5
