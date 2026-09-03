#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --job-name=vit1d_clip_realdata_midtrain2_from_humanml3d
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

# mem=220G / cpus=16: capture24 alone is 699M rows -- WindowDataset preloads
# every session's raw sensor arrays into RAM sequentially (no threading in
# this loader), so this will take a while during the initial data-loading
# phase before any training steps appear. Not a hang -- monitor via the
# "[preload N/M] ..." log lines this loader already emits per session.

BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/vit1d_clip_realdata_midtrain2_from_humanml3d.yaml"

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "$BASE_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --run-name vit1d_clip_realdata_midtrain2_from_humanml3d
