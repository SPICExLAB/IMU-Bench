#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_clip_humanml3d_probe_relcon_unimtssim
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "configs/base.yaml" \
    --model-config "configs/vit1d_clip_humanml3d_flu_probe_relcon_unimtssim.yaml" \
    --run-name vit1d_clip_humanml3d_probe_relcon_unimtssim
