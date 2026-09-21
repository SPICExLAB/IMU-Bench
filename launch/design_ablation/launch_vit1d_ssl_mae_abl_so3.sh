#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_abl_so3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Design ablation 'so3': continuous SO(3) rotation instead of the 24 octahedral rotations; otherwise identical to corpus arm 'ref'.
# Same recipe/budget as grid cell k1000_f10 (50,000 steps). --resume latest lets a TIMEOUT resubmit as-is.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/design_ablation/vit1d_ssl_mae_abl_so3.yaml \
    --run-name vit1d_ssl_mae_abl_so3 \
    --resume latest
