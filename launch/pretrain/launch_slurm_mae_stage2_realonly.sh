#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_stage2_realonly
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# --gres=gpu:1 (ANY gpu type), NOT gpu:h100:1 -- pinning to h100 put the
# mask-ratio arms ~15h back in the queue on 2026-08-07 for no reason; nothing
# in this workload needs a specific GPU, and arms are compared by step count.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_stage2_accgyro_realonly_flu.yaml \
    --run-name vit1d_ssl_mae_stage2_realonly
