#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_odom_finetune_nymeria_mae_recipe5pct
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Finetune-scaling companion to the odom probe already run for the 3-point
# recipe experiment (Section 11 of EXPERIMENTS.md). Unlike the full-body-pose
# PROBE config, this finetune config has early_stopping.enabled=true
# (patience=20), so it should terminate on its own well within budget --
# historical 100%-baseline run (vit1d_odom_finetune_nymeria_mae_multidata)
# reached its best checkpoint in well under 1h.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_odom_finetune_nymeria_mae_recipe5pct_multidata_flu.yaml \
    --run-name vit1d_odom_finetune_nymeria_mae_recipe5pct
