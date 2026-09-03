#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_arm_pose_probe_nymeria_mae_recipe10pct
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Arm-pose (5-joint) Nymeria linear-probe scaling check for the 3-point
# recipe experiment (session_fraction scaling, fixed-step 300K budget).
# Same methodology as EXPERIMENTS.md section 9c. early_stopping is enabled
# here (patience=20), so historically these finish in ~1-1.5h (jobs
# 7491647/7491648, 12h walltime never hit) -- 3h budget gives real margin.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_arm_pose_probe_nymeria_mae_recipe10pct_multidata_flu.yaml \
    --run-name vit1d_arm_pose_probe_nymeria_mae_recipe10pct
