#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --job-name=vis_odom_recovery_grid
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python analysis/mae_scaling_diagnostics/checks/vis_odom_recovery_grid.py \
    --model-config configs/vit1d_odom_finetune_nymeria_mae_multidata_flu.yaml \
    --run vit1d_odom_finetune_nymeria_mae_multidata --ckpt best \
    --n-sessions 9 --out data_vis/odom_recovery/grid_original_9sessions.png
