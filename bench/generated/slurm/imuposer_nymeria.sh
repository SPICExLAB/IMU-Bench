#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=180G
#SBATCH --exclude=qgpu3006,qgpu3013,qgpu2014,qgpu2006
#SBATCH --job-name=imuposer_nymeria
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# IMUPoser's BiLSTM trained and scored on Nymeria right-wrist IMU (task-specific baseline for the pose tables).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export WANDB_MODE=disabled
python baselines/full_body_pose/IMUPoser/train_nymeria_imuposer.py
echo "[imuposer] finished $(date -Is)"
