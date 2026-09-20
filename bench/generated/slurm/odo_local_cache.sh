#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --job-name=odo_local_cache
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export WANDB_MODE=disabled
cd baselines/odometry && python -u prepare_nymeria_windows_local.py --split val && python -u prepare_nymeria_windows_local.py --split train
echo "[odo_local_cache] finished $(date -Is)"
