#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=txmae_odom__frozen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
python scripts/run_pretrain.py --config configs/base.yaml --model-config configs/transfer_mae/txmae_odom__frozen.yaml --run-name txmae_odom__frozen
echo "[txmae_odom__frozen] finished $(date -Is)"
