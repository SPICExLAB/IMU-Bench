#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=txmae_odom__trainable
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
python scripts/run_pretrain.py --config configs/base.yaml --model-config configs/transfer_mae/txmae_odom__trainable.yaml --run-name txmae_odom__trainable
echo "[txmae_odom__trainable] finished $(date -Is)"
