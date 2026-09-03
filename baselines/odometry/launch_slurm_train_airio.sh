#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=odom_airio_nymeria
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

cd baselines/odometry/Air-IO
python3 -u train_nymeria_airio.py \
    --cache-dir ../data_cache \
    --out-dir runs_nymeria \
    --epochs 100 \
    --patience 15 \
    --batch-size 512 \
    --lr 1e-3
