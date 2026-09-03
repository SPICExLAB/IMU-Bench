#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=odom_probe_nymeria_mae_div1pct_redundant
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs data_vis/nymeria_odom_altattn_probe_div1pct_redundant

CUDA_VISIBLE_DEVICES=0 python scripts/eval_odom_nymeria_altattn.py \
    --run vit1d_ssl_mae_div1pct_redundant \
    --train-dir /projects/b53017/vbc1591/processed_data_body_Ty_rwrist \
    --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \
    --out-dir data_vis/nymeria_odom_altattn_probe_div1pct_redundant
