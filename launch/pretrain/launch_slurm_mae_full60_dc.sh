#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --job-name=mae_full60_dc
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
if [ -f runs/mae_full60_dc/.done ]; then exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml --model-config configs/nhanes_scale/mae_full60_dc.yaml --run-name mae_full60_dc --resume latest && touch runs/mae_full60_dc/.done
