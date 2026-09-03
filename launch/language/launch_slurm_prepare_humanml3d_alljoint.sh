#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=prepare_humanml3d_alljoint
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

cd baselines/language_alignment/UniMTS

# Separate output dirs for train/val -- pretrain.py's CLIPDataset has no
# internal split awareness (it trains on everything under --data_path), so
# train/val must be kept in physically separate directories to avoid val
# clips leaking into the from-scratch training data.
python prepare_humanml3d_alljoint.py --split train --out-dir ./data_ours_alljoint_humanml3d_train --progress-every 1000
python prepare_humanml3d_alljoint.py --split val   --out-dir ./data_ours_alljoint_humanml3d_val   --progress-every 1000
