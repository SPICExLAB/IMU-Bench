#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=recipe_val_rescore
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Rescores the 3 Section-11 recipe checkpoints on ONE identical,
# augmentation-free val set, fixing the val-augmentation confound documented
# in analysis/mae_scaling_diagnostics/12_recipe_val_rescore.py's docstring
# (5pct/25pct bypassed cache_dir via session_fraction -> mixed per-source val
# augmentation; 100pct loaded from cache -> CachedWindowDataset augments val
# unconditionally). Loads from the v3 cache, so no raw rebuild -- fast.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
python analysis/mae_scaling_diagnostics/12_recipe_val_rescore.py
