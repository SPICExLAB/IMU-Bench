#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=rebuild_humanml3d_cache_unimtssim
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

python scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_multidata_unimtssim_flu.yaml \
    --out-dir /projects/b53017/vbc1591/mae_multidata_cache_unimtssim \
    --sources humanml3d \
    --batch-size 256 \
    --num-workers 8
