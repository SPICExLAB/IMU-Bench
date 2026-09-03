#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --job-name=i1_5src_cache
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
DST=/projects/b53017/vbc1591/inertia1_5src30s_raw
mkdir -p "$DST"
python -u scripts/build_multidata_cache.py --config configs/base.yaml --model-config configs/inertia1/cache_5src_30s_raw.yaml --out-dir "$DST" --num-workers 12
