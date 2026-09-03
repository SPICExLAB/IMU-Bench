#!/bin/bash
#SBATCH --account=b1232
#SBATCH --partition=buyin
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --job-name=data_cache
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL

# Set up the venv once; reused (not rebuilt) on every subsequent job submission.
# Delete TransfHAR_env/ to force a full reinstall (e.g. after changing requirements.txt).
if [ ! -f TransfHAR_env/bin/activate ]; then
    python -m venv TransfHAR_env
    source TransfHAR_env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m pip install -e .
else
    source TransfHAR_env/bin/activate
fi

python scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml \
    --out-dir /projects/b53017/vbc1591/mae_multidata_cache \
    --num-workers 8
