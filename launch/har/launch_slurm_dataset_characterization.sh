#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=dataset_characterization
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs analysis/mae_scaling_diagnostics/09_dataset_characterization.py (per-source
# mean/var/hz/coordinate-frame/reconstruction-MSE comparison table). This is
# lightweight (reads cached .npy files + a few hundred forward passes per
# source) but still OOM-killed twice under the interactive session's memory
# cgroup (~1-4GB cap, confirmed via dmesg both times) -- moving it to SLURM
# like every other real-compute step in this investigation.

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL

if [ ! -f TransfHAR_env/bin/activate ]; then
    python -m venv TransfHAR_env
    source TransfHAR_env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m pip install -e .
else
    source TransfHAR_env/bin/activate
fi

mkdir -p logs
python analysis/mae_scaling_diagnostics/09_dataset_characterization.py \
    --cache-dir /projects/b53017/vbc1591/mae_multidata_cache_v3 \
    --out-subdir 09_dataset_characterization_v3_100pct
