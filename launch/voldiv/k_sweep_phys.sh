#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=vd_ksweep
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# CPU-only: k-means over cached 32-d physical PCA, no GPU. Answers "at which K
# does the physical feature space actually have modes?" -- the type axis was
# built at K=1000, which the encoder-space sweep found to be the WEAKEST point.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export PYTHONUNBUFFERED=1
python -u analysis/volume_diversity/11_k_sweep_phys.py
