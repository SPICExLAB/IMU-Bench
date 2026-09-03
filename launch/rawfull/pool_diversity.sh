#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=pool_diversity
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
# (a) every source at full cached size (NHANES = the 4,000 h candidate pool)
python -u analysis/pool_diversity.py --per-source 200000 --K 2000 --out analysis/pool_diversity/out_full
# (b) the pool the pretrain will actually see: NHANES restricted to the 272 h diversity-balanced subset
python -u analysis/pool_diversity.py --per-source 200000 --K 2000 --out analysis/pool_diversity/out_nhanes272 \
  --subset nhanes=analysis/nhanes_diversity/out/indices_balanced_272h.npy
