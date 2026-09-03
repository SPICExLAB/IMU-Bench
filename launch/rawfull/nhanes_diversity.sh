#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=nhanes_diversity
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python -u analysis/nhanes_diversity.py --cache /projects/b53017/vbc1591/mae_multidata_cache_raw --budget-hours 272 --K 2000 --fit-sample 300000 --out analysis/nhanes_diversity/out
