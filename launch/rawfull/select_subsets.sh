#!/bin/bash
# usage: sbatch launch/rawfull/select_subsets.sh "capture24=300 ocdetect=300 eatingspeed=100"
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --job-name=select_subsets
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
for kv in $1; do
  src=${kv%%=*}; h=${kv##*=}
  python -u analysis/nhanes_diversity.py --cache /projects/b53017/vbc1591/mae_multidata_cache_raw --source $src \
      --budget-hours $h --K 2000 --fit-sample 300000 --out analysis/subset_selection/$src
  echo "[select] $src -> analysis/subset_selection/$src/indices_balanced_${h%.*}h.npy"
done
