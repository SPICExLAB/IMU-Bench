#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=170G
#SBATCH --job-name=build_cache_v3raw_es
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# EatingSpeed slice of the un-normalised cache twin (the first build, 7031640, stopped at the gyro unit check), then the
# index file of the 'nozscore' arm. Everything else of mae_multidata_cache_v3raw is already in place.
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
DST=/projects/b53017/vbc1591/mae_multidata_cache_v3raw
echo "===== eatingspeed  $(date -Is)"
python -u scripts/build_multidata_cache.py --config configs/base.yaml --model-config configs/design_ablation/cache_v3raw.yaml \
    --out-dir "$DST" --sources eatingspeed --num-workers 30 || { echo "[build] eatingspeed FAILED"; exit 1; }
python launch/design_ablation/make_idx_ref_v3raw.py || exit 1
echo "[build_cache_v3raw_es] finished $(date -Is)"
