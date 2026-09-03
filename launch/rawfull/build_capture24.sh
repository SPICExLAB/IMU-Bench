#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name=rawfull_capture24
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
DST=/projects/b53017/vbc1591/mae_multidata_cache_raw
for f in x mask tmask; do rm -f "$DST/capture24_train_$f.npy" "$DST/capture24_val_$f.npy"; done
python -u scripts/build_multidata_cache.py --config configs/base.yaml --model-config configs/nhanes_scale/cache_raw_full.yaml --out-dir "$DST" --sources capture24 --num-workers 12
python - <<'PY'
import numpy as np
n=np.load("/projects/b53017/vbc1591/mae_multidata_cache_raw/capture24_train_x.npy",mmap_mode="r").shape[0]
print(f"[rawfull] capture24: {n} train windows = {n*5.12/3600:.0f} h")
PY
