#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name=rawfull_motionx
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
DST=/projects/b53017/vbc1591/mae_multidata_cache_raw
for f in x mask tmask; do rm -f "$DST/motionx_train_$f.npy" "$DST/motionx_val_$f.npy"; done
python -u scripts/build_multidata_cache.py --config configs/base.yaml --model-config configs/nhanes_scale/cache_raw_full.yaml --out-dir "$DST" --sources motionx --num-workers 12
python - <<'PY'
import numpy as np
import glob,os
D="/projects/b53017/vbc1591/mae_multidata_cache_raw"
n=np.load(f"{D}/motionx_train_x.npy",mmap_mode="r").shape[0]
tm=np.load(f"{D}/motionx_train_tmask.npy"); h=np.where(tm>=512,256,tm).sum()/50/3600 if os.path.exists(f"{D}/motionx_train_tmask.npy") else n*5.12/3600
print(f"[rawfull] motionx: {n} train windows = {h:.1f} h real signal")
PY
