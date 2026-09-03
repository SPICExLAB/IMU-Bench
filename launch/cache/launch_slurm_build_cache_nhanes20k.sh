#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=420G
#SBATCH --job-name=build_cache_nhanes20k
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# NHANES scale-up cache (2026-08-18): the 11-source pool with the NHANES slice raised
# 400 h -> 20,000 h (sample_fraction 0.25, max_hours 20000) => ~7M windows, ~43 GB.
# Everything else is byte-identical to the standard cache, so arms A/B differ only in
# how much NHANES they can draw. Only the nhanes source is (re)built; the other 11
# sources are hard-linked from the existing cache to avoid recomputing them.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
SRC=/projects/b53017/vbc1591/mae_multidata_cache
DST=/projects/b53017/vbc1591/mae_multidata_cache_nhanes20k
mkdir -p "$DST"
for f in "$SRC"/*.npy; do
  b=$(basename "$f")
  case "$b" in nhanes_*) continue;; esac
  [ -e "$DST/$b" ] || ln "$f" "$DST/$b" 2>/dev/null || cp "$f" "$DST/$b"
done
echo "[cache] linked $(ls "$DST" | grep -vc '^nhanes_') non-NHANES arrays"
python -u scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/nhanes_scale/cache_nhanes20k.yaml \
    --out-dir "$DST" --sources nhanes --num-workers 24
python - <<'PY'
import numpy as np, glob, os
tot=0
for f in sorted(glob.glob("/projects/b53017/vbc1591/mae_multidata_cache_nhanes20k/*_train_x.npy")):
    n=np.load(f, mmap_mode="r").shape[0]; h=n*10.24/3600; tot+=h
    print(os.path.basename(f).replace("_train_x.npy","").ljust(14), f"{n:>9} windows  {h:8.0f} h")
print(f"TOTAL {tot:.0f} h")
PY
