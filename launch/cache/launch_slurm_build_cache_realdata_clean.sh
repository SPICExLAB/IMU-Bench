#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --job-name=build_cache_realdata_clean
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# LEAKAGE FIX (2026-08-18): the `realdata` MAE source contained pamap2/recofit/utd_mhad/wisdm,
# which are also 4 of the 6 HAR *evaluation* sets. Rebuild that source from the 3 non-eval
# datasets only (samosa, shoaib, ut_watch) into realdataclean_*.npy, leaving the original
# realdata_*.npy in place for the historical runs.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
for DST in /projects/b53017/vbc1591/mae_multidata_cache /projects/b53017/vbc1591/mae_multidata_cache_nhanes20k; do
  mkdir -p "$DST/_clean_tmp"
  python -u scripts/build_multidata_cache.py --config configs/base.yaml \
      --model-config configs/nhanes_scale/cache_realdata_clean.yaml \
      --out-dir "$DST/_clean_tmp" --sources realdata --num-workers 16
  for f in "$DST/_clean_tmp"/realdata_*.npy; do mv "$f" "$DST/$(basename "$f" | sed 's/^realdata_/realdataclean_/')"; done
  rmdir "$DST/_clean_tmp" 2>/dev/null || true
  python - "$DST" <<'PY'
import numpy as np, sys, glob, os
d=sys.argv[1]
for f in sorted(glob.glob(os.path.join(d,"realdata*_train_x.npy"))):
    n=np.load(f, mmap_mode="r").shape[0]; print(os.path.basename(f), n, "windows", f"{n*5.12/3600:.0f} h train")
PY
done
