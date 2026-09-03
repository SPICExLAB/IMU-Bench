#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --job-name=rebuild_mocap_short
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# SHORT-CLIP RECOVERY (2026-08-21): rebuild sonic + motionx into the raw cache keeping clips
# of 1.28-10.24 s (previously discarded by `if n < T: continue`). Writes *_tmask.npy valid
# lengths; the MAE objective hides padded patches from encoder and loss.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
DST=/projects/b53017/vbc1591/mae_multidata_cache_raw
for S in motionx sonic; do
  rm -f "$DST/${S}_train_x.npy" "$DST/${S}_train_mask.npy" "$DST/${S}_val_x.npy" "$DST/${S}_val_mask.npy" "$DST/${S}_train_tmask.npy" "$DST/${S}_val_tmask.npy"
  echo "=== $S $(date -Is)"
  python -u scripts/build_multidata_cache.py --config configs/base.yaml \
      --model-config configs/nhanes_scale/cache_raw_shortclips.yaml \
      --out-dir "$DST" --sources $S --num-workers 16
done
python - <<'PY'
import numpy as np, glob, os
for s in ("motionx","sonic"):
    f=f"/projects/b53017/vbc1591/mae_multidata_cache_raw/{s}_train_x.npy"
    t=f.replace("_x.npy","_tmask.npy")
    n=np.load(f,mmap_mode="r").shape[0]
    if os.path.exists(t):
        v=np.asarray(np.load(t)); h=v.sum()/50/3600
        print(f"[rebuilt] {s}: {n} windows, {h:.1f} h real signal ({(v<512).sum()} short, {(v>=512).sum()} full)")
    else:
        print(f"[rebuilt] {s}: {n} windows, {n*5.12/3600:.1f} h (no short clips found?)")
PY
