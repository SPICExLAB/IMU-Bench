#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --job-name=fm_build_cache
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
#
# FRAME-MATCH ABLATION, step 0 (2026-08-20): build the 200 h OCDetect-only pretraining cache.
#
# Why OCDetect alone: the ablation's independent variable is the pretraining COORDINATE FRAME,
# so the pool must be in exactly one frame. It is not: measured mean-gravity axis in the
# existing 514 h cache is -Y for OCDetect, -X for SONIC/MotionX, +Z for TotalCapture and
# EatingSpeed, and uniformly spread for NHANES -- the "FLU" loaders do not agree with each
# other. A single source sidesteps that entirely, and OCDetect is both leak-free (no bench
# eval target touches it) and large enough (1,782 h available) to give 200 h.
#
# flu_convention: documented -- see configs/framematch/cache_ocd200.yaml and
# OCDetect_dataloader._load_recording. The legacy transform every earlier cache used is an
# acc-only mirror (det = -1) with the gyro untouched, i.e. acc and gyro in different frames.
#
# normalize is DISABLED in the build config: the cache stores impute+filter only, gravity
# intact, and the training config z-scores at read time (objective.cache_is_normalized: false).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

DST=/projects/b53017/vbc1591/mae_cache_frameabl_ocd200
mkdir -p "$DST"
if [ -f "$DST/.done" ]; then echo "[fm_cache] already built -> skip"; exit 0; fi

python -u scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/framematch/cache_ocd200.yaml \
    --out-dir "$DST" --sources ocdetect --num-workers 16

python - "$DST" <<'PY'
import numpy as np, os, sys
d = sys.argv[1]
tr = np.load(f"{d}/ocdetect_train_x.npy", mmap_mode="r")
va = np.load(f"{d}/ocdetect_val_x.npy",   mmap_mode="r")
g  = np.linalg.norm(np.asarray(tr[:2000])[:, :3].mean(-1), axis=1).mean()
print(f"[fm_cache] train={len(tr):,} win = {len(tr)*5.12/3600:.1f} h (hop-based)  val={len(va):,}")
print(f"[fm_cache] mean |acc| per window = {g:.2f} m/s^2  -> gravity {'PRESERVED' if g > 3 else 'MISSING (cache got normalised!)'}")
assert g > 3, "cache was normalised; the frame cue is gone -- do not train on this"
assert tr.shape[-1] == 512, f"expected T=512, got {tr.shape[-1]}"
PY
touch "$DST/.done"
echo "[fm_cache] done $(date -Is)"
