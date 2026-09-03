#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=420G
#SBATCH --job-name=build_cache_full
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# FULL-SIZE cache (2026-08-19): every hour of OCDetect / EatingSpeed / Capture24 / SONIC /
# MotionX / TotalCapture, no max_hours or max_files caps. Replaces the old cache whose
# OCDetect slice held 12 h of 1,782 h and EatingSpeed 82 h of 818 h (truncated writes).
# NHANES is capped at 60,000 h: the full 185,585 h is 130M windows (~1.6 TB fp32, ~19 h to
# build) and a 50k-step run only draws 6.4M windows, so beyond ~9,000 h nothing is seen twice.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
DST=/projects/b53017/vbc1591/mae_multidata_cache_full
mkdir -p "$DST"
for S in totalcapture motionx sonic eatingspeed ocdetect capture24 nhanes; do
  echo "=== $S  $(date -Is)"
  python -u scripts/build_multidata_cache.py --config configs/base.yaml \
      --model-config configs/nhanes_scale/cache_full.yaml \
      --out-dir "$DST" --sources $S --num-workers 32
  python - "$DST" "$S" <<'PY'
import numpy as np, sys, os
d,s=sys.argv[1],sys.argv[2]; f=os.path.join(d,f"{s}_train_x.npy")
if os.path.exists(f):
    n=np.load(f,mmap_mode="r").shape[0]
    print(f"[cache_full] {s}: {n} train windows = {n*5.12/3600:.0f} h", flush=True)
PY
done
python - <<'PY'
import numpy as np, glob, os
tot=0
for f in sorted(glob.glob("/projects/b53017/vbc1591/mae_multidata_cache_full/*_train_x.npy")):
    n=np.load(f, mmap_mode="r").shape[0]; h=n*5.12/3600; tot+=h
    print(f"{os.path.basename(f).replace('_train_x.npy',''):14s} {n:>10} win {h:8.0f} h")
print(f"TOTAL {tot:.0f} h")
PY
