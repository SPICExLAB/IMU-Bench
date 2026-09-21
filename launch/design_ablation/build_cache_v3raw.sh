#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=220G
#SBATCH --job-name=build_cache_v3raw
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Un-normalised twin of mae_multidata_cache_v3 for the five sources of the reference corpus ('MAE without z-norm' ablation).
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
DST=/projects/b53017/vbc1591/mae_multidata_cache_v3raw
mkdir -p "$DST" logs
for S in motionx sonic realdata eatingspeed ocdetect; do
  echo "===== $S  $(date -Is)"
  python -u scripts/build_multidata_cache.py --config configs/base.yaml --model-config configs/design_ablation/cache_v3raw.yaml \
      --out-dir "$DST" --sources $S --num-workers 30 || echo "[build] $S FAILED"
done
python - "$DST" <<'PY'
import sys, glob, os, numpy as np
dst=sys.argv[1]; ref="/projects/b53017/vbc1591/mae_multidata_cache_v3"
for s in ("realdata","ocdetect","eatingspeed","motionx","sonic"):
    for sp in ("train","val"):
        a=os.path.join(dst,f"{s}_{sp}_x.npy"); b=os.path.join(ref,f"{s}_{sp}_x.npy")
        na=np.load(a,mmap_mode="r").shape if os.path.exists(a) else None; nb=np.load(b,mmap_mode="r").shape if os.path.exists(b) else None
        print(f"[check] {s:12s} {sp:5s} raw={na} v3={nb} {'OK' if na==nb else 'MISMATCH'}")
PY
echo "[build_cache_v3raw] finished $(date -Is)"
