#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --job-name=rebuild_cache_v3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Rebuilds ocdetect/capture24/eatingspeed at 100% (2026-07-24: removed the
# sample_fraction caps entirely -- OCDetect 0.52->1.0, max_hours 3000->100000
# [no-op ceiling], Capture24 0.5->1.0, EatingSpeed 0.5->1.0) into a NEW cache
# dir (/projects/b53017/vbc1591/mae_multidata_cache_v3/) -- the other 8
# sources are symlinked in from the original mae_multidata_cache/ (unchanged,
# zero extra storage). Neither mae_multidata_cache/ nor
# mae_multidata_cache_{ocdetect_splitfix,v2}/ are touched by this job.

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL

if [ ! -f TransfHAR_env/bin/activate ]; then
    python -m venv TransfHAR_env
    source TransfHAR_env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m pip install -e .
else
    source TransfHAR_env/bin/activate
fi

OUT_DIR="/projects/b53017/vbc1591/mae_multidata_cache_v3"
mkdir -p "$OUT_DIR" logs

python scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml \
    --out-dir "$OUT_DIR" \
    --sources ocdetect capture24 eatingspeed \
    --num-workers 4

python - "$OUT_DIR" <<'PY'
import sys
import numpy as np
out_dir = sys.argv[1]
for name in ["ocdetect", "capture24", "eatingspeed"]:
    tr = np.load(f"{out_dir}/{name}_train_x.npy", mmap_mode="r")
    va = np.load(f"{out_dir}/{name}_val_x.npy", mmap_mode="r")
    hop_s = 5.12
    native_s = 10.24
    n = len(tr) + len(va)
    print(f"RESULT: {name} train={len(tr)} val={len(va)} windows "
          f"(~{n*hop_s/3600:.1f}h effective, ~{n*native_s/3600:.1f}h native-convention)")
PY
