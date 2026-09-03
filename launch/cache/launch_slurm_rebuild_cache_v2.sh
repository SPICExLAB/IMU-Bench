#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --job-name=rebuild_cache_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Rebuilds ocdetect/capture24/eatingspeed under their newly-raised
# sample_fraction (2026-07-23: OCDetect 0.1->0.52 [~3,000h of ~5,790h full,
# max_hours 500->3000], Capture24 0.1->0.5 [~2,000h of ~4,000h full],
# EatingSpeed 0.1->0.5 [~263h of ~526h full]) into a NEW cache dir
# (/projects/b53017/vbc1591/mae_multidata_cache_v2/) -- the other 8 sources
# are already symlinked in from the original mae_multidata_cache/ (unchanged,
# zero extra storage). Neither the original mae_multidata_cache/ nor
# mae_multidata_cache_ocdetect_splitfix/ are touched by this job.

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

OUT_DIR="/projects/b53017/vbc1591/mae_multidata_cache_v2"
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
    hrs = (len(tr) + len(va)) * hop_s / 3600
    print(f"RESULT: {name} train={len(tr)} val={len(va)} windows (~{hrs:.1f}h effective)")
PY
