#!/bin/bash
#SBATCH --account=b1232
#SBATCH --partition=b1232
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=rebuild_ocdetect_cache
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Rebuilds ONLY the OCDetect entry of the fixed-split cache
# (/projects/b53017/vbc1591/mae_multidata_cache_ocdetect_splitfix/) at the
# raised sample_fraction=0.1 (was 0.01 -- too few files for a representative
# file-disjoint val split, see analysis/mae_scaling_diagnostics/DIAGNOSIS.md).
# The interactive login-node session hung twice building OCDetect at >7 files
# (fork-based multiprocessing, reproducible), hence doing this via SLURM
# instead of retrying interactively. The other 10 sources in that cache dir
# are untouched symlinks into the original mae_multidata_cache/ -- this job
# only touches ocdetect_{train,val}_{x,mask}.npy.

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

OUT_DIR="/projects/b53017/vbc1591/mae_multidata_cache_ocdetect_splitfix"
mkdir -p "$OUT_DIR" logs

# Remove the previous (sample_fraction=0.01) ocdetect cache entries so the
# build script doesn't skip them as "already exists" -- everything else in
# this directory (symlinks to the other 10 sources) is left alone.
rm -f "$OUT_DIR"/ocdetect_*.npy "$OUT_DIR"/ocdetect_*.tmp.npy "$OUT_DIR"/ocdetect_*.progress.json

python scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml \
    --out-dir "$OUT_DIR" \
    --sources ocdetect \
    --num-workers 4

python - "$OUT_DIR" <<'PY'
import sys
import numpy as np
out_dir = sys.argv[1]
tr = np.load(f"{out_dir}/ocdetect_train_x.npy", mmap_mode="r")
va = np.load(f"{out_dir}/ocdetect_val_x.npy", mmap_mode="r")
print(f"RESULT: ocdetect train={len(tr)} windows, val={len(va)} windows")
PY
