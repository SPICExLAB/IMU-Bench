#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --job-name=nhanes_smoke_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# First real end-to-end test of the new NHANES PAX loader
# (imu_lm/data/nhanes_wrist_loader.py, wired into combined_mae_multidata_loader.py
# as source "nhanes") at production scale: builds the nhanes source only, via
# scripts/build_multidata_cache.py, using the sources.nhanes block already in
# configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml
# (sample_fraction=0.01 -> ~147/14,693 participants, max_hours=2000 safety cap).
#
# Why SLURM and not interactive: each participant .tar.bz2 archive takes
# ~100s to process (confirmed directly: 97.9s -> 135,105 windows for one real
# participant) -- bz2 has no random-access support, so tarfile.getmembers()
# must decompress the whole ~150MB archive just to build its member index.
# ~147 participants / 8 parallel workers x ~100s each is easily a login-node-
# hostile ~30+ min of sustained work, consistent with every other heavy
# raw-data build in this investigation needing SLURM rather than the
# interactive session.

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

OUT_DIR="/projects/b53017/vbc1591/mae_multidata_cache_nhanes_smoke"
mkdir -p "$OUT_DIR" logs

python scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml \
    --out-dir "$OUT_DIR" \
    --sources nhanes \
    --num-workers 8

python - "$OUT_DIR" <<'PY'
import sys
import numpy as np
out_dir = sys.argv[1]
tr = np.load(f"{out_dir}/nhanes_train_x.npy", mmap_mode="r")
va = np.load(f"{out_dir}/nhanes_val_x.npy", mmap_mode="r")
hop_s = 5.12
n = len(tr) + len(va)
print(f"RESULT: nhanes train={len(tr)} val={len(va)} windows (~{n*hop_s/3600:.1f}h effective)")
PY
