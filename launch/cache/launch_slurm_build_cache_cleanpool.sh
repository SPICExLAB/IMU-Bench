#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --job-name=build_cache_cleanpool
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ONE-TIME cache build for the leakage-free "clean pool" (6 sources, no
# overlap with any downstream eval target) -- see the header of
# configs/vit1d_ssl_mae_varlen_altattn_cleanpool_flu.yaml for the rationale.
#
# Cost is dominated by NHANES: its archives are .tar.bz2, which have no random
# access, so tarfile.getmembers() must fully decompress each one (~98s per
# participant measured). max_hours=4800 needs ~25 participants read -> ~45min,
# but the streaming cap in _collect_windows only stops once the budget is hit,
# so budget generously. capture24/ocdetect/eatingspeed are large but fast;
# sonic is slow per-session (~135k BVH sessions at 25-40 sessions/s -> ~1.5h).
# 12h with 300G covers all of it with real margin (the v3 build of only
# ocdetect/capture24/eatingspeed took 22min at 300G).
#
# NOTE: the cache stores CLEAN (deterministically preprocessed, NOT augmented)
# windows -- build_multidata_cache.py strips augmentation via
# _cfg_without_augmentation, and augmentation is re-applied fresh at
# __getitem__ time, train-split-only since the 2026-07-28 gating fix.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

OUT_DIR="/projects/b53017/vbc1591/mae_multidata_cache_cleanpool"
mkdir -p "$OUT_DIR"

python scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_cleanpool_flu.yaml \
    --out-dir "$OUT_DIR" \
    --sources capture24 ocdetect eatingspeed sonic motionx nhanes \
    --num-workers 8

echo "=== built cache contents ==="
python - "$OUT_DIR" <<'PY'
import sys, os, numpy as np
out = sys.argv[1]; HOP = 5.12
tot_tr = tot_va = 0
for name in ["capture24","ocdetect","eatingspeed","sonic","motionx","nhanes"]:
    tr_p, va_p = f"{out}/{name}_train_x.npy", f"{out}/{name}_val_x.npy"
    if not os.path.exists(tr_p):
        print(f"  {name:<12s} MISSING"); continue
    tr = np.load(tr_p, mmap_mode="r"); va = np.load(va_p, mmap_mode="r") if os.path.exists(va_p) else []
    tot_tr += len(tr); tot_va += len(va)
    print(f"  {name:<12s} train={len(tr):>10,d} ({len(tr)*HOP/3600:>7,.0f}h)  val={len(va):>9,d}")
print(f"  {'TOTAL':<12s} train={tot_tr:>10,d} ({tot_tr*HOP/3600:>7,.0f}h)  val={tot_va:>9,d}")
PY
