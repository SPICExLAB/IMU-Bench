#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=900G
#SBATCH --job-name=build_cache_stage1_acconly
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage-1 cache: capture24 (full) + NHANES @250,000h (~1,301 participants).
#
# WHY genhimem AND NOT A GPU PARTITION: cache building is pure CPU/IO (parse,
# resample, window) -- no GPU needed for these two sources (neither runs SMPL
# FK). The binding constraint is RAM, not disk or GPU:
# nhanes_wrist_loader._collect_windows accumulates every window in memory
# before anything is written, holding [M,3,T] fp32 = 6,144 bytes/window. At
# 250,000h that is ~1.08 TB live, which does NOT fit the 188 GB gengpu nodes
# (it would OOM exactly like the 2026-07-24 smoke test did at 64 GB).
#
# NOTE (2026-07-31): that function used to finish with np.concatenate(X_chunks),
# which holds the whole chunk list AND the output alive at once -- peak 2x the
# final array, i.e. ~2.16 TB here, more than the largest node on this cluster
# (1.51 TB), and it would have OOM'd only at the very END after ~4.4 h of
# irreversible decompression. It now preallocates and copies chunk-by-chunk,
# freeing each chunk as consumed, so peak is ~1.0x (verified bit-identical
# output). 1200G therefore has ~140 GB of real headroom over the 1.08 TB peak.
#
# SIZING -- why 250,000h and not "all of NHANES":
#   all 14,693 participants = 1.99e9 windows = 2,823,250h
#     -> 12.2 TB collection RAM  (no node has this)
#     -> 24.4 TB cache on disk   (only 10 TB free on /gpfs)
#     -> ~25 h decompression even at 16 workers
#   Using the full dataset is therefore not a config change -- it needs a
#   sharded/streaming loader (webdataset-style) so windows are never all
#   resident at once. 250,000h is the largest that fits the biggest node
#   available, and it is ~50x the Section 11 pool, which is precisely the
#   scale Section 11's fitted scaling law said was needed to halve the
#   remaining reducible loss (~258,000h). So this run directly tests that
#   extrapolation.
#
# COST: ~1,301 participants x ~98 s (tar.bz2 has no random access, so each is
# fully decompressed) / 8 pool workers = ~4.4 h, plus capture24. Cache is
# ~2.16 TB. 2-day limit gives ample margin.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
OUT="/projects/b53017/vbc1591/mae_multidata_cache_stage1_acconly"; mkdir -p "$OUT"

df -h /projects/b53017/vbc1591 | tail -1
free -g | head -2

python scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_stage1_acconly_flu.yaml \
    --out-dir "$OUT" --sources nhanes --num-workers 8

python - "$OUT" <<'PY'
import sys, os, numpy as np
o=sys.argv[1]; H=5.12; t=0
for n in ["capture24","nhanes"]:
    p=f"{o}/{n}_train_x.npy"
    if not os.path.exists(p): print(f"  {n:<10s} MISSING"); continue
    a=np.load(p, mmap_mode="r"); t+=len(a)
    print(f"  {n:<10s} train={len(a):>13,d} ({len(a)*H/3600:>9,.0f}h)  {a.nbytes/1e12:.2f} TB")
print(f"  {'TOTAL':<10s} train={t:>13,d} ({t*H/3600:>9,.0f}h)")
PY
df -h /projects/b53017/vbc1591 | tail -1
