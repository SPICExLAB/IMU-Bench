#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
#SBATCH --job-name=build_cache_stage2_accgyro
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Stage-1 cache: capture24 + NHANES @48,000h (~250 participants).
# NHANES .tar.bz2 has no random access -> ~98s/participant full decompression,
# so ~6.8h of pure I/O; cache is ~432 GB. 24h/400G budgeted with margin.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
OUT="/projects/b53017/vbc1591/mae_multidata_cache_stage2_accgyro"; mkdir -p "$OUT"
python scripts/build_multidata_cache.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_stage2_accgyro_flu.yaml \
    --out-dir "$OUT" --sources ocdetect eatingspeed sonic motionx --num-workers 8
python - "$OUT" <<'PY'
import sys,os,numpy as np
o=sys.argv[1];H=5.12;t=0
for n in ["ocdetect","eatingspeed","sonic","motionx"]:
    p=f"{o}/{n}_train_x.npy"
    if not os.path.exists(p): print(f"  {n}: MISSING"); continue
    a=np.load(p,mmap_mode="r"); t+=len(a)
    print(f"  {n:<10s} train={len(a):>12,d} ({len(a)*H/3600:>8,.0f}h)")
print(f"  {'TOTAL':<10s} train={t:>12,d} ({t*H/3600:>8,.0f}h)")
PY
