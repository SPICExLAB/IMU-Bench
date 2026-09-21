#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --qos=buyin
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --job-name=vit1d_ssl_mae_abl_nozscore
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Design ablation 'nozscore': physical-unit input instead of the per-window z-score; otherwise identical to corpus arm 'ref'.
# Same recipe/budget as grid cell k1000_f10 (50,000 steps). --resume latest lets a TIMEOUT resubmit as-is.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

# Preflight: the un-normalised cache must be a row-for-row twin of mae_multidata_cache_v3, or the remapped index file
# (idx_ref_v3raw.npy) would select different windows than the reference arm. Fail before training, not after.
python - <<'PY'
import os, sys, json, numpy as np
raw="/projects/b53017/vbc1591/mae_multidata_cache_v3raw"; ref="/projects/b53017/vbc1591/mae_multidata_cache_v3"
bad=0; tot=0
for s in ("realdata","ocdetect","eatingspeed","motionx","sonic"):
    for sp in ("train","val"):
        a=np.load(f"{raw}/{s}_{sp}_x.npy",mmap_mode="r"); b=np.load(f"{ref}/{s}_{sp}_x.npy",mmap_mode="r")
        ok=a.shape==b.shape; bad+=not ok; tot+=a.shape[0] if sp=="train" else 0
        x=np.asarray(a[:: max(1,len(a)//2000)][:2000]); g=np.linalg.norm(x[:,:3].mean(-1),axis=1).mean(); w=np.abs(x[:,3:]).mean()
        print(f"[preflight] {s:12s} {sp:5s} raw={a.shape} v3={b.shape} {'OK' if ok else 'MISMATCH'} | mean |acc| {g:.2f}  mean |gyro| {w:.3f}")
idx=np.load("/projects/b53017/vbc1591/diversity_saturation/corpus_ablation/idx_ref_v3raw.npy")
print(f"[preflight] pool {tot} windows, index file {len(idx)} windows, max index {idx.max()}")
sys.exit(1 if (bad or idx.max()>=tot or len(idx)!=197094) else 0)
PY

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/design_ablation/vit1d_ssl_mae_abl_nozscore.yaml \
    --run-name vit1d_ssl_mae_abl_nozscore \
    --resume latest
