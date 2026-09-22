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

# Preflight: the un-normalised cache twin must hold the five sources, the index file must fit its concatenation and
# the composition must match the reference arm (launch/design_ablation/make_idx_ref_v3raw.py writes both files and a
# JSON report). Fail before training, not after.
python - <<'PY'
import os, sys, json, numpy as np
raw="/projects/b53017/vbc1591/mae_multidata_cache_v3raw"
rep=json.load(open("/projects/b53017/vbc1591/diversity_saturation/corpus_ablation/idx_ref_v3raw.json"))
idx=np.load("/projects/b53017/vbc1591/diversity_saturation/corpus_ablation/idx_ref_v3raw.npy")
tot=0; bad=0
for s in ("realdata","ocdetect","eatingspeed","motionx","sonic"):
    a=np.load(f"{raw}/{s}_train_x.npy",mmap_mode="r"); tot+=a.shape[0]
    x=np.asarray(a[:: max(1,len(a)//1000)][:1000],dtype=np.float32)
    g=float(np.linalg.norm(x[:,:3].mean(-1),axis=1).mean()); w=float(np.percentile(np.linalg.norm(x[:,3:6],axis=1),99))
    r=rep["per_source"][s]; ok = a.shape[0]==r["raw_pool"] and 5.0<g<15.0 and w<40.0
    bad+= not ok
    print(f"[preflight] {s:12s} pool {a.shape[0]:>8d} selected {r['selected']:>6d} (ref {r['ref_count']:>6d}; {r['rule']}) | mean |acc| {g:5.2f} m/s^2, p99 |gyro| {w:5.2f} rad/s {'OK' if ok else 'BAD'}")
print(f"[preflight] index file {len(idx)} windows (reference {rep['reference_total']}), pool {tot}, max index {idx.max()}")
sys.exit(1 if (bad or idx.max()>=tot or len(idx)!=rep["total"] or abs(len(idx)-rep["reference_total"])>0.01*rep["reference_total"]) else 0)
PY

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/design_ablation/vit1d_ssl_mae_abl_nozscore.yaml \
    --run-name vit1d_ssl_mae_abl_nozscore \
    --resume latest
