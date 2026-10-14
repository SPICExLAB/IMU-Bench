#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=40G
#SBATCH --job-name=es_left_probe
#SBATCH --output=bench/generated/tmp/%x_%j.out
#SBATCH --error=bench/generated/tmp/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate
python - <<'PY'
import numpy as np, logging, yaml
logging.basicConfig(level=logging.ERROR)
from imu_lm.data import eatingspeed_wrist_loader as L
d=L.load_subset("/projects/b53017/vbc1591/EatingSpeed","FD-II",hand="both")
for i in range(len(d["X_L"])):
    for h in ("L","R"):
        a=np.asarray(d[f"X_{h}"][i],dtype=float); g=np.linalg.norm(a[:,3:6],axis=1); acc=np.linalg.norm(a[:,:3],axis=1)
        nan=np.isnan(a).mean()
        print(f"FD-II p{i:02d} {h}: T={len(a):8d} nan={nan:.4f} |acc| p50 {np.percentile(acc,50):6.2f} p99 {np.percentile(acc,99):6.2f} | |gyro| p50 {np.percentile(g,50):.3f} p99 {np.percentile(g,99):7.2f} max {g.max():7.1f} | col p99 {np.round(np.percentile(np.abs(a),99,axis=0),1)}", flush=True)
    same = np.array_equal(np.asarray(d["X_L"][i]), np.asarray(d["X_R"][i]))
    if same: print(f"   p{i:02d}: LEFT == RIGHT (identical arrays)", flush=True)
# windows through the cache path, per hand
from imu_lm.data.combined_mae_multidata_loader import _build_eatingspeed
def merge(a,b):
    for k,v in b.items(): a[k]=merge(a.get(k,{}),v) if isinstance(v,dict) and isinstance(a.get(k),dict) else v
    return a
cfg=merge(yaml.safe_load(open("configs/base.yaml")), yaml.safe_load(open("configs/design_ablation/cache_v3raw.yaml")))
src=dict(cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["eatingspeed"])
for h in ("L","R"):
    s=dict(src); s["subsets"]=["FD-II"]; s["hand"]=h; tr,va=_build_eatingspeed(cfg,s)
    n=len(tr); idx=np.random.default_rng(0).choice(n,size=min(n,800),replace=False); gy=[]; big=0
    for i in idx:
        x=tr[int(i)]; x=x[0] if isinstance(x,(tuple,list)) else x; x=np.asarray(x); g=np.linalg.norm(x[3:6],axis=0); gy.append(g); big+= g.max()>40
    gy=np.concatenate(gy); print(f"== FD-II hand {h}: cache-path windows {n}; |gyro| p50 {np.percentile(gy,50):.3f} p99 {np.percentile(gy,99):.2f} max {gy.max():.1f}; windows with |gyro|>40: {big}/{len(idx)}", flush=True)
PY
