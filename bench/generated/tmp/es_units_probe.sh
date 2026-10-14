#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G
#SBATCH --job-name=es_units_probe
#SBATCH --output=bench/generated/tmp/%x_%j.out
#SBATCH --error=bench/generated/tmp/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate
python - <<'PY'
import numpy as np, logging, yaml, time
logging.basicConfig(level=logging.WARNING)
from imu_lm.data.combined_mae_multidata_loader import _build_eatingspeed
from imu_lm.data import eatingspeed_wrist_loader as L
def merge(a,b):
    for k,v in b.items(): a[k]=merge(a.get(k,{}),v) if isinstance(v,dict) and isinstance(a.get(k),dict) else v
    return a
cfg=merge(yaml.safe_load(open("configs/base.yaml")), yaml.safe_load(open("configs/design_ablation/cache_v3raw.yaml")))
src=dict(cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["eatingspeed"])
for sub in ("FD-II","FD-I"):
    t0=time.time()
    d=L.load_subset("/projects/b53017/vbc1591/EatingSpeed", sub, hand="R")
    for i,a in enumerate(d["X_R"]):
        a=np.asarray(a,dtype=float); g=np.linalg.norm(a[:,3:6],axis=1); acc=np.linalg.norm(a[:,:3],axis=1)
        flag = " <-- suspicious" if np.percentile(g,99)>40 else ""
        print(f"{sub} participant {i:2d}: T={len(a):8d} cols={a.shape[1]} |acc| p50 {np.percentile(acc,50):6.2f} |gyro| p50 {np.percentile(g,50):.3f} p99 {np.percentile(g,99):7.2f} max {g.max():8.1f}{flag}", flush=True)
    s=dict(src); s["subsets"]=[sub]; tr,va=_build_eatingspeed(cfg,s)
    n=len(tr); idx=np.random.default_rng(0).choice(n,size=min(n,600),replace=False); gy=[]
    for i in idx:
        x=tr[int(i)]; x=x[0] if isinstance(x,(tuple,list)) else x; x=np.asarray(x); gy.append(np.linalg.norm(x[3:6],axis=0))
    gy=np.concatenate(gy); print(f"== {sub}: cache-path windows {n}; |gyro| p50 {np.percentile(gy,50):.3f} p99 {np.percentile(gy,99):.2f} max {gy.max():.1f}  ({time.time()-t0:.0f}s)", flush=True)
PY
