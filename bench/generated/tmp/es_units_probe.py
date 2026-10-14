import numpy as np, logging, yaml, sys, time
logging.basicConfig(level=logging.WARNING)
from imu_lm.data.combined_mae_multidata_loader import _build_eatingspeed
cfg=yaml.safe_load(open("configs/base.yaml")); m=yaml.safe_load(open("configs/design_ablation/cache_v3raw.yaml"))
def merge(a,b):
    for k,v in b.items():
        a[k]=merge(a.get(k,{}),v) if isinstance(v,dict) and isinstance(a.get(k),dict) else v
    return a
cfg=merge(cfg,m)
src=dict(cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["eatingspeed"])
for sub in ("MO","FD-II","FD-I"):
    t0=time.time(); s=dict(src); s["subsets"]=[sub]
    tr,va=_build_eatingspeed(cfg,s)
    n=len(tr); idx=np.random.default_rng(0).choice(n,size=min(n,400),replace=False)
    acc=[];gyr=[]
    for i in idx:
        x=tr[int(i)]; x=x[0] if isinstance(x,(tuple,list)) else x; x=np.asarray(x)
        acc.append(np.linalg.norm(x[:3],axis=0)); gyr.append(np.linalg.norm(x[3:6],axis=0))
    acc=np.concatenate(acc); gyr=np.concatenate(gyr)
    print(f"{sub}: n_windows {n}  |acc| p50 {np.percentile(acc,50):.2f} p99 {np.percentile(acc,99):.2f} | |gyro| p50 {np.percentile(gyr,50):.3f} p99 {np.percentile(gyr,99):.2f} max {gyr.max():.1f}  ({time.time()-t0:.0f}s)", flush=True)
