import numpy as np, logging, yaml, sys
logging.disable(logging.WARNING)
def du(a,b):
    for k,v in b.items():
        if isinstance(v,dict) and isinstance(a.get(k),dict): du(a[k],v)
        else: a[k]=v
    return a
cfg=du(yaml.safe_load(open("configs/base.yaml")), yaml.safe_load(open("configs/design_ablation/cache_v3raw.yaml")))
src=dict(cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["eatingspeed"]); src["max_participants_per_subset"]=2; src["subsets"]=["FD-I"]
from imu_lm.data.combined_mae_multidata_loader import _build_eatingspeed
from imu_lm.data.unit_check import check_gyro_units
tr,va=_build_eatingspeed(cfg,src)
print("train windows", len(tr), flush=True)
try:
    print(check_gyro_units(tr,"eatingspeed",n_windows=600), flush=True)
except Exception as e:
    print("CHECK FAILED:", e, flush=True)
xs=np.stack([np.asarray(tr[i][0]) for i in range(0,len(tr),max(1,len(tr)//400))])
gn=np.linalg.norm(xs[:,3:6],axis=1); print("sampled dataset items: gyro p99", np.percentile(gn,99), "max", gn.max(), "acc p50", np.percentile(np.linalg.norm(xs[:,:3],axis=1),50))
bad=np.where(gn.max(axis=1)>40)[0]; print("windows with |gyro|>40:", len(bad), "of", len(xs))
if len(bad): i=bad[0]; print("example window", i, "gyro abs max per channel", np.abs(xs[i,3:6]).max(axis=1), "acc abs max", np.abs(xs[i,:3]).max(axis=1))
