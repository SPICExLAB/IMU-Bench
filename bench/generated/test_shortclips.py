import sys, yaml, numpy as np, torch
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from imu_lm.utils.helpers import deep_update
cfg = deep_update(yaml.safe_load(open("configs/base.yaml")),
                  yaml.safe_load(open("configs/nhanes_scale/cache_raw.yaml")))
cfg["data"]["variable_length"]["include_short_clips"] = True
cfg["data"]["num_workers"] = 0
from imu_lm.data.motionx_wrist_loader_FLU import make_motionx_wrist_loaders_flu
L = make_motionx_wrist_loaders_flu(cfg=cfg, motionx_root="/projects/b53017/vbc1591/Motion-Xplusplus", add_gravity=True)
ds = L["train_loader"].dataset
wv = ds._window_valid
print(f"MotionX with short clips: {len(ds)} windows | short (<512): {(wv<512).sum()} | full: {(wv>=512).sum()}")
print(f"  hours: {wv.sum()/50/3600:.1f} (was 6.7 h with short clips discarded)")
si = int(np.argmax(wv < 512))
x, y = ds[si]
v = int(wv[si])
print(f"  sample window: valid {v}/512, tail all-zero: {bool((x[:, v:].abs().sum() == 0).item()) if v < 512 else '-'}")
from imu_lm.data.combined_mae_multidata_loader import XOnlyDataset
w = XOnlyDataset(ds, "motionx")
item = w[si]
print(f"  XOnlyDataset item arity: {len(item)} (3 = temporal mask attached), mask sum {int(item[2].sum()) if len(item)>2 else '-'}")
