#!/usr/bin/env python3
"""How much SONIC / MotionX / HumanML3D data is lost to the `if n < T: continue` rule?

Both mocap loaders drop any session shorter than one 10.24 s window. The model supports
variable length (64..512 samples), so those clips are recoverable in principle. Measure the
clip-length distribution to size the payoff before building anything.
"""
import sys, numpy as np, yaml
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from imu_lm.utils.helpers import deep_update
cfg = deep_update(yaml.safe_load(open("configs/base.yaml")),
                  yaml.safe_load(open("configs/nhanes_scale/cache_full.yaml")))
T, HOP = 512, 256
def report(name, lengths):
    L = np.asarray(lengths, dtype=np.int64)
    tot = L.sum() / 50 / 3600
    keep = L[L >= T]
    win_keep = ((keep - T) // HOP + 1).sum() if len(keep) else 0
    short = L[(L >= 64) & (L < T)]
    print(f"\n{name}: {len(L)} clips, {tot:.1f} h of raw motion")
    print(f"  clips >= 512 samples (10.24 s): {len(keep):7d}  ({L[L>=T].sum()/50/3600:8.1f} h)  -> {win_keep} windows currently cached")
    print(f"  clips 64..511 samples          : {len(short):7d}  ({short.sum()/50/3600:8.1f} h)  -> DISCARDED today")
    print(f"  clips < 64 samples             : {(L<64).sum():7d}  ({L[L<64].sum()/50/3600:8.1f} h)  -> unusable either way")
    if tot: print(f"  recoverable fraction of raw hours: {short.sum()/L.sum():.1%}")
for name, mod, fn, kw in (
    ("SONIC (bones-seed)", "imu_lm.data.bones_seed_wrist_full_body_loader_FLU", "make_bones_seed_wrist_full_body_loaders_flu",
     dict(bones_seed_root="/projects/b53017/vbc1591/bones-seed", train_files="all", val_files=None, add_gravity=True)),
    ("MotionX", "imu_lm.data.motionx_wrist_loader_FLU", "make_motionx_wrist_loaders_flu",
     dict(root="/projects/b53017/vbc1591/motionx", add_gravity=True)),
):
    try:
        m = __import__(mod, fromlist=[fn]); L = getattr(m, fn)(cfg=cfg, **kw)
        ds = L["train_loader"].dataset
        base = ds
        while hasattr(base, "_ds") or hasattr(base, "dataset"):
            base = getattr(base, "_ds", None) or getattr(base, "dataset")
        lens = getattr(base, "_session_lengths", None) or getattr(base, "session_lengths", None)
        if lens is None:
            print(f"\n{name}: could not read session lengths from {type(base).__name__}"); continue
        report(name, lens)
    except Exception as e:
        print(f"\n{name}: ERR {type(e).__name__}: {str(e)[:150]}")
