#!/usr/bin/env python3
"""Preflight for a native-preprocessing arm: fail before training, not after.

    python launch/nativenorm/preflight.py configs/nativenorm/<arm>.yaml

Checks, in order:
  1. every enabled source exists in the cache the config names, for both splits;
  2. the train index file fits the concatenated train pool exactly;
  3. each per-source val index file fits its own val array;
  4. the input actually reaches the model UN-z-scored -- the whole point of the arm.
     A per-window channel mean of ~0 and std of ~1 means the z-score is still on and the
     arm would silently be a duplicate of the row it is supposed to contrast with;
  5. for a config naming a stage-1 motif-distance run, that its checkpoint exists.
"""
import os, sys, logging
import numpy as np
import yaml
import torch

cfg_path = sys.argv[1]
cfg = yaml.safe_load(open(cfg_path))
mm = cfg["objective"]["mae_multidata_varlen_altattn_flu"]
cache = mm["cache_dir"]
enabled = [n for n, s in (mm.get("sources") or {}).items() if (s or {}).get("enabled", True)]
fail = []

# 1. cache files
pool = 0
for n in enabled:
    for split in ("train", "val"):
        for suf in ("x", "mask"):
            p = f"{cache}/{n}_{split}_{suf}.npy"
            if not os.path.isfile(p):
                fail.append(f"missing {p}")
    xp = f"{cache}/{n}_train_x.npy"
    if os.path.isfile(xp):
        pool += np.load(xp, mmap_mode="r").shape[0]
print(f"[preflight] cache {cache}: {len(enabled)} sources, train pool {pool} windows")

# 2. train index file
idxf = mm.get("subset_indices_file")
if mm.get("scaling_mode") == "index_file":
    if not idxf or not os.path.isfile(idxf):
        fail.append(f"scaling_mode=index_file but subset_indices_file missing: {idxf}")
    else:
        idx = np.load(idxf)
        print(f"[preflight] train index {os.path.basename(idxf)}: {len(idx)} windows, max {idx.max()} (pool {pool})")
        if idx.max() >= pool:
            fail.append(f"train index max {idx.max()} >= pool {pool} -- index space does not match this source set")

# 3. per-source val index files
val_total = 0
for n in enabled:
    vp = ((mm.get("sources") or {}).get(n) or {}).get("subset_index_file_val")
    if not vp:
        continue
    if not os.path.isfile(vp):
        fail.append(f"missing val index {vp}"); continue
    keep = np.load(vp)
    have = np.load(f"{cache}/{n}_val_x.npy", mmap_mode="r").shape[0]
    val_total += len(keep)
    if keep.max() >= have:
        fail.append(f"{n}: val index max {keep.max()} >= {have} val windows")
print(f"[preflight] val after per-source subsetting: {val_total} windows")

# 4. the input must NOT be per-window z-scored
logging.basicConfig(level=logging.ERROR)
from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders
probe_cfg = yaml.safe_load(open(cfg_path))
probe_cfg.setdefault("data", {})["num_workers"] = 0
r = make_combined_mae_multidata_loaders(probe_cfg)
b = next(iter(r["val_loader"]))
x = b[0] if isinstance(b, (tuple, list)) else b
m = float(x[:, :3].mean(-1).abs().mean())
s = float(x[:, :3].std(-1).mean())
print(f"[preflight] per-window acc |mean| {m:.4f}, std {s:.4f} "
      f"(per-window z-score would be 0.000 / 1.000)")
if m < 0.01 and abs(s - 1.0) < 0.02:
    fail.append("input is still per-window z-scored -- this arm would duplicate its z-score twin")

# 5. stage-1 checkpoint for RelCon
md = (cfg.get("objective") or {}).get("motifdist_run")
if md:
    ck = f"runs/{md}/checkpoints/best.pt"
    print(f"[preflight] motif-distance stage 1: {ck} {'found' if os.path.isfile(ck) else 'MISSING'}")
    if not os.path.isfile(ck):
        fail.append(f"stage-1 checkpoint {ck} not found -- run relcon_motifdist_5src_native first")

# 6. fft stem, when the arm asks for one
it = (cfg.get("vit1d") or {}).get("input_transform", "none")
if it != "none":
    from imu_lm.models.ViT1D.model import ViT1DEncoder
    enc = ViT1DEncoder(cfg)
    ps = enc.patch_size
    k = enc.patch_embed.kernel_size
    print(f"[preflight] input_transform={it}: patch_size {ps}, projection kernel {k}")
    if it == "fft" and k != 2 * ps:
        fail.append(f"fft stem expected a {2*ps}-wide kernel, got {k}")
    tok, _ = enc.forward_tokens(torch.randn(2, 6, 256))
    if tok.shape[1] != 6 * (256 // ps):
        fail.append(f"fft stem changed the token count: {tok.shape[1]} != {6*(256//ps)}")

if fail:
    print("\n[preflight] FAILED:")
    for f_ in fail:
        print("   -", f_)
    sys.exit(1)
print("[preflight] OK")
