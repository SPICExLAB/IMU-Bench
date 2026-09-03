#!/usr/bin/env python3
"""Convert the 5src 30 s raw cache into the Inertia-1 pretraining corpus format.

Input : /projects/b53017/vbc1591/inertia1_5src30s_raw/{src}_{train,val}_x.npy  [N, 6, 1500] float32,
        raw FLU m/s^2 @ 50 Hz (same five sources as mae_5src, incl. the realdata leak — deliberate:
        the comparison target mae_5src trained on it too).
Output: <out>/participants/<src>_<k>.{npy,json} — fp16 [M, 3, 600] windows (30 s @ 20 Hz, acc in g),
        pseudo-participants of <=128 windows so the pretrain script's 2 % participant hash-split works.
Per-source hour caps mirror the mae_5src cache composition (even subsample when over).
"""
import json, os, sys
import numpy as np
from scipy.signal import resample_poly

SRC_DIR = "/projects/b53017/vbc1591/inertia1_5src30s_raw"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/projects/b53017/vbc1591/inertia1_5src20hz"
CAPS_H = {"realdata": 150.0, "ocdetect": 14.0, "eatingspeed": 92.0, "motionx": 1e9, "sonic": 55.0}
G = 9.80665; CHUNK = 128
os.makedirs(os.path.join(OUT, "participants"), exist_ok=True)
rng = np.random.RandomState(42); total_h = 0.0
for src, cap in CAPS_H.items():
    parts = []
    for split in ("train", "val"):
        p = os.path.join(SRC_DIR, f"{src}_{split}_x.npy")
        if os.path.exists(p): parts.append(np.load(p, mmap_mode="r"))
    if not parts:
        print(f"[5src-corpus] {src}: MISSING", flush=True); continue
    n = sum(len(a) for a in parts); cap_n = int(cap * 3600 / 30)
    keep_per = [np.arange(len(a)) for a in parts]
    if n > cap_n:                       # even subsample across the concatenated pool
        sel = np.linspace(0, n - 1, cap_n).round().astype(np.int64); off = 0; keep_per = []
        for a in parts:
            keep_per.append(sel[(sel >= off) & (sel < off + len(a))] - off); off += len(a)
    kept = 0
    for ai, (a, keep) in enumerate(zip(parts, keep_per)):
        for ci in range(0, len(keep), CHUNK):
            idx = keep[ci:ci + CHUNK]
            if len(idx) == 0: continue
            x = np.asarray(a[idx][:, :3, :], dtype=np.float32)          # acc only, FLU
            y = resample_poly(x, 2, 5, axis=-1).astype(np.float32) / G  # 50 -> 20 Hz, m/s^2 -> g
            rv = y.var(axis=-1).max(axis=1)
            y = y[rv > 0]
            if len(y) == 0: continue
            seqn = f"{src}_{ai}_{ci // CHUNK:05d}"
            np.save(os.path.join(OUT, "participants", seqn + ".npy"), y.astype(np.float16))
            json.dump({"seqn": seqn, "n_windows": int(len(y)), "source": src},
                      open(os.path.join(OUT, "participants", seqn + ".json"), "w"))
            kept += len(y)
    print(f"[5src-corpus] {src}: {kept} windows = {kept*30/3600:.1f} h (cache had {n}, cap {cap if cap<1e8 else 'none'} h)", flush=True)
    total_h += kept * 30 / 3600
print(f"[5src-corpus] TOTAL {total_h:.1f} h -> {OUT}", flush=True)
