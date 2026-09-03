#!/usr/bin/env python
"""16_type_atlas.py -- pull the medoid window of each macro-family back out as a waveform.

The strongest answer to "what IS a motion type" is not a statistic, it is the
signal. For each of the 10 Ward macro-families this finds

  family centroid -> the type nearest it -> the window nearest that type's centroid

and saves the raw 6-channel waveform, so the families can be drawn rather than
described. Three exemplars per family (medoid + 2 nearest neighbours in
embedding space) so a reader can see whether the family is coherent.

Outputs /projects/b53017/vbc1591/diversity_saturation/type_atlas.npz
        analysis/diversity_saturation/out/type_atlas_meta.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "base.yaml")
MODEL_CONFIG = os.path.join(
    REPO_ROOT, "configs", "vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe100pct.yaml")

N_FAMILIES = 10
PER_FAMILY = 3
_AUG = ("random_rotation", "gaussian_noise", "jitter", "bias_drift", "sensor_dropout",
        "time_crop", "frequency_mask", "gain_jitter", "axis_bias", "time_warp")


def build_dataset():
    from imu_lm.utils.helpers import deep_update, load_yaml
    from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders
    cfg = deep_update(load_yaml(BASE_CONFIG), load_yaml(MODEL_CONFIG))
    for k in _AUG:
        b = cfg.get(k)
        if isinstance(b, dict):
            b["enabled"] = False
    return make_combined_mae_multidata_loaders(cfg)["train_loader"].dataset


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    man = json.load(open(os.path.join(WORK_DIR, "pool_manifest.json")))
    names, cum, n = man["sources"], np.array(man["cumulative_sizes"]), man["n"]
    labels = np.load(os.path.join(WORK_DIR, "labels11src_k1000.npy")).astype(np.int32)
    z = np.load(os.path.join(WORK_DIR, "pca64_f32.npy"), mmap_mode="r")

    cent = np.zeros((1000, z.shape[1]))
    for c in range(1000):
        cent[c] = np.asarray(z[labels == c]).mean(0)

    from scipy.cluster.hierarchy import linkage, fcluster
    fam = fcluster(linkage(cent, method="ward"), t=N_FAMILIES, criterion="maxclust") - 1

    # family label ordering must match 15_type_distribution.py: gyro-present
    # families first (ordered by tempo), then the accelerometer-only ones
    raw = np.load(os.path.join(WORK_DIR, "type_semantics_raw.npz"), allow_pickle=True)
    S, pt = raw["S"], raw["pick_type"]
    fstat = {f: {"dom_freq": float(S[fam[pt] == f, 2].mean()),
                 "gyro": float(S[fam[pt] == f, 1].mean())} for f in range(N_FAMILIES)}
    gyro_f = sorted([f for f in range(N_FAMILIES) if fstat[f]["gyro"] >= 0.01],
                    key=lambda f: fstat[f]["dom_freq"])
    acc_f = sorted([f for f in range(N_FAMILIES) if fstat[f]["gyro"] < 0.01],
                   key=lambda f: fstat[f]["dom_freq"])
    tag = {f: f"G{i + 1}" for i, f in enumerate(gyro_f)}
    tag.update({f: f"A{i + 1}" for i, f in enumerate(acc_f)})

    src_of = np.searchsorted(cum, np.arange(n), side="right")
    ds = build_dataset()

    waves, meta = [], []
    for f in gyro_f + acc_f:
        members = np.where(fam == f)[0]
        fc = cent[members].mean(0)
        rep_type = int(members[np.argmin(((cent[members] - fc) ** 2).sum(1))])
        widx = np.where(labels == rep_type)[0]
        zw = np.asarray(z[widx])
        d = ((zw - cent[rep_type]) ** 2).sum(1)
        pick = widx[np.argsort(d)[:PER_FAMILY]]
        for rank, gi in enumerate(pick):
            item = ds[int(gi)]
            x = item[0] if isinstance(item, (tuple, list)) else item
            x = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
            waves.append(x.astype(np.float32))
            meta.append({"family": tag[f], "family_raw_id": int(f), "type": rep_type,
                         "rank": rank, "global_idx": int(gi),
                         "source": names[int(src_of[gi])],
                         "type_size": int((labels == rep_type).sum())})
        print(f"{tag[f]}: type {rep_type} ({(labels == rep_type).sum()} windows), "
              f"exemplars from {[m['source'] for m in meta[-PER_FAMILY:]]}", flush=True)

    W = np.stack(waves)
    np.savez_compressed(os.path.join(WORK_DIR, "type_atlas.npz"), waves=W,
                        meta=np.array(json.dumps(meta)))
    with open(os.path.join(OUT_DIR, "type_atlas_meta.json"), "w") as fh:
        json.dump({"per_family": PER_FAMILY, "shape": list(W.shape), "exemplars": meta}, fh, indent=2)
    print(f"\nwrote atlas {W.shape}")


if __name__ == "__main__":
    main()
