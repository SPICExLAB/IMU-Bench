#!/usr/bin/env python
"""14_characterise_motion_types.py -- give the K=1000 "motion types" a physical reading.

The types are geometric (k-means centroids in the encoder's embedding space),
so on their own they are unnamed. This script fetches raw windows back out of
the pool by index and asks what physically separates them: how much linear
acceleration, how much rotation, how fast, how periodic.

UNITS CAVEAT: base.yaml preprocesses every window with per-channel z-scoring
(preprocess.normalize.method=zscore), so absolute amplitude and the gravity
offset are already divided out before the encoder ever sees a window. These
statistics are therefore SHAPE descriptors in normalised units, not g or
rad/s -- which also means a motion type cannot encode "vigorous vs gentle",
only waveform shape and cross-axis structure.

Per window (6ch x 512 @ 50 Hz = 10.24 s), on z-scored channels:
  acc_dyn      std of |acc| over time      -- burstiness of translation
  gyro_mag     mean of |gyro|              -- rotation content (0 => gyro absent)
  dom_freq     argmax of the |acc| spectrum (0.2-10 Hz) -- movement tempo
  spec_ent     spectral entropy of |acc|   -- periodic (low) vs erratic (high)
  still_frac   fraction of samples with |acc| within 5% of its median

Reports: variance in each statistic explained by type identity (eta^2 -- if
this is near 0 the types are not physical), and the extreme types on each
axis with their dominant source.

Outputs analysis/diversity_saturation/out/motion_type_semantics.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "base.yaml")
MODEL_CONFIG = os.path.join(
    REPO_ROOT, "configs", "vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe100pct.yaml")

PER_TYPE = 12          # windows sampled per cluster
FS = 50.0
SEED = 42

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


def window_stats(x):
    """x: [6, T] float32 (acc xyz, gyro xyz). Returns dict of scalars."""
    acc, gyr = x[:3], x[3:]
    amag = np.linalg.norm(acc, axis=0)
    gmag = np.linalg.norm(gyr, axis=0)
    a = amag - amag.mean()
    # spectrum of the demeaned acc magnitude, 0.2-10 Hz band
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a)))) ** 2
    freqs = np.fft.rfftfreq(len(a), 1.0 / FS)
    band = (freqs >= 0.2) & (freqs <= 10.0)
    sb = spec[band]
    if sb.sum() <= 0:
        dom, sent = 0.0, 0.0
    else:
        dom = float(freqs[band][int(np.argmax(sb))])
        p = sb / sb.sum()
        p = p[p > 0]
        sent = float(-(p * np.log(p)).sum() / np.log(len(sb)))
    g = float(np.median(amag))
    return {
        "acc_dyn": float(amag.std()),
        "gyro_mag": float(gmag.mean()),
        "dom_freq": dom,
        "spec_ent": sent,
        "still_frac": float((np.abs(amag - g) < 0.05 * max(g, 1e-6)).mean()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = json.load(open(os.path.join(WORK_DIR, "pool_manifest.json")))
    names, cum = manifest["sources"], np.array(manifest["cumulative_sizes"])
    labels = np.load(os.path.join(WORK_DIR, "labels11src_k1000.npy")).astype(np.int32)
    n = len(labels)

    rng = np.random.default_rng(SEED)
    by_type = [np.where(labels == c)[0] for c in range(1000)]
    picks, pick_type = [], []
    for c in range(1000):
        m = by_type[c]
        if len(m) == 0:
            continue
        take = rng.choice(m, size=min(PER_TYPE, len(m)), replace=False)
        picks.append(take)
        pick_type.append(np.full(len(take), c))
    picks = np.concatenate(picks)
    pick_type = np.concatenate(pick_type)
    order = np.argsort(picks)                    # sequential reads over the cache
    picks, pick_type = picks[order], pick_type[order]
    print(f"[sample] {len(picks):,} windows over {len(set(pick_type.tolist()))} types", flush=True)

    ds = build_dataset()
    keys = ["acc_dyn", "gyro_mag", "dom_freq", "spec_ent", "still_frac"]
    S = np.zeros((len(picks), len(keys)), dtype=np.float32)
    t0 = time.time()
    for i, gi in enumerate(picks):
        item = ds[int(gi)]
        x = item[0] if isinstance(item, (tuple, list)) else item
        x = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        st = window_stats(x.astype(np.float32))
        S[i] = [st[k] for k in keys]
        if i % 2000 == 0:
            print(f"[sample] {i}/{len(picks)} ({i / max(1e-9, time.time() - t0):.0f}/s)", flush=True)

    np.savez(os.path.join(WORK_DIR, "type_semantics_raw.npz"),
             S=S, pick_type=pick_type, picks=picks, keys=np.array(keys))

    # dominant source per type, for reading the extremes
    src_of_type = {}
    src_idx = np.searchsorted(cum, np.arange(n), side="right")
    tab = np.zeros((len(names), 1000), dtype=np.int64)
    np.add.at(tab, (src_idx, labels), 1)
    for c in range(1000):
        if tab[:, c].sum():
            src_of_type[c] = names[int(np.argmax(tab[:, c]))]

    # eta^2: fraction of each statistic's variance explained by type identity.
    # capture24 is accelerometer-only (gyro zero-filled), so it forms its own
    # island in embedding space -- reported both with and without it, because
    # the "with" number for gyro_mag is a sensor artifact, not motion structure.
    c24 = np.array([src_of_type.get(c) == "capture24" for c in range(1000)])
    keep = ~c24[pick_type]

    def eta_sq(mask):
        e, mus = {}, {}
        for j, k in enumerate(keys):
            v, pt = S[mask, j], pick_type[mask]
            gm = v.mean()
            means = np.array([v[pt == c].mean() if (pt == c).any() else np.nan
                              for c in range(1000)])
            cnts = np.array([(pt == c).sum() for c in range(1000)])
            ok = cnts > 0
            ssb = float((cnts[ok] * (means[ok] - gm) ** 2).sum())
            sst = float(((v - gm) ** 2).sum())
            e[k] = round(ssb / sst, 4) if sst > 0 else float("nan")
            mus[k] = means
        return e, mus

    eta2, per_type_mean = eta_sq(np.ones(len(S), bool))
    eta2_no_c24, _ = eta_sq(keep)

    extremes = {}
    for k in keys:
        m = per_type_mean[k]
        ok = ~np.isnan(m)
        idx = np.where(ok)[0]
        lo = idx[np.argsort(m[idx])[:5]]
        hi = idx[np.argsort(m[idx])[-5:]][::-1]
        extremes[k] = {
            "lowest": [{"type": int(c), "value": round(float(m[c]), 4),
                        "dominant_source": src_of_type.get(int(c))} for c in lo],
            "highest": [{"type": int(c), "value": round(float(m[c]), 4),
                         "dominant_source": src_of_type.get(int(c))} for c in hi],
        }

    # pooled distribution of the type means -- the span of the axis
    span = {k: {p: round(float(np.nanpercentile(per_type_mean[k], p)), 4)
                for p in (1, 25, 50, 75, 99)} for k in keys}

    out = {"n_sampled": int(len(picks)), "per_type": PER_TYPE, "fs_hz": FS,
           "eta_squared_variance_explained_by_type": eta2,
           "eta_squared_excluding_capture24_types": eta2_no_c24,
           "n_capture24_dominated_types": int(c24.sum()),
           "type_mean_percentiles": span, "extreme_types": extremes,
           "units_caveat": ("channels are per-window z-scored by base.yaml before the "
                            "encoder sees them, so amplitude and gravity are divided out; "
                            "all statistics below are shape descriptors in normalised "
                            "units, NOT g or rad/s"),
           "stat_definitions": {
               "acc_dyn": "std of |acc| (z-scored ch.) -- burstiness of translation",
               "gyro_mag": "mean |gyro| (z-scored ch.) -- rotation content; 0 => gyro absent",
               "dom_freq": "peak frequency of |acc| in 0.2-10 Hz (Hz) -- tempo",
               "spec_ent": "normalised spectral entropy of |acc| -- 0 periodic, 1 broadband",
               "still_frac": "fraction of samples within 5% of median |acc| -- stillness"}}
    p = os.path.join(OUT_DIR, "motion_type_semantics.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\neta^2 (variance explained by type identity)   "
          f"[{int(c24.sum())} capture24-dominated types excluded in col 2]:")
    for k in keys:
        print(f"  {k:11s} all={eta2[k]:.3f}  no-c24={eta2_no_c24[k]:.3f}   "
              f"span p1..p99 = {span[k][1]} .. {span[k][99]}")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
