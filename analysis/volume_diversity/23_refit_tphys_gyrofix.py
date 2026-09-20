#!/usr/bin/env python
"""Stage 23: refit the encoder-free motion-type partition (T-phys) with EatingSpeed's
gyroscope converted from deg/s to rad/s.

Why: the raw EatingSpeed Shimmer3 pickles store gyro in deg/s (still-block |gyro|
median 1.34, overall p99 204 -- impossible in rad/s), while OCDetect, SONIC and
Motion-X are in rad/s. No loader converts it. Every gyro-magnitude feature of
02_physical_features.py was therefore ~57x too large for EatingSpeed, which made
EatingSpeed separable by units alone (139 EatingSpeed-only T-phys clusters).
Per-window z-scored pretraining (mae_4src_z and every vd_* arm) is unaffected: a
constant per-channel scale cancels under z-scoring. Only the TYPE DEFINITION is.

Recipe is identical to 03_type_definitions.py except for the unit fix:
  40 rotation-invariant features -> robust median/IQR scaling fitted on a 500k
  sample -> clip +/-8 -> PCA-32 whitened -> MiniBatchKMeans K=1000, seed 42.
Only EatingSpeed's rows are recomputed; the other sources' feature rows are copied.
The old partition is NOT overwritten -- every completed arm keeps its provenance.

Writes $WORK/phys_feats_gyrofix_f32.npy, phys_pca32_gyrofix_f32.npy,
       labels_phys_gyrofix_k1000.npy, phys_scaler_gyrofix.npz
       out/type_definitions_gyrofix.json
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import CACHE_DIR, K_TYPES, OUT_DIR, PCA_PHYS, POOL_N, SEED, SOURCES, WORK_DIR, source_ranges

_spec = importlib.util.spec_from_file_location("physf", os.path.join(HERE, "02_physical_features.py"))
physf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(physf)

DEG2RAD = np.pi / 180.0
CHUNK = 8192



def _assert_eatingspeed_cache_is_deg_s(cache_dir):
    """This script converts EatingSpeed gyro deg/s -> rad/s itself. Caches built after 2026-09-17 already
    hold rad/s (the loader converts) and carry eatingspeed_units.json; converting again would be wrong."""
    import json as _json
    m = os.path.join(cache_dir, "eatingspeed_units.json")
    if os.path.exists(m) and _json.load(open(m)).get("gyro_units") == "rad/s":
        raise RuntimeError(f"{m} says the cache is already rad/s -- drop this script's deg/s conversion")

def main():
    t0 = time.time()
    _assert_eatingspeed_cache_is_deg_s(CACHE_DIR)
    r = source_ranges()
    old_f = np.load(os.path.join(WORK_DIR, "phys_feats_f32.npy"), mmap_mode="r")
    new_path = os.path.join(WORK_DIR, "phys_feats_gyrofix_f32.npy")
    F = np.lib.format.open_memmap(new_path, mode="w+", dtype=np.float32, shape=old_f.shape)
    for s in SOURCES:
        lo, hi = r[s]
        if s != "eatingspeed":
            for a in range(lo, hi, 200_000):
                b = min(hi, a + 200_000); F[a:b] = old_f[a:b]
            continue
        x = np.load(os.path.join(CACHE_DIR, f"{s}_train_x.npy"), mmap_mode="r")
        tpath = os.path.join(CACHE_DIR, f"{s}_train_tmask.npy")
        lens = (np.load(tpath).astype(np.int32) if os.path.exists(tpath)
                else np.full(len(x), x.shape[-1], dtype=np.int32))
        assert len(x) == hi - lo
        for a in range(0, len(x), CHUNK):
            b = min(len(x), a + CHUNK)
            w = np.asarray(x[a:b], dtype=np.float32).copy()
            w[:, 3:] *= DEG2RAD
            F[lo + a:lo + b] = physf.features(w, lens[a:b])
            if (a // CHUNK) % 10 == 0:
                print(f"[feat] eatingspeed {b:,}/{len(x):,}  {time.time()-t0:.0f}s", flush=True)
    F.flush()
    names = physf.NAMES
    lo, hi = r["eatingspeed"]
    samp = np.sort(np.random.default_rng(0).choice(hi - lo, 20000, replace=False)) + lo
    for n in ("gyr_mag_mean", "gyr_mag_p95", "gyr_log_power", "acc_mag_std"):
        j = names.index(n)
        print(f"[check] eatingspeed {n:16s} old median {np.median(old_f[samp, j]):9.3f} -> new {np.median(F[samp, j]):9.3f}")

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_mutual_info_score

    rng = np.random.default_rng(SEED)
    fit = np.sort(rng.choice(POOL_N, size=500_000, replace=False))
    raw = np.asarray(F[fit], dtype=np.float32)
    med = np.median(raw, axis=0)
    q = np.percentile(raw, [25, 75], axis=0)
    scale = np.where((q[1] - q[0]) > 1e-9, (q[1] - q[0]) / 1.349, np.maximum(raw.std(0), 1e-9))
    sample = np.clip((raw - med) / scale, -8, 8)
    pca = PCA(n_components=PCA_PHYS, whiten=True, random_state=SEED).fit(sample)
    var = float(pca.explained_variance_ratio_.sum())
    print(f"[pca] PCA-{PCA_PHYS} keeps {100*var:.2f}% variance  {time.time()-t0:.0f}s", flush=True)
    zpath = os.path.join(WORK_DIR, "phys_pca32_gyrofix_f32.npy")
    Z = np.lib.format.open_memmap(zpath, mode="w+", dtype=np.float32, shape=(POOL_N, PCA_PHYS))
    for s0 in range(0, POOL_N, 200_000):
        e0 = min(POOL_N, s0 + 200_000)
        Z[s0:e0] = pca.transform(np.clip((np.asarray(F[s0:e0], dtype=np.float32) - med) / scale, -8, 8))
    Z.flush()
    np.savez(os.path.join(WORK_DIR, "phys_scaler_gyrofix.npz"), med=med, scale=scale,
             components=pca.components_, mean=pca.mean_, explained_variance=pca.explained_variance_)

    km = MiniBatchKMeans(n_clusters=K_TYPES, batch_size=10_000, n_init=3, random_state=SEED,
                         max_no_improvement=50)
    zall = np.asarray(Z)
    km.fit(zall)
    labels = km.predict(zall).astype(np.int32)
    np.save(os.path.join(WORK_DIR, "labels_phys_gyrofix_k1000.npy"), labels)
    np.save(os.path.join(WORK_DIR, "centroids_phys_gyrofix_k1000.npy"), km.cluster_centers_.astype(np.float32))
    print(f"[kmeans] done  {time.time()-t0:.0f}s", flush=True)

    old = np.load(os.path.join(WORK_DIR, "labels_phys_k1000.npy"))
    src = np.zeros(POOL_N, np.int8)
    for j, s in enumerate(SOURCES):
        src[r[s][0]:r[s][1]] = j
    cmp_idx = np.sort(rng.choice(POOL_N, 200_000, replace=False))
    M = np.zeros((K_TYPES, 4), np.int64)
    Mo = np.zeros((K_TYPES, 4), np.int64)
    for j, s in enumerate(SOURCES):
        M[:, j] = np.bincount(labels[r[s][0]:r[s][1]], minlength=K_TYPES)
        Mo[:, j] = np.bincount(old[r[s][0]:r[s][1]], minlength=K_TYPES)

    def comp(Mx):
        size = Mx.sum(1); frac = Mx / np.maximum(size[:, None], 1)
        dom = frac.argmax(1); d = frac.max(1); npres = (Mx > 0).sum(1)
        return {"sizes": {"min": int(size.min()), "median": float(np.median(size)), "max": int(size.max()),
                          "nonempty": int((size > 0).sum())},
                "clusters_gt90pct_one_source": int((d > 0.9).sum()),
                "clusters_by_dominant_source": {s: int((dom == j).sum()) for j, s in enumerate(SOURCES)},
                "exclusive_clusters": {s: int(((npres == 1) & (dom == j)).sum()) for j, s in enumerate(SOURCES)}}

    res = {"k": K_TYPES, "seed": SEED, "pool_n": POOL_N, "unit_fix": "eatingspeed gyro deg/s -> rad/s",
           "phys_pca_var": var,
           "ami_new_old": float(adjusted_mutual_info_score(labels[cmp_idx], old[cmp_idx])),
           "ami_new_source": float(adjusted_mutual_info_score(labels[cmp_idx], src[cmp_idx])),
           "ami_old_source": float(adjusted_mutual_info_score(old[cmp_idx], src[cmp_idx])),
           "new": comp(M), "old": comp(Mo), "seconds": time.time() - t0}
    json.dump(res, open(os.path.join(OUT_DIR, "type_definitions_gyrofix.json"), "w"), indent=2)
    np.save(os.path.join(OUT_DIR, "cluster_source_matrix_phys_gyrofix.npy"), M)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
