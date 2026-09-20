#!/usr/bin/env python
"""Stage 3: define "motion type" twice, independently, and report how much the
two definitions agree.

  T-phys  k-means (K=1000) on PCA-32 of the 40 rotation-invariant physical
          features from 02. No learned model in the loop. **This is the
          definition the subset selection in 04 uses.**
  T-enc   k-means (K=1000) on PCA-64 of the frozen mae_4src_z CLS embedding
          from 01. Comparable to 11j/11l, used here only as a read-out.

Diagnostics written to out/type_definitions.json, each one answering a specific
objection to the 11j/11l diversity axis:

  ami_phys_enc      do the two definitions describe the same partition?
                    (if yes, the encoder-space result in 11j/11l was not an
                    artefact of the encoder)
  ami_*_source      are "motion types" secretly just "datasets"? if this were
                    high, the type axis and the source axis of the factorial
                    would be collinear and the design could not separate them
  stability_ari     is K=1000 k-means reproducible across seeds, and by how
                    much more than on a Gaussian null with the same covariance
                    (i.e. is there mode structure at all, or only a blob?)
  silhouette        same question, geometrically
  size_quantiles    the long tail the coverage statistic has to be weighted by

Writes $WORK/labels_phys_k1000.npy, $WORK/labels_enc_k1000.npy,
       $WORK/phys_pca32_f32.npy, out/type_definitions.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import (K_TYPES, OUT_DIR, PCA_PHYS, POOL_N, SEED, SOURCES, WORK_DIR,
                  source_ranges)

SEEDS = [42, 7, 2024]
CMP_N = 200_000
SIL_N = 20_000


def kmeans_labels(z, k, seed):
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=k, batch_size=10_000, n_init=3,
                         random_state=seed, max_no_improvement=50)
    km.fit(np.asarray(z))
    return km.predict(np.asarray(z)).astype(np.int32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, silhouette_score

    rng = np.random.default_rng(SEED)
    res = {"k": K_TYPES, "seed": SEED, "pool_n": POOL_N}

    # ---------- T-phys (the selection space) ----------
    phys_path = os.path.join(WORK_DIR, "phys_pca32_f32.npy")
    if os.path.exists(phys_path) and np.load(phys_path, mmap_mode="r").shape[0] == POOL_N:
        zp = np.load(phys_path, mmap_mode="r")
    else:
        feats = np.load(os.path.join(WORK_DIR, "phys_feats_f32.npy"), mmap_mode="r")
        fit = np.sort(rng.choice(POOL_N, size=min(500_000, POOL_N), replace=False))
        raw = np.asarray(feats[fit], dtype=np.float32)
        # robust (median/IQR) scaling, fitted once on this sample and then
        # applied to the whole pool -- a plain z-score would let the synthetic
        # sources' heavy-tailed jerk features dominate every principal axis
        med = np.median(raw, axis=0)
        q = np.percentile(raw, [25, 75], axis=0)
        scale = np.where((q[1] - q[0]) > 1e-9, (q[1] - q[0]) / 1.349,
                         np.maximum(raw.std(0), 1e-9))
        sample = np.clip((raw - med) / scale, -8, 8)
        del raw
        pca = PCA(n_components=PCA_PHYS, whiten=True, random_state=SEED).fit(sample)
        res["phys_pca_var"] = float(pca.explained_variance_ratio_.sum())
        zp = np.lib.format.open_memmap(phys_path, mode="w+", dtype=np.float32,
                                       shape=(POOL_N, PCA_PHYS))
        for s in range(0, POOL_N, 200_000):
            e = min(POOL_N, s + 200_000)
            blk = np.clip((np.asarray(feats[s:e], dtype=np.float32) - med) / scale, -8, 8)
            zp[s:e] = pca.transform(blk)
        zp.flush()
        np.savez(os.path.join(WORK_DIR, "phys_scaler.npz"), med=med, scale=scale,
                 components=pca.components_, mean=pca.mean_,
                 explained_variance=pca.explained_variance_)
        print(f"[phys] PCA-{PCA_PHYS} keeps {100 * res['phys_pca_var']:.1f}% variance", flush=True)

    lp_path = os.path.join(WORK_DIR, "labels_phys_k1000.npy")
    if os.path.exists(lp_path):
        labels_phys = np.load(lp_path)
    else:
        labels_phys = kmeans_labels(zp, K_TYPES, SEED)
        np.save(lp_path, labels_phys)
    sizes_p = np.bincount(labels_phys, minlength=K_TYPES)
    res["phys_cluster_sizes"] = {
        "nonempty": int((sizes_p > 0).sum()), "min": int(sizes_p.min()),
        "q10": int(np.percentile(sizes_p, 10)), "median": int(np.median(sizes_p)),
        "q90": int(np.percentile(sizes_p, 90)), "max": int(sizes_p.max())}
    print(f"[phys] k-means sizes {res['phys_cluster_sizes']}", flush=True)

    # ---------- T-enc (the read-out) ----------
    le_path = os.path.join(WORK_DIR, "labels_enc_k1000.npy")
    ze = np.load(os.path.join(WORK_DIR, "enc_pca64_f32.npy"), mmap_mode="r")
    if os.path.exists(le_path):
        labels_enc = np.load(le_path)
    else:
        labels_enc = kmeans_labels(ze, K_TYPES, SEED)
        np.save(le_path, labels_enc)
    sizes_e = np.bincount(labels_enc, minlength=K_TYPES)
    res["enc_cluster_sizes"] = {
        "nonempty": int((sizes_e > 0).sum()), "min": int(sizes_e.min()),
        "median": int(np.median(sizes_e)), "max": int(sizes_e.max())}

    # ---------- agreement ----------
    cmp_idx = np.sort(rng.choice(POOL_N, size=min(CMP_N, POOL_N), replace=False))
    src = np.zeros(POOL_N, dtype=np.int8)
    for j, (name, (lo, hi)) in enumerate(source_ranges().items()):
        src[lo:hi] = j
    res["ami_phys_enc"] = float(adjusted_mutual_info_score(labels_phys[cmp_idx], labels_enc[cmp_idx]))
    res["ami_phys_source"] = float(adjusted_mutual_info_score(labels_phys[cmp_idx], src[cmp_idx]))
    res["ami_enc_source"] = float(adjusted_mutual_info_score(labels_enc[cmp_idx], src[cmp_idx]))
    print(f"[agree] AMI(phys,enc)={res['ami_phys_enc']:.3f}  "
          f"AMI(phys,source)={res['ami_phys_source']:.3f}  "
          f"AMI(enc,source)={res['ami_enc_source']:.3f}", flush=True)

    # ---------- is there mode structure at all? real vs Gaussian null ----------
    sub = np.sort(rng.choice(POOL_N, size=min(CMP_N, POOL_N), replace=False))
    zs = np.asarray(zp[sub], dtype=np.float32)
    null = rng.multivariate_normal(zs.mean(0), np.cov(zs, rowvar=False),
                                   size=len(zs)).astype(np.float32)
    stab = {}
    for tag, data in (("real", zs), ("null", null)):
        labs = [kmeans_labels(data, K_TYPES, s) for s in SEEDS]
        aris = [float(adjusted_rand_score(labs[i], labs[j]))
                for i in range(len(SEEDS)) for j in range(i + 1, len(SEEDS))]
        sil_idx = rng.choice(len(data), min(SIL_N, len(data)), replace=False)
        stab[tag] = {"ari_mean": float(np.mean(aris)), "ari_pairs": aris,
                     "silhouette": float(silhouette_score(data[sil_idx], labs[0][sil_idx]))}
    stab["ari_excess"] = round(stab["real"]["ari_mean"] - stab["null"]["ari_mean"], 4)
    stab["sil_excess"] = round(stab["real"]["silhouette"] - stab["null"]["silhouette"], 4)
    res["stability_phys"] = stab
    print(f"[valid] phys ARI real {stab['real']['ari_mean']:.3f} vs null "
          f"{stab['null']['ari_mean']:.3f} (excess {stab['ari_excess']})", flush=True)

    # per-source type distribution -- how much of the pool's type inventory each
    # source contributes on its own (this is the "sources are near-redundant in
    # motion types" claim, restated for the 4-source pool)
    per_src = {}
    for name, (lo, hi) in source_ranges().items():
        cnt = np.bincount(labels_phys[lo:hi], minlength=K_TYPES)
        p = cnt / max(1, cnt.sum())
        nz = p[p > 0]
        per_src[name] = {"windows": int(hi - lo), "types_present": int((cnt > 0).sum()),
                         "effective_types": float(np.exp(-(nz * np.log(nz)).sum()))}
    res["per_source_types_phys"] = per_src
    res["sources"] = SOURCES

    with open(os.path.join(OUT_DIR, "type_definitions.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
