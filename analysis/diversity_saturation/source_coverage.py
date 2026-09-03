#!/usr/bin/env python3
"""Per-source diversity (cluster coverage) on the FULL cache -> pretraining mixture weights.

Method mirrors analysis/diversity_saturation: sample windows per source, embed with simple
scale-invariant descriptors, k-means the pooled sample into K clusters, then measure what
fraction of clusters each source covers and how *exclusively* it covers them. Mixture weight
is set by marginal coverage contribution, not by hours -- the saturation analysis showed the
pool's intrinsic diversity is fully covered by ~0.5 % of windows, so raw hours are the wrong
currency for balancing.
"""
import numpy as np, glob, os, json
from sklearn.cluster import MiniBatchKMeans
CACHE=os.environ.get("CACHE","/projects/b53017/vbc1591/mae_multidata_cache_full")
K=int(os.environ.get("K","1000")); PER=int(os.environ.get("PER","40000")); rng=np.random.RandomState(42)

def descriptors(x):                       # x [N, 6, T] -> scale/rotation-tolerant summary
    a, g = x[:, :3], x[:, 3:]
    f = [np.abs(np.fft.rfft(a, axis=-1))[:, :, 1:33].reshape(len(x), -1),
         np.abs(np.fft.rfft(g, axis=-1))[:, :, 1:33].reshape(len(x), -1),
         a.std(-1), g.std(-1), a.mean(-1), np.linalg.norm(a, axis=1).mean(-1, keepdims=True)]
    z = np.concatenate([v.reshape(len(x), -1) for v in f], 1)
    return (z - z.mean(0)) / (z.std(0) + 1e-6)

sources, X, owner = [], [], []
for f in sorted(glob.glob(os.path.join(CACHE, "*_train_x.npy"))):
    s = os.path.basename(f).replace("_train_x.npy", "")
    a = np.load(f, mmap_mode="r")
    # short-clip sources carry a valid-length array; zero-padded windows would form spurious
    # clusters (low std, truncated spectra) and inflate "exclusive" coverage -> sample only
    # full-length windows for the descriptor space. Hours still count the real signal.
    tpath = f.replace("_x.npy", "_tmask.npy")
    pool = np.arange(a.shape[0])
    if os.path.exists(tpath):
        vl = np.asarray(np.load(tpath)); pool = np.nonzero(vl >= a.shape[-1])[0]
    sub = os.environ.get(f"SUBSET_{s}")          # e.g. SUBSET_nhanes=indices_balanced_272h.npy
    if sub:
        keep = np.load(sub).astype(np.int64); pool = np.intersect1d(pool, keep)
    n = min(PER, len(pool))
    idx = np.sort(rng.choice(pool, n, replace=False))
    xb = np.asarray(a[idx], dtype=np.float32)
    if s in ("capture24", "nhanes"): xb = xb * 9.80665     # cached in g -> m/s^2 (else the unit gap masquerades as "exclusive" motion)
    X.append(descriptors(xb)); owner += [s]*n
    # 512-sample windows, 256-sample hop: a full window adds 5.12 s of new signal, a short
    # clip's single window adds its valid length (tmask)
    hrs = (np.where(np.asarray(np.load(tpath))>=512, 256, np.asarray(np.load(tpath))).sum()/50/3600) if os.path.exists(tpath) else a.shape[0]*5.12/3600
    if sub: hrs = len(keep) * 5.12 / 3600
    sources.append((s, len(keep) if sub else a.shape[0], hrs))
    print(f"{s:14s} sampled {n} full windows of {a.shape[0]} ({hrs:.0f} h real signal)", flush=True)
X = np.concatenate(X); owner = np.array(owner)
km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=4096, n_init=5).fit(X)
lab = km.labels_
out = {"k": K, "per_source_sample": PER, "sources": {}}
allc = set(range(K))
for s, n_win, hrs in sources:
    m = owner == s
    cl = set(np.unique(lab[m]).tolist())
    others = set(np.unique(lab[~m]).tolist())
    out["sources"][s] = {"windows": int(n_win), "hours": round(hrs, 1),
                         "clusters_covered": len(cl), "coverage": round(len(cl)/K, 4),
                         "exclusive_clusters": len(cl - others),
                         "exclusive_frac": round(len(cl - others)/max(len(cl), 1), 4)}
# weight = coverage + 2x exclusivity (unique motion types are what the saturation study says matters)
raw = {s: v["coverage"] + 2*v["exclusive_clusters"]/K for s, v in out["sources"].items()}
tot = sum(raw.values())
out["suggested_source_weights"] = {s: round(len(raw)*r/tot, 3) for s, r in raw.items()}   # 1.0 = equal share
json.dump(out, open(os.path.join("analysis/diversity_saturation/out", "source_coverage.json"), "w"), indent=1)
print(json.dumps(out["sources"], indent=1)); print("weights:", json.dumps(out["suggested_source_weights"], indent=1))
