#!/usr/bin/env python3
"""Motion-diversity analysis + diversity-preserving subset selection for NHANES.

Goal: keep ~0.01 % of NHANES (0.01 % x 2.72 M h = 272 h) while retaining most of its
motion diversity. Candidate pool = the cached NHANES train array (4,000 h strided evenly over
all 13,224 train participants, 8 days each), i.e. already a population sample.

Method
  1. descriptor per window (acc-only: |FFT| bins 1-32 per axis, per-axis std/mean, |a| mean,
     |a| spectrum) -> standardise -> PCA-32 (fit on a 300k sample)
  2. MiniBatchKMeans K clusters on the PCA space (fit on the sample, assign ALL windows)
  3. selection at budget B windows:
       random      : uniform sample (reference)
       strided     : evenly spaced in cache order (what max_hours_cached does today)
       balanced    : water-filling equal quota per cluster (rare motion types kept in full,
                     common ones thinned), within-cluster evenly strided over cache order
                     (cache order ~ participant order, so this also spreads participants)
  4. diversity metrics vs FULL pool: clusters covered / K, normalised cluster entropy,
     effective #clusters exp(H), rare-cluster retention, and a geometric coverage test:
     distance from 20k random full-pool windows to their nearest selected window
     (median / p95 in PCA space) -- lower = the subset "stands in" for the pool better.
  5. also: which NHANES clusters are NOT covered by the other pretraining sources
     (40k windows each) -> how many of those "exclusive" clusters each strategy keeps.
Outputs (OUT dir): indices_balanced_{H}h.npy (+ random/strided for ablation), report.md,
coverage_curve.png, cluster_occupancy.png, intensity_hist.png, pca_scatter.png, summary.json
"""
import argparse, glob, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/projects/b53017/vbc1591/mae_multidata_cache_raw")
ap.add_argument("--source", default="nhanes")
ap.add_argument("--budget-hours", type=float, default=272.0, help="0.01%% of 2.72 M h = 272 h")
ap.add_argument("--K", type=int, default=2000)
ap.add_argument("--fit-sample", type=int, default=300000)
ap.add_argument("--max-windows", type=int, default=None, help="debug: only use the first N windows")
ap.add_argument("--out", default="analysis/nhanes_diversity/out")
ap.add_argument("--no-other-sources", action="store_true")
ap.add_argument("--g-sources", nargs="*", default=["capture24", "nhanes"], help="sources cached in g (x9.80665 -> m/s^2 before descriptors)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
rng = np.random.RandomState(42); t0 = time.time()
HOP_S = 5.12

def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def descriptors(x, gain=1.0):            # x [n, 6, T] -> acc-only features [n, D]; gain: unit harmonisation
    a = x[:, :3].astype(np.float32) * np.float32(gain)
    mag = np.linalg.norm(a, axis=1)      # [n, T]
    fa = np.abs(np.fft.rfft(a, axis=-1))[:, :, 1:33].reshape(len(x), -1)
    fm = np.abs(np.fft.rfft(mag, axis=-1))[:, 1:33]
    return np.concatenate([fa, fm, a.std(-1), a.mean(-1), mag.mean(-1, keepdims=True), mag.std(-1, keepdims=True)], 1)

X = np.load(os.path.join(args.cache, f"{args.source}_train_x.npy"), mmap_mode="r")
SRC_GAIN = 9.80665 if args.source in args.g_sources else 1.0
N = X.shape[0] if args.max_windows is None else min(args.max_windows, X.shape[0])
log(f"{args.source}: {N} windows = {N*HOP_S/3600:.0f} h candidate pool (cache {X.shape})")

# ---- 1. descriptors for ALL windows (chunked)
CH = 20000; D = None; feats = None
for i in range(0, N, CH):
    d = descriptors(np.asarray(X[i:min(i+CH, N)]), SRC_GAIN)
    if feats is None: feats = np.empty((N, d.shape[1]), dtype=np.float32)
    feats[i:i+len(d)] = d
    if (i // CH) % 20 == 0: log(f"  descriptors {i+len(d)}/{N}")
mu, sd = feats.mean(0), feats.std(0) + 1e-6
feats = (feats - mu) / sd
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
fit_idx = np.sort(rng.choice(N, min(args.fit_sample, N), replace=False))
pca = PCA(n_components=32, random_state=42).fit(feats[fit_idx])
Z = np.empty((N, 32), dtype=np.float32)
for i in range(0, N, 200000): Z[i:i+200000] = pca.transform(feats[i:i+200000])
log(f"PCA-32 explained var {pca.explained_variance_ratio_.sum():.3f}")
del feats

# ---- 2. clusters
K = min(args.K, N // 20)
km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=8192, n_init=3).fit(Z[fit_idx])
lab = np.empty(N, dtype=np.int32)
for i in range(0, N, 200000): lab[i:i+200000] = km.predict(Z[i:i+200000])
sizes = np.bincount(lab, minlength=K)
log(f"K={K} cluster sizes min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")

# ---- other sources: which NHANES clusters do they already cover?
exclusive = np.ones(K, dtype=bool)
other_cov = {}
if not args.no_other_sources:
    for f in sorted(glob.glob(os.path.join(args.cache, "*_train_x.npy"))):
        s = os.path.basename(f).replace("_train_x.npy", "")
        if s in (args.source, "realdataclean"): continue
        a = np.load(f, mmap_mode="r"); tp = f.replace("_x.npy", "_tmask.npy")
        pool = np.arange(a.shape[0])
        if os.path.exists(tp): pool = np.nonzero(np.asarray(np.load(tp)) >= a.shape[-1])[0]
        idx = np.sort(rng.choice(pool, min(40000, len(pool)), replace=False))
        zo = pca.transform((descriptors(np.asarray(a[idx]), 9.80665 if s in args.g_sources else 1.0) - mu) / sd)
        lo = km.predict(zo); cov = np.zeros(K, bool); cov[np.unique(lo)] = True
        other_cov[s] = int(cov.sum()); exclusive &= ~cov
        log(f"  other source {s}: covers {cov.sum()}/{K} NHANES clusters")
    log(f"NHANES-exclusive clusters (no other source lands there): {exclusive.sum()}/{K}, "
        f"holding {sizes[exclusive].sum()/N*100:.1f}% of NHANES windows")

# ---- 3. selection strategies
B = int(round(args.budget_hours * 3600 / HOP_S)); B = min(B, N)
order_in_cluster = {c: np.nonzero(lab == c)[0] for c in range(K)}   # cache order = ~participant order

def sel_random(b): return np.sort(rng.choice(N, b, replace=False))
def sel_strided(b): return np.linspace(0, N - 1, b).round().astype(np.int64)
def sel_balanced(b, weights=None):
    """water-filling: equal quota per cluster (optionally weighted), clusters smaller than
    their quota are taken whole and the remainder is redistributed."""
    w = np.ones(K) if weights is None else np.asarray(weights, float)
    quota = np.zeros(K, dtype=np.int64); remaining = b; active = np.ones(K, bool)
    while remaining > 0 and active.any():
        share = w[active] / w[active].sum() * remaining
        q = np.floor(share).astype(np.int64); q[(share > 0) & (q == 0)] = 1
        cand = np.nonzero(active)[0]
        for c, qq in zip(cand, q):
            take = min(int(qq), int(sizes[c] - quota[c]))
            quota[c] += take; remaining -= take
            if quota[c] >= sizes[c]: active[c] = False
        if q.sum() == 0: break
    out = []
    for c in range(K):
        if quota[c] == 0: continue
        ids = order_in_cluster[c]
        out.append(ids[np.linspace(0, len(ids) - 1, quota[c]).round().astype(np.int64)])
    return np.sort(np.unique(np.concatenate(out)))[:b]

strategies = {"random": sel_random(B), "strided": sel_strided(B), "balanced": sel_balanced(B)}
if exclusive.any() and exclusive.sum() < K:
    strategies["balanced_exclusive2x"] = sel_balanced(B, np.where(exclusive, 2.0, 1.0))

# ---- 4. metrics
probe = np.sort(rng.choice(N, min(20000, N), replace=False))
from sklearn.neighbors import NearestNeighbors
def metrics(sel):
    cnt = np.bincount(lab[sel], minlength=K); p = cnt / cnt.sum(); nz = p > 0
    H = -(p[nz] * np.log(p[nz])).sum()
    rare = sizes < np.median(sizes)
    nn = NearestNeighbors(n_neighbors=1).fit(Z[sel]); d, _ = nn.kneighbors(Z[probe])
    return {"windows": int(len(sel)), "hours": round(len(sel) * HOP_S / 3600, 1),
            "clusters_covered": int(nz.sum()), "coverage": round(nz.sum() / K, 4),
            "norm_entropy": round(H / np.log(K), 4), "effective_clusters": round(float(np.exp(H)), 1),
            "rare_clusters_kept": round(float((cnt[rare] > 0).mean()), 4),
            "exclusive_clusters_kept": round(float((cnt[exclusive] > 0).mean()), 4) if exclusive.any() else None,
            "nn_dist_median": round(float(np.median(d)), 3), "nn_dist_p95": round(float(np.percentile(d, 95)), 3),
            "max_cluster_share": round(float(p.max()), 4)}
full = {"windows": N, "hours": round(N * HOP_S / 3600, 1), "clusters_covered": K, "coverage": 1.0,
        "norm_entropy": round(float(-((sizes/N)*np.log(sizes/N)).sum() / np.log(K)), 4),
        "effective_clusters": round(float(np.exp(-((sizes/N)*np.log(sizes/N)).sum())), 1),
        "rare_clusters_kept": 1.0, "exclusive_clusters_kept": 1.0 if exclusive.any() else None,
        "nn_dist_median": 0.0, "nn_dist_p95": 0.0, "max_cluster_share": round(float(sizes.max()/N), 4)}
res = {"full_pool": full}
for k, sel in strategies.items():
    res[k] = metrics(sel); log(k, res[k])
    np.save(os.path.join(args.out, f"indices_{k}_{int(round(args.budget_hours))}h.npy"), sel.astype(np.int64))

# coverage curve vs budget
budgets = [b for b in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5] if int(b * N) >= K // 10]
curve = {"budget_frac": budgets, "hours": [round(b * N * HOP_S / 3600, 1) for b in budgets]}
for k, fn in [("random", sel_random), ("strided", sel_strided), ("balanced", sel_balanced)]:
    curve[k] = [round(float(len(np.unique(lab[fn(int(b * N))])) / K), 4) for b in budgets]
res["coverage_curve"] = curve
res["meta"] = {"cache": args.cache, "source": args.source, "N": int(N), "K": int(K), "budget_hours": args.budget_hours,
               "budget_windows": int(B), "budget_frac_of_pool": round(B / N, 4), "other_source_cluster_coverage": other_cov,
               "exclusive_clusters": int(exclusive.sum()), "pca_explained": round(float(pca.explained_variance_ratio_.sum()), 3)}
json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)

# ---- 5. plots
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
for k in ["random", "strided", "balanced"]: ax.plot(curve["hours"], curve[k], marker="o", label=k)
ax.axvline(args.budget_hours, ls="--", c="k", lw=0.8, label=f"budget {args.budget_hours:.0f} h")
ax.set_xscale("log"); ax.set_xlabel("NHANES hours kept"); ax.set_ylabel(f"fraction of {K} motion clusters covered"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "coverage_curve.png"), dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4)); o = np.argsort(-sizes)
ax.plot(sizes[o] / N, label="full pool", c="k")
for k in ["random", "balanced"]:
    c = np.bincount(lab[strategies[k]], minlength=K); ax.plot(c[o] / c.sum(), label=k, alpha=.8)
ax.set_yscale("log"); ax.set_xlabel("cluster (sorted by full-pool size)"); ax.set_ylabel("share of windows"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "cluster_occupancy.png"), dpi=130); plt.close(fig)

inten = np.empty(N, np.float32)
for i in range(0, N, CH): inten[i:i+CH] = np.linalg.norm(np.asarray(X[i:min(i+CH, N)])[:, :3] * SRC_GAIN, axis=1).std(-1)
fig, ax = plt.subplots(figsize=(6, 4)); bins = np.logspace(-3, 1.5, 60)
ax.hist(inten, bins=bins, density=True, alpha=.4, label="full pool")
for k in ["random", "balanced"]: ax.hist(inten[strategies[k]], bins=bins, density=True, histtype="step", lw=1.5, label=k)
ax.set_xscale("log"); ax.set_xlabel("|acc| std per window (motion intensity)"); ax.set_ylabel("density"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "intensity_hist.png"), dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
bg = rng.choice(N, min(100000, N), replace=False)
for ax, (k, sel) in zip(axs, [("full pool (100k sample)", bg), ("random", strategies["random"]), ("balanced", strategies["balanced"])]):
    ax.scatter(Z[sel, 0], Z[sel, 1], s=1, alpha=.15); ax.set_title(f"{k}: {len(sel)} win"); ax.set_xlabel("PC1")
axs[0].set_ylabel("PC2"); fig.tight_layout(); fig.savefig(os.path.join(args.out, "pca_scatter.png"), dpi=130); plt.close(fig)

# ---- report
L = [f"# NHANES motion-diversity analysis ({args.source}, cache `{args.cache}`)", "",
     f"Candidate pool: {N:,} windows = {N*HOP_S/3600:,.0f} h (strided evenly over all train participants). "
     f"Budget: {args.budget_hours:.0f} h = {B:,} windows = {B/N*100:.2f} % of the pool "
     f"(0.01 % of the ~2.72 M h on disk). K = {K} motion clusters in PCA-32 descriptor space "
     f"(explained var {pca.explained_variance_ratio_.sum():.2f}).", "",
     "| selection | hours | clusters covered | norm. entropy | effective clusters | rare clusters kept | NHANES-exclusive kept | NN dist median / p95 | max cluster share |",
     "|---|---|---|---|---|---|---|---|---|"]
for k in ["full_pool"] + list(strategies):
    m = res[k]; exk = "-" if m["exclusive_clusters_kept"] is None else "%.1f %%" % (m["exclusive_clusters_kept"] * 100)
    L.append("| %s | %s | %d/%d (%.1f %%) | %s | %s | %.1f %% | %s | %s / %s | %.2f %% |" % (
        k, m["hours"], m["clusters_covered"], K, m["coverage"] * 100, m["norm_entropy"], m["effective_clusters"],
        m["rare_clusters_kept"] * 100, exk, m["nn_dist_median"], m["nn_dist_p95"], m["max_cluster_share"] * 100))
L += ["", "Coverage curve (fraction of clusters covered vs hours kept):", "", "| hours | random | strided | balanced |", "|---|---|---|---|"]
for i, h in enumerate(curve["hours"]): L.append(f"| {h} | {curve['random'][i]} | {curve['strided'][i]} | {curve['balanced'][i]} |")
if other_cov:
    L += ["", f"Other sources' coverage of the NHANES cluster space (40k windows each): " + ", ".join(f"{s} {v}/{K}" for s, v in other_cov.items()) +
          f". NHANES-exclusive clusters: {exclusive.sum()}/{K} holding {sizes[exclusive].sum()/N*100:.1f} % of NHANES windows."]
L += ["", "Files: `indices_balanced_{H}h.npy` (use as `sources.nhanes.subset_index_file`), coverage_curve.png, cluster_occupancy.png, intensity_hist.png, pca_scatter.png, summary.json"]
open(os.path.join(args.out, "report.md"), "w").write("\n".join(L) + "\n")
log("done ->", args.out)
