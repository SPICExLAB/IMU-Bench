#!/usr/bin/env python3
"""Motion-diversity analysis of the WHOLE pretraining pool (every source in the raw cache).

Shared descriptor space (acc-only so acc-only and 6-ch sources are comparable): per-axis and
|a| spectra (bins 1-32), std/mean, intensity -> standardise -> PCA-32 -> K motion clusters
fitted on a source-balanced sample. Then, per source:
  * hours / windows, clusters covered, exclusive clusters (no other source lands there),
    normalised entropy, effective #clusters, dominant-cluster share
  * saturation curve: clusters covered vs hours kept, random vs within-source balanced
    selection -> hours needed to retain 90 / 95 / 99 % of the source's own cluster set
Pool level:
  * asymmetric overlap matrix  cov[A][B] = share of A's clusters that B also covers
  * greedy source ordering by marginal new clusters (which sources actually add motion types)
  * motion-intensity distribution per source
Outputs: OUT/report.md, summary.json, saturation_curves.png, overlap_matrix.png,
         intensity_by_source.png, cluster_share_by_source.png
"""
import argparse, glob, json, os, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="/projects/b53017/vbc1591/mae_multidata_cache_raw")
ap.add_argument("--per-source", type=int, default=200000)
ap.add_argument("--K", type=int, default=2000)
ap.add_argument("--out", default="analysis/pool_diversity/out")
ap.add_argument("--exclude", nargs="*", default=["realdataclean"])
ap.add_argument("--subset", nargs="*", default=[], help="src=index.npy : restrict a source to selected windows")
ap.add_argument("--g-sources", nargs="*", default=["capture24", "nhanes"], help="sources cached in g (x9.80665 -> m/s^2 before descriptors)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); rng = np.random.RandomState(42); t0 = time.time()
HOP_S = 5.12
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def descriptors(x):
    a = x[:, :3].astype(np.float32); mag = np.linalg.norm(a, axis=1)
    fa = np.abs(np.fft.rfft(a, axis=-1))[:, :, 1:33].reshape(len(x), -1)
    fm = np.abs(np.fft.rfft(mag, axis=-1))[:, 1:33]
    return np.concatenate([fa, fm, a.std(-1), a.mean(-1), mag.mean(-1, keepdims=True), mag.std(-1, keepdims=True)], 1)

subsets = dict(kv.split("=", 1) for kv in args.subset)
S = {}      # name -> dict(feats, hours, n_total, has_gyro, inten)
for f in sorted(glob.glob(os.path.join(args.cache, "*_train_x.npy"))):
    s = os.path.basename(f).replace("_train_x.npy", "")
    if s in args.exclude: continue
    a = np.load(f, mmap_mode="r"); tp = f.replace("_x.npy", "_tmask.npy"); mp = f.replace("_x.npy", "_mask.npy")
    pool = np.arange(a.shape[0]); tm = None
    if os.path.exists(tp): tm = np.asarray(np.load(tp)); pool = np.nonzero(tm >= a.shape[-1])[0]
    if s in subsets: pool = np.intersect1d(pool, np.load(subsets[s]).astype(np.int64))
    hrs = (np.where(tm >= 512, 256, tm).sum() / 50 / 3600) if tm is not None else len(pool) * HOP_S / 3600
    if s in subsets: hrs = len(np.load(subsets[s])) * HOP_S / 3600
    idx = np.sort(rng.choice(pool, min(args.per_source, len(pool)), replace=False))
    feats, inten = [], []
    for i in range(0, len(idx), 20000):
        xb = np.asarray(a[idx[i:i+20000]]).astype(np.float32)
        if s in args.g_sources: xb = xb * 9.80665          # unit harmonisation: g -> m/s^2
        feats.append(descriptors(xb)); inten.append(np.linalg.norm(xb[:, :3], axis=1).std(-1))
    m = np.load(mp, mmap_mode="r"); has_gyro = bool(np.asarray(m[:1000, 3:]).any())
    S[s] = {"feats": np.concatenate(feats), "inten": np.concatenate(inten), "hours": float(hrs), "n_total": int(len(pool)), "has_gyro": has_gyro}
    log(f"{s:13s} sampled {len(idx):7d} of {len(pool):8d} windows ({hrs:7.1f} h) gyro={has_gyro}")
names = list(S)
allf = np.concatenate([S[s]["feats"] for s in names]); owner = np.concatenate([[s]*len(S[s]["feats"]) for s in names])
mu, sd = allf.mean(0), allf.std(0) + 1e-6; allf = (allf - mu) / sd
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
# source-balanced fit sample (so big sources don't own the cluster space)
fit = np.concatenate([np.nonzero(owner == s)[0][rng.permutation((owner == s).sum())[:40000]] for s in names])
pca = PCA(n_components=32, random_state=42).fit(allf[fit]); Z = pca.transform(allf).astype(np.float32)
K = args.K
km = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=8192, n_init=3).fit(Z[fit]); lab = km.predict(Z)
log(f"PCA var {pca.explained_variance_ratio_.sum():.3f}; K={K}")
del allf

cl = {s: np.bincount(lab[owner == s], minlength=K) for s in names}
cov = {s: cl[s] > 0 for s in names}
res = {"meta": {"cache": args.cache, "K": K, "per_source": args.per_source, "pca_explained": round(float(pca.explained_variance_ratio_.sum()), 3)}, "sources": {}}
for s in names:
    p = cl[s] / cl[s].sum(); nz = p > 0; H = -(p[nz] * np.log(p[nz])).sum()
    others = np.zeros(K, bool)
    for o in names:
        if o != s: others |= cov[o]
    res["sources"][s] = {"hours": round(S[s]["hours"], 1), "windows": S[s]["n_total"], "has_gyro": S[s]["has_gyro"],
                         "clusters_covered": int(nz.sum()), "coverage": round(nz.sum() / K, 4),
                         "exclusive_clusters": int((cov[s] & ~others).sum()), "exclusive_share_of_own_windows": round(float(p[cov[s] & ~others].sum()), 4),
                         "norm_entropy": round(float(H / np.log(K)), 4), "effective_clusters": round(float(np.exp(H)), 1),
                         "max_cluster_share": round(float(p.max()), 4),
                         "intensity_median_ms2": round(float(np.median(S[s]["inten"])), 3), "intensity_p90_ms2": round(float(np.percentile(S[s]["inten"], 90)), 3)}

# saturation curves per source (random vs balanced, in hours of that source)
fracs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
def balanced(labs, b):
    K_ = K; sizes = np.bincount(labs, minlength=K_); quota = np.zeros(K_, np.int64); rem = b; act = sizes > 0
    while rem > 0 and act.any():
        q = np.floor(rem / act.sum()); q = max(int(q), 1); moved = 0
        for c in np.nonzero(act)[0]:
            take = min(q, int(sizes[c] - quota[c]), rem); quota[c] += take; rem -= take; moved += take
            if quota[c] >= sizes[c]: act[c] = False
            if rem <= 0: break
        if moved == 0: break
    return int((quota > 0).sum())
curves = {}
for s in names:
    labs = lab[owner == s]; n = len(labs); own = int(cov[s].sum()); row = {"hours": [], "random": [], "balanced": []}
    for fr in fracs:
        b = max(1, int(fr * n)); row["hours"].append(round(fr * S[s]["hours"], 2))
        row["random"].append(round(float(len(np.unique(labs[rng.choice(n, b, replace=False)])) / own), 4))
        row["balanced"].append(round(balanced(labs, b) / own, 4))
    for tgt in (0.9, 0.95, 0.99):
        hr = next((h for h, c in zip(row["hours"], row["random"]) if c >= tgt), None)
        hb = next((h for h, c in zip(row["hours"], row["balanced"]) if c >= tgt), None)
        res["sources"][s][f"hours_for_{int(tgt*100)}pct_random"] = hr; res["sources"][s][f"hours_for_{int(tgt*100)}pct_balanced"] = hb
    curves[s] = row
res["saturation_curves"] = curves

# overlap matrix + greedy ordering
ov = {a: {b: round(float((cov[a] & cov[b]).sum() / max(cov[a].sum(), 1)), 3) for b in names} for a in names}
res["overlap_share_of_row_clusters_covered_by_col"] = ov
covered = np.zeros(K, bool); order = []
rest = list(names)
while rest:
    best = max(rest, key=lambda s: int((cov[s] & ~covered).sum())); gain = int((cov[best] & ~covered).sum())
    covered |= cov[best]; order.append({"source": best, "new_clusters": gain, "cum_coverage": round(float(covered.mean()), 4)}); rest.remove(best)
res["greedy_source_order"] = order
json.dump(res, open(os.path.join(args.out, "summary.json"), "w"), indent=1)

# plots
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for s in names:
    axs[0].plot(curves[s]["hours"], curves[s]["random"], marker=".", label=s); axs[1].plot(curves[s]["hours"], curves[s]["balanced"], marker=".", label=s)
for ax, t in zip(axs, ["random subset", "within-source balanced subset"]):
    ax.set_xscale("log"); ax.set_xlabel("hours of the source kept"); ax.set_title(t); ax.grid(alpha=.3); ax.axhline(0.95, ls="--", c="k", lw=.7)
axs[0].set_ylabel("fraction of the source's own motion clusters retained"); axs[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "saturation_curves.png"), dpi=130); plt.close(fig)
M = np.array([[ov[a][b] for b in names] for a in names])
fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
for i in range(len(names)):
    for j in range(len(names)): ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w" if M[i,j] < .6 else "k", fontsize=8)
ax.set_title("share of ROW source's clusters also covered by COL source"); fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(args.out, "overlap_matrix.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 4.5)); bins = np.logspace(-3, 1.7, 70)
for s in names: ax.hist(S[s]["inten"], bins=bins, density=True, histtype="step", lw=1.4, label=s)
ax.set_xscale("log"); ax.set_xlabel("|acc| std per window [m/s^2] (motion intensity)"); ax.set_ylabel("density"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "intensity_by_source.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 4)); tot = sum(cl[s] for s in names); o = np.argsort(-tot); bottom = np.zeros(K)
for s in names:
    v = cl[s][o] / np.maximum(tot[o], 1); ax.bar(range(K), v, bottom=bottom, width=1.0, label=s); bottom += v
ax.set_xlabel("motion cluster (sorted by pooled size)"); ax.set_ylabel("share of sampled windows per source"); ax.legend(fontsize=7, ncol=4); ax.set_xlim(0, K)
fig.tight_layout(); fig.savefig(os.path.join(args.out, "cluster_share_by_source.png"), dpi=130); plt.close(fig)

# report
L = [f"# Pretraining-pool motion-diversity analysis (`{args.cache}`)", "",
     f"Shared acc-only descriptor space, PCA-32 (explained var {res['meta']['pca_explained']}), K = {K} motion clusters fitted on a source-balanced sample; "
     f"up to {args.per_source:,} full-length windows sampled per source." + (f" Subsets: {subsets}" if subsets else ""), "",
     "| source | hours (train) | gyro | clusters covered | exclusive clusters (share of own windows) | norm. entropy | effective clusters | max cluster share | intensity median / p90 [m/s²] | h for 95 % of own clusters (random / balanced) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for s in names:
    r = res["sources"][s]
    L.append(f"| {s} | {r['hours']:,.0f} | {'yes' if r['has_gyro'] else 'no'} | {r['clusters_covered']}/{K} ({r['coverage']*100:.1f} %) | {r['exclusive_clusters']} ({r['exclusive_share_of_own_windows']*100:.1f} %) | "
             f"{r['norm_entropy']} | {r['effective_clusters']} | {r['max_cluster_share']*100:.2f} % | {r['intensity_median_ms2']} / {r['intensity_p90_ms2']} | {r['hours_for_95pct_random']} / {r['hours_for_95pct_balanced']} |")
L += ["", "Greedy source order by marginal new motion clusters:", "", "| rank | source | new clusters | cumulative coverage |", "|---|---|---|---|"]
for i, g in enumerate(order, 1): L.append(f"| {i} | {g['source']} | {g['new_clusters']} | {g['cum_coverage']*100:.1f} % |")
L += ["", "Overlap: share of ROW source's clusters also covered by COL source:", "", "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
for a in names: L.append(f"| {a} | " + " | ".join(f"{ov[a][b]:.2f}" for b in names) + " |")
L += ["", "Figures: saturation_curves.png, overlap_matrix.png, intensity_by_source.png, cluster_share_by_source.png; full numbers in summary.json"]
open(os.path.join(args.out, "report.md"), "w").write("\n".join(L) + "\n"); log("done ->", args.out)
