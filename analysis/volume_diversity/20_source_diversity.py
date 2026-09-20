#!/usr/bin/env python
"""Which pretraining source is the most DIVERSE?  -- matched-N comparison.

Raw "types touched" is not comparable across sources: OCDetect has 25x the
windows of Motion-X, and more draws always touch more types. So every number
here is computed on equal-size random draws from each source (rarefaction),
three draws per point, in both type spaces, with two families of metric:

  partition-based   effective types = exp(Shannon) over the shared K=1000
                    k-means partition. Caveat: that partition was fitted on a
                    pool that is 72 % OCDetect, so OCDetect's region is cut
                    finer -- this metric leans toward OCDetect.
  partition-free    Vendi score (exp entropy of the eigenvalues of the cosine
                    similarity kernel) -- "how many effectively distinct
                    directions does the sample occupy". No clustering involved.
                    Also median 10-NN distance (local spread).

Sensitivity: EatingSpeed gyroscope is very likely cached in deg/s (57x the
still-window gyro of the other sources). T-phys metrics are recomputed for
EatingSpeed with gyro / 57.2958 through the same scaler + PCA, and windows are
re-assigned to the nearest T-phys centroid.

Writes out/source_diversity.json, out/source_diversity_4src_{light,dark}.{png,pdf}
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import CACHE_DIR, OUT_DIR, SOURCES, WORK_DIR, source_ranges

_spec = importlib.util.spec_from_file_location("physf", os.path.join(HERE, "02_physical_features.py"))
physf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(physf)

NS = [250, 500, 1000, 2000, 5000, 10000, 20000, 50000]
N_REF = 20000
DRAWS = 3
NAMES = {"ocdetect": "OCDetect", "eatingspeed": "EatingSpeed", "motionx": "Motion-X", "sonic": "SONIC"}
COL = {"light": {"ocdetect": "#2a78d6", "eatingspeed": "#eb6834", "motionx": "#1baf7a", "sonic": "#eda100"},
       "dark": {"ocdetect": "#3987e5", "eatingspeed": "#d95926", "motionx": "#199e70", "sonic": "#c98500"}}
THEMES = {"light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7"),
          "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835")}



def _assert_eatingspeed_cache_is_deg_s(cache_dir):
    """This script converts EatingSpeed gyro deg/s -> rad/s itself. Caches built after 2026-09-17 already
    hold rad/s (the loader converts) and carry eatingspeed_units.json; converting again would be wrong."""
    import json as _json
    m = os.path.join(cache_dir, "eatingspeed_units.json")
    if os.path.exists(m) and _json.load(open(m)).get("gyro_units") == "rad/s":
        raise RuntimeError(f"{m} says the cache is already rad/s -- drop this script's deg/s conversion")

def eff_types(lab):
    c = np.bincount(lab, minlength=1000); p = c[c > 0] / c.sum()
    return float(np.exp(-(p * np.log(p)).sum())), int((c > 0).sum())


def vendi(z):
    x = z - 0.0                                   # PCA coords are already pool-centred
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    lam = np.linalg.eigvalsh((x.T @ x) / len(x))
    lam = lam[lam > 1e-12]; lam = lam / lam.sum()
    return float(np.exp(-(lam * np.log(lam)).sum()))


def knn_med(z, k=10, n=4000, rng=None):
    q = z[rng.choice(len(z), min(n, len(z)), replace=False)]
    d2 = (q ** 2).sum(1)[:, None] + (z ** 2).sum(1)[None, :] - 2 * q @ z.T
    d2 = np.partition(np.maximum(d2, 0), k, axis=1)[:, 1:k + 1]
    return float(np.median(np.sqrt(d2[:, -1])))


def main():
    rng = np.random.default_rng(42)
    r = source_ranges()
    ze = np.load(os.path.join(WORK_DIR, "enc_pca64_f32.npy"), mmap_mode="r")
    zp = np.load(os.path.join(WORK_DIR, "phys_pca32_f32.npy"), mmap_mode="r")
    le = np.load(os.path.join(WORK_DIR, "labels_enc_k1000.npy"))
    lp = np.load(os.path.join(WORK_DIR, "labels_phys_k1000.npy"))

    # T-phys centroids (k-means centroid = member mean) for re-assigning corrected windows
    zp_all = np.asarray(zp, dtype=np.float64)
    cnt = np.bincount(lp, minlength=1000)
    cent = np.stack([np.bincount(lp, weights=zp_all[:, d], minlength=1000) for d in range(zp_all.shape[1])], 1) / cnt[:, None]
    del zp_all
    sc = np.load(os.path.join(WORK_DIR, "phys_scaler.npz"))

    def phys_transform(feats):
        blk = np.clip((feats - sc["med"]) / sc["scale"], -8, 8)
        return ((blk - sc["mean"]) @ sc["components"].T) / np.sqrt(sc["explained_variance"])

    _assert_eatingspeed_cache_is_deg_s(CACHE_DIR)
    # EatingSpeed with gyro converted deg/s -> rad/s, on a fixed 50k sample
    lo, hi = r["eatingspeed"]
    es_idx = np.sort(rng.choice(hi - lo, 50000, replace=False))
    x = np.load(os.path.join(CACHE_DIR, "eatingspeed_train_x.npy"), mmap_mode="r")
    feats = []
    for a in range(0, len(es_idx), 5000):
        w = np.asarray(x[es_idx[a:a + 5000]], dtype=np.float32).copy()
        w[:, 3:] /= 57.2958
        feats.append(physf.features(w, np.full(len(w), w.shape[-1], dtype=np.int32)))
    zc = phys_transform(np.concatenate(feats)).astype(np.float32)
    lc = np.argmin((zc ** 2).sum(1)[:, None] + (cent ** 2).sum(1)[None, :] - 2 * zc @ cent.T, axis=1)
    # sanity: the same pipeline without the /57.3 must reproduce the stored labels
    w = np.asarray(x[es_idx[:3000]], dtype=np.float32)
    z0 = phys_transform(physf.features(w, np.full(len(w), 512, dtype=np.int32)))
    l0 = np.argmin((z0 ** 2).sum(1)[:, None] + (cent ** 2).sum(1)[None, :] - 2 * z0 @ cent.T, axis=1)
    agree = float((l0 == lp[lo + es_idx[:3000]]).mean())
    print(f"[check] nearest-centroid reproduces stored T-phys labels for {100*agree:.1f}% of EatingSpeed windows")

    groups = {s: ("stored", s) for s in SOURCES}
    groups["eatingspeed_radps"] = ("corrected", "eatingspeed")
    res = {}
    for g, (kind, s) in groups.items():
        lo, hi = r[s]; n_src = hi - lo if kind == "stored" else len(es_idx)
        rows = {}
        for n in NS:
            if n > n_src:
                continue
            vals = {k: [] for k in ("enc_eff", "enc_touched", "enc_vendi", "phys_eff", "phys_touched", "phys_vendi")}
            for d in range(DRAWS):
                pick = np.sort(rng.choice(n_src, n, replace=False))
                if kind == "stored":
                    gi = lo + pick
                    e, t = eff_types(le[gi]); vals["enc_eff"].append(e); vals["enc_touched"].append(t)
                    vals["enc_vendi"].append(vendi(np.asarray(ze[gi], dtype=np.float64)))
                    zph = np.asarray(zp[gi], dtype=np.float64); lph = lp[gi]
                else:
                    zph = zc[pick].astype(np.float64); lph = lc[pick]
                e, t = eff_types(lph); vals["phys_eff"].append(e); vals["phys_touched"].append(t)
                vals["phys_vendi"].append(vendi(zph))
            rows[n] = {k: float(np.mean(v)) for k, v in vals.items() if v}
        if kind == "stored":
            gi = lo + np.sort(rng.choice(n_src, N_REF, replace=False))
            rows["knn_enc"] = knn_med(np.asarray(ze[gi], dtype=np.float32), rng=rng)
            rows["knn_phys"] = knn_med(np.asarray(zp[gi], dtype=np.float32), rng=rng)
        else:
            rows["knn_phys"] = knn_med(zc[rng.choice(len(zc), N_REF, replace=False)], rng=rng)
        res[g] = rows

    print(f"\n=== matched N = {N_REF:,} windows per source (mean of {DRAWS} draws)")
    hdr = f"{'source':20s} {'T-enc eff':>9s} {'touched':>8s} {'Vendi':>6s} {'10-NN':>6s} | {'T-phys eff':>10s} {'touched':>8s} {'Vendi':>6s} {'10-NN':>6s}"
    print(hdr)
    for g, rows in res.items():
        m = rows[N_REF]
        print(f"{g:20s} {m.get('enc_eff', np.nan):9.1f} {m.get('enc_touched', np.nan):8.0f} {m.get('enc_vendi', np.nan):6.2f} "
              f"{rows.get('knn_enc', np.nan):6.2f} | {m['phys_eff']:10.1f} {m['phys_touched']:8.0f} {m['phys_vendi']:6.2f} {rows['knn_phys']:6.2f}")
    json.dump({"n_ref": N_REF, "draws": DRAWS, "label_reproduction_check": agree,
               "results": {g: {str(k): v for k, v in rows.items()} for g, rows in res.items()}},
              open(os.path.join(OUT_DIR, "source_diversity.json"), "w"), indent=1)

    # ---------------- figure ----------------
    panels = [("enc_eff", "T-enc · effective motion types"), ("enc_vendi", "T-enc · Vendi score (no clustering)"),
              ("phys_eff", "T-phys · effective motion types"), ("phys_vendi", "T-phys · Vendi score (no clustering)")]
    for theme, t in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": t["ink"],
                             "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"]})
        fig, axes = plt.subplots(1, 4, figsize=(19.5, 4.9), facecolor=t["surface"], gridspec_kw=dict(wspace=0.3))
        for ax, (key, title) in zip(axes, panels):
            ax.set_facecolor(t["surface"])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=t["ink2"], labelsize=8.5, length=3)
            ax.grid(True, color=t["grid"], lw=0.6); ax.set_axisbelow(True)
            ends = []
            for g, rows in res.items():
                ns = [n for n in NS if n in rows and key in rows[n]]
                if not ns:
                    continue
                s = "eatingspeed" if g.startswith("eatingspeed") else g
                ys = [rows[n][key] for n in ns]
                dashed = g == "eatingspeed_radps"
                ax.plot(ns, ys, color=COL[theme][s], lw=2, ls=(0, (4, 2.5)) if dashed else "-",
                        marker="o", ms=4.5, mec=t["surface"], mew=1.2)
                ends.append((ys[-1], ns[-1], NAMES[s] + (" (gyro→rad/s)" if dashed else ""), COL[theme][s]))
            ends.sort()
            lo_y, hi_y = ax.get_ylim(); gap = 0.075 * (hi_y - lo_y); placed = []
            for y, n, lab, c in ends:
                yy = y
                for p in placed:
                    if abs(yy - p) < gap:
                        yy = p + gap
                placed.append(yy)
                ax.annotate(lab, (n, y), xytext=(60000, yy), textcoords="data", fontsize=8.3, color=t["ink2"],
                            va="center", arrowprops=dict(arrowstyle="-", color=c, lw=0.8))
            ax.axvline(N_REF, color=t["muted"], lw=0.9, ls=(0, (2, 3)))
            ax.set_xscale("log"); ax.set_xlim(200, 4.2e5)
            ax.set_xticks([250, 1000, 5000, 20000, 50000]); ax.set_xticklabels(["250", "1k", "5k", "20k", "50k"])
            ax.set_xlabel("windows drawn from the source", fontsize=9)
            ax.set_title(title, fontsize=10.3, loc="left", pad=8, color=t["ink"])
        axes[0].set_ylabel("diversity (higher = more diverse)", fontsize=9)
        fig.suptitle("Diversity per source at matched sample size — rarefaction curves, mean of 3 random draws "
                     "(dotted line = 20k, the comparison point)", fontsize=11.5, x=0.01, ha="left", y=1.03, color=t["ink"])
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(OUT_DIR, f"source_diversity_4src_{theme}.{ext}"), dpi=165 if ext == "png" else None,
                        bbox_inches="tight", facecolor=t["surface"])
        plt.close(fig)
    print("\nwrote out/source_diversity.json and out/source_diversity_4src_{light,dark}.{png,pdf}")


if __name__ == "__main__":
    main()
