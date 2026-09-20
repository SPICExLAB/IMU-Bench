#!/usr/bin/env python
"""What does one T-enc motion-type cluster look like in the raw signal?

Picks six clusters by an explicit rule (not by eye), prints their physical
profile against pool percentiles, and draws the acc and gyro magnitude of three
random member windows each.

  1 largest cluster overall
  2 largest EatingSpeed-dominated cluster
  3 largest SONIC-dominated cluster
  4 largest Motion-X-dominated cluster
  5 stillest cluster (highest median stillness_frac, >=1,000 windows)
  6 most rhythmic cluster (highest median autocorr_peak, >=1,000 windows)

Writes out/cluster_portraits_4src_{light,dark}.{png,pdf}, out/cluster_portraits.json
"""
from __future__ import annotations

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

FS = 50.0
NAMES = {"ocdetect": "OCDetect", "eatingspeed": "EatingSpeed", "motionx": "Motion-X", "sonic": "SONIC"}
FEATS = ["acc_mag_std", "gyr_mag_mean", "stillness_frac", "acc_dom_hz", "autocorr_peak",
         "autocorr_peak_hz", "gravity_dir_change_deg"]
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", acc="#2a78d6", gyr="#eb6834"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", acc="#3987e5", gyr="#d95926"),
}


def pick(labels, M, F, fnames, rng):
    size = M.sum(1)
    frac = M / np.maximum(size[:, None], 1)
    dom = frac.argmax(1)
    samp = np.sort(rng.choice(len(labels), 400_000, replace=False))
    Ls, Fs = labels[samp], np.asarray(F[samp], dtype=np.float32)

    def med(c, name):
        v = Fs[Ls == c, fnames.index(name)]
        return float(np.median(v)) if len(v) else np.nan

    big = np.where(size >= 1000)[0]
    rules = [("largest overall", int(np.argmax(size)))]
    for j, s in enumerate(SOURCES):
        if s == "ocdetect":
            continue
        cand = np.where(dom == j)[0]
        rules.append((f"largest {NAMES[s]}-dominated", int(cand[np.argmax(size[cand])])))
    still = {c: med(c, "stillness_frac") for c in big}
    rules.append(("stillest", int(max(still, key=still.get))))
    rhy = {c: med(c, "autocorr_peak") for c in big}
    rules.append(("most rhythmic", int(max(rhy, key=rhy.get))))
    # keep the order: overall, eating, motionx, sonic -> reorder to overall, eating, sonic, motionx
    rules = [rules[0], rules[1], rules[3], rules[2], rules[4], rules[5]]
    prof = []
    for why, c in rules:
        prof.append({"rule": why, "cluster": c, "size": int(size[c]),
                     "source_pct": {s: round(100 * float(frac[c, j]), 1) for j, s in enumerate(SOURCES)},
                     "median": {f: round(med(c, f), 3) for f in FEATS}})
    return prof


def load_window(g, ranges):
    for s, (lo, hi) in ranges.items():
        if lo <= g < hi:
            x = np.load(os.path.join(CACHE_DIR, f"{s}_train_x.npy"), mmap_mode="r")
            w = np.asarray(x[g - lo], dtype=np.float32)
            tp = os.path.join(CACHE_DIR, f"{s}_train_tmask.npy")
            L = int(np.load(tp, mmap_mode="r")[g - lo]) if os.path.exists(tp) else w.shape[-1]
            return s, w[:, :L]
    raise IndexError(g)


def main():
    rng = np.random.default_rng(42)
    labels = np.load(os.path.join(WORK_DIR, "labels_enc_k1000.npy"))
    M = np.load(os.path.join(OUT_DIR, "cluster_source_matrix_enc.npy"))
    F = np.load(os.path.join(WORK_DIR, "phys_feats_f32.npy"), mmap_mode="r")
    fnames = json.load(open(os.path.join(WORK_DIR, "phys_feature_names.json")))["names"]
    ranges = source_ranges()
    prof = pick(labels, M, F, fnames, rng)

    pool = np.asarray(F[np.sort(rng.choice(len(labels), 300_000, replace=False))], dtype=np.float32)
    pct = {f: [float(np.percentile(pool[:, fnames.index(f)], q)) for q in (10, 50, 90)] for f in FEATS}
    for p in prof:
        dom_src = max(p["source_pct"], key=p["source_pct"].get)
        lo, hi = ranges[dom_src]
        members = np.where(labels[lo:hi] == p["cluster"])[0] + lo   # dominant source only
        p["examples"] = [int(g) for g in np.sort(rng.choice(members, 3, replace=False))]
        print(f"\n[{p['rule']}] cluster {p['cluster']}  n={p['size']:,}  sources {p['source_pct']}")
        for f in FEATS:
            print(f"   {f:24s} {p['median'][f]:8.3g}   pool p10/p50/p90 "
                  + " / ".join(f"{v:.3g}" for v in pct[f]))
    json.dump({"pool_percentiles_10_50_90": pct, "clusters": prof},
              open(os.path.join(OUT_DIR, "cluster_portraits.json"), "w"), indent=1)

    for theme, t in THEMES.items():
        plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": t["ink"],
                             "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"]})
        fig, axes = plt.subplots(2, 6, figsize=(18, 6.6), facecolor=t["surface"], sharex=True,
                                 gridspec_kw=dict(hspace=0.12, wspace=0.34))
        for col, p in enumerate(prof):
            dom_src = max(p["source_pct"], key=p["source_pct"].get)
            for k, g in enumerate(p["examples"]):
                s, w = load_window(g, ranges)
                tt = np.arange(w.shape[1]) / FS
                a = np.linalg.norm(w[:3], axis=0)
                gy = np.linalg.norm(w[3:], axis=0)
                alpha = [1.0, 0.65, 0.4][k]
                axes[0, col].plot(tt, a, color=t["acc"], lw=1.1, alpha=alpha)
                axes[1, col].plot(tt, gy, color=t["gyr"], lw=1.1, alpha=alpha)
            m = p["median"]
            axes[0, col].set_title(
                f"{p['rule']}\n#{p['cluster']} · {p['size']:,} win\n"
                f"{p['source_pct'][dom_src]:.0f}% {NAMES[dom_src]}\n"
                f"still {m['stillness_frac']:.2f} · rhythm {m['autocorr_peak']:.2f}",
                fontsize=8.6, loc="left", color=t["ink"], pad=6)
            for r in (0, 1):
                ax = axes[r, col]
                ax.set_facecolor(t["surface"])
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.tick_params(colors=t["ink2"], labelsize=7.5, length=2.5)
                ax.grid(True, color=t["grid"], lw=0.5)
                ax.set_axisbelow(True)
            axes[1, col].set_xlabel("time (s)", fontsize=8.5)
        axes[0, 0].set_ylabel("|acc|  (m/s²)", fontsize=9, color=t["acc"])
        axes[1, 0].set_ylabel("|gyro|  (cached units)", fontsize=9, color=t["gyr"])
        fig.suptitle("Six T-enc motion-type clusters in the raw signal — three random member windows each "
                     "from the dominant source (opacity separates the windows)",
                     fontsize=11.5, x=0.01, ha="left", y=1.04, color=t["ink"])
        for ext in ("png", "pdf"):
            pth = os.path.join(OUT_DIR, f"cluster_portraits_4src_{theme}.{ext}")
            fig.savefig(pth, dpi=160 if ext == "png" else None, bbox_inches="tight", facecolor=t["surface"])
        plt.close(fig)
    print("\nwrote out/cluster_portraits_4src_{light,dark}.{png,pdf}")


if __name__ == "__main__":
    main()
