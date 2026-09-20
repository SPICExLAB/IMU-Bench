#!/usr/bin/env python3
"""Motion-mode coverage vs pool fraction on the 5-SOURCE pool (paper Fig 7b, 5src version).

The paper's coverage panel (05_plot_coverage_curve.py) was measured on the 11-source pool.
The paper now reports the 5-source MAE, so this re-measures coverage on the pool that model
and the 11l grid were trained on: realdata, ocdetect, eatingspeed, motionx, sonic
(1,959,288 windows), with the same K=1000 k-means partition the grid used
(labels5src_k1000.npy, written by 10_grid_select.py's upstream stage).

Coverage = fraction of the 1000 clusters that a uniform random draw of that size touches;
5 draws per fraction (seed 42), mean and s.d. The 100% point is definitional.

Writes analysis/diversity_saturation/out/coverage_5src.json and
       analysis/diversity_saturation/coverage_curve_k1000_5src_{light,dark}.{png,pdf}
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
FRACTIONS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25]   # 100% is definitional
N_DRAWS = 5
K = 1000
GRID_K1000 = [0.1, 1, 10, 100]      # volumes of the grid's K=1000 row (%)
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", accent="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", accent="#3987e5"),
}


def measure():
    labels = np.load(os.path.join(WORK_DIR, "labels5src_k1000.npy")).astype(np.int32)
    n = len(labels)
    rng = np.random.default_rng(42)
    pts = []
    for f in FRACTIONS:
        m = max(1, int(round(f * n)))
        cov = [len(np.unique(labels[rng.choice(n, size=m, replace=False)])) / K * 100 for _ in range(N_DRAWS)]
        pts.append({"fraction_pct": f * 100, "n_windows": m, "coverage_pct": float(np.mean(cov)),
                    "sd_pct": float(np.std(cov))})
        print(f"{f*100:>8.3f}%  n={m:>9,}  coverage {np.mean(cov):6.2f}% +/- {np.std(cov):.2f}")
    sizes = np.bincount(labels, minlength=K)
    out = {"pool": "5-source (realdata, ocdetect, eatingspeed, motionx, sonic)", "n_pool": int(n), "k": K,
           "n_draws": N_DRAWS, "seed": 42, "points": pts,
           "cluster_size": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())}}
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "out", "coverage_5src.json"), "w"), indent=1)
    return out


def plot(res):
    pts = res["points"]
    xs = [p["fraction_pct"] for p in pts]
    ys = [p["coverage_pct"] for p in pts]
    sd = [p["sd_pct"] for p in pts]
    for theme, th in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=200, facecolor=th["surface"])
        ax.set_facecolor(th["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(th["axis"])
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=th["grid"], linewidth=0.8)
        ax.axvspan(GRID_K1000[0], 130, color=th["accent"], alpha=0.05, linewidth=0, zorder=0)
        ax.axhline(99, color=th["ink2"], linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)
        ax.text(0.0088, 96.4, "99% coverage", ha="left", va="top", fontsize=8.5, color=th["ink2"])
        ax.errorbar(xs, ys, yerr=sd, fmt="none", ecolor=th["accent"], elinewidth=1.4, capsize=3, zorder=3)
        ax.plot(xs + [100], ys + [100], color=th["accent"], linewidth=2.0, marker="o", markersize=7,
                markeredgecolor=th["surface"], markeredgewidth=1.8, zorder=4, clip_on=False)
        ax.annotate("definitional", (100, 100), textcoords="offset points", xytext=(4, -15), ha="right",
                    fontsize=7.5, color=th["muted"])
        for x, y in zip(xs, ys):
            if y < 99.5:
                ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(7, -12), ha="left",
                            fontsize=8.5, color=th["ink2"])
        ax.set_xscale("log")
        ax.set_xlim(0.0085, 130)
        ticks = [0.01, 0.1, 1, 10, 100]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}%" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=8.5, colors=th["muted"], length=0, pad=4)
        for lbl in ax.get_xticklabels():
            lbl.set_color(th["ink2"])
        ax.set_xlabel(f"Fraction of the 5-source pool ({res['n_pool']:,} windows, log scale)", fontsize=9,
                      color=th["muted"], labelpad=6)
        ax.set_ylabel("Motion-mode coverage (% of K=1000 clusters)", fontsize=9, color=th["muted"], labelpad=6)
        ax.text(GRID_K1000[0] * 1.12, 8, "grid volumes at K=1000\n(0.1% – 100%)", ha="left", va="bottom",
                fontsize=8.5, color=th["muted"], linespacing=1.5)
        ax.text(0.0, 1.03, "5-source pool: coverage saturates within the first percent of data",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=th["ink"])
        fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.145)
        for ext in ("png", "pdf"):
            p = os.path.join(HERE, f"coverage_curve_k1000_5src_{theme}.{ext}")
            fig.savefig(p, facecolor=th["surface"])
            print("wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    plot(measure())
