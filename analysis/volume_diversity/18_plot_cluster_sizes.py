#!/usr/bin/env python
"""How big are the 1,000 motion-type clusters, and what does that cost a
random-K arm?

The grid picks the K LARGEST clusters. That rule is what makes the K axis
confounded (see 17_plot_cluster_sources.py: every arm below K=1000 is OCDetect
plus a little EatingSpeed). The alternative is to pick K clusters at random,
which restores pool source proportions but buys far less data per K, because a
random cluster is average-sized rather than largest.

  (a) rank-size curve, log-log, both type spaces, against the flat 1,738-window
      line a perfectly even partition would give
  (b) size distribution, log-spaced bins
  (c) cumulative share of the pool held by the N largest clusters
  (d) what each rule can actually supply at each K (random: median and 5-95 %
      over 4,000 draws), against the five volume targets of the grid

Reads out/cluster_source_matrix_{enc,phys}.npy
Writes out/cluster_sizes_4src_{light,dark}.{png,pdf}
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import OUT_DIR

KS = [1, 5, 10, 25, 50, 100, 1000]
POOL = 1_738_094
VOLS = [(174, "0.01%"), (1738, "0.1%"), (17381, "1%"), (173809, "10%"), (POOL, "100%")]
SPACES = [("enc", "T-enc"), ("phys", "T-phys")]
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", cat=["#2a78d6", "#eb6834"], band="#2a78d6"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", cat=["#3987e5", "#d95926"], band="#3987e5"),
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=8.5, length=3)
    ax.grid(True, color=t["grid"], lw=0.6)
    ax.set_axisbelow(True)


def draw(theme):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": t["ink"],
                         "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"]})
    S = {k: np.load(os.path.join(OUT_DIR, f"cluster_source_matrix_{k}.npy")).sum(1) for k, _ in SPACES}
    fig, ax = plt.subplots(2, 2, figsize=(13.2, 8.6), facecolor=t["surface"],
                           gridspec_kw=dict(wspace=0.24, hspace=0.42))

    # (a) rank-size
    a = ax[0, 0]; style(a, t)
    for j, (sp, nm) in enumerate(SPACES):
        srt = np.sort(S[sp])[::-1]
        a.plot(np.arange(1, 1001), srt, color=t["cat"][j], lw=2, label=nm)
        a.annotate(nm, (1.15, srt[0]), xytext=(3, 6 if j == 0 else -12), textcoords="offset points",
                   color=t["cat"][j], fontsize=9.5, fontweight="bold")
    a.axhline(1738, color=t["muted"], lw=1.4, ls=(0, (4, 3)))
    a.annotate("1,738 = an even partition", (300, 1738), xytext=(0, 11), textcoords="offset points",
               ha="center", va="bottom", fontsize=8.5, color=t["muted"])
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks(KS); a.set_xticklabels([str(k) for k in KS])
    a.set_yticks([200, 500, 1000, 2000, 5000]); a.set_yticklabels(["200", "500", "1,000", "2,000", "5,000"])
    a.set_xlabel("cluster rank (largest first)", fontsize=9)
    a.set_ylabel("windows in cluster", fontsize=9)
    a.set_title("(a)  Sizes span 56× but fall off gently", fontsize=10.5, loc="left", pad=9, color=t["ink"])

    # (b) size distribution
    b = ax[0, 1]; style(b, t)
    bins = np.logspace(np.log10(100), np.log10(7000), 34)
    for j, (sp, nm) in enumerate(SPACES):
        h, e = np.histogram(S[sp], bins=bins)
        b.step(e[:-1], h, where="post", color=t["cat"][j], lw=2)
        b.fill_between(e[:-1], h, step="post", color=t["cat"][j], alpha=0.10, lw=0)
    b.set_ylim(0, 104)
    for j, (sp, nm) in enumerate(SPACES):
        med = np.median(S[sp])
        b.axvline(med, color=t["cat"][j], lw=1.2, ls=(0, (3, 2)))
        b.annotate(f"{nm} median\n{med:,.0f}", (med, 100), xytext=(-6 if j == 0 else 6, 0),
                   textcoords="offset points", ha="right" if j == 0 else "left", va="top",
                   fontsize=8.5, color=t["cat"][j])
    b.set_xscale("log")
    b.set_xticks([200, 500, 1000, 2000, 5000]); b.set_xticklabels(["200", "500", "1,000", "2,000", "5,000"])
    b.set_xlabel("windows in cluster", fontsize=9)
    b.set_ylabel("number of clusters", fontsize=9)
    b.set_title("(b)  Unimodal, no giant outlier", fontsize=10.5, loc="left", pad=9, color=t["ink"])

    # (c) cumulative mass
    c = ax[1, 0]; style(c, t)
    for j, (sp, nm) in enumerate(SPACES):
        srt = np.sort(S[sp])[::-1]
        cum = 100 * np.cumsum(srt) / srt.sum()
        c.plot(np.arange(1, 1001), cum, color=t["cat"][j], lw=2, label=nm)
    c.plot(np.arange(1, 1001), 100 * np.arange(1, 1001) / 1000, color=t["muted"], lw=1.4, ls=(0, (4, 3)))
    c.annotate("even partition", (700, 70), xytext=(4, -16), textcoords="offset points",
               fontsize=8.5, color=t["muted"])
    c.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=t["ink2"])
    srt = np.sort(S["enc"])[::-1]
    for k in (10, 100):
        v = 100 * srt[:k].sum() / srt.sum()
        c.plot([k], [v], "o", ms=7, color=t["cat"][0], mec=t["surface"], mew=2, zorder=5)
        c.annotate(f"top {k} = {v:.0f}%", (k, v), xytext=(-9, 6), textcoords="offset points",
                   ha="right", fontsize=8.5, color=t["ink2"])
    c.set_xscale("log"); c.set_xticks(KS); c.set_xticklabels([str(k) for k in KS])
    c.set_ylim(0, 100)
    c.set_xlabel("number of largest clusters", fontsize=9)
    c.set_ylabel("share of the pool (%)", fontsize=9)
    c.set_title("(c)  A mild head: 100 clusters hold a quarter of the pool", fontsize=10.5, loc="left", pad=9, color=t["ink"])

    # (d) what each rule supplies, T-enc
    d = ax[1, 1]; style(d, t)
    sz = S["enc"]; srt = np.sort(sz)[::-1]
    rng = np.random.default_rng(42)
    kk = np.array([1, 2, 3, 5, 8, 10, 15, 25, 40, 50, 75, 100, 150, 250, 400, 650, 1000])
    lf = np.array([srt[:k].sum() for k in kk], dtype=float)
    med, lo, hi = [], [], []
    for k in kk:
        dr = np.array([sz[rng.choice(1000, k, replace=False)].sum() for _ in range(4000)])
        med.append(np.median(dr)); lo.append(np.percentile(dr, 5)); hi.append(np.percentile(dr, 95))
    for n, lab in VOLS:
        d.axhline(n, color=t["muted"], lw=0.9, ls=(0, (2, 3)))
        d.annotate(lab, (1, n), xytext=(2, 3), textcoords="offset points", fontsize=8, color=t["muted"])
    d.fill_between(kk, lo, hi, color=t["band"], alpha=0.16, lw=0)
    d.plot(kk, med, color=t["cat"][0], lw=2.2)
    d.plot(kk, lf, color=t["ink2"], lw=2, ls=(0, (5, 2.5)))
    d.annotate("largest-first", (40, np.interp(40, kk, lf)), xytext=(-2, 10), textcoords="offset points",
               ha="right", fontsize=9.5, color=t["ink2"], fontweight="bold")
    d.annotate("random K\n(median, 5–95%)", (150, np.interp(150, kk, med)), xytext=(6, -26),
               textcoords="offset points", fontsize=9.5, color=t["cat"][0], fontweight="bold")
    d.set_xscale("log"); d.set_yscale("log")
    d.set_xticks(KS); d.set_xticklabels([str(k) for k in KS])
    d.set_yticks([1e2, 1e3, 1e4, 1e5, 1e6]); d.set_yticklabels(["100", "1k", "10k", "100k", "1M"])
    d.set_xlabel("K (motion types the arm may draw from)", fontsize=9)
    d.set_ylabel("windows the K clusters can supply", fontsize=9)
    d.set_title("(d)  T-enc · random K buys ~2.5× less data per K", fontsize=10.5, loc="left", pad=9, color=t["ink"])

    fig.suptitle("Cluster size distribution — 4-source pool, K=1000 partition, 1,738,094 windows",
                 fontsize=12.5, x=0.008, ha="left", color=t["ink"], y=0.985)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"cluster_sizes_4src_{theme}.{ext}")
        fig.savefig(p, dpi=170 if ext == "png" else None, bbox_inches="tight", facecolor=t["surface"])
        print("  wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    for th in THEMES:
        draw(th)
