#!/usr/bin/env python
"""Which data source is each of the 1,000 motion-type clusters made of?

For both type definitions on the 4-source pool (T-enc = mae_4src_z CLS -> PCA-64
-> k-means K=1000, used by the printed-grid replication; T-phys = 40-d physical
features -> PCA-32 -> k-means K=1000, used by the type ladders), three views:

  left    every cluster, largest first, stacked by source in absolute windows
          (log rank axis so the handful of clusters the K=1..100 grid arms draw
          from are readable);
  middle  the same clusters as a 100 % stack -- which source owns each one;
  right   the pooled composition of the K largest clusters, i.e. exactly what a
          largest-first grid arm at each K could sample from.

Reads out/cluster_source_matrix_{enc,phys}.npy (written by the composition pass).
Writes out/cluster_sources_4src_{light,dark}.{png,pdf}
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

SRC = ["ocdetect", "eatingspeed", "motionx", "sonic"]
NAMES = {"ocdetect": "OCDetect", "eatingspeed": "EatingSpeed", "motionx": "Motion-X", "sonic": "SONIC"}
KS = [1, 5, 10, 25, 50, 100, 1000]
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  cat=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835",
                  cat=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}
SPACES = [("enc", "T-enc  (MAE representation; used by the grid)"),
          ("phys", "T-phys  (physical features; used by the type ladders)")]


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], labelsize=8.5, length=3)
    ax.yaxis.grid(True, color=t["grid"], lw=0.6)
    ax.set_axisbelow(True)


def draw(theme):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": t["ink"],
                         "axes.labelcolor": t["ink2"], "axes.edgecolor": t["axis"]})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), facecolor=t["surface"],
                             gridspec_kw=dict(width_ratios=[1.3, 1.3, 1.12], wspace=0.24, hspace=0.5))
    for row, (space, title) in enumerate(SPACES):
        M = np.load(os.path.join(OUT_DIR, f"cluster_source_matrix_{space}.npy"))
        size = M.sum(1); order = np.argsort(-size); Ms = M[order]
        rank = np.arange(1, len(Ms) + 1)
        frac = Ms / size[order][:, None]
        dom = frac.max(1)

        # -- left: absolute stack, log rank
        ax = axes[row, 0]; style(ax, t)
        bottom = np.zeros(len(Ms))
        for j, s in enumerate(SRC):
            ax.bar(rank - 0.5, Ms[:, j], bottom=bottom, width=1.0, color=t["cat"][j],
                   lw=0, align="edge", label=NAMES[s])
            bottom += Ms[:, j]
        ax.set_xscale("log"); ax.set_xlim(0.5, 1000.5)
        ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("cluster rank (largest first)", fontsize=9)
        ax.set_ylabel("windows in cluster", fontsize=9)
        ax.set_title(f"{title}", fontsize=10.5, loc="left", color=t["ink"], pad=10)
        for k in KS[:-1]:
            ax.axvline(k + 0.5, color=t["axis"], lw=0.7, ls=(0, (2, 2)))
        ax.annotate(f"largest cluster = {size[order[0]]:,} windows", (1.5, size[order[0]]), xytext=(6, -3),
                    textcoords="offset points", fontsize=8, color=t["ink2"], va="top")
        if row == 0:
            ax.legend(frameon=False, fontsize=8.5, loc="center right", ncol=1, labelcolor=t["ink2"])

        # -- middle: 100 % stack, sorted by rank, log axis
        ax = axes[row, 1]; style(ax, t)
        bottom = np.zeros(len(Ms))
        for j, s in enumerate(SRC):
            ax.bar(rank - 0.5, 100 * frac[:, j], bottom=bottom, width=1.0,
                   color=t["cat"][j], lw=0, align="edge")
            bottom += 100 * frac[:, j]
        ax.set_xscale("log"); ax.set_xlim(0.5, 1000.5); ax.set_ylim(0, 100)
        ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("cluster rank (largest first)", fontsize=9)
        ax.set_ylabel("share of cluster (%)", fontsize=9)
        n90 = int((dom > 0.9).sum())
        ax.set_title(f"{n90} of 1,000 clusters are >90 % one source",
                     fontsize=10.5, loc="left", color=t["ink"], pad=10)
        for k in KS[:-1]:
            ax.axvline(k + 0.5, color=t["surface"], lw=0.9)

        # -- right: pooled composition of the K largest clusters
        ax = axes[row, 2]; style(ax, t)
        x = np.arange(len(KS)); bottom = np.zeros(len(KS))
        for j, s in enumerate(SRC):
            vals = np.array([100 * Ms[:k, j].sum() / Ms[:k].sum() for k in KS])
            ax.bar(x, vals, bottom=bottom, width=0.72, color=t["cat"][j], lw=0)
            for xi, v, b in zip(x, vals, bottom):
                if v >= 6:
                    ax.text(xi, b + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8,
                            color="#ffffff" if j in (0, 1) else t["ink"])
            bottom += vals
        tot = [Ms[:k].sum() for k in KS]
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}\n{n/1000:.0f}k" if n < 1e6 else f"K={k}\n{n/1e6:.2f}M" for k, n in zip(KS, tot)], fontsize=7.5)
        ax.set_ylim(0, 100); ax.set_ylabel("share of the K-cluster pool (%)", fontsize=9)
        ax.set_title("pool of the K largest clusters", fontsize=10.5, loc="left", color=t["ink"], pad=10)

    fig.suptitle("Source composition of the 1,000 motion-type clusters — 4-source pool (1,738,094 windows: "
                 "OCDetect 72.1 %, EatingSpeed 16.6 %, SONIC 8.5 %, Motion-X 2.9 %)",
                 fontsize=12, x=0.01, ha="left", color=t["ink"], y=1.0)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"cluster_sources_4src_{theme}.{ext}")
        fig.savefig(p, dpi=170 if ext == "png" else None, bbox_inches="tight", facecolor=t["surface"])
        print("  wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    for th in THEMES:
        draw(th)
