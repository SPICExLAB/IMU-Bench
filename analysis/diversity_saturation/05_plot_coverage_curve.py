#!/usr/bin/env python3
"""Cluster-coverage vs pool fraction at K=1000 (EXPERIMENTS.md 11j diagnostics).

Coverage = fraction of the K=1000 k-means motion modes that a uniform random
draw of that size touches. Numbers are the 5-draw means from
04_verify_coverage.py (K=1000, seed 42); the +/- s.d. is plotted as error
bars but is smaller than the markers everywhere except 0.1%.

The 100% point is definitional (the whole pool contains every cluster) rather
than a 5-draw measurement; it is annotated as such. It is NOT drawn as an open
marker: every marker carries a 2px surface ring, which reads as an open circle
wherever it sits on the shaded band, so that encoding was ambiguous.

The dashed rule is a THRESHOLD, not a gridline: coverage >= 99% is the level
at which adding data stops adding motion modes, and every fraction that was
actually pretrained (1.3% and up) sits above it.

Usage:
    python analysis/diversity_saturation/05_plot_coverage_curve.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same validated palette instance as the sibling figures.
ACCENT = "#2a78d6"       # slot 1, blue
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# fraction %, coverage %, s.d. %  -- 04_verify_coverage.py, K=1000 seed 42
POINTS = [
    (0.01,  28.84, 0.24),
    (0.10,  92.70, 0.73),
    (0.50,  99.94, 0.05),
    (1.30, 100.00, 0.00),
    (5.00, 100.00, 0.00),
    (10.60, 100.00, 0.00),
    (25.00, 100.00, 0.00),
]
DEFINITIONAL = (100.0, 100.00)   # whole pool, by construction
THRESHOLD = 99.0
PRETRAINED_FROM = 1.30           # smallest fraction any pretrain actually used


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).with_name("coverage_curve_k1000.png")

    xs = [p[0] for p in POINTS]
    ys = [p[1] for p in POINTS]
    sd = [p[2] for p in POINTS]

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=args.dpi, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.xaxis.grid(False)

    # band marking the range that was actually pretrained
    ax.axvspan(PRETRAINED_FROM, 130, color=ACCENT, alpha=0.04, linewidth=0, zorder=0)

    # the 99% threshold -- dashed because it IS a threshold, not a grid rule
    ax.axhline(THRESHOLD, color=INK_2, linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)
    ax.text(0.0088, THRESHOLD - 2.6, "99% coverage", ha="left", va="top",
            fontsize=8.5, color=INK_2)

    ax.errorbar(xs, ys, yerr=sd, fmt="none", ecolor=ACCENT, elinewidth=1.4,
                capsize=3, capthick=1.4, zorder=3)
    ax.plot(xs, ys, color=ACCENT, linewidth=2.0, marker="o", markersize=8,
            markerfacecolor=ACCENT, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=4, clip_on=False)
    ax.plot([xs[-1], DEFINITIONAL[0]], [ys[-1], DEFINITIONAL[1]], color=ACCENT,
            linewidth=2.0, zorder=3)
    ax.plot([DEFINITIONAL[0]], [DEFINITIONAL[1]], marker="o", markersize=8,
            markerfacecolor=ACCENT, markeredgecolor=SURFACE, markeredgewidth=2,
            zorder=4, clip_on=False)
    ax.annotate("definitional", (DEFINITIONAL[0], DEFINITIONAL[1]),
                textcoords="offset points", xytext=(0, -15), ha="center",
                fontsize=7.5, color=MUTED)

    ax.set_xscale("log")
    ax.set_xlim(0.0085, 130)
    ticks = [0.01, 0.1, 0.5, 1.3, 5, 10.6, 25, 100]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}%" for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0, 108)
    # the 99% level is carried by the dashed threshold + its own label, so the
    # ticks stay on round numbers and 100 is not silently missing
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=8.5, colors=MUTED, length=0, pad=4)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK_2)

    ax.set_xlabel("Fraction of the pretraining pool (log scale)", fontsize=9,
                  color=MUTED, labelpad=6)
    ax.set_ylabel("Motion-mode coverage (% of K=1000 clusters) ↑", fontsize=9,
                  color=MUTED, labelpad=6)

    # selective direct labels: the knee and the two sub-saturation points
    for x, y in ((0.01, 28.84), (0.10, 92.70), (0.50, 99.94)):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(7, -12), ha="left", fontsize=8.5, color=INK_2)

    ax.text(PRETRAINED_FROM * 1.12, 8, "fractions actually pretrained\n(1.3% – 100%): all at 100%",
            ha="left", va="bottom", fontsize=8.5, color=MUTED, linespacing=1.5)

    ax.text(0.0, 1.03, "Coverage saturates far below the smallest fraction ever trained",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=10, color=INK)

    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.145)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {out}")

    print(f"\n{'fraction':>10}{'coverage':>12}{'s.d.':>8}")
    for x, y, s in POINTS:
        print(f"{x:>9.2f}%{y:>11.2f}%{s:>8.2f}")
    print(f"{DEFINITIONAL[0]:>9.2f}%{DEFINITIONAL[1]:>11.2f}%{'--':>8}  (definitional)")


if __name__ == "__main__":
    main()
