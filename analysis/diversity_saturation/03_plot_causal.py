#!/usr/bin/env python3
"""Redundant-vs-diverse column chart for the diversity-saturation causal test.

Five metrics with incompatible units (val loss ~0.017, accuracy ~61%) and
opposite directions cannot share a raw axis, and a dual axis would be a lie.
They are indexed to a common base instead: each metric is expressed as a
percentage of the diverse arm, sign-corrected so LONGER IS ALWAYS BETTER.
The absolute value is printed on every bar, so nothing is hidden by the
normalization -- the chart carries the comparison, the labels carry the data.

Both arms are N=46,116 windows from the same pool at the same 105K fixed-step
budget; they differ only in how those windows were selected. See
EXPERIMENTS.md 11j.

Usage:
    python analysis/diversity_saturation/03_plot_causal.py
    python analysis/diversity_saturation/03_plot_causal.py -o out.png --dpi 400
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

# Palette slots 1-2, validated all-pairs in light mode (worst CVD dE 9.2,
# worst normal-vision dE 24.0) -- same instance analysis/plot_data_scaling.py
# uses, so the two figures read as one system.
DIVERSE = "#2a78d6"      # slot 1, blue
REDUNDANT = "#eb6834"    # slot 2, orange

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# (label, diverse, redundant, higher_is_better, value format)
# Arm pose keeps 5 decimals rather than 6: the two arms differ in the 5th
# (0.01704 vs 0.01845), and a 9-character label collides with its neighbour
# column once the bars are vertical.
METRICS = [
    ("PAMAP2\nprobe",     56.64,    30.28,    True,  "{:.2f}%"),
    ("WISDM\nprobe",      61.51,    35.31,    True,  "{:.2f}%"),
    ("Odometry\nprobe",   0.02739,  0.0327,   False, "{:.5f}"),
    ("MAE\npretrain",     0.7753,   0.9217,   False, "{:.4f}"),
    ("Arm pose\nprobe",   0.017038, 0.018452, False, "{:.5f}"),
]

# Column width is capped, not fitted to the slot: at this figure size 0.20
# data units lands just under the 24px ceiling, and the rest of the band is air.
BAR_W = 0.20
GAP = 0.025        # the 2px surface gap between the touching columns
DIV_LABEL_Y = 115  # one fixed line carries the reference arm's values


def rounded_column(ax, x0, width, height, color, radius_frac=0.35):
    """Column growing from the baseline: rounded cap, square at the baseline."""
    r = min(width * radius_frac, abs(height) * 0.5) if height else 0.0
    x1 = x0 + width
    if r <= 0:
        verts = [(x0, 0), (x0, height), (x1, height), (x1, 0), (x0, 0)]
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * 3 + [MplPath.CLOSEPOLY]
    else:
        verts = [
            (x0, 0), (x0, height - r),
            (x0, height), (x0 + r, height),
            (x1 - r, height),
            (x1, height), (x1, height - r),
            (x1, 0), (x0, 0),
        ]
        codes = [
            MplPath.MOVETO, MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3, MplPath.CURVE3,
            MplPath.LINETO, MplPath.CLOSEPOLY,
        ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color,
                           edgecolor="none", zorder=3, clip_on=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(__file__).with_name("causal_redundant_vs_diverse.png")

    rows = []
    for label, div, red, up, fmt in METRICS:
        # index to diverse = 100, sign-corrected so longer is always better
        rel = (red / div * 100.0) if up else (div / red * 100.0)
        rows.append((label, div, red, rel, fmt))

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=args.dpi, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.xaxis.grid(False)

    for i, (label, div, red, rel, fmt) in enumerate(rows):
        x_div = i - BAR_W - GAP / 2
        x_red = i + GAP / 2
        rounded_column(ax, x_div, BAR_W, 100.0, DIVERSE)
        rounded_column(ax, x_red, BAR_W, rel, REDUNDANT)

        # The diverse arm is the reference and always sits at exactly 100, so
        # its value labels go on ONE fixed line above every column rather than
        # riding each cap. Riding the cap put them at y~102, which collided
        # with the redundant labels on the metrics where the two arms are
        # close (arm pose: redundant is at 92) -- both in x and in y.
        ax.text(x_div + BAR_W / 2, DIV_LABEL_Y, fmt.format(div), ha="center",
                va="bottom", fontsize=8, color=INK_2)
        # Left-anchored, not centred: these labels are wider than the column,
        # and on the metrics where the redundant arm is short (PAMAP2, WISDM)
        # a centred label spills leftwards onto the full-height diverse column.
        # Growing rightwards keeps them over open surface.
        ax.text(x_red, rel + 2.5, fmt.format(red), ha="left",
                va="bottom", fontsize=8, color=INK_2)
        # the headline number rides only the redundant column -- the penalty
        ax.text(x_red, rel + 9.5, f"{rel - 100:+.0f}%", ha="left",
                va="bottom", fontsize=8.5, color=REDUNDANT, fontweight="bold")

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.set_xlim(-0.75, len(rows) - 0.25)

    ax.set_ylim(0, 134)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED, length=0, pad=4)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_ylabel("Performance relative to the diverse arm\n(diverse = 100, taller is better)",
                  fontsize=9, color=MUTED, labelpad=8)

    handles = [
        Line2D([], [], color=DIVERSE, linewidth=7, solid_capstyle="butt",
               label="diverse-1%  (all 1000 motion modes)"),
        Line2D([], [], color=REDUNDANT, linewidth=7, solid_capstyle="butt",
               label="redundant-1%  (3 of 1000 modes)"),
    ]
    leg = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.30),
                    frameon=False, fontsize=9, handlelength=1.3, ncol=2,
                    columnspacing=2.2, borderpad=0)
    for t in leg.get_texts():
        t.set_color(INK_2)

    ax.text(0.0, 1.02, "Same 46,116 windows, same 105K-step budget — only the selection differs",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, color=INK)

    fig.subplots_adjust(left=0.105, right=0.99, top=0.895, bottom=0.235)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {out}")

    print(f"\n{'metric':<18}{'diverse':>12}{'redundant':>12}{'rel. to diverse':>18}")
    for label, div, red, rel, fmt in rows:
        print(f"{label:<18}{fmt.format(div):>12}{fmt.format(red):>12}{rel:>17.1f}%")


if __name__ == "__main__":
    main()
