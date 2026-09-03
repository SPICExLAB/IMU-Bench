#!/usr/bin/env python3
"""Data-scaling trend, averaged over every task x evaluation protocol.

Each of the 17 task/protocol evaluations uses its own metric (accuracy, MPJPE,
MSE, ...) on its own scale, so raw values can't be averaged. Every evaluation is
therefore normalised to its own 5% baseline as a signed percent improvement
(higher-is-better metrics: (v - v5)/v5; lower-is-better: (v5 - v)/v5, so
positive is always better). The single plotted line is the mean of those 17
improvements at each subsampling fraction, with a +/- standard-error band.

The figure carries no title or caption — it is meant to drop into a paper under
its own LaTeX caption.

Usage
-----
    python analysis/plot_data_scaling.py                # one averaged trend
    python analysis/plot_data_scaling.py --grid         # original 17-panel grid
    python analysis/plot_data_scaling.py -o /path/out.png --dpi 400
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ---------------------------------------------------------------- palette ----
# Categorical slots 1-3. Validated all-pairs, light mode: worst CVD dE 9.2,
# worst normal-vision dE 24.0. Aqua sits below 3:1 on the surface, so the
# relief rule applies — every panel is titled with its protocol and the legend
# repeats it, so identity is never carried by colour alone.
PROBE = "#2a78d6"        # slot 1, blue
FINETUNE = "#eb6834"     # slot 2, orange
PRETRAIN = "#1baf7a"     # slot 3, aqua

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#006300"         # success text ink
CRITICAL = "#d03b3b"     # status: critical

# x positions are the EFFECTIVE kept-window fractions (session_fraction floors
# push the nominal 1%/10% to 1.3%/10.6% -- measured from jobs 8836196/8836197).
FRACTIONS = [1.3, 5, 10.6, 25, 100]

# ------------------------------------------------------------------- data ----
# 5-point recipe sweep (EXPERIMENTS.md 11 + 11i). Harvest rules validated
# against the original 3-point numbers: classification = metric at the
# best-val-loss step; pose/odom = min over training; MAE = 12_recipe_val_rescore
# (all five checkpoints scored in ONE job, 2026-08-10).
# mult rescales the y values purely so tick labels stay short; `unit` records it.
PANELS = [
    dict(task="WISDM", proto="Probe", metric="Accuracy (%)", up=True,
         values=[65.38, 62.96, 64.64, 62.94, 64.07]),
    dict(task="WISDM", proto="Finetune", metric="Accuracy (%)", up=True,
         values=[70.21, 71.08, 72.36, 71.22, 71.39]),
    dict(task="UTD-MHAD", proto="Probe", metric="Accuracy (%)", up=True,
         values=[68.60, 59.42, 63.77, 62.32, 63.77]),
    dict(task="UTD-MHAD", proto="Finetune", metric="Accuracy (%)", up=True,
         values=[70.05, 67.63, 76.33, 73.91, 68.12]),
    dict(task="PAMAP2", proto="Probe", metric="Accuracy (%)", up=True,
         values=[56.53, 53.63, 59.91, 57.92, 56.51]),
    dict(task="PAMAP2", proto="Finetune", metric="Accuracy (%)", up=True,
         values=[54.32, 52.03, 55.45, 53.20, 52.43]),

    dict(task="RecoFit", proto="Probe", metric="Accuracy (%)", up=True,
         values=[77.48, 75.88, 76.77, 75.15, 75.46]),
    dict(task="RecoFit", proto="Finetune", metric="Accuracy (%)", up=True,
         values=[87.06, 87.03, 88.23, 86.55, 86.42]),
    dict(task="OpenWatch gesture", proto="Probe", metric="Balanced acc. (%)",
         up=True, values=[61.76, 50.19, 59.38, 49.59, 53.52]),
    dict(task="OpenWatch gesture", proto="Finetune", metric="Accuracy (%)",
         up=True, values=[69.39, 66.33, 63.27, 63.27, 62.24]),
    dict(task="Arm pose", proto="Probe", metric="Val. loss", up=False,
         mult=1e3, unit=r"$\times 10^{-3}$",
         values=[0.016426, 0.016635, 0.016585, 0.016496, 0.016489]),
    dict(task="Arm pose", proto="Finetune", metric="Val. loss", up=False,
         mult=1e3, unit=r"$\times 10^{-3}$",
         values=[0.015815, 0.016398, 0.016009, 0.016276, 0.016160]),

    dict(task="Full-body pose", proto="Probe", metric="MPJPE (mm)", up=False,
         values=[97.29, 97.93, 98.12, 98.59, 97.61]),
    dict(task="Full-body pose", proto="Finetune", metric="MPJPE (mm)", up=False,
         values=[91.09, 92.49, 91.75, 92.14, 92.42]),
    dict(task="Odometry", proto="Probe", metric="Vel. MSE", up=False,
         mult=1e3, unit=r"$\times 10^{-3}$",
         values=[0.02542, 0.02627, 0.02591, 0.02642, 0.02615]),
    dict(task="Odometry", proto="Finetune", metric="Odom. MSE", up=False,
         mult=1e3, unit=r"$\times 10^{-3}$",
         values=[0.018011, 0.018127, 0.018294, 0.018277, 0.018389]),
    dict(task="MAE reconstruction", proto="Pretraining",
         metric="Clean val. MSE", up=False,
         values=[0.76043, 0.76057, 0.75793, 0.75840, 0.75670]),
]

COLOR = {"Probe": PROBE, "Finetune": FINETUNE, "Pretraining": PRETRAIN}

N_COLS = 6
PANEL_W = 2.16
PANEL_H = 1.92


# ---------------------------------------------------------- normalisation ----
def improvements(p):
    """Signed % improvement over the 5% baseline; positive is always better."""
    v5 = p["values"][0]
    if p["up"]:
        return [(v - v5) / v5 * 100.0 for v in p["values"]]
    return [(v5 - v) / v5 * 100.0 for v in p["values"]]


def mean_sem():
    """Per-fraction mean and standard error of the 17 normalised trends."""
    cols = list(zip(*(improvements(p) for p in PANELS)))
    n = len(PANELS)
    means = [sum(c) / n for c in cols]
    sems = [
        math.sqrt(sum((x - m) ** 2 for x in c) / (n - 1)) / math.sqrt(n)
        for c, m in zip(cols, means)
    ]
    return means, sems


# ------------------------------------------------------- averaged figure ----
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.9)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_xscale("log")
    ax.set_xlim(1.05, 130.0)
    ax.set_xticks(FRACTIONS)
    ax.set_xticklabels([f"{f:g}%" for f in FRACTIONS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.tick_params(labelsize=8.5, colors=MUTED, length=0, pad=4)
    for lbl in ax.get_xticklabels():
        lbl.set_color(INK_2)


def draw_aggregate(ax):
    means, sems = mean_sem()

    style_axes(ax)

    # the 5% baseline every trend is measured against
    ax.axhline(0.0, color=AXIS, linewidth=0.9, zorder=1)

    ax.fill_between(FRACTIONS,
                    [m - s for m, s in zip(means, sems)],
                    [m + s for m, s in zip(means, sems)],
                    color=PROBE, alpha=0.10, linewidth=0, zorder=2)
    ax.plot(FRACTIONS, means, color=PROBE, linewidth=2.0, marker="o",
            markersize=6.5, markerfacecolor=PROBE, markeredgecolor=SURFACE,
            markeredgewidth=1.6, clip_on=False, zorder=3)

    lo = min(m - s for m, s in zip(means, sems))
    hi = max(m + s for m, s in zip(means, sems))
    rng = (hi - lo) or 1.0
    ax.set_ylim(lo - 0.22 * rng, hi + 0.30 * rng)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.1f}"))

    # selective direct labels: the two non-baseline means
    for f, m in zip(FRACTIONS[1:], means[1:]):
        ax.annotate(f"{m:+.2f}%", (f, m), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, color=INK_2)
    ax.annotate("baseline", (FRACTIONS[0], means[0]),
                textcoords="offset points", xytext=(0, -14), ha="center",
                fontsize=8, color=MUTED)

    # single series: the in-plot key names it, no legend box needed
    ax.text(0.02, 0.97, f"Mean of {len(PANELS)} task × protocol evaluations",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color=INK_2)
    ax.text(0.02, 0.89, "shaded: ± standard error",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=MUTED)

    ax.set_ylabel(f"Avg. improvement over {FRACTIONS[0]}% data (%) ↑",
                  fontsize=9, color=MUTED, labelpad=6)
    ax.set_xlabel("Session-level subsampling fraction", fontsize=9,
                  color=MUTED, labelpad=5)


# ------------------------------------------------- original panel grid ----
def decimals_for(rng):
    """Enough decimals to keep four ticks distinct, and no more."""
    if rng <= 0:
        return 2
    return max(0, min(6, int(math.ceil(-math.log10(rng / 4.0)))))


def draw_panel(ax, p):
    color = COLOR[p["proto"]]
    mult = p.get("mult", 1.0)
    ys = [v * mult for v in p["values"]]

    style_axes(ax)
    ax.plot(FRACTIONS, ys, color=color, linewidth=2.0, marker="o",
            markersize=6.5, markerfacecolor=color, markeredgecolor=SURFACE,
            markeredgewidth=1.6, clip_on=False, zorder=3)

    lo, hi = min(ys), max(ys)
    rng = hi - lo
    pad = (rng * 0.30) if rng > 0 else (abs(hi) * 0.01 or 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    dec = decimals_for(rng)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.{dec}f}"))
    ax.tick_params(labelsize=8)

    arrow = "↑" if p["up"] else "↓"
    unit = f" {p['unit']}" if p.get("unit") else ""
    ax.set_ylabel(f"{p['metric']}{unit} {arrow}", fontsize=8, color=MUTED,
                  labelpad=4)

    # Task on its own line above a protocol/delta strip, so long task names
    # never run into the neighbouring panel.
    ax.set_title(p["task"], fontsize=9.5, color=INK, fontweight="bold", pad=17)
    ax.text(0.0, 1.02, p["proto"], transform=ax.transAxes, ha="left",
            va="bottom", fontsize=7.5, color=MUTED)

    # Selective direct label: the 5% -> 100% change, the headline of the panel.
    # A glyph carries better/worse so the judgement is never colour-alone.
    delta = ys[-1] - ys[0]
    better = (delta > 0) if p["up"] else (delta < 0)
    badge = f"{delta:+.{dec + 1}f} {'▲' if better else '▼'}"
    ax.text(1.0, 1.02, badge, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color=GOOD if better else CRITICAL)


def render_grid(out, dpi):
    n_rows = math.ceil((len(PANELS) + 1) / N_COLS)  # +1 leaves a slot for the legend
    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(N_COLS * PANEL_W, n_rows * PANEL_H),
        dpi=dpi, facecolor=SURFACE,
    )
    flat = axes.ravel()

    for ax, p in zip(flat, PANELS):
        draw_panel(ax, p)

    # legend in the trailing empty slot; any slot beyond that is removed
    for ax in flat[len(PANELS):]:
        ax.set_facecolor(SURFACE)
        ax.axis("off")

    legend_ax = flat[len(PANELS)] if len(PANELS) < len(flat) else None
    if legend_ax is not None:
        handles = [
            Line2D([], [], color=COLOR[k], linewidth=2.0, marker="o",
                   markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.6,
                   label=k)
            for k in ("Probe", "Finetune", "Pretraining")
        ]
        leg = legend_ax.legend(
            handles=handles, loc="center", bbox_to_anchor=(0.5, 0.58),
            frameon=False, fontsize=9, handlelength=1.7, labelspacing=0.7,
            borderpad=0,
        )
        for t in leg.get_texts():
            t.set_color(INK_2)
        legend_ax.text(0.5, 0.95, "▲ better   ▼ worse", transform=legend_ax.transAxes,
                       ha="center", va="top", fontsize=8.5, color=MUTED)
        legend_ax.text(0.5, 0.0, "x: session-level\nsubsampling fraction",
                       transform=legend_ax.transAxes, ha="center", va="bottom",
                       fontsize=8.5, color=MUTED, linespacing=1.5)

    fig.subplots_adjust(left=0.045, right=0.995, top=0.915, bottom=0.065,
                        wspace=0.60, hspace=0.85)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    print(f"wrote {out}  ({n_rows}x{N_COLS} grid, {len(PANELS)} panels)")


def render_aggregate(out, dpi):
    fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=dpi, facecolor=SURFACE)
    draw_aggregate(ax)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.96, bottom=0.15)
    fig.savefig(out, dpi=dpi, facecolor=SURFACE)
    print(f"wrote {out}  (averaged trend, {len(PANELS)} evaluations)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--grid", action="store_true",
                    help="render the original per-task panel grid instead")
    args = ap.parse_args()

    default_name = "data_scaling.png" if args.grid else "data_scaling_avg.png"
    out = Path(args.out) if args.out else Path(__file__).with_name(default_name)

    if args.grid:
        render_grid(out, args.dpi)
    else:
        render_aggregate(out, args.dpi)

    # the non-visual reading of the same numbers: normalised % improvement
    means, sems = mean_sem()
    print(f"\n{'evaluation':<34}" + "".join(f"{f'{f}%':>10}" for f in FRACTIONS))
    for p in PANELS:
        imp = improvements(p)
        print(f"{p['task'] + ' · ' + p['proto']:<34}"
              + "".join(f"{v:>+10.2f}" for v in imp))
    print(f"{'mean':<34}" + "".join(f"{m:>+10.2f}" for m in means))
    print(f"{'s.e.m.':<34}" + "".join(f"{s:>10.2f}" for s in sems))


if __name__ == "__main__":
    main()
