"""The 11l figure: performance vs NUMBER OF MOTION TYPES, one line per data volume.

x = motion types actually present in the pretraining set (1..1000, log)
y = probe performance in native units
lines = data volume (0.01 / 0.1 / 1 / 10 / 100 % of the 1.96M 5-source pool)

Small multiples, one panel per probe, because the eight probes are in four
different units (accuracy %, balanced accuracy %, val loss, mm, vel MSE) and
putting them on one axis would either need a second y-scale or a normalisation
that hides the actual numbers. The ninth panel carries the normalised mean.

Volume is a MAGNITUDE, not an identity, so the five lines use an ordinal ramp of
one hue (light = little data, dark = all of it) rather than five categorical
hues. Steps start at blue-250, the lightest that still clears 2:1 on white.

Cells that do not exist are simply absent -- the grid is triangular, since K
motion types cap how much data can be drawn (see 09_grid_feasibility.py), and
the 100% volume exists only at K=1000, so it is drawn as a labelled point rather
than a line.

Reads  out/grid_selection.json, out/grid_results.json
Writes analysis/diversity_saturation/grid_{light,dark}.png/.pdf
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835",
                  ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]),
}

FRACS = [("0p01", 0.0001, "0.01%"), ("0p1", 0.001, "0.1%"), ("1", 0.01, "1%"),
         ("10", 0.10, "10%"), ("100", 1.00, "100%")]
K_TYPES = [1, 5, 10, 25, 50, 100, 1000]

# label, higher_is_better, unit
PROBES = [
    ("MAE val loss (pretrain)", False),
    ("WISDM (acc %)", True), ("UTD-MHAD (acc %)", True), ("PAMAP2 (acc %)", True),
    ("RecoFit (acc %)", True), ("Gesture (bal acc %)", True),
    ("Arm pose (val loss)", False), ("Full-body (mpjpe mm)", False),
    ("Odometry (vel MSE)", False),
]


def main():
    sel_p = os.path.join(OUT_DIR, "grid_selection.json")
    res_p = os.path.join(OUT_DIR, "grid_results.json")
    if not os.path.exists(sel_p):
        raise SystemExit("run 10_grid_select.py first")
    sel = json.load(open(sel_p))["cells"]
    raw = json.load(open(res_p)) if os.path.exists(res_p) else {}
    res = raw.get("probes", {})

    # The pretraining objective is available as soon as a cell finishes, long
    # before its probes clear the queue, so it leads the figure. Only cells that
    # reached the full 50K budget are comparable -- a mid-run val loss is not.
    pre = {c: v["final_val_loss"] for c, v in raw.get("cells", {}).items()
           if v.get("final_step") == 50000 and v.get("final_val_loss") is not None}
    if pre:
        res = {"MAE val loss (pretrain)": pre, **res}

    for theme in ("light", "dark"):
        th = THEMES[theme]
        fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.6), facecolor=th["surface"])
        axes = axes.ravel()

        norm_store = {f[0]: {} for f in FRACS}

        for pi, (label, hib) in enumerate(PROBES):
            ax = axes[pi]
            blk = res.get(label, {})
            for fi, (ftag, fval, flab) in enumerate(FRACS):
                pts = []
                for K in K_TYPES:
                    cell = f"k{K}_f{ftag}"
                    if cell not in sel:
                        continue
                    v = blk.get(cell)
                    if v is None:
                        continue
                    pts.append((K, v))
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                for K, v in pts:
                    norm_store[ftag].setdefault(K, []).append((label, hib, v))
                if len(pts) == 1:
                    # A single-point series (the 100% volume exists only at
                    # K=1000) still needs a legend entry, otherwise the reader
                    # sees a stray diamond with nothing naming it. Diamond with
                    # no connecting line is itself the signal that this volume
                    # is one cell rather than a curve.
                    ax.plot(xs, ys, marker="D", ms=9, color=th["ramp"][fi],
                            markeredgecolor=th["surface"], markeredgewidth=1.4,
                            linestyle="none", label=flab, zorder=5)
                else:
                    ax.plot(xs, ys, marker="o", ms=5, lw=2, color=th["ramp"][fi],
                            markeredgecolor=th["surface"], markeredgewidth=1.2,
                            label=flab, zorder=3)
            ax.set_title(label, fontsize=10, color=th["ink"], fontweight="bold", pad=6)
            if not blk:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10, color=th["muted"])

        # 9 panels exactly: the pretraining objective + the 8 probes. (The
        # normalised-mean panel this layout used to carry was dropped when the
        # MAE panel was added -- it would have overwritten the 9th axes.)
        for ax in axes:
            ax.set_facecolor(th["surface"])
            ax.set_xscale("log")
            ax.set_xticks(K_TYPES)
            ax.set_xticklabels([str(k) for k in K_TYPES], fontsize=8.5)
            ax.minorticks_off()
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(th["axis"])
            ax.tick_params(colors=th["muted"], labelsize=8.5)
            ax.grid(True, color=th["grid"], lw=0.8, zorder=0)
            ax.set_axisbelow(True)
        for ax in axes[6:]:
            ax.set_xlabel("motion types in the pretraining set", fontsize=9,
                          color=th["ink2"])

        handles, labels_ = axes[0].get_legend_handles_labels()
        if not handles:
            handles, labels_ = axes[0].get_legend_handles_labels()
        if handles:
            leg = fig.legend(handles, labels_, loc="lower center", ncol=5, frameon=False,
                             fontsize=9.5, bbox_to_anchor=(0.5, 0.005),
                             title="pretraining data (fraction of the 1,959,288-window 5-source pool)")
            leg.get_title().set_fontsize(9)
            leg.get_title().set_color(th["ink2"])
            for t in leg.get_texts():
                t.set_color(th["ink"])

        fig.suptitle("Performance vs number of motion types, at five data volumes",
                     fontsize=14.5, fontweight="bold", color=th["ink"], x=0.055,
                     ha="left", y=0.985)
        fig.text(0.055, 0.955,
                 "5-source pool, 22 realisable cells, equal 50K-step budget (EXPERIMENTS.md 11l). "
                 "Missing points are cells that cannot exist: K types cap how much data can be drawn.",
                 fontsize=9.2, color=th["ink2"], ha="left")

        fig.tight_layout(rect=[0, 0.055, 1, 0.945])
        for ext in ("png", "pdf"):
            p = os.path.join(HERE, f"grid_{theme}.{ext}")
            fig.savefig(p, dpi=300, facecolor=th["surface"], bbox_inches="tight",
                        pad_inches=0.25)
            print("wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
