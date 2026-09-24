#!/usr/bin/env python3
"""Table 2 as a radar chart: every IMU-Bench task on one axis, one polygon per encoder.

Reads paper_results/tab2_main_results/main_results_long.csv (written by make_main_table.py), so the
figure and Table 2 can never disagree -- run make_main_table.py first.

Two panels, the two protocols of Table 2 (frozen probe, fine-tune). Seven axes, one per task family,
with the four activity-recognition sets collapsed into their mean so that HAR does not take 4/10 of the
perimeter. The Scratch row of Table 2 is drawn in both panels as the no-pretraining reference.

Normalisation. The axes carry cm, mm, m, F1 and R@1, so each is mapped to a relative score in [0, 1]:

    higher-is-better   s = v / v_best          lower-is-better   s = v_best / v

where v_best is the best value on that axis among the encoders drawn in that panel, Scratch included.
1.0 is therefore the best cell of the panel and the outer ring, and the score keeps the *ratio* between
methods: a 7 % spread on arm pose draws as a 7 % ring, not as a full sweep of the axis the way a
per-axis min-max would. Scores are written to paper_results/tab2_main_results/radar_normalized.csv.

Sizing. The figure is drawn K times larger than the \\textwidth it is included at, so every font size,
line width and marker below is the size it has ON THE PAGE, in points.

    python analysis/plot_main_radar.py [--outdir <paper>/Pictures] [--with-ablations]
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "paper_results/tab2_main_results/main_results_long.csv")
OUT_DIR = "/gpfs/home/vbc1591/paper_iclr/Pictures"
NAME = "main_radar"

# The figure is drawn at the width it is included at (ICLR \textwidth = 5.5 in) and saved with the full
# canvas, no tight crop, so \includegraphics[width=\textwidth] scales it by exactly 1 and every font size
# below is the size it has on the page, in points. Keep the margins wide enough that nothing is clipped.
PRINT_IN = 5.5
RMIN = 0.30             # radial floor: the axis runs RMIN..1.00, not 0..1.00 (declared in the caption)

# house theme, shared with the other paper figures
th = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7")
# categorical slots 1, 3, 4, 7, 8 of the documented palette: this five-hue set is the one that clears the
# all-pairs normal-vision floor (worst 16.3). Its worst CVD pair is 6.9 (deuteranopia), inside the 6-8
# band that is legal only with secondary encoding -- hence a line style and a marker shape per series.
# Aqua and yellow sit below 3:1 on this surface; the relief is Table 2 itself and radar_normalized.csv.
SERIES = [  # (encoder id, label, colour, line style, marker, line width pt, fill?)
    ("mae_5src", "WristMAE (Ours)", "#2a78d6", "-", "o", 1.7, True),
    ("limubert_5src", "LIMU-BERT", "#1baf7a", "-", "s", 1.0, False),
    ("simmtm_5src", "SimMTM", "#eda100", (0, (5, 2)), "^", 1.0, False),
    ("focal_5src", "FOCAL", "#4a3aa7", (0, (6, 2, 1, 2)), "D", 1.0, False),
    ("relcon_5src", "RelCon", "#e34948", (0, (1.6, 1.6)), "v", 1.0, False),
    # slot 6 (green) is the only unused hue that clears the all-pairs gates with the five above:
    # validate_palette.js --pairs all gives worst normal-vision dE 15.6 (>=15 floor) and worst CVD 6.9.
    # Orange (slot 2) and magenta (slot 5) both FAIL the normal-vision floor against red (dE 7.1 / 13.2).
    # 6.9 sits in the 6-8 band that is legal only with secondary encoding -- the dash pattern and the
    # marker below are that encoding, as for every other series here.
    ("sslwear_5src", "SSL-Wearables", "#008300", (0, (5, 1.5, 1, 1.5, 1, 1.5)), "P", 1.0, False),
]
ABLATIONS = [
    ("mae_5src_noaltattn", "WristMAE, plain attn.", "#2a78d6", (0, (5, 2)), "o", 0.9, False),
    ("mae_5src_acconly", "WristMAE, acc. only", "#2a78d6", (0, (1.6, 1.6)), "o", 0.9, False),
]
REFERENCE = ("scratch", "Scratch", th["muted"], (0, (3, 2.5)), None, 0.9, False)
HAR = ["har_hhar", "har_mhealth", "har_prism", "har_userwatch"]
# (axis key, axis label, unit shown under the best value, scale for that unit, higher_is_better)
AXES = [("arm_pose_nymeria", "Arm", "cm", 1, 0),
        ("full_body_pose_nymeria", "Body", "mm", 1, 0),
        ("odometry_nymeria_local", "Odom.", "m", 1, 0),
        ("microgesture_openwatch", "Gesture", "F1", 100, 1),
        ("language_humanml3d", "Text", "R@1", 1, 1),
        ("pd_pads", "PD", "F1", 100, 1),
        ("__har", "Activity", "F1", 100, 1)]
PANELS = [("probe", "Frozen probe"), ("finetune", "Fine-tune")]


def raw_value(d, enc, protocol, axis):
    """Table 2's primary metric for one cell; the HAR axis is the mean of the four activity sets."""
    proto = "scratch" if enc == "scratch" else protocol
    tasks = HAR if axis == "__har" else [axis]
    vals = [d.value[(d.encoder == enc) & (d.protocol == proto) & (d.task == t)].to_numpy() for t in tasks]
    if any(len(v) != 1 or not np.isfinite(v[0]) for v in vals):
        return np.nan
    return float(np.mean([v[0] for v in vals]))


def panel_scores(d, protocol, series):
    """rows x axes of relative scores in [0, 1], and the best raw value per axis."""
    raw = np.array([[raw_value(d, e, protocol, ax) for ax, *_ in AXES] for e, *_ in series])
    best, scores = [], np.full_like(raw, np.nan)
    for j, (_, _, _, _, hib) in enumerate(AXES):
        col = raw[:, j]
        b = np.nanmax(col) if hib else np.nanmin(col)
        best.append(b)
        scores[:, j] = col / b if hib else b / col
    return raw, scores, np.array(best)


def draw(ax, series, scores, best, title, rmin):
    n = len(AXES)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    loop = np.concatenate([ang, ang[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor(th["surface"])
    ax.set_ylim(rmin, 1.0)
    ax.set_rgrids([0.4, 0.6, 0.8, 1.0], labels=[], angle=0)
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    ax.grid(color=th["grid"], linewidth=0.4)
    ax.spines["polar"].set_color(th["axis"])
    ax.spines["polar"].set_linewidth(0.55)

    for (_, label, colour, ls, marker, lw, fill), row in zip(series, scores):
        y = np.concatenate([row, row[:1]])
        ax.plot(loop, y, color=colour, linestyle=ls, linewidth=lw, marker=marker, markersize=2.9,
                markerfacecolor=colour, markeredgecolor=th["surface"], markeredgewidth=0.5,
                clip_on=False, zorder=4 if fill else 3, label=label)
        if fill:
            ax.fill(loop, y, color=colour, alpha=0.11, linewidth=0, zorder=2)

    for a, (_, label, unit, scale, _), b in zip(ang, AXES, best):
        deg = np.degrees(a) % 360
        ha = "center" if deg < 12 or abs(deg - 180) < 12 else ("left" if deg < 180 else "right")
        fmt = "{:.3f}" if unit == "m" else ("{:.0f}" if unit == "mm" else "{:.1f}")
        ax.annotate(label, (a, 1.17), ha=ha, va="center", fontsize=7.0, color=th["ink"],
                    fontweight="bold", annotation_clip=False)
        ax.annotate(f"{fmt.format(b * scale)} {unit}", (a, 1.17), textcoords="offset points",
                    xytext=(0, -5.0), ha=ha, va="center", fontsize=5.7, color=th["muted"],
                    annotation_clip=False)
    for r in (0.4, 0.6, 0.8, 1.0):
        ax.text(np.pi / n, r, f"{r:.1f}", ha="center", va="center", fontsize=5.0, color=th["muted"],
                bbox=dict(facecolor=th["surface"], edgecolor="none", pad=0.6), zorder=5)
    ax.set_title(title, fontsize=8.5, color=th["ink"], fontweight="bold", pad=23)


def build(d, series, ncol, rows):
    """One pass at the figure, with page-point sizes drawn K times larger."""
    fig, axes = plt.subplots(1, 2, figsize=(PRINT_IN, 0.50 * PRINT_IN), subplot_kw=dict(polar=True),
                             facecolor=th["surface"])
    bm = 0.225 + 0.085 * (rows - 1)          # bottom margin: the note, then the legend under it
    fig.subplots_adjust(left=0.100, right=0.872, top=0.825, bottom=bm, wspace=0.70)
    export = []
    for ax, (proto, title) in zip(axes, PANELS):
        raw, scores, best = panel_scores(d, proto, series)
        lo = np.nanmin(scores)
        if lo < RMIN:
            sys.exit(f"RMIN={RMIN} would clip a value ({lo:.3f}) in the {proto} panel -- lower it")
        draw(ax, series, scores, best, title, RMIN)
        for i, (enc, label, *_) in enumerate(series):
            for j, (axis, axlabel, unit, scale, hib) in enumerate(AXES):
                export.append([proto, axis, axlabel, enc, label, unit, hib, raw[i, j], best[j], scores[i, j]])

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
                     bbox_to_anchor=(0.5, bm * 0.40), fontsize=6.9, handlelength=2.4,
                     columnspacing=1.6, handletextpad=0.6)
    for t in leg.get_texts():
        t.set_color(th["ink2"])
    fig.text(0.5, bm * 0.55,
             f"further out is better; each axis scaled to its panel's best (= 1.00); "
             f"radial axis {RMIN:.2f}\u20131.00",
             ha="center", fontsize=5.7, color=th["muted"])
    return fig, export


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=OUT_DIR)
    p.add_argument("--with-ablations", action="store_true", help="add the two WristMAE ablation rows of Table 2")
    a = p.parse_args()
    if not os.path.exists(SRC):
        sys.exit(f"{SRC} not found -- run analysis/paper_tables/make_main_table.py first")
    d = pd.read_csv(SRC)
    series = SERIES + (ABLATIONS if a.with_ablations else []) + [REFERENCE]
    ncol = min(len(series), 4 if len(series) > 6 else 6)
    rows = -(-len(series) // ncol)  # legend rows, so the note never lands on the legend

    fig, export = build(d, series, ncol, rows)

    os.makedirs(a.outdir, exist_ok=True)
    stem = NAME + ("_ablations" if a.with_ablations else "")
    for ext in ("png", "pdf"):
        q = os.path.join(a.outdir, f"{stem}.{ext}")
        fig.savefig(q, dpi=300 if ext == "png" else None, facecolor=th["surface"])
        print("wrote", q)

    sys.path.insert(0, os.path.join(REPO, "paper_results"))
    from paper_io import write_csv
    print("wrote", write_csv("tab2_main_results", f"{stem.replace('main_radar', 'radar')}_normalized.csv",
                             ["protocol", "task", "task_label", "encoder", "encoder_label", "unit",
                              "higher_is_better", "value", "panel_best", "score"], export))


if __name__ == "__main__":
    main()
