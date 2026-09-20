#!/usr/bin/env python3
"""Paper Fig (scaling b): the 11l motion-types x volume grid, protocol-clean panels only.

Same data and layout idea as 11_plot_grid.py (x = motion types, one line per data volume, ordinal blue
ramp), but only the probes that are valid under the IMU-Bench protocol:
  * MAE validation loss, arm-pose val loss, full-body MPJPE, odometry velocity MSE (grid_results.json);
  * HHAR / MHEALTH / PRISM macro-F1 from the bench (g5_k{K}_f{tag} encoders).
Dropped from the original figure: WISDM, UTD-MHAD, PAMAP2, RecoFit (inside the pool's realdata source,
removed from IMU-Bench) and the OpenWatch gesture probe (pre-dedup split with duplicated subjects).
K=1 is shaded: modes are added largest-first and the largest mode is device-at-rest noise, so the
drop at K=1 reflects that one degenerate cluster, not a lack of "diversity".

Writes <paper>/Pictures/grid_clean_5src.{png,pdf}
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = "/gpfs/home/vbc1591/paper_iclr/Pictures"
FRACS = [("0p01", "0.01%"), ("0p1", "0.1%"), ("1", "1%"), ("10", "10%"), ("100", "100%")]
KS = [1, 5, 10, 25, 50, 100, 1000]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
BAND = "#f1efe9"

g = json.load(open(os.path.join(HERE, "out", "grid_results.json")))
cells = g["cells"]


def grid_probe(key):
    return lambda c: g["probes"][key].get(c)


def bench_probe(task):
    def f(c):
        p = os.path.join(REPO, "bench", "results", task, f"g5_{c}__probe__s42.json")
        return json.load(open(p))["metrics"]["macro_f1"] if os.path.exists(p) else None
    return f


PANELS = [("MAE validation loss", lambda c: cells[c]["final_val_loss"], False),
          ("Arm pose, probe val. loss", grid_probe("Arm pose (val loss)"), False),
          ("Full-body pose, MPJPE (mm)", grid_probe("Full-body (mpjpe mm)"), False),
          ("Odometry, velocity MSE", grid_probe("Odometry (vel MSE)"), False),
          ("HHAR, macro-F1", bench_probe("har_hhar"), True),
          ("MHEALTH, macro-F1", bench_probe("har_mhealth"), True),
          ("PRISM, macro-F1", bench_probe("har_prism"), True)]

fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.4), facecolor=SURF)
axes = axes.ravel()
for ax, (title, get, hib) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    ax.axvspan(0.72, 1.45, color=BAND, lw=0, zorder=0)
    n, allv = 0, []
    for fi, (tag, _) in enumerate(FRACS):
        pts = [(K, get(f"k{K}_f{tag}")) for K in KS if f"k{K}_f{tag}" in cells]
        pts = [(k, v) for k, v in pts if v is not None]
        n += len(pts)
        allv += [p[1] for p in pts]
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if tag == "100":
            ax.plot(xs, ys, marker="D", ms=8, ls="none", color=RAMP[fi], markeredgecolor=SURF, markeredgewidth=1.3, zorder=5)
        else:
            ax.plot(xs, ys, marker="o", ms=5, lw=2, color=RAMP[fi], markeredgecolor=SURF, markeredgewidth=1.1, zorder=3)
    ax.set_title(title + (" ↑" if hib else " ↓") + ("" if n == len(cells) else f"  ({n}/{len(cells)} cells)"),
                 fontsize=9.5, color=INK, loc="left", pad=6)
    if hib and allv and max(allv) - min(allv) < 0.10:   # keep single-seed noise from reading as a trend
        mid = (max(allv) + min(allv)) / 2
        ax.set_ylim(mid - 0.055, mid + 0.055)
    if n == 0:
        ax.set_yticks([])
        ax.text(0.5, 0.5, "probes still running", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.72, 1400)
    ax.set_xticks(KS)
    ax.set_xticklabels([str(k) for k in KS])
    ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
for ax in axes[3:7]:
    ax.set_xlabel("motion types in the pretraining set", fontsize=8.5, color=INK2)
axes[0].annotate("K=1: at-rest mode", xy=(1.0, 0.0), xycoords=("data", "axes fraction"), xytext=(4, 4),
                 textcoords="offset points", ha="left", va="bottom", fontsize=7.5, color=MUTED)
leg_ax = axes[-1]
leg_ax.axis("off")
handles = [Line2D([], [], color=RAMP[i], lw=2, marker="o", ms=5) for i in range(4)]
handles.append(Line2D([], [], color=RAMP[4], ls="none", marker="D", ms=8))
leg = leg_ax.legend(handles, [f for _, f in FRACS], loc="center left", frameon=False, fontsize=9,
                    title="share of the 1,959,288-\nwindow five-source pool")
leg.get_title().set_fontsize(8.5)
leg.get_title().set_color(INK2)
fig.tight_layout(w_pad=1.4, h_pad=1.6)
for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"grid_clean_5src.{ext}")
    fig.savefig(p, dpi=220 if ext == "png" else None, facecolor=SURF)
    print("wrote", p)
