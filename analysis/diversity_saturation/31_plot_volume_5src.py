#!/usr/bin/env python3
"""Paper Fig (scaling a): pretraining volume at full motion diversity on the 5-source pool.

The K=1000 row of the 11l grid: 0.1 / 1 / 10 / 100 % of the 1,959,288-window 5-source pool
(v3 cache), every cell pretrained for the same 50k steps. One small multiple per metric, each on its
own axis. Only protocol-clean probes are shown:
  * MAE validation loss, arm-pose probe val loss, full-body MPJPE, odometry velocity MSE
    (analysis/diversity_saturation/out/grid_results.json, legacy probes);
  * HHAR / MHEALTH / PRISM macro-F1 from the bench (g5_k1000_* encoders) -- the leak-free HAR sets.
Omitted on purpose: WISDM, UTD-MHAD, PAMAP2, RecoFit (inside the pool's `realdata` source) and the
grid's OpenWatch gesture probe (pre-dedup split, 10.24 s windows -- not the IMU-Bench protocol).
The dashed rule marks the reference corpus size (197,094 windows = 10.06 % of the pool).

Last panel: coverage of the K=1000 modes by uniform draws (out/coverage_5src.json, 30_coverage_5src.py).\nWrites <paper>/Pictures/volume_scaling_5src.{png,pdf}
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = "/gpfs/home/vbc1591/paper_iclr/Pictures"
POOL = 1959288
REF = 197094
FR = [("f0p1", 0.1), ("f1", 1.0), ("f10", 10.0), ("f100", 100.0)]
INK, INK2, MUTED, GRID, AXIS, SURF, BLUE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff", "#2a78d6"

g = json.load(open(os.path.join(HERE, "out", "grid_results.json")))
P = g["probes"]


def grid_series(key):
    return [P[key].get(f"k1000_{t}") for t, _ in FR]


def bench_series(task):
    out = []
    for t, _ in FR:
        p = os.path.join(REPO, "bench", "results", task, f"g5_k1000_{t}__probe__s42.json")
        out.append(json.load(open(p))["metrics"]["macro_f1"] if os.path.exists(p) else None)
    return out


PANELS = [
    ("MAE validation loss", [g["cells"][f"k1000_{t}"]["final_val_loss"] for t, _ in FR], False, 0.02),
    ("Arm pose, probe val. loss", grid_series("Arm pose (val loss)"), False, 0.0010),
    ("Full-body pose, MPJPE (mm)", grid_series("Full-body (mpjpe mm)"), False, 3.0),
    ("Odometry, velocity MSE", grid_series("Odometry (vel MSE)"), False, 0.0020),
    ("HHAR, macro-F1", bench_series("har_hhar"), True, 0.10),
    ("MHEALTH, macro-F1", bench_series("har_mhealth"), True, 0.10),
    ("PRISM, macro-F1", bench_series("har_prism"), True, 0.10),
]

# ---- numbers behind the figure -> paper_results/appendix_volume_scaling ---------------------------
import sys as _sys
_sys.path.insert(0, os.path.join(REPO, "paper_results"))
from paper_io import write_csv  # noqa: E402
write_csv("appendix_volume_scaling", "volume_k1000.csv",
          ["panel", "higher_is_better"] + [f"pool_{f:g}pct" for _, f in FR],
          [[t, int(h)] + list(ys) for t, ys, h, _ in PANELS])
_cov = json.load(open(os.path.join(HERE, "out", "coverage_5src.json")))["points"]
write_csv("appendix_volume_scaling", "motion_mode_coverage.csv", list(_cov[0].keys()),
          [list(c.values()) for c in _cov])

fig, axes = plt.subplots(2, 4, figsize=(11.0, 4.9), facecolor=SURF)
axes = axes.ravel()
xs = [f for _, f in FR]
for ax, (title, ys, hib, span) in zip(axes, PANELS):
    ax.set_facecolor(SURF)
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    ax.axvline(REF / POOL * 100, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=2, marker="o", ms=7,
                markeredgecolor=SURF, markeredgewidth=2, zorder=3, clip_on=False)
        lo, hi = min(p[1] for p in pts), max(p[1] for p in pts)
        if hi - lo < span:
            mid = (hi + lo) / 2
            lo, hi = mid - span / 2, mid + span / 2
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    n = len(pts)
    ax.set_title(title + (" \u2191" if hib else " \u2193") + ("" if n == 4 else f"  ({n}/4 cells)"),
                 fontsize=9.5, color=INK, loc="left", pad=6)
    if not pts:
        ax.set_yticks([])
        ax.text(0.5, 0.5, "probes still running", transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.07, 140)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0.1%", "1%", "10%", "100%"])
    ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].text(REF / POOL * 100 * 1.12, 0.97, "reference\ncorpus", transform=axes[0].get_xaxis_transform(),
             ha="left", va="top", fontsize=7.5, color=MUTED)
ax = axes[-1]
cov = json.load(open(os.path.join(HERE, "out", "coverage_5src.json")))["points"]
cx = [c["fraction_pct"] for c in cov] + [100.0]
cy = [c["coverage_pct"] for c in cov] + [100.0]
ax.set_facecolor(SURF)
ax.axvline(REF / POOL * 100, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
ax.plot(cx, cy, color=BLUE, lw=2, marker="o", ms=6, markeredgecolor=SURF, markeredgewidth=1.6, zorder=3, clip_on=False)
for x, y in zip(cx, cy):
    if y < 99.5:
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=7.5, color=INK2)
ax.set_title("Motion-mode coverage (%) \u2191", fontsize=9.5, color=INK, loc="left", pad=6)
ax.set_xscale("log"); ax.set_xlim(0.007, 140); ax.set_ylim(0, 105)
ax.set_xticks([0.01, 0.1, 1, 10, 100]); ax.set_xticklabels(["0.01%", "0.1%", "1%", "10%", "100%"]); ax.minorticks_off()
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(AXIS)
ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=3)
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.6, h_pad=1.8)
for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"volume_scaling_5src.{ext}")
    fig.savefig(p, dpi=220 if ext == "png" else None, facecolor=SURF)
    print("wrote", p)
for title, ys, _, _ in PANELS:
    print(f"  {title:30s} " + "  ".join("--" if y is None else f"{y:.4g}" for y in ys))
