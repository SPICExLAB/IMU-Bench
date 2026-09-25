#!/usr/bin/env python3
"""Section 6 main figure in two simple panels, each one average over the eight downstream probes.

Reads paper_results/fig3_scaling_grid/grid_long.csv (written by 33_plot_grid_nonleaky.py, which stays as the
per-task breakdown in the appendix). Each probe is expressed against the full-pool, all-modes model (k1000_f100),
signed so that positive = better, then averaged over the eight tasks with equal weight. MAE validation loss is a
pretraining quantity and is not in the average.

Normalisation (--norm, default gain, user decision 2026-09-24):
  gain   100 * (v - v_ref) / (v_ref - v_rand), sign flipped for errors: the change in percent of what the full
         pool gains over a NEVER-TRAINED encoder under the same probe (0 = full pool, -100 = never trained).
         A plain ratio hid the kinematic tasks: 3 mm of full-body error is ~3 % of the error but a large share of
         what pretraining buys, while 3 F1 points on HHAR is a small share of its 24-point gain. v_rand is bench's
         control_random probe for the benchmark panels and out/grid_random_anchor.json (35_harvest_...) for the
         grid's own arm/full-body/odometry probes. EVERY probe is clipped to +-100 (user, 2026-09-25): no task
         counts for more than one full pretraining gain either way, and -100 also stands for "worse than never
         trained". Needed because two probes barely beat the never-trained encoder -- Parkinson's by 1.2 window-F1
         points, the grid's arm probe by 1.1 % of its val loss -- so their unclipped values run from -440 to +182.
  ratio  the earlier 100 * (v - v_ref) / v_ref.

  (a) data amount      the uniform draws (all 1,000 modes available) at 0.1 / 1 / 10 / 100 % of the pool
  (b) motion diversity  motion modes 1 -> 1000, one line per volume. 0.01, 0.1 and 1 % have the full ladder;
                        10 % exists only at 100 and 1,000 modes and 100 % only at 1,000 modes, where it is the
                        reference every value is expressed against, so it sits on the zero line by construction

Writes <paper>/Pictures/scaling_avg_5src.{png,pdf} and paper_results/fig3_scaling_grid/scaling_avg.csv
"""
import argparse
import collections
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.environ.get("PAPER_PICTURES", "/gpfs/home/vbc1591/paper_iclr/Pictures")  # override: write into a worktree
sys.path.insert(0, os.path.join(REPO, "paper_results"))
from paper_io import write_csv  # noqa: E402

th = dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
          ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"],
          # panel (b): one categorical hue per volume (validated: CVD dE 9.2, normal-vision dE 27.6), plus a
          # distinct marker per line so identity is not colour alone
          series=["#2a78d6", "#eb6834", "#1baf7a"], markers=["o", "s", "^"],
          # the two volumes that do NOT have the full mode ladder (10 % has 100 and 1,000 modes, 100 % only
          # 1,000, where it is the reference) are drawn in neutral ink so they read as different in kind and
          # the three validated hues keep their meaning
          sparse="#6f6e69", ref_ink="#0b0b0b")
REF = "k1000_f100"
FRACS = [("0p01", 0.01), ("0p1", 0.1), ("1", 1.0), ("10", 10.0), ("100", 100.0)]
K_TYPES = [1, 5, 10, 25, 50, 100, 1000]

ap = argparse.ArgumentParser()
ap.add_argument("--norm", choices=["gain", "ratio"], default="gain")
ap.add_argument("--drop", default="", help="comma-separated panel-label prefixes to leave out of the average")
_a = ap.parse_args()
NORM, DROP = _a.norm, tuple(s for s in _a.drop.split(",") if s)
CLIP = 100.0                            # every probe capped at one full gain either way; see the docstring

per_task, hib, source = collections.defaultdict(dict), {}, {}
for r in csv.DictReader(open(os.path.join(REPO, "paper_results", "fig3_scaling_grid", "grid_long.csv"))):
    if r["panel"].startswith(("InrtlWrist", "WristMAE", "MAE")):
        continue
    per_task[r["panel"]][r["cell"]] = float(r["value"])
    hib[r["panel"]] = r["higher_is_better"] == "1"
    source[r["panel"]] = r["source"]
tasks = sorted(t for t in per_task if not t.startswith(DROP)) if DROP else sorted(per_task)
cells = sorted(set.intersection(*(set(per_task[t]) for t in tasks)))
assert len(tasks) == 8 - len(DROP) and len(cells) == 22, (len(tasks), len(cells))


def never_trained(task):
    """The never-trained encoder's value under the same probe as the panel."""
    s = source[task]
    if s.startswith("bench probe: "):
        bt, metric = s[len("bench probe: "):].split()
        return json.load(open(os.path.join(REPO, "bench", "results", bt, "control_random__probe__s42.json")))["metrics"][metric]
    anchors = json.load(open(os.path.join(HERE, "out", "grid_random_anchor.json")))["values"]
    if task not in anchors:
        sys.exit(f"no never-trained anchor for {task!r} yet -- run 35_harvest_grid_random_anchor.py once "
                 f"jobs 7318003/05/07 finish (or pass --norm ratio)")
    return anchors[task]


RAND = {t: never_trained(t) for t in tasks} if NORM == "gain" else {}
for t in RAND:
    gain = (per_task[t][REF] - RAND[t]) * (1 if hib[t] else -1)
    assert gain > 0, f"{t}: the full pool is not better than the never-trained encoder ({gain})"


def rel(task, cell):
    v, ref = per_task[task][cell], per_task[task][REF]
    if NORM == "ratio":
        return 100.0 * ((v - ref) / ref if hib[task] else (ref - v) / ref)
    x = 100.0 * (v - ref) / (ref - RAND[task])      # same sign convention for errors: both deltas flip
    return max(-CLIP, min(CLIP, x))


avg = {c: float(np.mean([rel(t, c) for t in tasks])) for c in cells}
write_csv("fig3_scaling_grid", "scaling_avg.csv", ["cell", f"avg_change_pct_{NORM}"] + tasks,
          [[c, round(avg[c], 3)] + [round(rel(t, c), 3) for t in tasks] for c in cells])
if RAND:
    write_csv("fig3_scaling_grid", "scaling_avg_anchors.csv",
              ["panel", "source", "higher_is_better", "full_pool", "never_trained", "gain", "clip"],
              [[t, source[t], int(hib[t]), per_task[t][REF], RAND[t], abs(per_task[t][REF] - RAND[t]),
                CLIP] for t in tasks])

fig, (ax_v, ax_d) = plt.subplots(1, 2, figsize=(9.6, 3.5), facecolor=th["surface"])

# (a) data amount, all modes available
pts = [(pct, avg[f"k1000_f{tag}"]) for tag, pct in FRACS if f"k1000_f{tag}" in avg]
ax_v.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=6, lw=2.2, color=th["ramp"][3],
          markeredgecolor=th["surface"], markeredgewidth=1.2, zorder=3)
ax_v.set_xscale("log")
ax_v.set_xticks([p[0] for p in pts])
ax_v.set_xticklabels(["0.1%\n1,959", "1%\n19.6k", "10%\n196k", "100%\n1.96M"])
ax_v.set_xlabel("pretraining data (fraction of pool / windows)", fontsize=9.5, color=th["ink2"])
ax_v.set_title("(a) Data amount (all 1,000 modes)", fontsize=10.5, color=th["ink"], fontweight="bold",
               loc="left", pad=6)

# (b) motion diversity, one line per volume
for fi, (tag, pct) in enumerate(FRACS[:3]):
    pts = [(K, avg[f"k{K}_f{tag}"]) for K in K_TYPES if f"k{K}_f{tag}" in avg]
    ax_d.plot([p[0] for p in pts], [p[1] for p in pts], marker=th["markers"][fi], ms=5.5, lw=2, color=th["series"][fi],
              markeredgecolor=th["surface"], markeredgewidth=1.1, zorder=3, label=f"{pct:g}% of pool")
# the two larger volumes, at the mode counts the triangular grid reaches: 10 % at 100 and 1,000 modes, and
# 100 % at 1,000 modes only, which is the reference and therefore lies on the dashed line by construction
pts = [(K, avg[f"k{K}_f10"]) for K in K_TYPES if f"k{K}_f10" in avg]
ax_d.plot([p[0] for p in pts], [p[1] for p in pts], marker="D", ms=5.5, lw=1.8, ls=(0, (4, 2)),
          color=th["sparse"], markeredgecolor=th["surface"], markeredgewidth=1.1, zorder=4,
          label="10% of pool")
# 100 % is a single cell (all 1,000 modes) and it is the reference, so it lies on the dashed line at
# exactly 0 by construction: drawn as a point, with no line, so it is not read as a trend
ax_d.plot([1000], [avg["k1000_f100"]], marker="*", ms=15, lw=0, color=th["ref_ink"],
          markeredgecolor=th["surface"], markeredgewidth=1.3, zorder=6,
          label="100% of pool (reference)")
ax_d.set_xscale("log")
ax_d.set_xticks(K_TYPES)
ax_d.set_xticklabels([str(k) for k in K_TYPES])
ax_d.set_xlim(0.8, 1000 * 1.45)          # room for the reference star, which sits on the last tick
ax_d.set_xlabel("motion modes in the pretraining set", fontsize=9.5, color=th["ink2"])
ax_d.set_title("(b) Motion diversity", fontsize=10.5, color=th["ink"], fontweight="bold", loc="left", pad=6)
leg = ax_d.legend(frameon=False, fontsize=8.2, loc="lower right", labelspacing=0.35, handlelength=2.0)
for t in leg.get_texts():
    t.set_color(th["ink"])

pad = max(1.5, 0.05 * (max(avg.values()) - min(avg.values())))
lo, hi = min(avg.values()) - pad, max(4.0 if NORM == "ratio" else 10.0, max(avg.values()) + pad)
ax_d.set_ylim(lo, hi)
# (a): same y span as (b), centred on its own line so the line sits in the middle of the panel
ya = [avg[f"k1000_f{tag}"] for tag, _ in FRACS if f"k1000_f{tag}" in avg]
mid, half = (min(ya) + max(ya)) / 2, (hi - lo) / 2
ax_v.set_ylim(mid - half, mid + half)
for ax in (ax_v, ax_d):
    ax.axhline(0, color=th["muted"], lw=1.0, ls="--", zorder=2)
    if NORM == "gain" and ax.get_ylim()[0] < -100:
        ax.axhline(-100, color=th["muted"], lw=0.9, ls=":", zorder=2)
        ax.text(1.3 if ax is ax_d else ax.get_xlim()[0], -100, " never-trained encoder", va="bottom", ha="left",
                fontsize=7.5, color=th["muted"], zorder=2)
    ax.set_facecolor(th["surface"])
    ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelsize=8.5)
    ax.grid(True, color=th["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
ax_v.set_ylabel("avg. change vs. full pool (%, up = better)" if NORM == "ratio" else
               "avg. change vs. full pool\n(% of its gain over never-trained)", fontsize=9.3, color=th["ink2"])
fig.tight_layout(w_pad=2.5)
for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"scaling_avg_5src.{ext}")
    fig.savefig(p, dpi=300 if ext == "png" else None, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.05)
    print("wrote", p)
for c in sorted(avg, key=lambda c: (int(c.split("_")[0][1:]), c)):
    print(f"  {c:12s} {avg[c]:+6.2f}")
