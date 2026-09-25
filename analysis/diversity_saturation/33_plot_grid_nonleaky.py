#!/usr/bin/env python3
"""The 11l grid figure in its original 3x3 layout, with every probe panel leak-free.

Identical to 11_plot_grid.py in layout, styling and x-axis (motion types, largest-first), with the four
activity-recognition panels that sit inside the pretraining pool (WISDM, UTD-MHAD, PAMAP2, RecoFit)
replaced by leak-free probes from the bench, run on the same 22 grid checkpoints (g5_k{K}_f{tag}):
HHAR, MHEALTH, PRISM (macro-F1) and PADS Parkinson's (window macro-F1; see PANELS).
Gesture: the grid's original OpenWatch probe predates the de-duplicated subject split. It is replaced by
the IMU-Bench protocol probe (clip F1) once all 22 cells have landed; until then the old probe is drawn
and its title says so.

Writes <paper>/Pictures/grid_5src_nonleaky.{png,pdf}
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT_DIR = os.environ.get("PAPER_PICTURES", "/gpfs/home/vbc1591/paper_iclr/Pictures")  # override: write into a worktree
th = dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
          ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
FRACS = [("0p01", "0.01%"), ("0p1", "0.1%"), ("1", "1%"), ("10", "10%"), ("100", "100%")]
K_TYPES = [1, 5, 10, 25, 50, 100, 1000]

raw = json.load(open(os.path.join(HERE, "out", "grid_results.json")))
cells = raw["cells"]


def from_grid(key):
    return {c: v for c, v in raw["probes"][key].items() if v is not None}


def from_bench(task, metric):
    out = {}
    for c in cells:
        p = os.path.join(REPO, "bench", "results", task, f"g5_{c}__probe__s42.json")
        if os.path.exists(p):
            v = json.load(open(p))["metrics"].get(metric)
            if v is not None:
                out[c] = v
    return out


gesture_clean = from_bench("microgesture_openwatch", "clip_macro_f1")
if len(gesture_clean) == len(cells):
    gesture = ("Gesture (clip F1)", gesture_clean, True)
else:
    gesture = ("Gesture (bal acc %, old split: re-run pending)", from_grid("Gesture (bal acc %)"), True)

PANELS = [
    ("inrtlWrist val loss (pretrain)", {c: v["final_val_loss"] for c, v in cells.items() if v.get("final_step") == 50000}, False),
    ("HHAR (macro-F1)", from_bench("har_hhar", "macro_f1"), True),
    ("MHEALTH (macro-F1)", from_bench("har_mhealth", "macro_f1"), True),
    ("PRISM (macro-F1)", from_bench("har_prism", "macro_f1"), True),
    # window, not subject, macro-F1 (2026-09-24): Table 2 ranks PD on subject F1, but over 94 participants it moves in
    # steps of a few points and put the single-mode 0.01 % cell (56.5) above the full pool (45.4); the grid and its
    # average compare cells a few percent apart, so they use the per-window metric. Variants: fig4_variants/.
    ("Parkinson's (window F1)", from_bench("pd_pads", "window_macro_f1"), True),
    gesture,
    ("Arm pose (val loss)", from_grid("Arm pose (val loss)"), False),
    ("Full-body (mpjpe mm)", from_grid("Full-body (mpjpe mm)"), False),
    ("Odometry (vel MSE)", from_grid("Odometry (vel MSE)"), False),
]

# ---- numbers behind the figure -> paper_results/fig3_scaling_grid ----------------------------------
import sys as _sys
_sys.path.insert(0, os.path.join(REPO, "paper_results"))
from paper_io import write_csv  # noqa: E402
_BENCH = {"HHAR": "har_hhar macro_f1", "MHEALTH": "har_mhealth macro_f1", "PRISM": "har_prism macro_f1",
          "Parkinson": "pd_pads window_macro_f1", "Gesture (clip": "microgesture_openwatch clip_macro_f1"}
_rows = []
for _label, _blk, _hib in PANELS:
    _src = next((f"bench probe: {v}" for k, v in _BENCH.items() if _label.startswith(k)), None)
    if _src is None:
        _src = "pretraining (grid validation set)" if _label.startswith(("inrtlWrist", "WristMAE", "MAE")) else "legacy grid probe (lighter decoder)"
    for _c, _v in sorted(_blk.items()):
        _cell = cells.get(_c, {})
        _rows.append([_c, _cell.get("K"), _cell.get("fraction"), _cell.get("n_windows"), _label, _v, int(_hib), _src])
write_csv("fig3_scaling_grid", "grid_long.csv",
          ["cell", "motion_types_K", "pool_fraction", "n_windows", "panel", "value", "higher_is_better", "source"], _rows)

fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.6), facecolor=th["surface"])
axes = axes.ravel()
for ax, (label, blk, hib) in zip(axes, PANELS):
    for fi, (ftag, flab) in enumerate(FRACS):
        pts = [(K, blk[f"k{K}_f{ftag}"]) for K in K_TYPES if f"k{K}_f{ftag}" in blk]
        if not pts:
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        if ftag == "100":
            ax.plot(xs, ys, marker="D", ms=9, color=th["ramp"][fi], markeredgecolor=th["surface"],
                    markeredgewidth=1.4, linestyle="none", label=flab, zorder=5)
        else:
            ax.plot(xs, ys, marker="o", ms=5, lw=2, color=th["ramp"][fi], markeredgecolor=th["surface"],
                    markeredgewidth=1.2, label=flab, zorder=3)
    n = len(blk)
    ax.set_title(label + ("" if n >= len(cells) else f"  ({n}/{len(cells)})"), fontsize=10, color=th["ink"],
                 fontweight="bold", pad=6)
    if hib and blk and max(blk.values()) - min(blk.values()) < 0.10 and max(blk.values()) <= 1.0:
        mid = (max(blk.values()) + min(blk.values())) / 2      # keep single-seed noise from reading as a trend
        ax.set_ylim(mid - 0.055, mid + 0.055)
    if not blk:
        ax.set_yticks([])
        ax.text(0.5, 0.5, "probes still running", transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color=th["muted"])
for ax in axes:
    ax.set_facecolor(th["surface"])
    ax.set_xscale("log")
    ax.set_xlim(0.8, 1300)
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
    ax.set_xlabel("motion types in the pretraining set", fontsize=9, color=th["ink2"])
handles, labels_ = axes[0].get_legend_handles_labels()
leg = fig.legend(handles, labels_, loc="lower center", ncol=5, frameon=False, fontsize=9.5,
                 bbox_to_anchor=(0.5, 0.005), title="pretraining data (fraction of the 1,959,288-window 5-source pool)")
leg.get_title().set_fontsize(9)
leg.get_title().set_color(th["ink2"])
for t in leg.get_texts():
    t.set_color(th["ink"])
fig.suptitle("Performance vs number of motion types, at five data volumes", fontsize=14.5, fontweight="bold",
             color=th["ink"], x=0.055, ha="left", y=0.985)
clean_note = ("Every probe is leak-free: none of its evaluation sets is in the pretraining pool." if gesture[0].startswith("Gesture (clip")
              else "Every probe except the old-split gesture panel is leak-free.")
fig.text(0.055, 0.955, "5-source pool, 22 realisable cells, equal 50K-step budget. " + clean_note +
         " Missing points are cells that cannot exist.",
         fontsize=9.2, color=th["ink2"], ha="left")
fig.tight_layout(rect=[0, 0.055, 1, 0.945])
for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"grid_5src_nonleaky.{ext}")
    fig.savefig(p, dpi=300 if ext == "png" else None, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.25)
    print("wrote", p)
for label, blk, _ in PANELS:
    print(f"  {label:48s} {len(blk)}/{len(cells)} cells")
