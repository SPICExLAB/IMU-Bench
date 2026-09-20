#!/usr/bin/env python
"""The paper's "Data scaling and motion coverage analysis" figure, rebuilt on the
leak-free 4-source corpus.

The printed figure (Sections/6_Scaling.tex, fig:scaling_coverage) has three panels
and all three come from the OLD pools -- panels (a) and (b) from the 11-source
mixture, panel (c) from the 5-source one. Both overlap the evaluation datasets, so
the printed figure is a transductive measurement and the caption says so. This
script produces the same three panels from the 4-source factorial, where no
evaluation dataset is in the pretraining pool.

  (a) data_scaling_4src   performance vs pretraining windows, one panel per task.
                          The line is the natural-proportion column; the shaded
                          band is the min-max across all four diversity levels at
                          that volume, so the reader sees the volume effect and the
                          diversity spread in the same frame instead of having to
                          hold one of them in their head.

  (b) coverage_4src       fraction of the K=1000 physical motion-type clusters a
                          uniform draw of n windows touches. Measured, not
                          modelled: 5 independent draws per n, s.d. as error bars.
                          The dashed rule at 99 % is a THRESHOLD, not a gridline --
                          it is where adding windows stops adding motion modes.

  (c) grid_4src           performance vs the motion types a subset ACTUALLY
                          contains, one line per volume. This is the analogue of
                          the printed panel (c), with two differences that matter:
                          x is the realised effective type count (Shannon) rather
                          than a nominal K, and the four volumes are genuinely
                          matched in every other respect. The dedicated type
                          ladders (T-phys and T-enc, both at N=12,500) are overlaid
                          as open markers -- same volume, different selection rule.

Volume is a magnitude, not an identity, so volumes use an ordinal ramp of one hue
(light = little data) rather than categorical colours. Theme dict, ramp and dpi are
copied from analysis/diversity_saturation/11_plot_grid.py so the new panels can sit
beside the old ones in the same document without a visual seam.

Writes analysis/volume_diversity/out/{data_scaling,coverage,grid}_4src_{light,dark}.png/.pdf
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import IDX_DIR, K_TYPES, OUT_DIR, POOL_N, WORK_DIR

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", band="#e9eef6",
                  ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", band="#22293a",
                  ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]),
}

# (task id, metric key, panel label, higher_is_better)
TASKS = [
    ("arm_pose_nymeria",            "pos_err_cm",     "Arm pose (pos err, cm)",   False),
    ("odometry_nymeria_worldframe", "ate_m",          "Odometry (ATE, m)",        False),
    ("har_wisdm",                   "macro_f1",       "HAR · WISDM (macro-F1)", True),
    ("microgesture_openwatch",      "clip_macro_f1",  "Microgesture (clip F1)",   True),
]
DIVS = [("nar", "narrow band"), ("one", "uniform, 1 source"),
        ("nat", "natural, 4 sources"), ("bal", "balanced, 4 sources")]
VOLS = [("n12k", 12_500), ("n50k", 50_000), ("n200k", 200_000), ("n800k", 800_000)]
LADDERS = [("typ", "T-phys ladder", "s"), ("etyp", "T-enc ladder", "^")]
DIV_MARKERS = ["o", "v", "D", "P"]   # nar / one / nat / bal
LEVELS = ["3", "30", "300", "all"]


def metric(task, key, enc):
    p = os.path.join(REPO, "bench", "results", task, f"{enc}__probe__s42.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["metrics"].get(key)


def arms_manifest():
    return json.load(open(os.path.join(OUT_DIR, "arms.json")))["arms"]


def style(ax, th):
    ax.set_facecolor(th["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelsize=8.5)
    ax.grid(True, color=th["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, th, theme):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{name}_4src_{theme}.{ext}")
        fig.savefig(p, dpi=300, facecolor=th["surface"], bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  (a) data scaling
# --------------------------------------------------------------------------- #
def fig_data_scaling(A):
    xs_all = [n for _, n in VOLS] + [POOL_N]
    for theme, th in THEMES.items():
        fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), facecolor=th["surface"])
        for ax, (task, key, label, hib) in zip(axes.ravel(), TASKS):
            lo, hi, mid = [], [], []
            for vt, _n in VOLS:
                col = [v for d, _ in DIVS if (v := metric(task, key, f"vd_{d}_{vt}")) is not None]
                lo.append(min(col)); hi.append(max(col))
                mid.append(metric(task, key, f"vd_nat_{vt}"))
            fulls = [v for a in ("vd_nat_nfull", "vd_bal_nfull")
                     if (v := metric(task, key, a)) is not None]
            lo.append(min(fulls)); hi.append(max(fulls))
            mid.append(metric(task, key, "vd_nat_nfull"))
            ax.fill_between(xs_all, lo, hi, color=th["band"], zorder=1,
                            label="spread across the 4 diversity levels")
            ax.plot(xs_all, mid, marker="o", ms=5, lw=2, color=th["ramp"][3],
                    markeredgecolor=th["surface"], markeredgewidth=1.2, zorder=3,
                    label="natural proportions")
            ax.set_xscale("log")
            ax.set_title(label, fontsize=10, color=th["ink"], fontweight="bold", pad=6)
            ax.set_xlabel("pretraining windows", fontsize=8.5, color=th["ink2"])
            if not hib:
                ax.invert_yaxis()          # so "up" always means "better"
            style(ax, th)
        h, l = axes[0][0].get_legend_handles_labels()
        leg = fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
                         fontsize=9, bbox_to_anchor=(0.5, -0.035))
        for t in leg.get_texts():
            t.set_color(th["ink"])
        fig.suptitle("Data scaling on the leak-free 4-source corpus", y=1.005,
                     fontsize=13.5, fontweight="bold", color=th["ink"], x=0.02, ha="left")
        fig.text(0.02, 0.952,
                 "Fixed 50,000-step budget. In every panel the y-axis is oriented so that up is better.",
                 fontsize=9.0, color=th["ink2"], ha="left")
        fig.tight_layout(rect=[0, 0.02, 1, 0.915])
        save(fig, "data_scaling", th, theme)


# --------------------------------------------------------------------------- #
#  (b) motion coverage
# --------------------------------------------------------------------------- #
def fig_coverage(labels, draws=5, seed=42):
    rng = np.random.default_rng(seed)
    ns = np.unique(np.round(np.logspace(np.log10(200), np.log10(POOL_N), 22)).astype(int))
    mean, sd = [], []
    for n in ns:
        c = [len(np.unique(labels[rng.choice(POOL_N, size=int(n), replace=False)])) / K_TYPES
             for _ in range(draws)]
        mean.append(float(np.mean(c))); sd.append(float(np.std(c)))
    mean = np.array(mean); sd = np.array(sd)
    knee = int(ns[np.argmax(mean >= 0.99)]) if (mean >= 0.99).any() else None

    for theme, th in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=th["surface"])
        ax.axhline(0.99, ls="--", lw=1.4, color=th["ramp"][4], zorder=2)
        ax.text(ns[0] * 1.15, 0.995, "99 % of motion modes covered",
                fontsize=8.8, color=th["ramp"][4], va="bottom")
        ax.errorbar(ns, mean, yerr=sd, marker="o", ms=5, lw=2, capsize=2.5,
                    color=th["ramp"][3], markeredgecolor=th["surface"],
                    markeredgewidth=1.2, zorder=3)
        if knee:
            ax.axvline(knee, ls=":", lw=1.3, color=th["muted"], zorder=2)
            ax.text(knee * 1.12, 0.30, f"coverage saturates\nat ~{knee:,} windows",
                    fontsize=8.8, color=th["ink2"], va="center")
        for vt, n in VOLS:
            ax.plot([n, n], [0.0, 0.055], lw=1.6, color=th["ink2"], zorder=4)
            ax.text(n, 0.075, vt.replace("n", "").replace("k", "k"), fontsize=7.6,
                    color=th["ink2"], ha="center")
        ax.set_xscale("log")
        ax.set_xlabel("windows drawn uniformly from the 4-source pool", fontsize=9.5, color=th["ink2"])
        ax.set_ylabel("fraction of K=1000 motion types touched", fontsize=9.5, color=th["ink2"])
        ax.set_ylim(0, 1.06)
        style(ax, th)
        ax.set_title("Motion coverage — 4-source pool, encoder-free types",
                     fontsize=12.5, fontweight="bold", color=th["ink"], loc="left", pad=8)
        fig.text(0.005, -0.02,
                 f"Mean of {draws} independent draws per point; error bars are 1 s.d. "
                 "Ticks mark the volumes the factorial actually pretrained.",
                 fontsize=8.6, color=th["ink2"], ha="left")
        fig.tight_layout()
        save(fig, "coverage", th, theme)
    return ns, mean, knee


# --------------------------------------------------------------------------- #
#  (c) the grid: performance vs realised motion types, one line per volume
# --------------------------------------------------------------------------- #
def fig_grid(A):
    def eff(enc):
        e = A.get(enc)
        return None if e is None else e["phys"]["effective_types_shannon"]

    for theme, th in THEMES.items():
        fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8), facecolor=th["surface"])
        for ax, (task, key, label, hib) in zip(axes.ravel(), TASKS):
            # The four diversity levels are NOT a ladder: nat and bal land within
            # ~10 % of each other in type count and differ in source balance, so a
            # connecting line would assert an ordering along x that does not exist.
            # Marker shape carries the level, colour carries the volume.
            for vi, (vt, n) in enumerate(VOLS):
                for d, mk in zip([d for d, _ in DIVS], DIV_MARKERS):
                    enc = f"vd_{d}_{vt}"
                    x, y = eff(enc), metric(task, key, enc)
                    if x is None or y is None:
                        continue
                    ax.plot([x], [y], marker=mk, ms=7, ls="none",
                            color=th["ramp"][vi], markeredgecolor=th["surface"],
                            markeredgewidth=1.0, zorder=3,
                            label=f"{n:,} windows" if d == "nar" else None)
            for pre, lab, mk in LADDERS:
                pts = []
                for lv in LEVELS:
                    enc = f"vd_{pre}{lv}_n12k"
                    x, y = eff(enc), metric(task, key, enc)
                    if x is not None and y is not None:
                        pts.append((x, y))
                pts.sort()
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=6,
                            lw=1.2, ls=":", mfc="none", color=th["muted"], zorder=2,
                            label=f"{lab} (12,500)")
            ax.set_xscale("log")
            ax.set_title(label, fontsize=10, color=th["ink"], fontweight="bold", pad=6)
            ax.set_xlabel("effective motion types present", fontsize=8.5, color=th["ink2"])
            if not hib:
                ax.invert_yaxis()
            style(ax, th)
        h, l = axes[0][0].get_legend_handles_labels()
        leg = fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.8,
                         bbox_to_anchor=(0.5, -0.06))
        for t in leg.get_texts():
            t.set_color(th["ink"])
        fig.suptitle("Motion types × volume on the leak-free 4-source corpus", y=1.005,
                     fontsize=13.5, fontweight="bold", color=th["ink"], x=0.02, ha="left")
        fig.text(0.02, 0.952,
                 "x is the effective type count each subset realises, not a nominal K. Marker shape = "
                 "diversity level (○ narrow, ▽ uniform, ◇ natural, ✚ balanced); up is better everywhere.",
                 fontsize=9.0, color=th["ink2"], ha="left")
        fig.tight_layout(rect=[0, 0.03, 1, 0.915])
        save(fig, "grid", th, theme)


def main():
    A = arms_manifest()
    print("[a] data scaling"); fig_data_scaling(A)
    print("[b] motion coverage")
    labels = np.load(os.path.join(WORK_DIR, "labels_phys_k1000.npy"))
    ns, mean, knee = fig_coverage(labels)
    print(f"    coverage reaches 99% at ~{knee:,} windows" if knee else "    never reaches 99%")
    print("[c] grid"); fig_grid(A)
    json.dump({"coverage_n": ns.tolist(), "coverage_mean": mean.tolist(),
               "coverage_knee_windows": knee},
              open(os.path.join(OUT_DIR, "coverage_4src.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
