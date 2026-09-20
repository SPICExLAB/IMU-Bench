#!/usr/bin/env python
"""The 4-source counterpart of the printed panel (c): performance vs NUMBER OF
MOTION TYPES, one line per data volume.

Deliberately a direct restyling of analysis/diversity_saturation/11_plot_grid.py --
same axes, same ordinal volume ramp, same small-multiples layout, same treatment of
the triangular gaps -- so the two figures can be read side by side and the only
thing that differs between them is the corpus.

  x      motion types the pretraining subset may draw from (K largest T-enc
         clusters of a K=1000 partition), log scale
  y      probe performance in native units
  lines  volume, 0.01 / 0.1 / 1 / 10 / 100 % of the 1,738,094-window 4-source pool

Six panels: the pretraining objective first (it is available the moment a cell
finishes, long before its probes clear the queue), then the four probes, then a
normalised mean so the four different units can be read as one number.

Volume is a magnitude, not an identity, so the five volumes use an ordinal ramp of
one hue rather than five categorical colours. Cells that cannot exist are simply
absent: K types cap how much data can be drawn without resampling.

Reads out/enc_type_volume_grid.json, runs/<arm>/logs/metrics.txt, bench/results/.
Writes out/enc_grid_4src_{light,dark}.png/.pdf
"""
from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import OUT_DIR, POOL_N

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835",
                  ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]),
}

KS = [1, 5, 10, 25, 50, 100, 1000]
# Scratch reference and a MINIMUM y-span per panel. Without the floor, matplotlib
# scales to whatever few cells exist and a 0.004 m odometry spread fills the panel
# top to bottom, which reads as a violent effect when it is inside noise. The span
# is chosen per task so the axis shows a range a reader would call meaningful.
REF = {"HAR · WISDM (macro-F1)": 0.6056, "HAR · UTD-MHAD (macro-F1)": 0.7010,
       "HAR · PAMAP2 (macro-F1)": 0.2190, "HAR · RecoFit (macro-F1)": 0.6630,
       "Microgesture (clip F1)": 0.4876, "Arm pose (pos err, cm)": 8.1529,
       "Full-body pose (MPJPE, mm)": 92.428, "Odometry (ATE, m)": 0.1400}
MIN_SPAN = {"HAR · WISDM (macro-F1)": 0.15, "HAR · UTD-MHAD (macro-F1)": 0.20,
            "HAR · PAMAP2 (macro-F1)": 0.20, "HAR · RecoFit (macro-F1)": 0.20,
            "Microgesture (clip F1)": 0.20, "Arm pose (pos err, cm)": 0.40,
            "Full-body pose (MPJPE, mm)": 12.0, "Odometry (ATE, m)": 0.020}
# measured run-to-run spread of a byte-identical config; only arm pose has one
NOISE = {"Arm pose (pos err, cm)": 0.108}
FRACS = [("0p01", "0.01%  (174)"), ("0p1", "0.1%  (1,738)"), ("1", "1%  (17,381)"),
         ("10", "10%  (173,809)"), ("100", "100%  (1,738,094)")]
# label, higher_is_better, (task, metric) or None for the pretraining loss
PROBES = [
    ("MAE val loss (pretrain)", False, None),
    ("HAR · WISDM (macro-F1)", True, ("har_wisdm", "macro_f1")),
    ("HAR · UTD-MHAD (macro-F1)", True, ("har_utd_mhad", "macro_f1")),
    ("HAR · PAMAP2 (macro-F1)", True, ("har_pamap2", "macro_f1")),
    ("HAR · RecoFit (macro-F1)", True, ("har_recofit", "macro_f1")),
    ("Microgesture (clip F1)", True, ("microgesture_openwatch", "clip_macro_f1")),
    ("Arm pose (pos err, cm)", False, ("arm_pose_nymeria", "pos_err_cm")),
    ("Full-body pose (MPJPE, mm)", False, ("full_body_pose_nymeria", "mpjpe_mm")),
    ("Odometry (ATE, m)", False, ("odometry_nymeria_worldframe", "ate_m")),
]


def arm(K, ftag):
    return f"vd_ek{K}_f{ftag}"


def pretrain_loss(a):
    """Final val loss at the full 50K budget. A mid-run value is not comparable."""
    p = os.path.join(REPO, "runs", a, "logs", "metrics.txt")
    if not os.path.exists(p):
        return None
    last = None
    for line in open(p, errors="ignore"):
        if "split=val" not in line:
            continue
        m = re.search(r"step=(\d+).*?loss=([0-9.]+)", line)
        if m and int(m.group(1)) >= 50_000:
            last = float(m.group(2))
    return last


def probe(a, task, key):
    p = os.path.join(REPO, "bench", "results", task, f"{a}__probe__s42.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["metrics"].get(key)


def main():
    cells = set(json.load(open(os.path.join(OUT_DIR, "enc_type_volume_grid.json")))["arms"])

    # gather every value first, so the normalised panel can scale per probe
    raw = {}
    for label, hib, spec in PROBES:
        raw[label] = {}
        for K in KS:
            for ftag, _ in FRACS:
                a = arm(K, ftag)
                if a not in cells:
                    continue
                v = pretrain_loss(a) if spec is None else probe(a, *spec)
                if v is not None:
                    raw[label][(K, ftag)] = v
    n_have = sum(len(v) for lab, v in raw.items() if lab != "MAE val loss (pretrain)")
    print(f"[grid] {len(raw['MAE val loss (pretrain)'])}/24 pretrain cells, {n_have}/192 probe cells")

    for theme, th in THEMES.items():
        fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.6), facecolor=th["surface"])
        axes = axes.ravel()
        for pi, (label, hib, _spec) in enumerate(PROBES):
            ax = axes[pi]
            blk = raw.get(label, {})
            for fi, (ftag, flab) in enumerate(FRACS):
                pts = [(K, blk[(K, ftag)]) for K in KS if (K, ftag) in blk]
                if not pts:
                    continue
                xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                if len(pts) == 1:
                    # a one-cell volume still needs a legend entry; a diamond with
                    # no line is itself the signal that this volume is one cell
                    ax.plot(xs, ys, marker="D", ms=9, color=th["ramp"][fi], linestyle="none",
                            markeredgecolor=th["surface"], markeredgewidth=1.4,
                            label=flab, zorder=5)
                else:
                    ax.plot(xs, ys, marker="o", ms=5, lw=2, color=th["ramp"][fi],
                            markeredgecolor=th["surface"], markeredgewidth=1.2,
                            label=flab, zorder=3)
            ref = REF.get(label)
            if ref is not None:
                band = NOISE.get(label)
                if band:
                    ax.axhspan(ref - band / 2, ref + band / 2, color=th["muted"],
                               alpha=0.13, zorder=1)
                ax.axhline(ref, ls="--", lw=1.2, color=th["muted"], zorder=2)
                ax.text(1.02, ref, "scratch" + ("  \u00b1 noise" if band else ""),
                        transform=ax.get_yaxis_transform(), va="center",
                        fontsize=7.8, color=th["muted"])
            if blk:
                vals = list(blk.values()) + ([ref] if ref is not None else [])
                lo, hi = min(vals), max(vals)
                need = MIN_SPAN.get(label)
                if need and (hi - lo) < need:
                    mid = 0.5 * (lo + hi)
                    lo, hi = mid - need / 2, mid + need / 2
                pad = 0.06 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(label, fontsize=10, color=th["ink"], fontweight="bold", pad=6)
            if not blk:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                        va="center", fontsize=10, color=th["muted"])
        for ax in axes:
            ax.set_facecolor(th["surface"])
            ax.set_xscale("log")
            ax.set_xticks(KS)
            ax.set_xticklabels([str(k) for k in KS], fontsize=8.5)
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

        handles, labels_ = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels_:
                    handles.append(hh); labels_.append(ll)
        order = [flab for _, flab in FRACS]
        pairs = sorted(zip(handles, labels_), key=lambda p: order.index(p[1]))
        if pairs:
            leg = fig.legend([p[0] for p in pairs], [p[1] for p in pairs],
                             loc="lower center", ncol=5, frameon=False, fontsize=9.5,
                             bbox_to_anchor=(0.5, -0.025),
                             title=f"pretraining data (fraction of the {POOL_N:,}-window 4-source pool)")
            leg.get_title().set_fontsize(9); leg.get_title().set_color(th["ink2"])
            for t in leg.get_texts():
                t.set_color(th["ink"])

        fig.suptitle("Performance vs number of motion types, at five data volumes",
                     fontsize=13.5, fontweight="bold", color=th["ink"], x=0.02, ha="left", y=1.015)
        fig.text(0.02, 0.978,
                 "Leak-free 4-source pool, 24 realisable cells, equal 50K-step budget. Motion types are the K largest "
                 "clusters of a K=1000 partition of the MAE (T-enc) representation. "
                 "Missing points are cells that cannot exist: K types cap how much data can be drawn.",
                 fontsize=9.2, color=th["ink2"], ha="left")
        fig.tight_layout(rect=[0, 0.01, 1, 0.965])
        for ext in ("png", "pdf"):
            p = os.path.join(OUT_DIR, f"enc_grid_4src_{theme}.{ext}")
            fig.savefig(p, dpi=300, facecolor=th["surface"], bbox_inches="tight")
            print(f"  wrote {p}")
        plt.close(fig)


if __name__ == "__main__":
    main()
