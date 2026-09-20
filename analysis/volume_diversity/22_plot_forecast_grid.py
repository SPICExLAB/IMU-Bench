#!/usr/bin/env python
"""Forecast counterpart of the printed nine-panel grid figure.

Same layout, axes, ordinal volume ramp, scratch lines and arm-pose noise band as
16_plot_enc_grid.py, drawn from out/phys_random_grid_forecast.json instead of
measured probes:
  x      motion-type clusters an arm may draw from (10 / 100 / 500 / 1000 of the
         K=1000 T-phys partition, random nested, source-stratified), log scale
  lines  pretraining volume, 0.1 / 1 / 10 / 100 % of the 1,738,094-window pool
  marks  filled = forecast inside measured diversity, hollow = extrapolated,
         diamond = the one cell already measured (whole pool)
  error  modelled quantities carry +/- leave-one-out error bars; flat quantities
         (model no better than the average) carry one shaded +/- band
Writes out/forecast_grid_4src_{light,dark}.{png,pdf}
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import OUT_DIR, POOL_N

THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", ramp=["#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", ramp=["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]),
}
CLUSTERS = {1: 10, 10: 100, 50: 500, 100: 1000}
KX = [10, 100, 500, 1000]
VOLS = [(0.1, "0.1%  (1,738)"), (1, "1%  (17,381)"), (10, "10%  (173,809)"), (100, "100%  (1,738,094)")]
PANELS = [("val_loss", "MAE val loss (pretrain)"), ("har_wisdm", "HAR · WISDM (macro-F1)"),
          ("har_utd_mhad", "HAR · UTD-MHAD (macro-F1)"), ("har_pamap2", "HAR · PAMAP2 (macro-F1)"),
          ("har_recofit", "HAR · RecoFit (macro-F1)"), ("microgesture_openwatch", "Microgesture (clip F1)"),
          ("arm_pose_nymeria", "Arm pose (pos err, cm)"), ("full_body_pose_nymeria", "Full-body pose (MPJPE, mm)"),
          ("odometry_nymeria_worldframe", "Odometry (ATE, m)")]
MIN_SPAN = {"har_wisdm": 0.15, "har_utd_mhad": 0.20, "har_pamap2": 0.20, "har_recofit": 0.20,
            "microgesture_openwatch": 0.20, "arm_pose_nymeria": 0.40, "full_body_pose_nymeria": 12.0,
            "odometry_nymeria_worldframe": 0.020, "val_loss": 0.30}


def main():
    F = json.load(open(os.path.join(OUT_DIR, "phys_random_grid_forecast.json")))
    cells = F["cells"]
    T = {t["task"]: t for t in F["tasks"]}
    for theme, th in THEMES.items():
        fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.6), facecolor=th["surface"])
        axes = axes.ravel()
        for pi, (task, label) in enumerate(PANELS):
            ax, t = axes[pi], T[task]
            flat, err = not t["model_informative"], t["interval"]
            ys_all = []
            if flat:
                c = t["const_value"]
                ax.axhspan(c - err, c + err, color=th["ramp"][1], alpha=0.10, lw=0, zorder=1)
                ax.text(0.98, 0.97, "flat forecast: same value in every cell", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7.8, color=th["ink2"])
                ys_all += [c - err, c + err]
            for vi, (npct, vlab) in enumerate(VOLS):
                idx = [i for i, cl in enumerate(cells) if cl["n_pct"] == npct]
                idx.sort(key=lambda i: cells[i]["k_pct"])
                col = th["ramp"][vi]
                if npct == 100:
                    i = idx[0]; x = CLUSTERS[cells[i]["k_pct"]]; m = t["measured"][i]
                    ax.plot([x], [m], marker="D", ms=9, color=col, linestyle="none",
                            markeredgecolor=th["surface"], markeredgewidth=1.4, zorder=6)
                    ys_all.append(m)
                    continue
                xs = [CLUSTERS[cells[i]["k_pct"]] for i in idx]
                ys = [t["pred"][i] for i in idx]
                ax.plot(xs, ys, lw=2, color=col, zorder=3)
                for x, y, i in zip(xs, ys, idx):
                    ex = t["extrapolated"][i]
                    if not flat:
                        ax.errorbar([x], [y], yerr=[[err], [err]], fmt="none", ecolor=col, elinewidth=1,
                                    capsize=2.5, alpha=0.55, zorder=2)
                    ax.plot([x], [y], marker="o", ms=5.5, linestyle="none", zorder=4,
                            color=col, markerfacecolor=(th["surface"] if ex else col),
                            markeredgecolor=(col if ex else th["surface"]), markeredgewidth=(1.5 if ex else 1.2))
                ys_all += [y + (0 if flat else s * err) for y in ys for s in (-1, 1)]
            ref = t["scratch"]
            if ref is not None:
                band = t["noise_band"]
                if band:
                    ax.axhspan(ref - band / 2, ref + band / 2, color=th["muted"], alpha=0.13, zorder=1)
                ax.axhline(ref, ls="--", lw=1.2, color=th["muted"], zorder=2)
                ax.text(1.02, ref, "scratch" + ("  ± noise" if band else ""),
                        transform=ax.get_yaxis_transform(), va="center", fontsize=7.8, color=th["muted"])
                ys_all.append(ref)
            lo, hi = min(ys_all), max(ys_all)
            need = MIN_SPAN.get(task)
            if need and (hi - lo) < need:
                mid = 0.5 * (lo + hi); lo, hi = mid - need / 2, mid + need / 2
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(label, fontsize=10, color=th["ink"], fontweight="bold", pad=6)
        for ax in axes:
            ax.set_facecolor(th["surface"])
            ax.set_xscale("log"); ax.set_xlim(7, 1450)
            ax.set_xticks(KX); ax.set_xticklabels([str(k) for k in KX], fontsize=8.5)
            ax.minorticks_off()
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(th["axis"])
            ax.tick_params(colors=th["muted"], labelsize=8.5)
            ax.grid(True, color=th["grid"], lw=0.8, zorder=0)
            ax.set_axisbelow(True)
        for ax in axes[6:]:
            ax.set_xlabel("motion-type clusters in the pretraining set", fontsize=9, color=th["ink2"])

        handles = [Line2D([], [], color=th["ramp"][i], lw=2, marker="o", ms=5.5, markeredgecolor=th["surface"])
                   for i in range(3)]
        handles.append(Line2D([], [], color=th["ramp"][3], lw=0, marker="D", ms=8, markeredgecolor=th["surface"]))
        labels = [VOLS[0][1], VOLS[1][1], VOLS[2][1], VOLS[3][1] + " · measured"]
        handles += [Line2D([], [], color=th["ink2"], lw=0, marker="o", ms=6, markerfacecolor=th["surface"],
                           markeredgecolor=th["ink2"], markeredgewidth=1.5),
                    Line2D([], [], color=th["ink2"], lw=1, marker="|", ms=9),
                    Patch(facecolor=th["ramp"][1], alpha=0.18, edgecolor="none")]
        labels += ["extrapolated", "± forecast error", "flat forecast ± error"]
        leg = fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False, fontsize=9.2,
                         bbox_to_anchor=(0.5, -0.03),
                         title=f"pretraining data (fraction of the {POOL_N:,}-window 4-source pool)")
        leg.get_title().set_fontsize(9); leg.get_title().set_color(th["ink2"])
        for tx in leg.get_texts():
            tx.set_color(th["ink"])

        fig.suptitle("Forecast: performance vs number of motion types, at four data volumes",
                     fontsize=13.5, fontweight="bold", color=th["ink"], x=0.02, ha="left", y=1.015)
        fig.text(0.02, 0.978,
                 "Not yet run. Leak-free 4-source pool, 12 realisable cells, equal 50K-step budget. Motion types are random, "
                 "source-stratified clusters of a K=1000 partition of 40 physical features (realised effective types "
                 "8.6 / 83 / 415 / 808 at 1% data). Forecast fitted on 58 completed arms.",
                 fontsize=9.2, color=th["ink2"], ha="left", wrap=True)
        fig.tight_layout(rect=[0, 0.01, 1, 0.965])
        for ext in ("png", "pdf"):
            p = os.path.join(OUT_DIR, f"forecast_grid_4src_{theme}.{ext}")
            fig.savefig(p, dpi=(160 if ext == "png" else None), facecolor=th["surface"], bbox_inches="tight")
            print("  wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
