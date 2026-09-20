#!/usr/bin/env python
"""Measured random physical-type x volume grid, drawn over its pre-registered forecast.

Same nine-panel layout as 16_plot_enc_grid.py / 22_plot_forecast_grid.py:
  x       motion-type clusters an arm may draw from (10 / 100 / 500 / 1000)
  colour  pretraining volume (0.1 / 1 / 10 % ramp; the whole pool is the diamond)
  solid   MEASURED cells (vd_pk*, and vd_ek1000_f100 for the whole pool)
  dashed  the forecast from out/phys_random_grid_forecast.json, unchanged since it
          was written; flat forecasts keep their shaded +/- band
Cells not yet scored are simply absent; each panel reports how many of the 12 exist.
Writes out/grid_vs_forecast_4src_{light,dark}.{png,pdf}
"""
from __future__ import annotations

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import OUT_DIR, POOL_N

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", ramp=["#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", ramp=["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"]),
}
CLUSTERS = {1: 10, 10: 100, 50: 500, 100: 1000}
KX = [10, 100, 500, 1000]
VOLS = [(0.1, "0.1%  (1,738)"), (1, "1%  (17,381)"), (10, "10%  (173,809)"), (100, "100%  (1,738,094)")]
ARM = {(1, 0.1): "vd_pk1_n0p1", (1, 1): "vd_pk1_n1", (10, 0.1): "vd_pk10_n0p1", (10, 1): "vd_pk10_n1",
       (10, 10): "vd_pk10_n10", (50, 0.1): "vd_pk50_n0p1", (50, 1): "vd_pk50_n1", (50, 10): "vd_pk50_n10",
       (100, 0.1): "vd_pk100_n0p1", (100, 1): "vd_pk100_n1", (100, 10): "vd_pk100_n10", (100, 100): "vd_ek1000_f100"}
PANELS = [("val_loss", "MAE val loss (pretrain)", None), ("har_wisdm", "HAR · WISDM (macro-F1)", "macro_f1"),
          ("har_utd_mhad", "HAR · UTD-MHAD (macro-F1)", "macro_f1"), ("har_pamap2", "HAR · PAMAP2 (macro-F1)", "macro_f1"),
          ("har_recofit", "HAR · RecoFit (macro-F1)", "macro_f1"), ("microgesture_openwatch", "Microgesture (clip F1)", "clip_macro_f1"),
          ("arm_pose_nymeria", "Arm pose (pos err, cm)", "pos_err_cm"), ("full_body_pose_nymeria", "Full-body pose (MPJPE, mm)", "mpjpe_mm"),
          ("odometry_nymeria_worldframe", "Odometry (ATE, m)", "ate_m")]
MIN_SPAN = {"har_wisdm": 0.15, "har_utd_mhad": 0.20, "har_pamap2": 0.20, "har_recofit": 0.20,
            "microgesture_openwatch": 0.20, "arm_pose_nymeria": 0.40, "full_body_pose_nymeria": 12.0,
            "odometry_nymeria_worldframe": 0.020, "val_loss": 0.30}


def val_loss(arm):
    p = os.path.join(REPO, "runs", arm, "logs", "metrics.txt")
    if not os.path.exists(p):
        return None
    last = None
    for line in open(p, errors="ignore"):
        if "split=val" in line:
            m = re.search(r"step=(\d+).*?loss=([0-9.]+)", line)
            if m and int(m.group(1)) >= 50_000:
                last = float(m.group(2))
    return last


def probe(arm, task, key):
    p = os.path.join(REPO, "bench", "results", task, f"{arm}__probe__s42.json")
    return json.load(open(p))["metrics"].get(key) if os.path.exists(p) else None


def main():
    F = json.load(open(os.path.join(OUT_DIR, "phys_random_grid_forecast.json")))
    cells = F["cells"]
    T = {t["task"]: t for t in F["tasks"]}
    measured = {}
    for task, _, key in PANELS:
        for c in cells:
            arm = ARM[(c["k_pct"], c["n_pct"])]
            v = val_loss(arm) if key is None else probe(arm, task, key)
            if v is not None:
                measured[(task, c["k_pct"], c["n_pct"])] = v
    n_probe = sum(1 for k in measured if k[0] != "val_loss")
    print(f"[plot] measured: {sum(1 for k in measured if k[0]=='val_loss')}/12 pretrain cells, {n_probe}/96 probe cells")

    for theme, th in THEMES.items():
        fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.6), facecolor=th["surface"])
        axes = axes.ravel()
        for pi, (task, label, _key) in enumerate(PANELS):
            ax, t = axes[pi], T[task]
            flat, err = not t["model_informative"], t["interval"]
            ys = []
            if flat:
                c0 = t["const_value"]
                ax.axhspan(c0 - err, c0 + err, color=th["ramp"][1], alpha=0.09, lw=0, zorder=1)
                ys += [c0 - err, c0 + err]
            have = 0
            for vi, (npct, _vlab) in enumerate(VOLS):
                col = th["ramp"][vi]
                idx = sorted([i for i, c in enumerate(cells) if c["n_pct"] == npct], key=lambda i: cells[i]["k_pct"])
                if npct == 100:
                    i = idx[0]; x = CLUSTERS[cells[i]["k_pct"]]
                    m = measured.get((task, cells[i]["k_pct"], npct))
                    if m is not None:
                        have += 1
                        ax.plot([x], [m], marker="D", ms=9, color=col, linestyle="none",
                                markeredgecolor=th["surface"], markeredgewidth=1.4, zorder=6)
                        ys.append(m)
                    continue
                xs = [CLUSTERS[cells[i]["k_pct"]] for i in idx]
                fc = [t["pred"][i] for i in idx]
                if not flat:
                    ax.plot(xs, fc, lw=1.3, ls=(0, (4, 3)), color=col, alpha=0.75, zorder=2)
                    ys += fc
                mx = [(CLUSTERS[cells[i]["k_pct"]], measured[(task, cells[i]["k_pct"], npct)]) for i in idx
                      if (task, cells[i]["k_pct"], npct) in measured]
                have += len(mx)
                if mx:
                    ax.plot([p[0] for p in mx], [p[1] for p in mx], lw=2.2, color=col, marker="o", ms=6,
                            markeredgecolor=th["surface"], markeredgewidth=1.2, zorder=4)
                    ys += [p[1] for p in mx]
            ref = t["scratch"]
            if ref is not None:
                band = t["noise_band"]
                if band:
                    ax.axhspan(ref - band / 2, ref + band / 2, color=th["muted"], alpha=0.13, zorder=1)
                ax.axhline(ref, ls="--", lw=1.1, color=th["muted"], zorder=2)
                ax.text(1.02, ref, "scratch" + ("  ± noise" if band else ""), transform=ax.get_yaxis_transform(),
                        va="center", fontsize=7.8, color=th["muted"])
                ys.append(ref)
            lo, hi = min(ys), max(ys)
            need = MIN_SPAN.get(task)
            if need and hi - lo < need:
                mid = 0.5 * (lo + hi); lo, hi = mid - need / 2, mid + need / 2
            pad = 0.08 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(label, fontsize=10, color=th["ink"], fontweight="bold", pad=6)
            ax.text(0.98, 0.97, f"{have}/12 cells measured" + (" · flat forecast" if flat else ""),
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=th["ink2"])
        for ax in axes:
            ax.set_facecolor(th["surface"]); ax.set_xscale("log"); ax.set_xlim(7, 1450)
            ax.set_xticks(KX); ax.set_xticklabels([str(k) for k in KX], fontsize=8.5); ax.minorticks_off()
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(th["axis"])
            ax.tick_params(colors=th["muted"], labelsize=8.5)
            ax.grid(True, color=th["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
        for ax in axes[6:]:
            ax.set_xlabel("motion-type clusters in the pretraining set", fontsize=9, color=th["ink2"])
        handles = [Line2D([], [], color=th["ramp"][i], lw=2.2, marker="o", ms=6, markeredgecolor=th["surface"]) for i in range(3)]
        handles.append(Line2D([], [], color=th["ramp"][3], lw=0, marker="D", ms=8, markeredgecolor=th["surface"]))
        labels = [VOLS[0][1], VOLS[1][1], VOLS[2][1], VOLS[3][1]]
        handles += [Line2D([], [], color=th["ink2"], lw=1.3, ls=(0, (4, 3))), Patch(facecolor=th["ramp"][1], alpha=0.16, edgecolor="none")]
        labels += ["forecast (dashed)", "flat forecast ± error"]
        leg = fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=9.2, bbox_to_anchor=(0.5, -0.03),
                         title=f"solid = measured · pretraining data (fraction of the {POOL_N:,}-window 4-source pool)")
        leg.get_title().set_fontsize(9); leg.get_title().set_color(th["ink2"])
        for tx in leg.get_texts():
            tx.set_color(th["ink"])
        fig.suptitle("Measured vs forecast: performance vs number of motion types, at four data volumes",
                     fontsize=13.5, fontweight="bold", color=th["ink"], x=0.02, ha="left", y=1.015)
        fig.text(0.02, 0.978, f"Random, source-stratified physical-type clusters (gyro-unit fixed), 50K-step budget, seed 42. "
                 + (f"All 96 probe cells scored. " if n_probe == 96 else f"{n_probe}/96 probe cells scored so far; missing points are still running. ") + f"Dashed lines are the forecast "
                 f"written before any cell was trained.", fontsize=9.2, color=th["ink2"], ha="left")
        fig.tight_layout(rect=[0, 0.01, 1, 0.965])
        for ext in ("png", "pdf"):
            p = os.path.join(OUT_DIR, f"grid_vs_forecast_4src_{theme}.{ext}")
            fig.savefig(p, dpi=(160 if ext == "png" else None), facecolor=th["surface"], bbox_inches="tight")
            print("  wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
