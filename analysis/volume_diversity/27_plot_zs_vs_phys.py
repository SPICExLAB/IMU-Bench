#!/usr/bin/env python
"""Probe-input protocol contrast on the vd_ek grid: z-score vs physical scaling.

Every vd_ek cell was pretrained on per-window z-scored input. It was first probed with
PHYSICAL-scaled input (the bench default from 2026-08-20 to 2026-09-14), then re-probed with
Z-SCORE input matching its pretraining (alias encoders `vd_ek*_zs`, submitted 2026-09-15).
Same checkpoint, same probe recipe, same seed -- only the downstream input scaling differs.

  panel 1   mean delta per task (z-score minus physical, signed so + = z-score better)
  panels 2-9  the nine-panel grid layout: x = clusters the arm may draw from, one line per
              volume; solid = z-score input, dashed = physical input
Cells whose z-score probe has not finished are simply absent.
Writes out/zs_vs_phys_4src_{light,dark}.{png,pdf} and out/zs_vs_phys_deltas.json
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import OUT_DIR, POOL_N

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", ramp=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"], pos="#1baf7a", neg="#eb6834"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                  axis="#383835", ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab"], pos="#199e70", neg="#d95926"),
}
KS = [1, 5, 10, 25, 50, 100, 1000]
FRACS = [("0p01", "0.01%  (174)"), ("0p1", "0.1%  (1,738)"), ("1", "1%  (17,381)"),
         ("10", "10%  (173,809)"), ("100", "100%  (1,738,094)")]
PROBES = [("HAR · WISDM (macro-F1)", True, ("har_wisdm", "macro_f1")),
          ("HAR · UTD-MHAD (macro-F1)", True, ("har_utd_mhad", "macro_f1")),
          ("HAR · PAMAP2 (macro-F1)", True, ("har_pamap2", "macro_f1")),
          ("HAR · RecoFit (macro-F1)", True, ("har_recofit", "macro_f1")),
          ("Microgesture (clip F1)", True, ("microgesture_openwatch", "clip_macro_f1")),
          ("Arm pose (pos err, cm)", False, ("arm_pose_nymeria", "pos_err_cm")),
          ("Full-body pose (MPJPE, mm)", False, ("full_body_pose_nymeria", "mpjpe_mm")),
          ("Odometry (ATE, m)", False, ("odometry_nymeria_worldframe", "ate_m")),
          ("HAR · HHAR (macro-F1)", True, ("har_hhar", "macro_f1")),
          ("HAR · MHEALTH (macro-F1)", True, ("har_mhealth", "macro_f1")),
          ("HAR · SAMoSA (macro-F1)", True, ("har_samosa", "macro_f1")),
          ("HAR · UT-Watch (macro-F1)", True, ("har_ut_watch", "macro_f1")),
          ("HAR · PRISM (macro-F1)", True, ("har_prism", "macro_f1"))]
N_CELLS = 24 * len(PROBES)
REF = {"HAR · WISDM (macro-F1)": 0.6056, "HAR · UTD-MHAD (macro-F1)": 0.7010, "HAR · PAMAP2 (macro-F1)": 0.2190,
       "HAR · RecoFit (macro-F1)": 0.6630, "Microgesture (clip F1)": 0.4876, "Arm pose (pos err, cm)": 8.1529,
       "Full-body pose (MPJPE, mm)": 92.428, "Odometry (ATE, m)": 0.1400}
MIN_SPAN = {"HAR · WISDM (macro-F1)": 0.15, "HAR · UTD-MHAD (macro-F1)": 0.20, "HAR · PAMAP2 (macro-F1)": 0.20,
            "HAR · RecoFit (macro-F1)": 0.20, "Microgesture (clip F1)": 0.20, "Arm pose (pos err, cm)": 0.40,
            "Full-body pose (MPJPE, mm)": 12.0, "Odometry (ATE, m)": 0.020}
NOISE = {"Arm pose (pos err, cm)": 0.108}
for _l, _h, (_t, _k) in PROBES[8:]:
    MIN_SPAN[_l] = 0.20
    _p = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
                      "bench", "results", _t, "scratch__scratch__s42.json")
    if os.path.exists(_p):
        REF[_l] = json.load(open(_p))["metrics"][_k]


def probe(arm, task, key):
    p = os.path.join(REPO, "bench", "results", task, f"{arm}__probe__s42.json")
    return json.load(open(p))["metrics"].get(key) if os.path.exists(p) else None


def main():
    cells = sorted(json.load(open(os.path.join(OUT_DIR, "enc_type_volume_grid.json")))["arms"])
    data, deltas = {}, {}
    for label, hib, (task, key) in PROBES:
        d = {}
        for K in KS:
            for ftag, _ in FRACS:
                arm = f"vd_ek{K}_f{ftag}"
                if arm not in cells:
                    continue
                ph, zs = probe(arm, task, key), probe(arm + "_zs", task, key)
                if ph is not None or zs is not None:
                    d[(K, ftag)] = (zs, ph)
        data[label] = d
        pair = [(z - p) * (1 if hib else -1) for z, p in d.values() if z is not None and p is not None]
        phys_vals = [p for z, p in d.values() if z is not None and p is not None]
        deltas[label] = {"n": len(pair), "mean": float(np.mean(pair)) if pair else None,
                         "z_better": int(sum(x > 0 for x in pair)),
                         "min": float(min(pair)) if pair else None, "max": float(max(pair)) if pair else None,
                         "unit": label.split("(")[-1].rstrip(")"),
                         "phys_mean": float(np.mean(phys_vals)) if phys_vals else None}
    n_zs = sum(1 for d in data.values() for z, _ in d.values() if z is not None)
    print(f"[plot] z-score cells scored: {n_zs}/{N_CELLS}")
    json.dump(deltas, open(os.path.join(OUT_DIR, "zs_vs_phys_deltas.json"), "w"), indent=1)

    for theme, th in THEMES.items():
        fig, axes = plt.subplots(4, 4, figsize=(16.5, 13.6), facecolor=th["surface"])
        axes = axes.ravel()
        ax0 = axes[0]
        axes[-1].axis("off")
        names = [l for l, _, _ in PROBES if deltas[l]["mean"] is not None]
        vals = [deltas[l]["mean"] / abs(deltas[l]["phys_mean"]) * 100 for l in names]
        y = np.arange(len(names))[::-1]
        ax0.barh(y, vals, color=[th["pos"] if v > 0 else th["neg"] for v in vals], height=0.6)
        ax0.axvline(0, color=th["axis"], lw=1)
        ax0.set_yticks(y); ax0.set_yticklabels([n.split(" (")[0].replace("HAR · ", "") for n in names], fontsize=8.5)
        for yy, v, l in zip(y, vals, names):
            d = deltas[l]
            ax0.text(max(v, 0) + 1.5, yy, f"{d['mean']:+.3g} {d['unit']}  ({d['z_better']}/{d['n']} cells)",
                     va="center", ha="left", fontsize=7.0, color=th["ink2"])
        ax0.set_xlim(min(vals + [0]) * 1.15 - 2, max(vals + [0]) + 48)
        ax0.set_xlabel("mean effect of z-score input, % of the physical-input mean", fontsize=8.5, color=th["ink2"])
        ax0.set_title("z-score minus physical input\n(+ = z-score better; cells z-better/paired)",
                      fontsize=10, color=th["ink"], fontweight="bold", pad=6, loc="left")

        for pi, (label, hib, _spec) in enumerate(PROBES):
            ax = axes[pi + 1]
            blk = data[label]
            ys = []
            for fi, (ftag, flab) in enumerate(FRACS):
                col = th["ramp"][fi]
                pts_z = [(K, blk[(K, ftag)][0]) for K in KS if (K, ftag) in blk and blk[(K, ftag)][0] is not None]
                pts_p = [(K, blk[(K, ftag)][1]) for K in KS if (K, ftag) in blk and blk[(K, ftag)][1] is not None]
                if pts_p:
                    ax.plot([p[0] for p in pts_p], [p[1] for p in pts_p], lw=1.3, ls=(0, (4, 3)),
                            color=col, alpha=0.7, marker="o", ms=3.5, markeredgecolor=th["surface"], markeredgewidth=0.8, zorder=2)
                    ys += [p[1] for p in pts_p]
                if pts_z:
                    ax.plot([p[0] for p in pts_z], [p[1] for p in pts_z], lw=2.2, color=col, marker="o", ms=5.5,
                            markeredgecolor=th["surface"], markeredgewidth=1.2, zorder=4)
                    ys += [p[1] for p in pts_z]
            ref = REF.get(label)
            if ref is not None:
                band = NOISE.get(label)
                if band:
                    ax.axhspan(ref - band / 2, ref + band / 2, color=th["muted"], alpha=0.13, zorder=1)
                ax.axhline(ref, ls="--", lw=1.1, color=th["muted"], zorder=2)
                ax.text(1.02, ref, "scratch", transform=ax.get_yaxis_transform(), va="center", fontsize=7.8, color=th["muted"])
                ys.append(ref)
            if ys:
                lo, hi = min(ys), max(ys)
                need = MIN_SPAN.get(label)
                if need and hi - lo < need:
                    mid = 0.5 * (lo + hi); lo, hi = mid - need / 2, mid + need / 2
                pad = 0.08 * (hi - lo); ax.set_ylim(lo - pad, hi + pad)
            n_z = sum(1 for k in blk if blk[k][0] is not None)
            if not blk:
                ax.set_yticks([])
                ax.text(0.5, 0.5, "queued — no cells scored yet", transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color=th["muted"])
            ax.set_title(label, fontsize=10, color=th["ink"], fontweight="bold", pad=6)
            ax.text(0.98, 0.97, f"{n_z}/24 z-score cells", transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.8, color=th["ink2"])
        for ax in axes[:-1]:
            ax.set_facecolor(th["surface"])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(th["axis"])
            ax.tick_params(colors=th["muted"], labelsize=8.5)
            ax.grid(True, color=th["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
        ax0.grid(axis="y", b=False) if False else None
        for ax in axes[1:-1]:
            ax.set_xscale("log"); ax.set_xlim(0.8, 1300); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS], fontsize=8.5); ax.minorticks_off()
        for ax in axes[10:14]:
            ax.set_xlabel("motion-type clusters in the pretraining set", fontsize=9, color=th["ink2"])
        for ax in axes[len(PROBES) + 1:-1]:
            ax.axis("off")
        handles = [Line2D([], [], color=th["ramp"][i], lw=2.2, marker="o", ms=5) for i in range(len(FRACS))]
        labels = [f for _, f in FRACS]
        handles += [Line2D([], [], color=th["ink2"], lw=2.2), Line2D([], [], color=th["ink2"], lw=1.3, ls=(0, (4, 3)))]
        labels += ["z-score input (matches pretraining)", "physical input (old protocol)"]
        leg = axes[-1].legend(handles, labels, loc="center", ncol=1, frameon=False, fontsize=9.2,
                         title=f"pretraining data\n(fraction of the {POOL_N:,}-window\n4-source pool)")
        leg.get_title().set_fontsize(9); leg.get_title().set_color(th["ink2"])
        for t in leg.get_texts():
            t.set_color(th["ink"])
        fig.suptitle("Probe input scaling: z-score (matches pretraining) vs physical — same checkpoints, same recipe",
                     fontsize=13.5, fontweight="bold", color=th["ink"], x=0.02, ha="left", y=1.015)
        fig.text(0.02, 0.978, f"vd_ek grid, 4-source leak-free pool, 24 cells x {len(PROBES)} tasks, seed 42. "
                 + (f"All {N_CELLS} z-score cells scored. " if n_zs == N_CELLS else f"{n_zs}/{N_CELLS} z-score cells scored; the rest are still queued. ") + "Every encoder was pretrained on "
                 f"per-window z-scored input, so the solid lines are the protocol-consistent evaluation.",
                 fontsize=9.2, color=th["ink2"], ha="left")
        fig.tight_layout(rect=[0, 0.01, 1, 0.972])
        for ext in ("png", "pdf"):
            p = os.path.join(OUT_DIR, f"zs_vs_phys_4src_{theme}.{ext}")
            fig.savefig(p, dpi=(160 if ext == "png" else None), facecolor=th["surface"], bbox_inches="tight")
            print("  wrote", p)
        plt.close(fig)


if __name__ == "__main__":
    main()
