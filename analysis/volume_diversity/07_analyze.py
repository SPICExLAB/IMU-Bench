#!/usr/bin/env python
"""Stage 7: turn the factorial into an EFFECT-SIZE comparison, not an ordering.

"Diversity matters more than volume" is only a claim if both effects are
measured on the same scale, in the same units, over a stated range. This does
three things, in increasing order of strength:

  (1) Marginal contrasts. For each volume level, the nar -> bal gap at FIXED N
      (pure diversity). For each diversity level, the 12.5K -> 800K gap at FIXED
      diversity (pure volume, 64x). Both in native metric units.

  (2) A joint model per task:
          y  ~  b_N * log2(N)  +  b_T * log2(effective motion types)
                                +  b_S * log2(effective sources)
      fitted over the cells, with a bootstrap CI on every coefficient and on the
      ratio |b_T| / |b_N|. Diversity is entered as the MEASURED effective type
      count of each realised subset (04), not as the design label.

  (3) The exchange rate: how many DOUBLINGS of data volume the observed
      diversity gap is worth, = (nar -> bal gap) / b_N. When b_N is not
      distinguishable from zero the honest output is a lower bound ("more than
      the 6 doublings tested"), and that is what gets printed -- an unbounded
      "diversity beats volume" is exactly the over-claim this experiment exists
      to replace.

Writes out/effects.json and analysis/volume_diversity/voldiv_{light,dark}.{png,pdf}
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import DIVERSITIES, DIV_LABEL, OUT_DIR, VOLUMES, arm_name

HERE = os.path.dirname(os.path.abspath(__file__))
THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", grid="#e1e0d9",
                  axis="#c3c2b7", ramp=["#c6d6e8", "#86b6ef", "#2a78d6", "#0d366b"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#2c2c2a",
                 axis="#383835", ramp=["#cde2fb", "#9ec5f4", "#6da7ec", "#1c5cab"]),
}
PANELS = [
    ("pretrain_val_loss", None, "MAE val loss", False),
    ("probe", "arm_pose_nymeria", "Arm pose (cm)", False),
    ("probe", "odometry_nymeria_worldframe", "Odometry ATE (m)", False),
    ("probe", "har_wisdm", "WISDM macro-F1", True),
    ("probe", "microgesture_openwatch", "Gesture clip macro-F1", True),
]
BOOT = 2000


def value(cell, kind, task):
    if kind == "pretrain_val_loss":
        return cell.get("pretrain_val_loss")
    p = (cell.get("probes") or {}).get(task)
    return p.get("value") if p else None


def fit(y, X, seed=42):
    """OLS with an intercept + a bootstrap over cells. Returns coefficients and
    percentile CIs. Deliberately simple: with ~16 cells anything fancier would
    be fitting noise, and the point is a ratio of two slopes, not prediction."""
    rng = np.random.default_rng(seed)
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    boots = []
    for _ in range(BOOT):
        s = rng.integers(0, len(y), len(y))
        try:
            b, *_ = np.linalg.lstsq(A[s], y[s], rcond=None)
            boots.append(b)
        except np.linalg.LinAlgError:
            continue
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return beta, lo, hi, boots


def main():
    with open(os.path.join(OUT_DIR, "results.json")) as f:
        res = json.load(f)
    cells = res["cells"]
    report = {"contrasts": {}, "models": {}, "exchange_rate": {}}

    print("=" * 92)
    print("(1) MARGINAL CONTRASTS -- each factor with the other one held fixed")
    print("=" * 92)
    for kind, task, label, higher in PANELS:
        key = label
        report["contrasts"][key] = {"higher_is_better": higher, "diversity": {}, "volume": {}}
        print(f"\n{label}   ({'higher' if higher else 'lower'} is better)")
        for tag, n in VOLUMES:                      # diversity at fixed N
            a, b = value(cells.get(arm_name("nar", tag), {}), kind, task), \
                   value(cells.get(arm_name("bal", tag), {}), kind, task)
            if a is None or b is None:
                continue
            report["contrasts"][key]["diversity"][tag] = {"nar": a, "bal": b, "delta": b - a}
            print(f"   diversity  N={n:>9,}   nar {a:>9.4f} -> bal {b:>9.4f}   "
                  f"delta {b - a:>+9.4f}  ({100 * (b - a) / abs(a):+.1f}%)")
        for div in DIVERSITIES:                     # volume at fixed diversity
            a = value(cells.get(arm_name(div, VOLUMES[0][0]), {}), kind, task)
            b = value(cells.get(arm_name(div, VOLUMES[-1][0]), {}), kind, task)
            if a is None or b is None:
                continue
            report["contrasts"][key]["volume"][div] = {"lo": a, "hi": b, "delta": b - a}
            print(f"   volume     {div:<4s} 64x    {a:>9.4f} -> {b:>9.4f}   "
                  f"delta {b - a:>+9.4f}  ({100 * (b - a) / abs(a):+.1f}%)")

    # ---- (1b) the subject ladder, if 09 built it -----------------------
    subs = sorted((a for a in cells if a.startswith("vd_sub")),
                  key=lambda a: cells[a].get("effective_subjects") or 0)
    if subs:
        print()
        print("=" * 92)
        print("(1b) SUBJECT LADDER -- N, source and motion-type mix held fixed, wearers varied")
        print("=" * 92)
        print("     the eff-types column is the CONTROL: if it is flat across rows, the only")
        print("     thing that moved is who wore the watch")
        report["subject_ladder"] = {}
        for kind, task, label, higher in PANELS:
            vals = [(a, value(cells[a], kind, task)) for a in subs]
            vals = [(a, v) for a, v in vals if v is not None]
            if not vals:
                continue
            print(f"\n{label}")
            for a, v in vals:
                c = cells[a]
                print(f"   {a:16s} subjects={str(c.get('n_subjects_known')):>4s} "
                      f"eff-subjects={(c.get('effective_subjects') or float('nan')):>6.2f}  "
                      f"eff-types={(c.get('eff_types_phys') or float('nan')):>7.1f}  "
                      f"{v:>10.4f}")
            report["subject_ladder"][label] = {a: v for a, v in vals}
            spread = max(v for _, v in vals) - min(v for _, v in vals)
            print(f"   spread across the ladder: {spread:+.4f}")

    print()
    print("=" * 92)
    print("(2) JOINT MODEL   y ~ log2(N) + log2(effective types) + log2(effective sources)")
    print("=" * 92)
    for kind, task, label, higher in PANELS:
        rows = [(c, value(c, kind, task)) for a, c in cells.items()
                if not a.startswith("vd_sub")]      # the ladder is a separate design
        rows = [(c, v) for c, v in rows if v is not None
                and c.get("eff_types_phys") and c.get("effective_sources")]
        if len(rows) < 8:
            print(f"\n{label}: only {len(rows)} cells scored, model skipped")
            continue
        y = np.array([v for _, v in rows], dtype=float)
        rows_max_n = max(c["n_windows"] for c, _ in rows)
        X = np.column_stack([
            np.log2([c["n_windows"] for c, _ in rows]),
            np.log2([c["eff_types_phys"] for c, _ in rows]),
            np.log2([c["effective_sources"] for c, _ in rows]),
        ])
        beta, lo, hi, boots = fit(y, X)
        names = ["log2 N", "log2 eff-types", "log2 eff-sources"]
        # the two diversity predictors can only be read separately if the design
        # actually separated them -- print the evidence rather than assume it
        corr = np.corrcoef(X, rowvar=False)
        cond = float(np.linalg.cond(np.column_stack([np.ones(len(y)), X])))
        print(f"\n{label}  (n={len(rows)} cells)")
        print(f"   predictor corr: N~types {corr[0, 1]:+.2f}  N~sources {corr[0, 2]:+.2f}  "
              f"types~sources {corr[1, 2]:+.2f}   design condition number {cond:.1f}"
              + ("   [WARNING: collinear, do not read the two diversity coefficients apart]"
                 if abs(corr[1, 2]) > 0.9 else ""))
        for i, nm in enumerate(names, start=1):
            print(f"   b[{nm:<16s}] = {beta[i]:>+9.4f}   95% CI [{lo[i]:>+8.4f}, {hi[i]:>+8.4f}]"
                  f"{'   (CI spans 0)' if lo[i] <= 0 <= hi[i] else ''}")
        ratio = np.abs(boots[:, 2]) / np.maximum(np.abs(boots[:, 1]), 1e-12)
        rlo, rmed, rhi = np.percentile(ratio, [2.5, 50, 97.5])
        print(f"   |b_types| / |b_N| = {rmed:.2f}   95% CI [{rlo:.2f}, {rhi:.2f}]"
              f"   (per doubling; >1 means one doubling of motion-type diversity "
              f"moves the metric more than one doubling of data)")
        report["models"][label] = {
            "n_cells": len(rows),
            "coefficients": {nm: float(beta[i + 1]) for i, nm in enumerate(names)},
            "ci_lo": {nm: float(lo[i + 1]) for i, nm in enumerate(names)},
            "ci_hi": {nm: float(hi[i + 1]) for i, nm in enumerate(names)},
            "ratio_types_over_volume": {"median": float(rmed), "ci": [float(rlo), float(rhi)]},
        }

        # (3) exchange rate at the middle volume level
        mid = VOLUMES[1][0]
        a = value(cells.get(arm_name("nar", mid), {}), kind, task)
        b = value(cells.get(arm_name("bal", mid), {}), kind, task)
        if a is not None and b is not None:
            gap = b - a
            bn = beta[1]
            if lo[1] <= 0 <= hi[1]:
                msg = (f"   exchange rate: the diversity gap ({gap:+.4f}) cannot be bought with "
                       f"volume -- b_N is indistinguishable from 0 over the 64x tested")
                report["exchange_rate"][label] = {"gap": float(gap), "doublings": None,
                                                  "note": "volume slope CI spans 0"}
            else:
                d = gap / bn
                # never extrapolate past what was actually swept: the volume axis
                # spans 12.5K -> 1.74M windows, i.e. log2(1738094/12500) doublings.
                tested = float(np.log2(rows_max_n / VOLUMES[0][1]))
                if abs(d) > tested:
                    msg = (f"   exchange rate: the diversity gap ({gap:+.4f}) exceeds what the "
                           f"whole {tested:.1f}-doubling volume sweep buys -- reported as a lower "
                           f"bound (> {tested:.1f} doublings), NOT extrapolated")
                    report["exchange_rate"][label] = {"gap": float(gap),
                                                      "doublings_lower_bound": tested,
                                                      "note": "gap exceeds the swept volume range"}
                else:
                    msg = (f"   exchange rate: the diversity gap is worth {d:,.1f} doublings of "
                           f"data volume ({2 ** abs(d):,.0f}x), within the {tested:.1f} doublings swept")
                    report["exchange_rate"][label] = {"gap": float(gap), "doublings": float(d),
                                                      "doublings_swept": tested}
            print(msg)

    with open(os.path.join(OUT_DIR, "effects.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---------------- figure ----------------
    for theme, th in THEMES.items():
        fig, axes = plt.subplots(2, len(PANELS), figsize=(4.0 * len(PANELS), 7.2),
                                 facecolor=th["surface"])
        for j, (kind, task, label, higher) in enumerate(PANELS):
            # top: metric vs volume, one line per diversity level
            ax = axes[0, j]
            for i, div in enumerate(DIVERSITIES):
                xs, ys = [], []
                for tag, n in VOLUMES:
                    v = value(cells.get(arm_name(div, tag), {}), kind, task)
                    if v is not None:
                        xs.append(n)
                        ys.append(v)
                if xs:
                    ax.plot(xs, ys, "o-", color=th["ramp"][i], lw=1.8, ms=4,
                            label=DIV_LABEL[div].split(",")[0])
            ax.set_xscale("log")
            ax.set_xlabel("distinct pretraining windows", fontsize=8)
            ax.set_title(label, fontsize=9, color=th["ink"])
            if j == 0:
                ax.legend(fontsize=6, frameon=False, labelcolor=th["ink2"])
            # bottom: metric vs MEASURED diversity, one line per volume
            ax2 = axes[1, j]
            for i, (tag, n) in enumerate(VOLUMES):
                xs, ys = [], []
                for div in DIVERSITIES:
                    c = cells.get(arm_name(div, tag), {})
                    v = value(c, kind, task)
                    if v is not None and c.get("eff_types_phys"):
                        xs.append(c["eff_types_phys"])
                        ys.append(v)
                if xs:
                    o = np.argsort(xs)
                    ax2.plot(np.array(xs)[o], np.array(ys)[o], "o-", color=th["ramp"][i],
                             lw=1.8, ms=4, label=f"N={n:,}")
            ax2.set_xscale("log")
            ax2.set_xlabel("effective motion types (measured)", fontsize=8)
            if j == 0:
                ax2.legend(fontsize=6, frameon=False, labelcolor=th["ink2"])
            for a in (ax, ax2):
                a.set_facecolor(th["surface"])
                a.grid(True, color=th["grid"], lw=0.5)
                a.tick_params(colors=th["ink2"], labelsize=7)
                for s in a.spines.values():
                    s.set_color(th["axis"])
        fig.suptitle("Volume and diversity, decoupled (full-size 4-source pool, equal 50K-step budget)",
                     color=th["ink"], fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(HERE, f"voldiv_{theme}.{ext}"), dpi=300,
                        facecolor=th["surface"], bbox_inches="tight")
        plt.close(fig)
    print(f"\nwrote {os.path.join(OUT_DIR, 'effects.json')} and voldiv_{{light,dark}}.png/pdf")


if __name__ == "__main__":
    main()
