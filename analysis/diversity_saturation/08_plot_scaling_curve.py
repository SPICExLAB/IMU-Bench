"""The 11k figure: does the performance knee sit where coverage saturates?

Two panels on one shared log-x axis -- deliberately NOT a dual-axis chart, so
coverage and performance are never plotted against two y-scales in one frame:

  top     cluster coverage vs data fraction   (the mechanism)
  bottom  probe performance vs data fraction  (the outcome)

A vertical rule at the coverage-saturation point crosses both, which is the
whole claim: if the bottom curve's knee lands on that rule, performance
saturates exactly where motion-mode coverage does.

The 1.3% -> 100% span is SHADED, not drawn as a connected line: those are 11i's
300K-step numbers and are not budget-matched to this 105K sweep. It becomes a
real line only once the matched-budget 100% arm is run.

Reads  out/subpct_selection.json, out/subpct_results.json, out/coverage_verification.json
Writes analysis/diversity_saturation/scaling_curve_{light,dark}.png/.pdf
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")

THEMES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835",
                  series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]),
}

PROBE_ORDER = ["WISDM (acc %)", "UTD-MHAD (acc %)", "PAMAP2 (acc %)", "RecoFit (acc %)",
               "Gesture (bal acc %)", "Arm pose (val loss)", "Full-body (mpjpe mm)",
               "Odometry (vel MSE)"]
ARM_FRAC = {"0.01%": 0.0001, "0.05%": 0.0005, "0.1%": 0.001, "0.5%": 0.005, "1.3%": 0.013}


def load():
    def maybe(name):
        p = os.path.join(OUT_DIR, name)
        return json.load(open(p)) if os.path.exists(p) else None

    cov = {}
    ver = maybe("coverage_verification.json")
    if ver:
        c = ver["configs"]["K1000_seed42"]["coverage"]
        cov = {float(k): v["empirical_mean"] for k, v in c.items()}
    sel = maybe("subpct_selection.json")
    if sel:
        for arm in sel["arms"].values():
            cov[float(arm["effective_fraction"])] = arm["coverage"]
    cov[1.0] = 1.0
    return cov, maybe("subpct_results.json")


def saturation_point(cov, thresh=0.999):
    xs = sorted(cov)
    for x in xs:
        if cov[x] >= thresh:
            return x
    return None


def draw(theme, res, cov, stem):
    th = THEMES[theme]
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(8.6, 7.4), sharex=True, facecolor=th["surface"],
        gridspec_kw={"height_ratios": [1.0, 1.55], "hspace": 0.13})

    for ax in (ax0, ax1):
        ax.set_facecolor(th["surface"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(th["axis"])
        ax.tick_params(colors=th["muted"], labelsize=9)
        ax.grid(True, which="major", axis="both", color=th["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    xs = np.array(sorted(cov))
    ys = np.array([cov[x] * 100 for x in xs])
    sat = saturation_point(cov)

    # ---- shaded "not budget-matched" span, drawn on both panels ------------
    for ax in (ax0, ax1):
        ax.axvspan(0.013, 1.0, color=th["muted"], alpha=0.07, zorder=0, lw=0)
        if sat:
            ax.axvline(sat, color=th["ink2"], lw=1.1, ls=(0, (5, 3)), zorder=2)

    # ---- panel A: coverage ------------------------------------------------
    ax0.plot(xs, ys, marker="o", ms=5, lw=2, color=th["series"][0],
             markeredgecolor=th["surface"], markeredgewidth=1.2, zorder=3)
    ax0.set_ylabel("motion-mode coverage\n(% of 1000 k-means clusters)",
                   fontsize=9.5, color=th["ink2"])
    ax0.set_ylim(0, 108)
    if sat:
        ax0.annotate(f"coverage $\\geq$99.9% from\n{sat:.2%} of the pool onward",
                     xy=(sat, 100), xytext=(sat * 1.9, 52),
                     fontsize=9, color=th["ink"],
                     arrowprops=dict(arrowstyle="->", color=th["ink2"], lw=1.1))

    # ---- panel B: performance --------------------------------------------
    # Eight probes is past the point where distinct hues stay separable, so the
    # ensemble is drawn muted and unlabelled and the MEAN carries the one
    # emphasised colour -- same summary 11i already reports.
    plotted, curves = 0, {}
    if res:
        for label in PROBE_ORDER:
            blk = res["probes"].get(label)
            if not blk:
                continue
            vals = blk["values"]
            pts = sorted((ARM_FRAC[a], v) for a, v in vals.items()
                         if v is not None and a in ARM_FRAC)
            if len(pts) < 2:
                continue
            px = np.array([p for p, _ in pts])
            pv = np.array([v for _, v in pts], dtype=float)
            lo, hi = pv.min(), pv.max()
            norm = np.full_like(pv, 0.5) if hi == lo else (pv - lo) / (hi - lo)
            if not blk["higher_better"]:
                norm = 1.0 - norm
            curves[label] = (px, norm)
            ax1.plot(px, norm, lw=1.1, color=th["muted"], alpha=0.55, zorder=2,
                     label="individual probes" if plotted == 0 else None)
            plotted += 1

    if curves:
        # mean +/- s.e.m. over the probes that have a value at each fraction
        all_x = sorted({x for px, _ in curves.values() for x in px})
        mx, mu, se = [], [], []
        for x in all_x:
            vs = [n[list(px).index(x)] for px, n in curves.values() if x in list(px)]
            if not vs:
                continue
            mx.append(x)
            mu.append(float(np.mean(vs)))
            se.append(float(np.std(vs, ddof=1) / np.sqrt(len(vs))) if len(vs) > 1 else 0.0)
        mx, mu, se = np.array(mx), np.array(mu), np.array(se)
        ax1.fill_between(mx, mu - se, mu + se, color=th["series"][0], alpha=0.18,
                         lw=0, zorder=3)
        ax1.plot(mx, mu, marker="o", ms=6, lw=2.4, color=th["series"][0],
                 markeredgecolor=th["surface"], markeredgewidth=1.4, zorder=4,
                 label=f"mean of {len(curves)} probes ($\\pm$s.e.m.)")

    ax1.set_ylabel("probe performance\n(normalised per task, 1 = best in sweep)",
                   fontsize=9.5, color=th["ink2"])
    ax1.set_xlabel("pretraining data (fraction of the 3,555,159-window pool, log scale)",
                   fontsize=9.5, color=th["ink2"])
    ax1.set_xscale("log")
    ax1.set_xlim(6e-5, 1.4)
    ax1.set_ylim(-0.08, 1.12)
    ax1.set_xticks([1e-4, 5e-4, 1e-3, 5e-3, 1.3e-2, 1e-1, 1.0])
    ax1.set_xticklabels(["0.01%", "0.05%", "0.1%", "0.5%", "1.3%", "10%", "100%"])

    if plotted:
        # the rising limb occupies the lower LEFT and the shaded band the right,
        # so the legend sits in the empty lower-centre
        leg = ax1.legend(loc="lower center", frameon=False, fontsize=9, ncol=1,
                         bbox_to_anchor=(0.42, 0.02))
        for t in leg.get_texts():
            t.set_color(th["ink"])
    else:
        ax1.text(0.5, 0.5, "no probe results harvested yet\n(run 07_harvest_subpct.py)",
                 transform=ax1.transAxes, ha="center", va="center",
                 fontsize=10, color=th["muted"])

    # annotation lives inside the shaded band, well clear of both the curve and
    # the legend
    ax1.text(0.86, 0.30, "11i reference range\n300K-step budget,\nnot matched to this sweep",
             transform=ax1.transAxes, fontsize=8.2, color=th["muted"],
             ha="center", va="center", linespacing=1.5)

    # The title states the QUESTION until the probe data is in; only once the
    # bottom panel has points is it honest to assert the answer.
    title = ("Performance saturates where motion-mode coverage saturates"
             if plotted else
             "Does performance saturate where motion-mode coverage does?")
    fig.suptitle(title, fontsize=13.5, fontweight="bold", color=th["ink"],
                 x=0.055, ha="left", y=0.975)
    fig.text(0.055, 0.933,
             "Sub-1% volume sweep, nested subsets, matched 105K-step budget (EXPERIMENTS.md 11k).",
             fontsize=9.2, color=th["ink2"], ha="left")

    for ext in ("png", "pdf"):
        p = f"{stem}.{ext}"
        fig.savefig(p, dpi=300, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.25)
        print("wrote", p)
    plt.close(fig)


def main():
    cov, res = load()
    if not cov:
        raise SystemExit("no coverage data found -- run 04_verify_coverage.py / 06_select_subpct.py")
    for theme in ("light", "dark"):
        draw(theme, res, cov, os.path.join(HERE, f"scaling_curve_{theme}"))


if __name__ == "__main__":
    main()
