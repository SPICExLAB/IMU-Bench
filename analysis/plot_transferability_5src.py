#!/usr/bin/env python3
"""Cross-task transfer matrices for the 5-source MAE (paper Section 4), read from bench results.

Same figure design as plot_transferability.py (whose drawing code is reused), but every cell is read
from bench/results instead of being typed in:

  source "MAE"       mae_5src            (probe / finetune)
  source <task>      tx5_<task>          mae_5src fine-tuned on <task>, then frozen (probe) or used as
                                         the initialisation (finetune) for the target
  reference column   scratch             the target trained from random initialisation

Targets / sources: PRISM activity recognition (the leak-free HAR set used as the HAR source). The compact
panels label that row and column "HAR" for width: it is PRISM ALONE, not the four-set mean that Figure 2's
"Activity" axis shows, and the caption has to say so. Also Nymeria
full-body pose, OpenWatch microgesture, Nymeria world-frame odometry, PADS Parkinson's. Classification
metrics are shown in %.

    python analysis/plot_transferability_5src.py [--outdir <paper>/Pictures]
"""
import argparse
import json
from pathlib import Path

import yaml

import plot_transferability as pt

REPO = Path(__file__).resolve().parents[1]
TASKS = yaml.safe_load(open(REPO / "bench/registry/tasks.yaml"))["tasks"]

SOURCES = [("prism", "PRISM", "activity"), ("armpose", "Arm", "pose"), ("fullbody", "Full-body", "pose"),
           ("gesture", "Gesture", "OpenWatch"), ("odom", "Odometry", "Nymeria"), ("pd", "Parkinson's", "PADS"),
           ("language", "IMU–text", "HumanML3D")]
# (source key of the same task, bench task id, row name, metric label, scale, fmt)
TARGETS = [("prism", "har_prism", "PRISM activity", "Macro-F1 (%)", 100, "{:.1f}"),
           ("armpose", "arm_pose_nymeria", "Nymeria arm pose", "Pos. err (cm)", 1, "{:.2f}"),
           ("fullbody", "full_body_pose_nymeria", "Nymeria full-body pose", "MPJPE (mm)", 1, "{:.2f}"),
           ("gesture", "microgesture_openwatch", "OpenWatch gesture", "Clip macro-F1 (%)", 100, "{:.1f}"),
           ("odom", "odometry_nymeria_local", "Nymeria odometry", "ATE (m)", 1, "{:.4f}"),
           ("pd", "pd_pads", "PADS Parkinson's", "Subject macro-F1 (%)", 100, "{:.1f}"),
           ("language", "language_humanml3d", "HumanML3D IMU–text", "R@1, G=100 (%)", 1, "{:.1f}")]


def value(task, enc, proto, scale):
    p = REPO / "bench/results" / task / f"{enc}__{proto}__s42.json"
    if not p.exists():
        return pt.PENDING
    v = json.load(open(p))["metrics"].get(TASKS[task]["metrics"]["primary"])
    return pt.PENDING if v is None else v * scale


# reference = the same protocol WITHOUT any source: (encoder, protocol, full label, sub-label, short header, colour bar)
REF_SCRATCH = ("scratch", "scratch", "From scratch", "(baseline)", "Scratch", "Gain over from-scratch (%)")
# a never-trained encoder, frozen, under the probe decoder: the zero point for frozen FROM-SCRATCH sources
REF_RANDOM = ("control_random", "probe", "Random init", "(frozen probe)", "Random", "Gain over the random-init probe (%)")


# Masked reconstruction (the pretraining task) as a TARGET: 20,000 steps on the reference corpus with a fresh decoder,
# encoder frozen (decoder only) or trainable. Run names: txmae_<src> for MAE-fine-tuned sources, txsmae_<src> for
# from-scratch sources; the no-source references are txsmae_random__frozen (never-trained encoder, frozen) and
# txmae_scratch__scratch (random init, trained end to end). The metric is the best validation loss of the run.
MAE_STEPS = 20000
MAE_PREFIX = {"tx5_": "txmae_", "txs_": "txsmae_"}


def recon_loss(run):
    """Best validation reconstruction loss of a FINISHED masked-reconstruction run, else PENDING."""
    import re
    logs = sorted((REPO / "runs" / run / "wandb").glob("run-*/files/output.log"), key=lambda q: q.stat().st_mtime)
    if not logs:
        return pt.PENDING
    val = [(int(m.group(1)), float(m.group(2)))
           for m in re.finditer(r"step=(\d+) split=val loss=([0-9.]+)", logs[-1].read_text(errors="ignore"))]
    if not val or max(st for st, _ in val) < MAE_STEPS:
        return pt.PENDING
    return min(v for _, v in val)


def mae_row(proto, prefix, ref):
    mode = "frozen" if proto == "probe" else "trainable"
    vals = {"mae": None}          # MAE -> masked reconstruction is the diagonal (source task = target task): left empty
    for key, _, _ in SOURCES:
        vals[key] = recon_loss(f"{MAE_PREFIX[prefix]}{key}__{mode}")
    ref_run = "txsmae_random__frozen" if ref is REF_RANDOM else "txmae_scratch__scratch"
    return dict(name="Masked reconstruction", metric="Val. loss", arrow="↓", higher_better=False, fmt="{:.3f}",
                ref=recon_loss(ref_run), values=vals)


def build(proto, mae_label, mae_sub, out, prefix="tx5_", ref=REF_SCRATCH, with_mae_row=False):
    """prefix tx5_: sources are the reference MAE fine-tuned on each task; txs_: sources are the
    benchmark's from-scratch run on each task (no pretraining) -- same targets."""
    targets = []
    for own, task, name, metric, scale, fmt in TARGETS:
        m = TASKS[task]["metrics"]
        hib = m[m["primary"]].get("higher_is_better", True)
        vals = {"mae": value(task, "mae_5src", proto, scale)}
        for key, _, _ in SOURCES:
            vals[key] = None if key == own else value(task, f"{prefix}{key}", proto, scale)
        targets.append(dict(name=name, metric=metric, arrow="↑" if hib else "↓", higher_better=hib, fmt=fmt,
                            ref=value(task, ref[0], ref[1], scale), values=vals))
    if with_mae_row:
        # FIRST, not last: the column order is Random | MAE | <the six task sources>, so the MAE
        # source sits in column 2. Putting its own target (masked reconstruction) in row 1 lines
        # every source-is-target dash up on one diagonal; appending it instead left that dash two
        # columns out of line while the other six ran diagonally.
        targets.insert(0, mae_row(proto, prefix, ref))
    return dict(out=out, ref_label=ref[2], ref_sub=ref[3], ref_short=ref[4], cbar=ref[5],
                sources=[("mae", mae_label, mae_sub)] + SOURCES, targets=targets)



# ---- compact renderer for the side-by-side layout ----------------------------------------------------------
# The full renderer draws a 15-inch-wide panel; two of those side by side in a 5.5-inch text block would print
# their numbers at about 2 pt. This one is laid out at twice the PRINTED size of a half-width panel (5.4 x 3.5 in
# for 2.7 x 1.75 in on the page), so a 10-pt value prints at 5 pt. Same colour ramp and ink as the full
# renderer; short labels, integer gains and no colour bar (every cell prints its gain).
SHORT_SRC = {"PRISM": "HAR", "Full-body": "Body", "Gesture": "Gest.", "Odometry": "Odom.", "Parkinson's": "PD",
             "IMU-text": "Text", "IMU–text": "Text", "MAE probe": "MAE", "MAE finetune": "MAE"}
SHORT_TGT = {"PRISM activity": ("HAR", "F1 %"), "Nymeria full-body pose": ("Body", "mm"),
             "Nymeria arm pose": ("Arm", "cm"),
             "OpenWatch gesture": ("Gesture", "F1 %"), "Nymeria odometry": ("Odom.", "ATE m"),
             "PADS Parkinson's": ("PD", "F1 %"), "HumanML3D IMU–text": ("Text", "R@1 %"), "HumanML3D IMU-text": ("Text", "R@1 %"),
             "Masked reconstruction": ("MAE", "val loss")}


# Diverging encoding of the gain over training from scratch: green = benefit, orange-pink = harm, neutral grey at
# zero (dataviz method: two hues + a neutral midpoint, equal steps per arm). Validated with the dataviz palette
# validator: the poles #1a8f5f / #f2917f clear the CVD target (protan dE 9.1, normal 28.3), but LIGHT green and LIGHT
# pink tints are indistinguishable under protanopia (dE ~1), so colour is never the only channel -- every cell
# prints its SIGNED gain, which is what carries the sign for a reader who cannot separate the two hues.
# The diagonal hatch on negative cells was a second such channel; it was dropped on request (2026-09-22) because
# it made the matrix busy. The printed sign still means colour is not load-bearing on its own, but this figure now
# has one redundant channel instead of two -- do not also drop the per-cell gain.
# The dark end stops at #2f9e6b rather than #0f6b45 so that ONE ink colour is readable on every
# fill: against #0b0b0b the darkest stops of both ramps are 5.8:1 and 7.1:1, clear of WCAG AA (4.5).
# The old ramp reached 3.0:1, which forced white text on the darkest cells and black elsewhere --
# the per-cell switching that made the matrix look noisy.
GAIN_POS = ["#f0efec", "#bde5d1", "#8ed0af", "#5cba8f", "#2f9e6b"]
GAIN_NEG = ["#f0efec", "#fbd3ca", "#f7b3a5", "#f2917f", "#ec7a66"]
GAIN_SAT = 40.0           # |gain| in % at which the colour saturates (the number in the cell is never clipped)


def render_compact(spec, outdir, dpi, vmax, title, ref="scratch"):
    """ref="scratch": gains over the target trained from scratch (one grey reference column, MAE is a coloured
    column). ref="mae": gains over the UNADAPTED MAE under the same protocol -- the effect of the source task's
    supervision alone; scratch and MAE are both grey context columns and only the six sources are coloured."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    sources, targets = spec["sources"], spec["targets"]
    vs_mae = ref == "mae"
    if vs_mae:
        sources = [src for src in sources if src[0] != "mae"]
    nr, nc = len(targets), len(sources)
    LAB, REF, GAP, CW, CH, HEAD, PAD = 0.62, 0.62, 0.10, 0.58, 0.50, (0.46 if title else 0.24), 0.04
    NREF = 2 if vs_mae else 1
    fw = PAD + LAB + NREF * REF + GAP + nc * CW + PAD
    fh = PAD + HEAD + nr * CH + PAD
    fig = plt.figure(figsize=(fw, fh), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, fw); ax.set_ylim(fh, 0); ax.axis("off")
    from matplotlib.colors import LinearSegmentedColormap
    cm_pos = LinearSegmentedColormap.from_list("gain_pos", GAIN_POS)
    cm_neg = LinearSegmentedColormap.from_list("gain_neg", GAIN_NEG)
    x_ref, x0, y0 = PAD + LAB, PAD + LAB + NREF * REF + GAP, PAD + HEAD

    def cell(x, y, w, h, **kw):
        ax.add_patch(plt.Rectangle((x + 0.015, y + 0.015), w - 0.03, h - 0.03, **kw))

    def short(fmt, v):
        t = fmt.format(v)
        return t[1:] if t.startswith("0.") and len(t) > 5 else t         # 0.1667 -> .1667 keeps four digits in a narrow cell

    if title:
        ax.text(PAD, PAD + 0.02, title, ha="left", va="top", fontsize=9.5, fontweight="bold", color=pt.INK)
    ax.text(x_ref + REF / 2, y0 - 0.05, spec.get("ref_short", "Scratch"), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=pt.INK_2)
    if vs_mae:
        ax.text(x_ref + 1.5 * REF, y0 - 0.05, "MAE", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=pt.INK)
    for j, (_, lab, _) in enumerate(sources):
        ax.text(x0 + (j + 0.5) * CW, y0 - 0.05, SHORT_SRC.get(lab, lab), ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=pt.INK)
    for i, t in enumerate(targets):
        y = y0 + i * CH
        name, unit = SHORT_TGT.get(t["name"], (t["name"], t["metric"]))
        ax.text(x_ref - 0.06, y + CH * 0.30, name, ha="right", va="center", fontsize=8.5, fontweight="bold", color=pt.INK)
        ax.text(x_ref - 0.06, y + CH * 0.70, f"{unit} {t['arrow']}", ha="right", va="center", fontsize=6.5, color=pt.MUTED)
        cell(x_ref, y, REF, CH, facecolor="#f4f3f0", edgecolor="#e1e0d9", linewidth=0.6)
        ref_pending = t["ref"] is pt.PENDING
        ax.text(x_ref + REF / 2, y + CH / 2, "…" if ref_pending else short(t["fmt"], t["ref"]), ha="center", va="center",
                fontsize=9, color=pt.MUTED if ref_pending else pt.INK_2)
        base = t["values"]["mae"] if vs_mae else t["ref"]
        if vs_mae:     # the reference the colours are measured against: darker frame, bold value
            cell(x_ref + REF, y, REF, CH, facecolor="#f4f3f0", edgecolor=pt.INK_2, linewidth=0.9)
            ax.text(x_ref + 1.5 * REF, y + CH / 2, short(t["fmt"], base), ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=pt.INK)
        for j, (key, _, _) in enumerate(sources):
            x, v = x0 + j * CW, t["values"][key]
            if v is None or v is pt.PENDING:
                cell(x, y, CW, CH, facecolor="#f4f3f0", edgecolor="#e1e0d9", linewidth=0.6)
                ax.text(x + CW / 2, y + CH / 2, "—" if v is None else "…", ha="center", va="center", fontsize=9, color=pt.MUTED)
                continue
            if base is pt.PENDING:        # no reference yet: the value alone, on a neutral cell
                cell(x, y, CW, CH, facecolor="#f4f3f0", edgecolor="#e1e0d9", linewidth=0.6)
                ax.text(x + CW / 2, y + CH / 2, short(t["fmt"], v), ha="center", va="center", fontsize=9.5,
                        fontweight="bold", color=pt.INK)
                continue
            g = pt.transfer_gain(v, base, t["higher_better"])
            mag = min(abs(g), GAIN_SAT) / GAIN_SAT
            fill = (cm_neg if g < 0 else cm_pos)(mag)
            cell(x, y, CW, CH, facecolor=fill, edgecolor="none", linewidth=0)
            # One ink for every cell, value and gain alike: the ramps above are capped so black clears
            # WCAG AA on the darkest fill of either sign, so nothing has to switch per cell.
            ax.text(x + CW / 2, y + CH * 0.36, short(t["fmt"], v), ha="center", va="center", fontsize=9.5,
                    fontweight="bold", color=pt.INK)
            ax.text(x + CW / 2, y + CH * 0.74, f"{g:+.0f}%", ha="center", va="center", fontsize=7, color=pt.INK)
    out = Path(outdir) / spec["out"].replace(".png", "_vsmae_compact.png" if vs_mae else "_compact.png")
    fig.savefig(out, dpi=dpi, facecolor="white"); fig.savefig(str(out).replace(".png", ".pdf"), facecolor="white"); plt.close(fig)
    print(f"wrote {out}  ({fw:.2f} x {fh:.2f} in; prints at half size)")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="/gpfs/home/vbc1591/paper_iclr/Pictures")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    pt.TABLES = {"scratch": build("probe", "MAE probe", "self-supervised", "transfer5_frozen.png"),
                 "finetune": build("finetune", "MAE finetune", "direct", "transfer5_trainable.png"),
                 # sources trained from scratch per task: does supervision transfer without pretraining?
                 # frozen from-scratch sources are measured against the never-trained frozen encoder, the same
                 # protocol without a source task (trainable: that reference IS the from-scratch run)
                 # with_mae_row is off (user decision 2026-09-23): the masked-reconstruction row is a differently
                 # trained source and reads as one of the six task sources, which it is not.
                 "scratch_src_probe": build("probe", "MAE probe", "self-supervised",
                                            "transfer_scratchsrc_frozen.png", prefix="txs_", ref=REF_RANDOM),
                 "scratch_src_finetune": build("finetune", "MAE finetune", "direct",
                                               "transfer_scratchsrc_trainable.png", prefix="txs_")}
    export_csv()
    def complete(spec):
        return not any(t["ref"] is pt.PENDING for t in spec["targets"])
    # one colour scale for every full-size figure, set by the scratch-referenced tables only: gains over the
    # random-init probe reach +190 % and would wash the others out (they saturate instead)
    tables = pt.TABLES
    pt.TABLES = {k: v for k, v in tables.items() if v["ref_short"] == "Scratch" and complete(v)}
    vmax = pt.nice_ceiling(pt.global_gain_max())
    pt.TABLES = tables
    for spec in pt.TABLES.values():
        if complete(spec):
            pt.render(spec, "absolute", a.outdir, a.dpi, vmax)
        else:
            print(f"{spec['out']}: a reference cell is pending -> full-size figure skipped, compact panel drawn")
        render_compact(spec, a.outdir, a.dpi, vmax, "")          # panel titles are the LaTeX subcaptions
        if spec["out"].startswith("transfer5_") and not any(t["values"]["mae"] is pt.PENDING for t in spec["targets"]):
            render_compact(spec, a.outdir, a.dpi, vmax, "", ref="mae")


def export_csv():
    """Numbers behind Figure 2 -> paper_results/fig2_transfer (raw metric units, no % scaling)."""
    import sys
    sys.path.insert(0, str(REPO / "paper_results"))
    from paper_io import write_csv
    rows = []
    for proto in ("probe", "finetune"):
        for own, task, name, metric, scale, fmt in TARGETS:
            m = TASKS[task]["metrics"]
            ref = value(task, "scratch", "scratch", 1)
            for key, lab in [("mae", "MAE (no source task)")] + [(k, l) for k, l, _ in SOURCES]:
                if key == own:
                    continue
                enc = "mae_5src" if key == "mae" else f"tx5_{key}"
                v = value(task, enc, proto, 1)
                rows.append([proto, task, name, key, lab, enc, m["primary"],
                             None if v is pt.PENDING else v, None if ref is pt.PENDING else ref,
                             int(bool(m[m["primary"]].get("higher_is_better", True))),
                             "pending" if v is pt.PENDING else "ok"])
    write_csv("fig2_transfer", "transfer_long.csv",
              ["protocol", "target_task", "target_label", "source_key", "source_label", "encoder", "metric",
               "value", "scratch_reference", "higher_is_better", "status"], rows)


if __name__ == "__main__":
    main()
