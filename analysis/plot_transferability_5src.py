#!/usr/bin/env python3
"""Cross-task transfer matrices for the 5-source MAE (paper Section 4), read from bench results.

Same figure design as plot_transferability.py (whose drawing code is reused), but every cell is read
from bench/results instead of being typed in:

  source "MAE"       mae_5src            (probe / finetune)
  source <task>      tx5_<task>          mae_5src fine-tuned on <task>, then frozen (probe) or used as
                                         the initialisation (finetune) for the target
  reference column   scratch             the target trained from random initialisation

Targets / sources: PRISM activity recognition (the leak-free HAR set used as the HAR source), Nymeria
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

SOURCES = [("prism", "PRISM", "activity"), ("fullbody", "Full-body", "pose"), ("gesture", "Gesture", "OpenWatch"),
           ("odom", "Odometry", "Nymeria"), ("pd", "Parkinson's", "PADS"), ("language", "IMU–text", "HumanML3D")]
# (source key of the same task, bench task id, row name, metric label, scale, fmt)
TARGETS = [("prism", "har_prism", "PRISM activity", "Macro-F1 (%)", 100, "{:.1f}"),
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


def build(proto, mae_label, mae_sub, out, prefix="tx5_"):
    """prefix tx5_: sources are the reference MAE fine-tuned on each task; txs_: sources are the
    benchmark's from-scratch run on each task (no pretraining) -- same targets, same reference."""
    targets = []
    for own, task, name, metric, scale, fmt in TARGETS:
        m = TASKS[task]["metrics"]
        hib = m[m["primary"]].get("higher_is_better", True)
        vals = {"mae": value(task, "mae_5src", proto, scale)}
        for key, _, _ in SOURCES:
            vals[key] = None if key == own else value(task, f"{prefix}{key}", proto, scale)
        targets.append(dict(name=name, metric=metric, arrow="↑" if hib else "↓", higher_better=hib, fmt=fmt,
                            ref=value(task, "scratch", "scratch", scale), values=vals))
    return dict(out=out, ref_label="From scratch", ref_sub="(baseline)", cbar="Gain over from-scratch (%)",
                sources=[("mae", mae_label, mae_sub)] + SOURCES, targets=targets)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="/gpfs/home/vbc1591/paper_iclr/Pictures")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    pt.TABLES = {"scratch": build("probe", "MAE probe", "self-supervised", "transfer5_frozen.png"),
                 "finetune": build("finetune", "MAE finetune", "direct", "transfer5_trainable.png"),
                 # sources trained from scratch per task: does supervision transfer without pretraining?
                 "scratch_src_probe": build("probe", "MAE probe", "self-supervised",
                                            "transfer_scratchsrc_frozen.png", prefix="txs_"),
                 "scratch_src_finetune": build("finetune", "MAE finetune", "direct",
                                               "transfer_scratchsrc_trainable.png", prefix="txs_")}
    export_csv()
    if any(t["ref"] is pt.PENDING for spec in pt.TABLES.values() for t in spec["targets"]):
        print("a from-scratch reference cell is pending -> figure not re-rendered (CSV exported)")
        return
    vmax = pt.nice_ceiling(pt.global_gain_max())
    for spec in pt.TABLES.values():
        pt.render(spec, "absolute", a.outdir, a.dpi, vmax)


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
