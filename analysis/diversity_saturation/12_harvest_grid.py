"""Harvest the motion-types x volume grid (EXPERIMENTS.md 11l).

Same validated rules as 02_harvest.py / 07_harvest_subpct.py:
  classification -> metric at the best-val-loss step
  regression     -> min val loss over training
  odometry       -> last vel_mse= line in the probe's SLURM log

Writes analysis/diversity_saturation/out/grid_results.json in the shape
11_plot_grid.py expects: {"probes": {label: {cell: value}}}.
"""
from __future__ import annotations

import glob
import json
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")
MAX_STEPS = 50000

PROBES = [
    ("WISDM (acc %)",       "vit1d_har_probe_wisdm_mae_{c}",        "acc",  "at_best_loss", 100),
    ("UTD-MHAD (acc %)",    "vit1d_har_probe_utd_mhad_mae_{c}",     "acc",  "at_best_loss", 100),
    ("PAMAP2 (acc %)",      "vit1d_har_probe_pamap2_mae_{c}",       "acc",  "at_best_loss", 100),
    ("RecoFit (acc %)",     "vit1d_har_probe_recofit_mae_{c}",      "acc",  "at_best_loss", 100),
    ("Gesture (bal acc %)", "vit1d_gesture_probe_openwatch_6class_mae_{c}", "bal_acc",
     "at_best_loss", 100),
    ("Arm pose (val loss)", "vit1d_arm_pose_probe_nymeria_mae_{c}", "loss", "min", 1),
    ("Full-body (mpjpe mm)", "vit1d_full_body_pose_probe_nymeria_mae_{c}", "mpjpe_mm", "min", 1),
]


def val_rows(run):
    p = os.path.join(REPO_ROOT, "runs", run, "logs", "metrics.txt")
    if not os.path.exists(p):
        return None
    rows = []
    for line in open(p):
        if "split=val" not in line:
            continue
        d = {}
        for kv in line.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
        rows.append(d)
    return rows or None


def main():
    sel = json.load(open(os.path.join(OUT_DIR, "grid_selection.json")))
    cells = sel["cells"]
    report = {"max_steps": MAX_STEPS, "cells": {}, "probes": {}, "warnings": []}

    print("=" * 78)
    print("PRETRAIN (MAE) -- equal 50K budget")
    print("=" * 78)
    for name, meta in sorted(cells.items(), key=lambda kv: (kv[1]["K"], kv[1]["fraction"])):
        run = f"vit1d_ssl_mae_grid_{name}"
        rows = val_rows(run)
        if rows is None:
            report["warnings"].append(f"{run}: no metrics.txt yet")
            continue
        last = rows[-1]
        report["cells"][name] = {"K": meta["K"], "fraction": meta["fraction"],
                                 "n_windows": meta["n_windows"],
                                 "final_step": last.get("step"),
                                 "final_val_loss": last.get("loss")}
        print(f"  {name:14s} K={meta['K']:<5d} n={meta['n_windows']:>9,}  "
              f"step={last.get('step')}  val_loss={last.get('loss')}")

    print()
    print("=" * 78)
    print("DOWNSTREAM PROBES")
    print("=" * 78)
    for label, tmpl, key, how, scale in PROBES:
        vals = {}
        for name in cells:
            rows = val_rows(tmpl.format(c=name))
            if not rows:
                continue
            cand = [r for r in rows if key in r and "loss" in r]
            if not cand:
                continue
            vals[name] = (min(r[key] for r in cand) * scale if how == "min"
                          else min(cand, key=lambda r: r["loss"])[key] * scale)
        report["probes"][label] = vals
        print(f"  {label:22s} {len(vals):>2d}/{len(cells)} cells")

    # See 07_harvest_subpct.py: the odom eval falls back to latest.pt when
    # best.pt is missing and still reports a number, so a log that did not load
    # best.pt must be refused rather than harvested.
    odom = {}
    for name in cells:
        val = None
        for h in sorted(glob.glob(os.path.join(
                REPO_ROOT, "logs", f"odom_probe_nymeria_mae_{name}_*.out"))):
            text = open(h, errors="ignore").read()
            if "checkpoints/best.pt" not in text:
                report["warnings"].append(
                    f"{os.path.basename(h)}: loaded a non-best.pt checkpoint, ignored")
                continue
            for line in text.splitlines():
                m = re.search(r"vel_mse=([\d.]+)", line)
                if m:
                    val = float(m.group(1))
        if val is not None:
            odom[name] = val
    report["probes"]["Odometry (vel MSE)"] = odom
    print(f"  {'Odometry (vel MSE)':22s} {len(odom):>2d}/{len(cells)} cells")

    if report["warnings"]:
        print(f"\n{len(report['warnings'])} pretrains not finished yet")

    p = os.path.join(OUT_DIR, "grid_results.json")
    with open(p, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
