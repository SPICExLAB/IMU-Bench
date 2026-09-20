#!/usr/bin/env python
"""Stage 6: collect the factorial's numbers into one table.

Pretraining: the val row at the equal 50,000-step budget from
runs/<arm>/logs/metrics.txt. Cells whose last val row is below budget are
reported as incomplete rather than harvested, and rows past the budget are
ignored -- metrics.txt is append-only across resumes (11j trap 2), so
min-over-file would hand a resumed cell an out-of-budget number.

Downstream: the four bench result JSONs per cell. Nothing is re-derived from
logs; if bench has not scored a cell, it is missing here.

Writes out/results.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import OUT_DIR, REPO_ROOT, STEPS, manifest_arms

PANEL = {
    "arm_pose_nymeria": ("pos_err_cm", "min"),
    "odometry_nymeria_worldframe": ("ate_m", "min"),
    "har_wisdm": ("macro_f1", "max"),
    "microgesture_openwatch": ("clip_macro_f1", "max"),
}


def val_rows(run):
    path = os.path.join(REPO_ROOT, "runs", run, "logs", "metrics.txt")
    if not os.path.exists(path):
        return None
    rows = []
    for line in open(path, errors="ignore"):
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
    with open(os.path.join(OUT_DIR, "arms.json")) as f:
        manifest = json.load(f)
    out = {"steps": STEPS, "panel": list(PANEL), "cells": {}, "incomplete": [], "missing": []}

    for arm, div, vol_tag, n in manifest_arms(manifest):
        info = manifest["arms"].get(arm, {})
        cell = {"diversity": div, "volume_tag": vol_tag, "n_windows": n,
                "hours": n * 5.12 / 3600.0,
                "n_sources": info.get("n_sources"),
                "types_phys": (info.get("phys") or {}).get("types_present"),
                "eff_types_phys": (info.get("phys") or {}).get("effective_types_shannon"),
                "eff_types_enc": (info.get("enc") or {}).get("effective_types_shannon"),
                "pool_mass_covered_phys": (info.get("phys") or {}).get("pool_mass_covered"),
                "effective_sources": info.get("effective_sources"),
                "effective_subjects": info.get("effective_subjects"),
                "effective_sessions": info.get("effective_sessions"),
                "n_subjects_known": info.get("n_subjects_known"),
                "vendi_enc": info.get("vendi_enc"),
                "participation_ratio_enc": info.get("participation_ratio_enc"),
                "fill_median": (info.get("pool_coverage") or {}).get("fill_median"),
                "probes": {}}

        rows = val_rows(arm)
        if rows is None:
            out["missing"].append(arm)
        else:
            inb = [r for r in rows if r.get("step", 0) <= STEPS]
            last = inb[-1] if inb else None
            cell["pretrain_step"] = last.get("step") if last else None
            cell["pretrain_val_loss"] = last.get("loss") if last else None
            if not last or last.get("step", 0) < STEPS:
                out["incomplete"].append(f"{arm} @ step {cell['pretrain_step']}")

        for task, (metric, _) in PANEL.items():
            p = os.path.join(REPO_ROOT, "bench", "results", task, f"{arm}__probe__s42.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                r = json.load(f)
            if r.get("status") != "ok":
                continue
            cell["probes"][task] = {"metric": metric, "value": r["metrics"].get(metric),
                                    "primary": r.get("primary_metric"),
                                    "all": r.get("metrics")}
        out["cells"][arm] = cell

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'cell':16s} {'N':>9s} {'src':>4s} {'effTyp':>8s} {'val_loss':>9s}  " +
          "  ".join(f"{t[:12]:>12s}" for t in PANEL))
    for arm, c in out["cells"].items():
        pr = "  ".join(f"{(c['probes'].get(t) or {}).get('value', float('nan')):>12.4f}"
                       for t in PANEL)
        print(f"{arm:16s} {c['n_windows']:>9,} {str(c['n_sources']):>4s} "
              f"{(c['eff_types_phys'] or float('nan')):>8.1f} "
              f"{(c.get('pretrain_val_loss') or float('nan')):>9.4f}  {pr}")
    if out["incomplete"]:
        print("\nincomplete (below the equal budget, NOT comparable):")
        for x in out["incomplete"]:
            print("  " + x)
    if out["missing"]:
        print("\nnot started: " + ", ".join(out["missing"]))


if __name__ == "__main__":
    main()
