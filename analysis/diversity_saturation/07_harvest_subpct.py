"""Harvest the sub-1% volume sweep (EXPERIMENTS.md 11k).

Five points at the SAME 105K-step budget:

    0.01%  0.05%  0.1%  0.5%   (this sweep, nested subsets)
    1.3%                        (11j's random-1% arm, reused as the anchor)

Harvest rules are the validated ones from 02_harvest.py, unchanged:
  classification -> metric at the best-val-loss step
  regression     -> min val loss over training
  odometry       -> last vel_mse= line in the probe's SLURM log
`metrics.txt` is append-only across resumes, so every read is over parsed val
rows rather than a file-wide min on the raw text.

Writes analysis/diversity_saturation/out/subpct_results.json.
"""
from __future__ import annotations

import glob
import json
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")
MAX_STEPS = 105000

# (arm key, pretrain run, probe-run infix) -- ascending data amount
ARMS = [
    ("0.01%", "vit1d_ssl_mae_sub0p01pct", "sub0p01pct"),
    ("0.05%", "vit1d_ssl_mae_sub0p05pct", "sub0p05pct"),
    ("0.1%",  "vit1d_ssl_mae_sub0p1pct",  "sub0p1pct"),
    ("0.5%",  "vit1d_ssl_mae_sub0p5pct",  "sub0p5pct"),
    ("1.3%",  "vit1d_ssl_mae_div1pct_random", "div1pct_random"),   # 11j anchor
]

# 11i's probe set. The first three exist for the 11j arms too, so the 1.3%
# anchor is populated for them; the rest are 11k-only below 1.3% and the anchor
# column will read "--" for those until a div1pct_random clone is run.
PROBES = [
    ("WISDM (acc %)",       "vit1d_har_probe_wisdm_mae_{i}",        "acc",  "at_best_loss", 100, True),
    ("UTD-MHAD (acc %)",    "vit1d_har_probe_utd_mhad_mae_{i}",     "acc",  "at_best_loss", 100, True),
    ("PAMAP2 (acc %)",      "vit1d_har_probe_pamap2_mae_{i}",       "acc",  "at_best_loss", 100, True),
    ("RecoFit (acc %)",     "vit1d_har_probe_recofit_mae_{i}",      "acc",  "at_best_loss", 100, True),
    ("Gesture (bal acc %)", "vit1d_gesture_probe_openwatch_6class_mae_{i}", "bal_acc",
     "at_best_loss", 100, True),
    ("Arm pose (val loss)", "vit1d_arm_pose_probe_nymeria_mae_{i}", "loss", "min", 1, False),
    ("Full-body (mpjpe mm)", "vit1d_full_body_pose_probe_nymeria_mae_{i}", "mpjpe_mm", "min", 1, False),
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


def check_checkpoint(run):
    import torch
    p = os.path.join(REPO_ROOT, "runs", run, "checkpoints", "best.pt")
    if not os.path.exists(p):
        return {"exists": False}
    s = torch.load(p, map_location="cpu", weights_only=False)
    step = int(s.get("step", -1))
    return {"exists": True, "step": step, "within_budget": step <= MAX_STEPS,
            "best_val_loss": s.get("best_val_loss")}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sel_path = os.path.join(OUT_DIR, "subpct_selection.json")
    sel = json.load(open(sel_path)) if os.path.exists(sel_path) else {"arms": {}}
    report = {"max_steps": MAX_STEPS, "selection": sel.get("arms", {}),
              "pretrain": {}, "probes": {}, "warnings": []}

    print("=" * 78)
    print("PRETRAIN (MAE) -- val loss at the end of the equal 105K budget")
    print("=" * 78)
    for label, run, _ in ARMS:
        rows = val_rows(run)
        ck = check_checkpoint(run)
        if rows is None:
            report["warnings"].append(f"{run}: no metrics.txt yet")
            print(f"  {label:6s} {run:32s} --")
            continue
        last = rows[-1]
        report["pretrain"][label] = {
            "run": run, "final_step": last.get("step"),
            "final_val_loss": last.get("loss"), "n_val_rows": len(rows),
            "checkpoint": ck}
        if ck.get("exists") and not ck.get("within_budget"):
            report["warnings"].append(
                f"{run}: best.pt step {ck['step']} exceeds the {MAX_STEPS} budget")
        print(f"  {label:6s} {run:32s} step={last.get('step')}  "
              f"val_loss={last.get('loss')}  best.pt@{ck.get('step')}")

    print()
    print("=" * 78)
    print("DOWNSTREAM PROBES (frozen encoder)")
    print("=" * 78)
    for label, tmpl, key, how, scale, higher_better in PROBES:
        vals = {}
        for arm, _, infix in ARMS:
            rows = val_rows(tmpl.format(i=infix))
            if not rows:
                vals[arm] = None
                continue
            cand = [r for r in rows if key in r and "loss" in r]
            if not cand:
                vals[arm] = None
            elif how == "min":
                vals[arm] = min(r[key] for r in cand) * scale
            else:
                vals[arm] = min(cand, key=lambda r: r["loss"])[key] * scale
        report["probes"][label] = {"higher_better": higher_better, "values": vals}
        dec = 6 if scale == 1 else 2
        print(f"  {label:22s} " + "  ".join(
            f"{a}={vals[a]:.{dec}f}" if vals[a] is not None else f"{a}=--" for a, _, _ in ARMS))

    # The odom eval script silently falls back to latest.pt when best.pt is
    # absent (scripts/eval_odom_nymeria_altattn.py:94-96), so unlike the
    # config-driven probes it does NOT fail loudly on a bad checkpoint -- it
    # reports a number from the wrong weights. Refuse any log that did not load
    # best.pt rather than harvesting it.
    odom = {}
    for arm, _, infix in ARMS:
        val = None
        for h in sorted(glob.glob(os.path.join(
                REPO_ROOT, "logs", f"odom_probe_nymeria_mae_{infix}_*.out"))):
            text = open(h, errors="ignore").read()
            if "checkpoints/best.pt" not in text:
                report["warnings"].append(
                    f"{os.path.basename(h)}: loaded a non-best.pt checkpoint, ignored")
                continue
            for line in text.splitlines():
                m = re.search(r"vel_mse=([\d.]+)", line)
                if m:
                    val = float(m.group(1))
        odom[arm] = val
    report["probes"]["Odometry (vel MSE)"] = {"higher_better": False, "values": odom}
    print(f"  {'Odometry (vel MSE)':22s} " + "  ".join(
        f"{a}={odom[a]:.5f}" if odom[a] is not None else f"{a}=--" for a, _, _ in ARMS))

    if report["warnings"]:
        print("\nWARNINGS:")
        for w in report["warnings"]:
            print(f"  ! {w}")

    out = os.path.join(OUT_DIR, "subpct_results.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out}")
    return 1 if report["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
