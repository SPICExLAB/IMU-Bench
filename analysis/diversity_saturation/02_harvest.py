#!/usr/bin/env python
"""02_harvest.py -- collect the diversity-saturation causal results (EXPERIMENTS.md 11j).

Guards against the two budget-cut traps documented in 11j:
  1. `best.pt` must be at step <= MAX_STEPS. The arms were cancelled at
     different steps (random ~100K, redundant 104K, diverse 109K) before
     being resumed to a common 105K, so a stale over-budget best.pt would
     silently give one arm extra training.
  2. `metrics.txt` is append-only across runs, so a killed run's rows sit in
     the file ahead of the resumed rows. Val numbers are therefore taken
     from the LAST row (end of the resumed run), never min-over-file.

Probe metrics use the same rules validated on the recipe sweep:
classification = metric at the best-val-loss step; regression = min val loss
(probe runs are single-shot, no resume, so min-over-file is safe there).

Usage:  TransfHAR_env/bin/python analysis/diversity_saturation/02_harvest.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")
VARIANTS = ["redundant", "random", "diverse"]   # predicted worst -> best
MAX_STEPS = 105000

PROBES = [
    # label, run template, metric key, selection ("at_best_loss" | "min"), scale
    ("Arm pose (val loss)", "vit1d_arm_pose_probe_nymeria_mae_div1pct_{v}", "loss", "min", 1),
    ("WISDM (acc %)",       "vit1d_har_probe_wisdm_mae_div1pct_{v}",        "acc",  "at_best_loss", 100),
    ("PAMAP2 (acc %)",      "vit1d_har_probe_pamap2_mae_div1pct_{v}",       "acc",  "at_best_loss", 100),
]

# recipe-sweep reference rows (EXPERIMENTS.md 11i), for the "diverse-1% ~= 100%?" test.
# NOTE trained at 300K steps -- absolute values are NOT comparable to the 105K
# arms; included only so the report shows them side by side, explicitly flagged.
REFERENCE_300K = {
    "Arm pose (val loss)": {"session-1.3%": 0.016426, "random-100%": 0.016489},
    "WISDM (acc %)":       {"session-1.3%": 65.38,    "random-100%": 64.07},
    "PAMAP2 (acc %)":      {"session-1.3%": 56.53,    "random-100%": 56.51},
}


def val_rows(run):
    p = os.path.join(REPO_ROOT, "runs", run, "logs", "metrics.txt")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as f:
        for line in f:
            if "split=val" not in line:
                continue
            d = {}
            for kv in line.split():
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
            rows.append(d)
    return rows or None


def check_checkpoint(run):
    """Trap 1: best.pt must be within budget."""
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
    report = {"max_steps": MAX_STEPS, "pretrain": {}, "probes": {}, "warnings": []}

    print("=" * 74)
    print("PRETRAIN (MAE) -- val loss at END of the equal 105K budget")
    print("=" * 74)
    for v in VARIANTS:
        run = f"vit1d_ssl_mae_div1pct_{v}"
        rows = val_rows(run)
        ck = check_checkpoint(run)
        if rows is None:
            print(f"  {v:10s} -- no metrics yet")
            report["pretrain"][v] = {"status": "pending"}
            continue
        # Trap 2: LAST row, not min-over-file.
        last = rows[-1]
        entry = {"final_step": last.get("step"), "final_val_loss": last.get("loss"),
                 "n_val_rows": len(rows), "checkpoint": ck}
        report["pretrain"][v] = entry
        flag = ""
        if ck.get("exists") and not ck.get("within_budget"):
            flag = f"  *** OVER BUDGET: best.pt at step {ck['step']} > {MAX_STEPS} ***"
            report["warnings"].append(f"{run}: best.pt step {ck['step']} exceeds budget")
        elif not ck.get("exists"):
            flag = "  (best.pt missing)"
        print(f"  {v:10s} step={last.get('step'):>7} val_loss={last.get('loss'):.6f} "
              f"best.pt@{ck.get('step')}{flag}")

    print()
    print("=" * 74)
    print("DOWNSTREAM PROBES")
    print("=" * 74)
    for label, tmpl, key, how, scale in PROBES:
        vals = {}
        for v in VARIANTS:
            rows = val_rows(tmpl.format(v=v))
            if rows is None:
                vals[v] = None
                continue
            if how == "min":
                cand = [r[key] for r in rows if key in r]
                vals[v] = min(cand) * scale if cand else None
            else:
                cand = [r for r in rows if "loss" in r and key in r]
                vals[v] = min(cand, key=lambda r: r["loss"])[key] * scale if cand else None
        report["probes"][label] = vals
        # decimals from whatever numbers we have, including the reference row --
        # otherwise a not-yet-landed probe renders its 0.0164 reference as "0.02"
        scale_ref = [abs(x) for x in vals.values() if x is not None]
        scale_ref += [abs(x) for x in REFERENCE_300K.get(label, {}).values()]
        dec = 6 if (min(scale_ref) if scale_ref else 1) < 1 else 2
        cells = "  ".join(f"{v}={vals[v]:.{dec}f}" if vals[v] is not None else f"{v}=--"
                          for v in VARIANTS)
        print(f"  {label:22s} {cells}")
        ref = REFERENCE_300K.get(label)
        if ref:
            print(f"  {'':22s} [300K ref, NOT budget-matched] "
                  + "  ".join(f"{k}={x:.{dec}f}" for k, x in ref.items()))

    # odom probe: parse the eval script's summary line from the SLURM log
    print()
    print("  Odometry probe (vel MSE):")
    odom = {}
    for v in VARIANTS:
        val = None
        for h in sorted(glob.glob(os.path.join(REPO_ROOT, "logs",
                                               f"odom_probe_nymeria_mae_div1pct_{v}_*.out"))):
            for line in open(h, errors="ignore"):
                m = re.search(r"vel_mse=([\d.]+)", line)
                if m:
                    val = float(m.group(1))
        odom[v] = val
        print(f"    {v:10s} {val if val is not None else '--'}")
    report["probes"]["Odometry (vel MSE)"] = odom

    # verdict
    print()
    print("=" * 74)
    pre = report["pretrain"]
    if all(pre.get(v, {}).get("final_val_loss") is not None for v in VARIANTS):
        r, d, red = (pre["random"]["final_val_loss"], pre["diverse"]["final_val_loss"],
                     pre["redundant"]["final_val_loss"])
        ok = red > r >= d or red > d >= r
        print(f"Pretrain ordering redundant > random ~ diverse : "
              f"{'HOLDS' if ok else 'DOES NOT HOLD'} "
              f"(redundant {red:.4f}, random {r:.4f}, diverse {d:.4f})")
    if report["warnings"]:
        print("\nWARNINGS:")
        for w in report["warnings"]:
            print(f"  ! {w}")

    with open(os.path.join(OUT_DIR, "causal_results.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'causal_results.json')}")
    return 1 if report["warnings"] else 0


if __name__ == "__main__":
    sys.exit(main())
