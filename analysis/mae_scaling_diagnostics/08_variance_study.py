#!/usr/bin/env python
"""08_variance_study.py -- TODO §8 (estimate downstream variance).

Checks whether the differences reported between 5%/25%/50%/100% probe results
in EXPERIMENTS.md are bigger than ordinary training noise, by generating (and,
by default, only PRINTING -- see --launch) the commands to rerun one HAR probe
and one pose probe x 3 seeds x {5%, 25%, 50%, 100%} pretrained checkpoints.

Relies on the `--seed` CLI flag added to scripts/run_pretrain.py (this repo
had no seed control for probe reruns before -- trainer.seed was fixed at 42
in every config). Probe jobs are short (existing configs show max_steps
10-20k with early stopping / patience 20), unlike full pretraining reruns, but
still real GPU jobs -- this script does not call sbatch/subprocess itself, it
only prints what to run, consistent with the "no SLURM submission from
Steps 1-5 without confirmation" plan (variance reruns are a Step 6 item
requiring separate go-ahead).

Once results exist (either via the printed commands or --launch), pass
--summarize to aggregate the per-seed val_loss/accuracy from each run's
logs/metrics.txt and report mean/std per fraction, plus flag whether the
5%->100% spread exceeds observed seed-to-seed noise.

Usage:
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/08_variance_study.py \
        --har-config configs/vit1d_har_probe_wisdm_mae_data{frac}_multidata_flu.yaml \
        --pose-config configs/vit1d_arm_pose_probe_nymeria_mae_data{frac}_multidata_flu.yaml \
        --fractions 5pct,25pct,50pct,100pct --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import OUT_DIR, REPO_ROOT  # noqa: E402

# Fraction-name -> {frac} token used by the existing per-fraction probe configs
# (configs/vit1d_har_probe_wisdm_mae_data{X}pct_multidata_flu.yaml,
#  configs/vit1d_arm_pose_probe_nymeria_mae_data{X}pct_multidata_flu.yaml). The
# un-suffixed 100% probe configs don't follow the "data{X}pct" naming --
# handled as a special case below.
FRAC_TOKENS = {"0.1pct": "0.1pct", "1pct": "1pct", "5pct": "5pct", "25pct": "25pct", "50pct": "50pct"}


def resolve_config(template: str, frac: str) -> str:
    if frac == "100pct":
        # 100% configs drop the "_data{X}pct" infix entirely, e.g.
        # vit1d_har_probe_wisdm_mae_multidata_flu.yaml (no data{X}pct token).
        return re.sub(r"_data\{frac\}", "", template)
    return template.replace("{frac}", FRAC_TOKENS.get(frac, frac))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--har-config", default="configs/vit1d_har_probe_wisdm_mae_data{frac}_multidata_flu.yaml")
    ap.add_argument("--pose-config", default="configs/vit1d_arm_pose_probe_nymeria_mae_data{frac}_multidata_flu.yaml")
    ap.add_argument("--fractions", default="5pct,25pct,50pct,100pct")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--launch", action="store_true",
                     help="actually submit the runs (real GPU jobs) instead of just printing commands")
    ap.add_argument("--summarize", action="store_true",
                     help="instead of generating commands, read back results from already-run "
                          "runs/<run_name>/logs/metrics.txt and report mean/std per fraction")
    args = ap.parse_args()
    fractions = args.fractions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    plan = []
    for task_name, template in [("har_wisdm", args.har_config), ("pose_nymeria", args.pose_config)]:
        for frac in fractions:
            cfg_path = resolve_config(template, frac)
            if not os.path.exists(os.path.join(REPO_ROOT, cfg_path)):
                print(f"[08] WARNING: {cfg_path} does not exist, skipping {task_name}/{frac}")
                continue
            for seed in seeds:
                run_name = f"08variance_{task_name}_{frac}_seed{seed}"
                plan.append({"task": task_name, "fraction": frac, "seed": seed,
                             "config": cfg_path, "run_name": run_name})

    out_dir = os.path.join(OUT_DIR, "08_variance_study")
    os.makedirs(out_dir, exist_ok=True)

    if args.summarize:
        summarize(plan, out_dir)
        return

    print(f"[08] {len(plan)} probe reruns planned "
          f"({len(set(p['task'] for p in plan))} tasks x {len(fractions)} fractions x {len(seeds)} seeds)")
    for p in plan:
        cmd = [
            sys.executable, "scripts/run_pretrain.py",
            "--config", "configs/base.yaml",
            "--model-config", p["config"],
            "--run-name", p["run_name"],
            "--seed", str(p["seed"]),
        ]
        print("  $", " ".join(cmd))
        if args.launch:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    with open(os.path.join(out_dir, "variance_study_plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\n[08] wrote {out_dir}/variance_study_plan.json")
    if not args.launch:
        print("[08] NOT launched (pass --launch to actually submit these -- real GPU jobs, "
              "confirm compute budget first: "
              f"{len(plan)} short probe runs, each ~max_steps 10-20k w/ early stopping)")


def summarize(plan, out_dir):
    import numpy as np

    rows = []
    for p in plan:
        metrics_path = os.path.join(REPO_ROOT, "runs", p["run_name"], "logs", "metrics.txt")
        if not os.path.exists(metrics_path):
            print(f"[08] no metrics found for {p['run_name']} (not run yet?) -- skipping")
            continue
        # metrics.txt format is whatever Trainer._run_val logs -- read the last
        # line's val_loss (and val_acc/mpjpe_mm if present) as this run's final score.
        with open(metrics_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue
        last = lines[-1]
        m = dict(re.findall(r"(\w+)=([\-0-9.eE]+)", last))
        row = {**p, **{k: float(v) for k, v in m.items()}}
        rows.append(row)

    if not rows:
        print("[08] no results to summarize yet")
        return

    print("\n[08] per-fraction mean +/- std across seeds:")
    metric_keys = sorted({k for r in rows for k in r if k not in
                           ("task", "fraction", "seed", "config", "run_name")})
    by_task_frac = {}
    for r in rows:
        by_task_frac.setdefault((r["task"], r["fraction"]), []).append(r)

    summary = []
    for (task, frac), rs in sorted(by_task_frac.items()):
        entry = {"task": task, "fraction": frac, "n_seeds": len(rs)}
        for key in metric_keys:
            vals = [r[key] for r in rs if key in r]
            if vals:
                entry[f"{key}_mean"] = float(np.mean(vals))
                entry[f"{key}_std"] = float(np.std(vals))
        summary.append(entry)
        print(f"  {task:16s} {frac:8s} n_seeds={len(rs)} " +
              "  ".join(f"{k}={entry[k]:.5g}" for k in entry if k.endswith("_mean")))

    # Flag whether the fraction-to-fraction spread exceeds seed noise, per task/metric
    print("\n[08] fraction spread vs. seed noise (is a fraction difference bigger than its own variance?):")
    for task in {s["task"] for s in summary}:
        task_rows = [s for s in summary if s["task"] == task]
        for key in metric_keys:
            mean_key, std_key = f"{key}_mean", f"{key}_std"
            pts = [(r["fraction"], r.get(mean_key), r.get(std_key)) for r in task_rows if mean_key in r]
            if len(pts) < 2:
                continue
            means = [p[1] for p in pts]
            spread = max(means) - min(means)
            avg_std = np.mean([p[2] for p in pts if p[2] is not None])
            verdict = "SPREAD > NOISE (likely real difference)" if spread > 2 * avg_std else \
                      "spread within ~2x seed noise -- NOT clearly distinguishable from noise"
            print(f"  {task}/{key}: fraction spread={spread:.5g}  avg seed std={avg_std:.5g}  -> {verdict}")

    with open(os.path.join(out_dir, "variance_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[08] wrote {out_dir}/variance_summary.json")


if __name__ == "__main__":
    main()
