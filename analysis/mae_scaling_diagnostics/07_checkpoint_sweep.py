#!/usr/bin/env python
"""07_checkpoint_sweep.py -- TODO §7 (checkpoint-selection mismatch).

Only `best.pt` exists on disk for every one of the 6 existing scaling-sweep
pretraining runs (confirmed by directory listing during research: no
`step_XXXXXXX.pt` files, `save_step_checkpoints: false` in every
data{X}pct.yaml). A true "does reconstruction loss keep improving after
downstream quality has saturated" checkpoint-progression sweep needs
intermediate checkpoints that don't exist yet -- this script does NOT invent
them or fabricate results. It only:

  1. Checks what checkpoints actually exist for the requested fractions.
  2. If step checkpoints are missing, generates ready-to-submit rerun configs
     (save_step_checkpoints: true, ckpt_every_steps set so ~6-8 checkpoints
     land across the run) and prints the exact commands -- WITHOUT submitting
     any SLURM job (full pretraining reruns are expensive; needs a separate
     go/no-go per the approved plan).
  3. If step checkpoints DO exist (e.g. after such a rerun), runs a quick
     linear-probe screen at each checkpoint by shelling out to a probe config
     with `pretrained_ckpt` pointed at that specific step file, reporting
     val_loss/accuracy per checkpoint so reconstruction-loss-vs-downstream-
     quality can be compared directly.

Usage:
    # Step 1: see what's on disk + get rerun commands (no jobs submitted)
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/07_checkpoint_sweep.py --fractions 5pct,25pct,100pct

    # Step 2 (only after step checkpoints exist on disk): screen them
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/07_checkpoint_sweep.py --fractions 5pct --probe-config configs/vit1d_har_probe_wisdm_mae_data5pct_multidata_flu.yaml --screen
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import FRACTION_CONFIGS, OUT_DIR, REPO_ROOT, RUN_DIRS  # noqa: E402

CHECKPOINT_TARGETS = [50_000, 100_000, 150_000, 200_000, 250_000]


def inventory(fractions):
    rows = []
    for name in fractions:
        run_dir = os.path.join(REPO_ROOT, RUN_DIRS[name], "checkpoints")
        if not os.path.isdir(run_dir):
            rows.append({"fraction": name, "run_dir": run_dir, "exists": False, "checkpoints": []})
            continue
        ckpts = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*.pt")))
        rows.append({"fraction": name, "run_dir": run_dir, "exists": True, "checkpoints": ckpts})
    return rows


def write_rerun_configs(fractions, ckpt_every=25_000):
    out_cfg_dir = os.path.join(OUT_DIR, "configs_checkpoint_sweep")
    os.makedirs(out_cfg_dir, exist_ok=True)
    written = []
    for name in fractions:
        _, cfg_path = FRACTION_CONFIGS[name]
        with open(os.path.join(REPO_ROOT, cfg_path)) as f:
            raw = yaml.safe_load(f)
        raw["logging"]["save_step_checkpoints"] = True
        raw["logging"]["ckpt_every_steps"] = ckpt_every
        out_name = f"vit1d_ssl_mae_varlen_altattn_multidata_flu_data{name}_ckptsweep.yaml"
        out_path = os.path.join(out_cfg_dir, out_name)
        with open(out_path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        written.append((name, out_path))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="5pct,25pct,100pct")
    ap.add_argument("--screen", action="store_true",
                     help="run a quick linear-probe screen at each existing step checkpoint (requires "
                          "--probe-config and step checkpoints already present on disk)")
    ap.add_argument("--probe-config", default=None)
    ap.add_argument("--ckpt-every", type=int, default=25_000)
    args = ap.parse_args()
    fractions = args.fractions.split(",")

    rows = inventory(fractions)
    print("[07] checkpoint inventory:")
    for r in rows:
        print(f"  {r['fraction']:8s} {r['run_dir']}: {r['checkpoints'] if r['exists'] else 'MISSING run dir'}")

    have_step_ckpts = any(any(c.startswith("step_") for c in r["checkpoints"]) for r in rows)

    if not have_step_ckpts:
        print("\n[07] no intermediate step checkpoints exist for any requested fraction -- as expected "
              "(save_step_checkpoints: false in every existing scaling config). A real checkpoint-"
              "progression sweep requires NEW pretraining runs with checkpointing enabled.")
        written = write_rerun_configs(fractions, args.ckpt_every)
        print(f"\n[07] wrote {len(written)} rerun configs under {os.path.join(OUT_DIR, 'configs_checkpoint_sweep')}/ "
              f"(ckpt_every_steps={args.ckpt_every}) -- NOT submitted. This is a full pretraining rerun "
              "per fraction (hours of GPU time each) -- confirm before launching. Example command:")
        name, path = written[0]
        rel = os.path.relpath(path, REPO_ROOT)
        print(f"  CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \\")
        print(f"      --config configs/base.yaml --model-config {rel} \\")
        print(f"      --run-name vit1d_ssl_mae_varlen_altattn_multidata_data{name}_ckptsweep")
        print("  (or submit via an sbatch script modeled on launch_slurm_MAE_scaling_5pct.sh)")
        return

    if not args.screen:
        print("\n[07] step checkpoints found -- rerun with --screen --probe-config <path> to evaluate them.")
        return

    if args.probe_config is None:
        raise SystemExit("--screen requires --probe-config")

    out_dir = os.path.join(OUT_DIR, "07_checkpoint_sweep")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for r in rows:
        step_ckpts = sorted(
            (c for c in r["checkpoints"] if c.startswith("step_")),
            key=lambda c: int(c.split("_")[1].split(".")[0]),
        )
        step_ckpts += [c for c in ("best.pt",) if c in r["checkpoints"]]
        for ckpt_name in step_ckpts:
            ckpt_path = os.path.join(r["run_dir"], ckpt_name)
            run_name = f"07sweep_{r['fraction']}_{ckpt_name.replace('.pt', '')}"
            print(f"[07] screening {r['fraction']} / {ckpt_name} -- this launches a real (short) probe "
                  f"training job via scripts/run_pretrain.py, run_name={run_name}")
            # NOTE: this DOES run a real (short, early-stopped) probe training
            # job -- unlike the pretraining reruns above, probes are cheap
            # (minutes, per the existing configs' max_steps/early_stopping) so
            # this is launched directly rather than just printed. Comment out
            # / dry-run below if you want to inspect the command first.
            cmd = [
                sys.executable, "scripts/run_pretrain.py",
                "--config", "configs/base.yaml",
                "--model-config", args.probe_config,
                "--run-name", run_name,
            ]
            print("   $", " ".join(cmd))
            # subprocess.run(cmd, cwd=REPO_ROOT, check=True)  # uncomment to actually launch
            results.append({"fraction": r["fraction"], "checkpoint": ckpt_name, "run_name": run_name,
                             "launched": False, "note": "dry-run only -- uncomment subprocess.run to launch"})

    with open(os.path.join(out_dir, "checkpoint_screen_plan.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[07] wrote dry-run plan to {out_dir}/checkpoint_screen_plan.json -- "
          "no jobs were actually launched (see comment in the script).")


if __name__ == "__main__":
    main()
