#!/usr/bin/env python
"""06_budget_protocols.py -- TODO §6 (training-budget comparability).

Prints, for every data-scaling fraction, the actual epoch/step numbers behind
the existing EXPERIMENTS.md §9 sweep (already documented: 9,667 epochs @0.1%
down to 77 epochs @100%, because trainer.max_steps is effectively unbounded
and early stopping is keyed on a FIXED step count -- val_every_steps=1000,
patience=10 -- regardless of how many epochs that spans at a given fraction).

Then computes two controlled-budget protocols and prints the resulting
max_steps each fraction would need, plus writes ready-to-use override config
YAMLs under out/configs_budget_controlled/. Does NOT submit any SLURM job and
does NOT modify any existing config -- see the approved plan's "Explicit
non-goals" section.

Usage:
    TransfHAR_env/bin/python analysis/mae_scaling_diagnostics/06_budget_protocols.py \
        [--target-epochs 77] [--target-steps 291000] [--composition out/01_composition/window_counts.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import FRACTION_CONFIGS, OUT_DIR, REPO_ROOT, load_merged_cfg  # noqa: E402

# Documented in EXPERIMENTS.md §9 (2026 pretraining scaling sweep) -- used as
# the fallback total/best-step source when 01_composition_audit.py hasn't been
# run yet (its window_counts.json, if present, is authoritative and preferred).
DOCUMENTED_TOTAL_TRAIN_WINDOWS = 485_125
DOCUMENTED_BEST_STEP = {
    "0.1pct": 29_000, "1pct": 135_000, "5pct": 221_000,
    "25pct": 221_000, "50pct": 221_000, "100pct": 291_000,
}
DOCUMENTED_BEST_EPOCH = {
    "0.1pct": 9_667, "1pct": 3_649, "5pct": 1_170,
    "25pct": 234, "50pct": 117, "100pct": 77,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-epochs", type=int, default=77,
                     help="epoch budget for the fixed-epoch-budget protocol (default: matches the 100%% run)")
    ap.add_argument("--target-steps", type=int, default=291_000,
                     help="step budget for the fixed-step-budget protocol (default: matches the 100%% run's best step)")
    ap.add_argument("--composition", default=os.path.join(OUT_DIR, "01_composition", "window_counts.json"))
    args = ap.parse_args()

    window_counts = {}
    if os.path.exists(args.composition):
        with open(args.composition) as f:
            window_counts = json.load(f)["total_train_by_fraction"]
        print(f"[06] using live window counts from {args.composition}")
    else:
        print("[06] 01_composition_audit.py output not found -- falling back to the "
              "DOCUMENTED_TOTAL_TRAIN_WINDOWS pooled-fraction arithmetic from EXPERIMENTS.md "
              "(each fraction's train windows = round(485125 * fraction), pooled mode)")
        for name, (frac, _path) in FRACTION_CONFIGS.items():
            window_counts[name] = max(1, round(DOCUMENTED_TOTAL_TRAIN_WINDOWS * frac))

    rows = []
    for name, (frac, cfg_path) in FRACTION_CONFIGS.items():
        cfg = load_merged_cfg(cfg_path)
        batch_size = int(cfg["data"]["batch_size"])
        n_train = window_counts.get(name, max(1, round(DOCUMENTED_TOTAL_TRAIN_WINDOWS * frac)))
        steps_per_epoch = max(1, n_train // batch_size)

        actual_best_step = DOCUMENTED_BEST_STEP.get(name)
        actual_best_epoch = DOCUMENTED_BEST_EPOCH.get(name)

        fixed_epoch_steps = args.target_epochs * steps_per_epoch
        fixed_step_epochs_equiv = args.target_steps / steps_per_epoch

        rows.append({
            "fraction": name, "overall_fraction": frac, "n_train": n_train,
            "batch_size": batch_size, "steps_per_epoch": steps_per_epoch,
            "actual_best_step": actual_best_step, "actual_best_epoch": actual_best_epoch,
            "fixed_epoch_budget_max_steps": fixed_epoch_steps,
            "fixed_step_budget_max_steps": args.target_steps,
            "fixed_step_budget_epochs_equiv": round(fixed_step_epochs_equiv, 2),
        })

    print()
    hdr = f"{'fraction':8s} {'n_train':>9s} {'steps/epoch':>11s} {'actual best_step':>17s} {'actual best_epoch':>18s} {'fixed-EPOCH max_steps':>22s} {'fixed-STEP epochs_equiv':>24s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['fraction']:8s} {r['n_train']:9d} {r['steps_per_epoch']:11d} "
              f"{str(r['actual_best_step']):>17s} {str(r['actual_best_epoch']):>18s} "
              f"{r['fixed_epoch_budget_max_steps']:22d} {r['fixed_step_budget_epochs_equiv']:24.2f}")
    print()
    print(f"Fixed-EPOCH-budget protocol: every fraction trains for exactly {args.target_epochs} epochs "
          f"(matches the 100% run) -- max_steps above.")
    print(f"Fixed-STEP-budget protocol: every fraction trains for exactly {args.target_steps} steps "
          f"(matches the 100% run's best step) -- 'epochs_equiv' above shows how many times each "
          f"fraction's pool gets revisited under that budget (still very uneven, deliberately -- this "
          f"protocol isolates 'same gradient-update count' rather than 'same data-exposure count').")

    # ---- write override configs (both protocols), no SLURM submission ----
    out_cfg_dir = os.path.join(OUT_DIR, "configs_budget_controlled")
    os.makedirs(out_cfg_dir, exist_ok=True)
    written = []
    for r in rows:
        name = r["fraction"]
        _, cfg_path = FRACTION_CONFIGS[name]
        import yaml
        with open(os.path.join(REPO_ROOT, cfg_path)) as f:
            raw = yaml.safe_load(f)

        for protocol, max_steps in [
            ("fixed_epoch_budget", r["fixed_epoch_budget_max_steps"]),
            ("fixed_step_budget", r["fixed_step_budget_max_steps"]),
        ]:
            c = copy.deepcopy(raw)
            c["trainer"]["max_steps"] = int(max_steps)
            # Scale early-stopping patience (in val-check units) proportionally to
            # the new step budget so "no improvement for N%% of the budget" stays
            # comparable across fractions instead of the fixed absolute
            # patience=10 (10,000 steps) that causes the epoch-budget confound
            # documented in EXPERIMENTS.md §9 in the first place.
            val_every = int(c["trainer"]["val_every_steps"])
            c["trainer"]["early_stopping"]["patience"] = max(2, int(max_steps * 0.2) // val_every)
            out_name = f"vit1d_ssl_mae_varlen_altattn_multidata_flu_data{name}_{protocol}.yaml"
            out_path = os.path.join(out_cfg_dir, out_name)
            with open(out_path, "w") as f:
                yaml.safe_dump(c, f, sort_keys=False)
            written.append(out_path)

    print(f"\nWrote {len(written)} budget-controlled config variants under {out_cfg_dir}/")
    print("No SLURM job submitted -- these are ready to launch manually once reviewed, e.g.:")
    print("  CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \\")
    print("      --config configs/base.yaml \\")
    print(f"      --model-config {os.path.relpath(written[0], REPO_ROOT)} \\")
    print("      --run-name <pick_a_name>")

    summary_path = os.path.join(OUT_DIR, "06_budget_protocols.json")
    with open(summary_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nMachine-readable summary: {summary_path}")


if __name__ == "__main__":
    main()
