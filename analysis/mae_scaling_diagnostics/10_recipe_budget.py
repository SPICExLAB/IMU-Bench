#!/usr/bin/env python
"""10_recipe_budget.py -- budget planning for the approved 3-point "does more
data help" experiment: session_fraction scaling x {natural, balanced}
sampling at 5%/25%/100%, on the CURRENT production sources block (v3-scale
ocdetect/capture24/eatingspeed, i.e. configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml
as it stands today), with NHANES excluded (its own scaling behavior is
uncharacterized -- see DIAGNOSIS.md).

Builds every source ONCE with provenance (the expensive step -- raw parse +
windowing, bypasses cache_dir the same way 01_composition_audit.py does),
then for each fraction in {0.05, 0.25, 1.0} computes the session_fraction
subset per source and prints both a 'natural' epoch definition (sum of
per-source counts) and a 'balanced' epoch definition
(n_sources * min(per-source counts), matching the WeightedRandomSampler
formula in combined_mae_multidata_loader.make_combined_mae_multidata_loaders)
so we can see BEFORE launching whether balanced sampling collapses to a
tiny epoch size once tiny sources (totalcapture/imuposer) get floored to a
single kept session under session_fraction.

Does NOT submit any training job or write final configs -- print/inspect
only. Run under SLURM (raw source build needs real memory/CPU, OOMs under
the interactive shell's cgroup).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from lib.common import OUT_DIR, REPO_ROOT, load_merged_cfg  # noqa: E402

PROD_CONFIG = "configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml"
FRACTIONS = [0.05, 0.25, 1.0]
BATCH_SIZE = 128
TARGET_EPOCHS = 77  # matches the existing 100%/window_fraction/natural run's actual best epoch


def main():
    from imu_lm.data import combined_mae_multidata_loader as cm

    cfg = load_merged_cfg(PROD_CONFIG)
    mm_cfg = cfg["objective"]["mae_multidata_varlen_altattn_flu"]
    mm_cfg["sources"]["nhanes"]["enabled"] = False
    print("[10] nhanes disabled for this experiment (uncharacterized scaling behavior)")

    print("[10] building every source once at full (current production) resolution "
          "with provenance -- this is the expensive step...")
    from lib.common import build_all_sources_once
    train_parts, val_parts, names = build_all_sources_once(cfg)
    base_counts = {n: len(p) for n, p in zip(names, train_parts)}
    total_train = sum(base_counts.values())
    print(f"[10] built {len(names)} sources, total_train={total_train} windows: "
          + ", ".join(f"{k}={v}" for k, v in base_counts.items()))

    rows = []
    for frac in FRACTIONS:
        frac_name = f"{frac*100:g}pct"
        per_source = {}
        for name, part in zip(names, train_parts):
            if frac >= 1.0:
                per_source[name] = len(part)
                continue
            sub, warn = cm._group_fraction_subset(name, part, frac, level="session", seed=42)
            if sub is None:
                print(f"[10]   {frac_name}/{name}: {warn}")
                sub = cm._fraction_subset(part, frac, seed=42)
            per_source[name] = len(sub)

        n_train_natural = sum(per_source.values())
        min_source = min(per_source.values())
        min_source_name = min(per_source, key=per_source.get)
        n_train_balanced_epoch = len(per_source) * min_source

        steps_per_epoch_natural = max(1, n_train_natural // BATCH_SIZE)
        steps_per_epoch_balanced = max(1, n_train_balanced_epoch // BATCH_SIZE)

        row = {
            "fraction": frac_name, "overall_fraction": frac,
            "per_source": per_source,
            "n_train_natural_epoch": n_train_natural,
            "min_source": min_source_name, "min_source_n": min_source,
            "n_train_balanced_epoch": n_train_balanced_epoch,
            "steps_per_epoch_natural": steps_per_epoch_natural,
            "steps_per_epoch_balanced": steps_per_epoch_balanced,
            "fixed_epoch_budget_max_steps_natural": TARGET_EPOCHS * steps_per_epoch_natural,
            "fixed_epoch_budget_max_steps_balanced": TARGET_EPOCHS * steps_per_epoch_balanced,
        }
        rows.append(row)
        print(f"\n[10] === fraction={frac_name} (session_fraction={frac}) ===")
        print("  per-source train windows kept: " + ", ".join(f"{k}={v}" for k, v in per_source.items()))
        print(f"  natural epoch size = {n_train_natural} windows -> {steps_per_epoch_natural} steps/epoch "
              f"-> {row['fixed_epoch_budget_max_steps_natural']} steps @ {TARGET_EPOCHS} epochs")
        print(f"  balanced epoch size = {len(per_source)} sources * min({min_source_name}={min_source}) "
              f"= {n_train_balanced_epoch} windows -> {steps_per_epoch_balanced} steps/epoch "
              f"-> {row['fixed_epoch_budget_max_steps_balanced']} steps @ {TARGET_EPOCHS} epochs")

    out_dir = os.path.join(OUT_DIR, "10_recipe_budget")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "recipe_budget.json")
    with open(out_path, "w") as f:
        json.dump({"base_counts": base_counts, "total_train": total_train, "rows": rows}, f, indent=2)
    print(f"\n[10] wrote {out_path}")


if __name__ == "__main__":
    main()
