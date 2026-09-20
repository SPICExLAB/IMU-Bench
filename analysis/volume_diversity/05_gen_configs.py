#!/usr/bin/env python
"""Stage 5: emit everything needed to RUN the factorial.

  configs/voldiv/<arm>.yaml         pretrain config, one per cell
  launch/voldiv/<arm>.sh            sbatch script, one per cell
  launch/voldiv/submit.sh           submits the cells and chains the probe
                                    panel behind each with afterok
  bench/registry/encoders.yaml      one hidden encoder entry per cell, so the
                                    probes run through the SAME controlled
                                    bench recipe every other row of the paper
                                    uses (--write-registry)

Every cell differs from `configs/nhanes_scale/mae_4src_z.yaml` in exactly two
places -- the window index file and the step budget. Same architecture, mask
ratio, LR, schedule, augmentation, normalisation, val set and seed. The val set
is the untouched 4-source val split for every cell, so MAE val loss is a common
yardstick and a narrow-diversity cell is scored on the diverse val it did not
train on, which is the point.

Compute is held fixed at 50,000 steps x batch 128 = 6.4 M window presentations
for every cell, so a 12,500-window cell revisits its windows ~512 times and the
full-pool cell ~3.7 times. That is the control the claim needs: at MATCHED
COMPUTE, does more distinct data help? Matching epochs instead would confound
volume with compute and is what the original scaling curve did.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import (BATCH, DIV_LABEL, IDX_DIR, OUT_DIR, POOL_CONFIG, POOL_N, REPO_ROOT,
                  SEED, STEPS, idx_path, manifest_arms)

CFG_DIR = os.path.join(REPO_ROOT, "configs", "voldiv")
LAUNCH_DIR = os.path.join(REPO_ROOT, "launch", "voldiv")
REGISTRY = os.path.join(REPO_ROOT, "bench", "registry", "encoders.yaml")

# the downstream panel: four tasks, four different output types, all scored by
# the existing bench scorers on the existing splits
PANEL = ["arm_pose_nymeria", "odometry_nymeria_worldframe", "har_wisdm",
         "microgesture_openwatch"]

def _label(div):
    """`sub<k>` arms are added by 09 after the factorial, so fall back rather
    than KeyError on a diversity level arms.py does not enumerate."""
    if div in DIV_LABEL:
        return DIV_LABEL[div]
    if div.startswith("sub"):
        k = div[3:]
        return f"ocdetect only, {'every' if k == 'all' else k} subject(s), type-matched"
    if div.startswith("pk"):
        # 24_phys_random_grid.py: random, source-stratified T-phys types x volume
        return f"4-source pool, {div[2:]}% of T-phys types (random, source-stratified, gyro-unit fixed)"
    if div.startswith("ek"):
        # 15_enc_type_volume_grid.py: the paper's types x volume grid, T-enc types
        return f"4-source pool, {div[2:]} largest T-enc type(s), water-filled"
    if div.startswith("typ"):
        # the T-phys type ladder, now crossed with volume by 14_type_volume_grid.py
        k = div[3:]
        return f"ocdetect only, {'every' if k == 'all' else k} T-phys type(s), water-filled"
    if div.startswith("etyp"):
        # added by 12_enc_type_arms.py: the type ladder selected in T-enc (the
        # mae_4src_z representation) instead of T-phys
        k = div[4:]
        return f"MAE-space types, {'every' if k == 'all' else k} T-enc type(s), ocdetect only"
    return div


SBATCH = """#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name={arm}
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# volume x diversity factorial, cell {arm} (EXPERIMENTS.md 11m)
# {label}, N={n:,} windows ({hours:,.0f} h), equal {steps:,}-step budget.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
RUN={arm}
if [ -f runs/$RUN/.done ]; then echo "$RUN already complete"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \\
  --model-config configs/voldiv/{arm}.yaml --run-name "$RUN" --resume latest \\
  && touch runs/$RUN/.done
"""


def build_cfg(arm, div, vol_tag, n, manifest):
    with open(POOL_CONFIG) as f:
        cfg = yaml.safe_load(f)
    mm = cfg["objective"]["mae_multidata_varlen_altattn_flu"]
    info = manifest["arms"][arm]

    if info.get("index_file"):
        mm["scaling_mode"] = "index_file"
        mm["subset_indices_file"] = info["index_file"]
        mm["sampling_mode"] = "natural"
        # balanced-mode leftovers would be silently ignored under natural
        # sampling; drop them so the config cannot be misread
        mm.pop("source_weights", None)
        mm.pop("epoch_windows", None)
        sampler_note = "natural sampling over the index file"
    else:
        mm["sampling_mode"] = "balanced"          # the vd_bal_nfull anchor
        sampler_note = ("balanced WeightedRandomSampler with the production "
                        "source_weights -- same 100 % of the pool as vd_nat_nfull, "
                        "different sampling law")

    cfg["trainer"]["max_steps"] = STEPS
    cfg["data"]["batch_size"] = BATCH
    cfg["trainer"]["seed"] = SEED
    # on but un-triggerable: `enabled: false` stops best.pt from ever being
    # written (see the trap box in EXPERIMENTS.md 11k)
    cfg["trainer"]["early_stopping"] = {"enabled": True, "monitor": "val_loss",
                                        "patience": 999999, "min_delta": 1.0e-4}

    phys = info.get("phys", {})
    enc = info.get("enc", {})
    cfg["_recipe_note"] = (
        f"volume x diversity factorial cell {arm} (EXPERIMENTS.md 11m). "
        f"Diversity level '{div}' = {_label(div)}; volume {vol_tag} = "
        f"{n:,} windows ({n * 5.12 / 3600:,.0f} h, {100.0 * n / POOL_N:.3g}% of the "
        f"1,738,094-window full-size 4-source pool). {sampler_note}. "
        f"Realised: sources={info.get('n_sources', 4)}, "
        f"T-phys types={phys.get('types_present', '-')} "
        f"(effective {phys.get('effective_types_shannon', float('nan')):.1f}), "
        f"T-enc effective types={enc.get('effective_types_shannon', float('nan')):.1f}. "
        f"Everything except the {'sampler' if not info.get('index_file') else 'index file'} "
        f"and max_steps={STEPS} is identical to "
        f"configs/nhanes_scale/mae_4src_z.yaml. Equal-compute protocol: "
        f"{STEPS:,} steps x batch {BATCH} for every cell."
    )
    return cfg


def registry_entry(arm, div, vol_tag, n, manifest, order):
    info = manifest["arms"][arm]
    phys = info.get("phys", {})
    return {
        "name": f"VolDiv {_label(div)}, N={n:,}",
        "short": arm.replace("vd_", ""),
        "group": "ablation",
        "order": order,
        "hide_in_summary": True,
        "run": arm,
        "ckpt": "best",
        "alt_attention": True,
        "channels": 6,
        "shared_kernel": "acc_gyro",
        "params_m": 21,
        "normalize_method": "zscore",
        "preprocess_from": f"configs/voldiv/{arm}.yaml",
        "pretrain_data": (f"volume x diversity cell: {_label(div)}, {n:,} windows "
                          f"({n * 5.12 / 3600:,.0f} h) of the full-size 4-source pool; "
                          f"{info.get('n_sources', 4)} sources, "
                          f"{phys.get('types_present', '-')} T-phys motion types "
                          f"(effective {phys.get('effective_types_shannon', float('nan')):.1f})"),
        "pretrain_objective": f"MAE mask 0.60, z-score input, octahedral rotation, lr 1e-4, {STEPS:,} steps",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-registry", action="store_true",
                    help="append the cells to bench/registry/encoders.yaml")
    args = ap.parse_args()

    os.makedirs(CFG_DIR, exist_ok=True)
    os.makedirs(LAUNCH_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "arms.json")) as f:
        manifest = json.load(f)

    # the full-pool natural anchor needs an index file covering the whole pool
    full = os.path.join(IDX_DIR, "vd_nat_nfull.npy")
    if not os.path.exists(full):
        np.save(full, np.arange(POOL_N, dtype=np.int64))

    lines, entries = [], {}
    for order, (arm, div, vol_tag, n) in enumerate(manifest_arms(manifest)):
        cfg = build_cfg(arm, div, vol_tag, n, manifest)
        with open(os.path.join(CFG_DIR, f"{arm}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        sh = os.path.join(LAUNCH_DIR, f"{arm}.sh")
        with open(sh, "w") as f:
            f.write(SBATCH.format(arm=arm, label=_label(div), n=n,
                                  hours=n * 5.12 / 3600, steps=STEPS,
                                  account="p32139", partition="gengpu"))
        os.chmod(sh, 0o755)
        entries[arm] = registry_entry(arm, div, vol_tag, n, manifest, 900 + order)
        lines.append((arm, n))

    # submit script: pretrain, then the probe panel chained behind it
    sub = os.path.join(LAUNCH_DIR, "submit.sh")
    with open(sub, "w") as f:
        f.write("#!/bin/bash\n"
                "# Submit the volume x diversity factorial (EXPERIMENTS.md 11m).\n"
                "#   ./launch/voldiv/submit.sh [--probes-only] [--arms a,b,c] [--buyin]\n"
                "# Each cell's four probes are chained behind its pretrain with afterok,\n"
                "# and run through the standard bench recipe (bench/registry/tasks.yaml).\n"
                "set -euo pipefail\n"
                "cd /gpfs/home/vbc1591/IMU-SSL\n"
                "source TransfHAR_env/bin/activate\n"
                "PROBES_ONLY=0; ONLY=\"\"; BUYIN=\"\"\n"
                "while [ $# -gt 0 ]; do case \"$1\" in\n"
                "  --probes-only) PROBES_ONLY=1;; --buyin) BUYIN=--buyin;;\n"
                "  --arms) ONLY=\"$2\"; shift;; *) echo \"unknown arg $1\"; exit 1;; esac; shift; done\n"
                f"PANEL=\"{' '.join(PANEL)}\"\n"
                f"ARMS=\"{' '.join(a for a, _ in lines)}\"\n"
                "for ARM in $ARMS; do\n"
                "  if [ -n \"$ONLY\" ] && [[ \",$ONLY,\" != *\",$ARM,\"* ]]; then continue; fi\n"
                "  DEP=\"\"\n"
                "  if [ \"$PROBES_ONLY\" -eq 0 ]; then\n"
                "    JID=$(sbatch --parsable launch/voldiv/$ARM.sh)\n"
                "    echo \"pretrain $ARM -> $JID\"\n"
                "    DEP=\"--dependency afterok:$JID\"\n"
                "  fi\n"
                "  for T in $PANEL; do\n"
                "    python -m bench.launch --cell $T $ARM probe $DEP $BUYIN\n"
                "  done\n"
                "done\n")
    os.chmod(sub, 0o755)

    if args.write_registry:
        with open(REGISTRY) as f:
            reg = yaml.safe_load(f)
        added = 0
        for arm, entry in entries.items():
            if arm not in reg["encoders"]:
                added += 1
            reg["encoders"][arm] = entry
        with open(REGISTRY, "w") as f:
            yaml.safe_dump(reg, f, sort_keys=False, default_flow_style=False, width=100)
        print(f"[registry] wrote {len(entries)} entries ({added} new) to {REGISTRY}")
    else:
        path = os.path.join(OUT_DIR, "encoder_entries.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({"encoders": entries}, f, sort_keys=False, default_flow_style=False, width=100)
        print(f"[registry] preview written to {path} (re-run with --write-registry to install)")

    print(f"[gen] {len(lines)} configs in {CFG_DIR}, launch scripts in {LAUNCH_DIR}")
    print(f"[gen] submit with: {sub}")


if __name__ == "__main__":
    main()
