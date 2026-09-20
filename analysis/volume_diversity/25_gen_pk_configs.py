#!/usr/bin/env python
"""Stage 25: configs, sbatch scripts, registry entries and a submit script for the
random physical-type grid cells ONLY (vd_pk*), reusing 05_gen_configs.py's builders so
every cell is byte-for-byte the same recipe as the earlier vd_* arms.

Unlike 05 --write-registry, this does not re-dump the whole registry (that would strip
its comments); new entries are appended as text, and an existing entry is never touched.

Writes configs/voldiv/vd_pk*.yaml, launch/voldiv/vd_pk*.sh, launch/voldiv/submit_pk.sh,
       appends to bench/registry/encoders.yaml
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from arms import OUT_DIR, STEPS

_spec = importlib.util.spec_from_file_location("gen", os.path.join(HERE, "05_gen_configs.py"))
gen = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gen)

PANEL = ["arm_pose_nymeria", "odometry_nymeria_worldframe", "har_wisdm", "microgesture_openwatch",
         "har_utd_mhad", "har_pamap2", "har_recofit", "full_body_pose_nymeria"]
EXCLUDE = "#SBATCH --exclude=qgpu3006,qgpu3013,qgpu2014,qgpu2006\n"


def main():
    grid = json.load(open(os.path.join(OUT_DIR, "phys_random_grid.json")))
    manifest = json.load(open(os.path.join(OUT_DIR, "arms.json")))
    arms = [c["arm"] for c in grid["cells"] if not c["reused"]]
    reg_text = open(gen.REGISTRY).read()
    reg = yaml.safe_load(reg_text)["encoders"]
    new_entries = {}
    for order, arm in enumerate(arms):
        info = manifest["arms"][arm]
        div, tag, n = info["diversity"], info["volume_tag"], int(info["n_windows"])
        cfg = gen.build_cfg(arm, div, tag, n, manifest)
        with open(os.path.join(gen.CFG_DIR, f"{arm}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        sh = gen.SBATCH.format(arm=arm, label=gen._label(div), n=n, hours=n * 5.12 / 3600, steps=STEPS,
                               account="p32139", partition="gengpu")
        sh = sh.replace("#SBATCH --job-name", EXCLUDE + "#SBATCH --job-name", 1)
        p = os.path.join(gen.LAUNCH_DIR, f"{arm}.sh")
        open(p, "w").write(sh); os.chmod(p, 0o755)
        if arm not in reg:
            new_entries[arm] = gen.registry_entry(arm, div, tag, n, manifest, 1000 + order)
    if new_entries:
        block = yaml.safe_dump(new_entries, sort_keys=False, default_flow_style=False, width=100)
        block = "".join("  " + line if line.strip() else line for line in block.splitlines(True))
        with open(gen.REGISTRY, "a") as f:
            if not reg_text.endswith("\n"):
                f.write("\n")
            f.write("  # --- random physical-type x volume grid (24_phys_random_grid.py) ---\n" + block)
    check = yaml.safe_load(open(gen.REGISTRY))["encoders"]
    assert all(a in check for a in arms), "registry append failed"

    sub = os.path.join(gen.LAUNCH_DIR, "submit_pk.sh")
    with open(sub, "w") as f:
        f.write("#!/bin/bash\n# Random physical-type x volume grid: pretrain each cell, chain the 8-task probe panel.\n"
                "#   ./launch/voldiv/submit_pk.sh [--probes-only] [--arms a,b] [--buyin]\n"
                "set -euo pipefail\ncd /gpfs/home/vbc1591/IMU-SSL\nsource TransfHAR_env/bin/activate\n"
                "PROBES_ONLY=0; ONLY=\"\"; BUYIN=\"\"\n"
                "while [ $# -gt 0 ]; do case \"$1\" in\n  --probes-only) PROBES_ONLY=1;; --buyin) BUYIN=--buyin;;\n"
                "  --arms) ONLY=\"$2\"; shift;; *) echo \"unknown arg $1\"; exit 1;; esac; shift; done\n"
                f"PANEL=\"{' '.join(PANEL)}\"\nARMS=\"{' '.join(arms)}\"\n"
                "for ARM in $ARMS; do\n  if [ -n \"$ONLY\" ] && [[ \",$ONLY,\" != *\",$ARM,\"* ]]; then continue; fi\n"
                "  DEP=\"\"\n  if [ \"$PROBES_ONLY\" -eq 0 ]; then\n"
                "    if squeue -u \"$USER\" -h -o '%j' | grep -qx \"$ARM\"; then echo \"$ARM already queued\"; continue; fi\n"
                "    JID=$(sbatch --parsable launch/voldiv/$ARM.sh)\n    echo \"pretrain $ARM -> $JID\"\n"
                "    DEP=\"--dependency afterok:$JID\"\n  fi\n"
                "  for T in $PANEL; do\n    python -m bench.launch --cell $T $ARM probe $DEP $BUYIN\n  done\ndone\n")
    os.chmod(sub, 0o755)
    print(f"[gen] {len(arms)} cells: configs + sbatch written, {len(new_entries)} registry entries appended")
    print(f"[gen] submit with: {sub}")


if __name__ == "__main__":
    main()
