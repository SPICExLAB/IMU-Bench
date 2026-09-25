#!/usr/bin/env python3
"""Harvest the random-init anchor of the grid's own (legacy) kinematic probes.

Figure 4 expresses every probe as a share of the full-pool model's gain over a never-trained encoder under
the same probe (34_plot_scaling_avg.py). The benchmark probes already have that encoder scored
(bench/results/<task>/control_random__probe__s42.json); the grid's arm, full-body and odometry probes did
not, so it is run through them unchanged: bench's `control_random_frozen_5src` checkpoint (the InrtlWrist
architecture, never trained) in place of `vit1d_ssl_mae_grid_k1000_f100`, every other setting identical
(configs/grid5src/*_control_random_*, launch/grid5src/launch_*control_random.sh, jobs 7318003/05/07).

The values are read with 12_harvest_grid.py's own rules (min over training for the pose probes, the last
vel_mse= line of a best.pt-loading log for odometry), so anchor and cells are harvested identically.

    python analysis/diversity_saturation/35_harvest_grid_random_anchor.py   -> out/grid_random_anchor.json
"""
import glob
import importlib.util
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("harvest_grid", os.path.join(HERE, "12_harvest_grid.py"))
hg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hg)

NAME = "control_random"
out, missing = {}, []
for label, tmpl, key, how, scale in hg.PROBES:
    if not label.startswith(("Arm pose", "Full-body")):
        continue                                  # the classification panels use benchmark probes
    rows = hg.val_rows(tmpl.format(c=NAME))
    cand = [r for r in rows or [] if key in r and "loss" in r]
    if not cand:
        missing.append(label)
        continue
    out[label] = (min(r[key] for r in cand) * scale if how == "min"
                  else min(cand, key=lambda r: r["loss"])[key] * scale)

val = None
for h in sorted(glob.glob(os.path.join(hg.REPO_ROOT, "logs", f"odom_probe_nymeria_mae_{NAME}_*.out"))):
    text = open(h, errors="ignore").read()
    if "checkpoints/best.pt" not in text:
        continue
    for line in text.splitlines():
        m = re.search(r"vel_mse=([\d.]+)", line)
        if m:
            val = float(m.group(1))
if val is None:
    missing.append("Odometry (vel MSE)")
else:
    out["Odometry (vel MSE)"] = val

p = os.path.join(HERE, "out", "grid_random_anchor.json")
json.dump({"encoder": "control_random_frozen_5src", "values": out, "missing": missing}, open(p, "w"), indent=1)
print(json.dumps(out, indent=1), "\nmissing:", missing, "\nwrote", p)
