#!/usr/bin/env python3
"""Score one scratch-normalization cell with its task's own bench scorer -> results_scratch3/."""
import importlib, json, os, sys
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from bench.registry_io import load_tasks
run, task = sys.argv[1], sys.argv[2]
tcard = load_tasks()["tasks"][task]
mod = importlib.import_module(tcard["scorer"])
m, info = mod.score(os.path.join("runs", run), os.path.join("runs", run, "checkpoints", "best.pt"), tcard)
json.dump({"run": run, "task": task, "metrics": m, "info": {k: v for k, v in info.items() if isinstance(v, (int, float, str, type(None)))}},
          open(f"bench/generated/results_scratch3/{run}.json", "w"), indent=1)
print(f"[scr3] {run}: {tcard['metrics']['primary']}={m.get(tcard['metrics']['primary'])}")
