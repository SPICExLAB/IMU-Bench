#!/usr/bin/env python3
"""Score one LR-tuning run (best.pt = max val bal_acc checkpoint) into results_lrtune/."""
import json, os, sys
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from bench.score.har import score
from bench.registry_io import load_tasks
run, task = sys.argv[1], sys.argv[2]
m, info = score(os.path.join("runs", run), os.path.join("runs", run, "checkpoints", "best.pt"), load_tasks()["tasks"][task])
json.dump({"run": run, "task": task, "metrics": m, "info": info}, open(f"bench/generated/results_lrtune/{run}.json", "w"), indent=1)
print(f"[lrtune] {run}: macro_f1={m['macro_f1']:.4f} bal_acc={m['bal_acc']:.4f}")
