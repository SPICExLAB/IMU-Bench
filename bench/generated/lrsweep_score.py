#!/usr/bin/env python3
"""Score one LR-sweep run into results_lrsweep/, dispatching on the task card's
own scorer so the number is produced exactly the way the leaderboard produces it.

`lrtune_score.py` hard-codes `bench.score.har`; this sweep covers pose, odometry,
gesture, language and PD, each of which has its own scorer and its own
checkpoint-selection rule.

    python bench/generated/lrsweep_score.py <run_name> <task_id>
"""
import importlib, json, os, sys

sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from bench.registry_io import load_tasks

run, task_id = sys.argv[1], sys.argv[2]
task = load_tasks()["tasks"][task_id]
mod = importlib.import_module(task["scorer"])

run_dir = os.path.join("runs", run)
ckpt = os.path.join(run_dir, "checkpoints", "best.pt")
m, info = mod.score(run_dir, ckpt, task)

out = {"run": run, "task": task_id, "metrics": m, "info": info}
os.makedirs("bench/generated/results_lrsweep", exist_ok=True)
with open(f"bench/generated/results_lrsweep/{run}.json", "w") as f:
    json.dump(out, f, indent=1)
pk = task["metrics"]["primary"]
print(f"[lrsweep] {run}: {pk}={m.get(pk)}")
