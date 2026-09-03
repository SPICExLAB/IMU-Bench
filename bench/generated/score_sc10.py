#!/usr/bin/env python3
"""Score one sc10 scratch cell under BOTH checkpoint-selection rules (min val_loss, max val macro_f1)."""
import json, os, sys
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from bench.score.har import score
from bench.registry_io import load_tasks
run, task = sys.argv[1], sys.argv[2]
tasks = load_tasks()["tasks"]; out = {"run": run, "task": task}
for sel, ck in (("val_loss", "best.pt"), ("macro_f1", "best_macro_f1.pt")):
    p = os.path.join("runs", run, "checkpoints", ck)
    if not os.path.exists(p):
        continue
    m, info = score(os.path.join("runs", run), p, tasks[task])
    out[sel] = {"macro_f1": m["macro_f1"], "bal_acc": m["bal_acc"], "acc": m["acc"], "step": info.get("checkpoint_step")}
    print(f"{run} [{sel}] macro_f1={m['macro_f1']:.4f}", flush=True)
json.dump(out, open(f"bench/generated/results_sc10/{run}.json", "w"), indent=1)
