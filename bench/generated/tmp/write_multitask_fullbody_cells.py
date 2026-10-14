#!/usr/bin/env python
"""Write the joint runs' own full-body head as a benchmark cell.

bench.score cannot harvest these: bench/score/full_body_pose.py reads the best MPJPE val row out of
runs/<r>/logs/metrics.txt, and scripts/train_multitask_joint.py logs per-task LOSSES to the slurm log instead. The
metric is therefore recomputed from the joint checkpoint with the benchmark's own evaluator
(bench.rescore_full_body_extra.evaluate, the same code that produced the MobilePoser-protocol columns of every other
full-body row), and written in the scorer's schema so Table 10 reads from bench/results like every other number.
"""
import datetime, json, os, sys
sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")   # the script lives three levels down, and python puts the SCRIPT dir on sys.path, not cwd
from bench.rescore_full_body_extra import evaluate
from bench.schema import RESULTS_ROOT

TASK = "full_body_pose_nymeria"
for enc in ("mt_body_odom", "mt_body_gesture"):
    run = f"{enc}__{TASK}"
    ck = os.path.join("runs", run, "checkpoints", "best.pt")
    if not os.path.exists(ck):
        print(f"[skip] {enc}: no checkpoint"); continue
    m, info = evaluate(os.path.join("runs", run), ck)
    out = os.path.join(RESULTS_ROOT, TASK, f"{enc}__finetune__s42.json")
    json.dump({
        "task": TASK, "encoder": enc, "protocol": "finetune", "seed": 42,
        "metrics": {"mpjpe_mm": m["mpjpe_mm_recomputed"], **{k: v for k, v in m.items() if k != "mpjpe_mm_recomputed"}},
        "primary_metric": "mpjpe_mm", "eval_split": "val", "run": run, "checkpoint": "best.pt",
        "checkpoint_step": info["step"],
        "selection": {"split": "val", "monitor": "joint val loss", "mode": "min",
                      "via": "best.pt of the joint run (the trainer selects on the summed per-task val loss)"},
        "provenance": {"source": "bench.rescore_full_body_extra.evaluate (bench.score cannot read this run's "
                                 "metrics.txt: the joint trainer logs per-task losses, not the benchmark metric)",
                       "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                       "info": info},
        "status": "ok",
        "notes": "joint multi-task fine-tune; this cell is the joint model's own full-body head, comparable with the "
                 "single-task fine-tune row of the same MAE",
    }, open(out, "w"), indent=2)
    print(f"[wrote] {out}  mpjpe_mm={m['mpjpe_mm_recomputed']:.2f}  step={info['step']}  n_windows={info['n_windows']}")
