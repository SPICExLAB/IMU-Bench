"""Score one benchmark cell and write bench/results/<task>/<cell>.json.

    python -m bench.score --task arm_pose_nymeria --encoder focal --protocol probe [--seed 42]
                          [--run <existing run name>]     # score a legacy run into this cell
                          [--ckpt best|latest|best_mpjpe] # override the task's checkpoint rule
                          [--provenance-note "..."]

The task card's `scorer:` names the module; every scorer returns
(metrics: dict, info: dict) and this wrapper does the bookkeeping.
"""
from __future__ import annotations

import argparse
import importlib
import os
import traceback
from typing import Any, Dict, Optional

from bench.registry_io import REPO, load_encoders, load_tasks, run_name as bench_run_name
from bench.schema import Result, save_result, stamp_provenance
from bench.score.common import resolve_ckpt, run_dir_of, slurm_job_id_for_run


def _train_end_reason(run_dir: str) -> str:
    """Last '[train] run complete: <reason>' line of logs/stdout.log, or 'unknown/incomplete'."""
    import glob as _glob
    run = os.path.basename(run_dir.rstrip("/"))
    cands = [os.path.join(run_dir, "logs", "stdout.log")]
    outs = _glob.glob(os.path.join(REPO, "logs", f"{run}_*.out"))
    import re as _re
    pat = _re.compile(r"^" + _re.escape(run) + r"_(\d+)\.out$")
    outs = sorted([o for o in outs if pat.match(os.path.basename(o))], key=os.path.getmtime)
    cands += outs
    reason = "unknown/incomplete"
    for p in cands:
        try:
            with open(p, errors="ignore") as f:
                for line in f:
                    if "[train] run complete:" in line:
                        reason = line.split("run complete:", 1)[1].strip()
        except OSError:
            pass
    return reason


def score_cell(task_id: str, encoder_id: str, protocol: str, seed: int = 42, run: Optional[str] = None,
               ckpt: Optional[str] = None, note: str = "", legacy: bool = False,
               tasks=None, encoders=None) -> Result:
    tasks = tasks or load_tasks()
    encoders = encoders or load_encoders()
    task = tasks["tasks"][task_id]
    run = run or bench_run_name(task_id, encoder_id, protocol, seed)
    run_dir = run_dir_of(run)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    prefer = ckpt or task.get("checkpoint", "best")
    ckpt_path = resolve_ckpt(run_dir, prefer)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint ({prefer}) in {run_dir}/checkpoints")

    mod = importlib.import_module(task["scorer"])
    metrics, info = mod.score(run_dir, ckpt_path, task)
    info["train_end"] = _train_end_reason(run_dir)

    sel = dict(task.get("selection", {}))
    if os.path.basename(ckpt_path) == "best_mpjpe.pt":
        sel = {"split": "val", "monitor": "mpjpe_mm", "mode": "min", "via": "best_mpjpe.pt (legacy)"}
    if os.path.basename(ckpt_path) == "latest.pt" and sel.get("mode") != "latest":
        sel = {"split": "none", "monitor": "fixed_budget", "mode": "latest"}

    r = Result(
        task=task_id, encoder=encoder_id, protocol=protocol, seed=int(seed),
        metrics={k: (float(v) if v is not None else None) for k, v in metrics.items()},
        primary_metric=task["metrics"]["primary"],
        eval_split=task.get("eval_split", "val"),
        run=run,
        checkpoint=os.path.basename(ckpt_path),
        checkpoint_step=info.pop("checkpoint_step", None),
        selection=sel,
        job_ids={"train": slurm_job_id_for_run(run), "score": os.environ.get("SLURM_JOB_ID")},
        provenance=stamp_provenance(f"scorer:{task['scorer']}", legacy_run=bool(legacy), info=info),
        status="ok",
        notes=note,
    )
    if legacy:
        r.notes = (r.notes + " " if r.notes else "") + "harvested from a pre-harness run with a recipe-equivalent config."
    save_result(r)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", default=None, help="score this existing run into the cell (legacy harvest)")
    ap.add_argument("--ckpt", default=None, choices=[None, "best", "latest", "best_mpjpe"])
    ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    r = score_cell(args.task, args.encoder, args.protocol, args.seed, run=args.run, ckpt=args.ckpt,
                   note=args.note, legacy=args.legacy)
    print(f"[bench.score] wrote {r.path}")
    print(f"[bench.score] {r.task} {r.cell_id}: {r.primary_metric}={r.metrics[r.primary_metric]}")


if __name__ == "__main__":
    main()
