"""Re-score EXISTING benchmark cells to produce their per-unit sidecars
(bench/results/_units/<task>/<cell>.npz) without retraining and WITHOUT touching
the result JSONs.

For every result JSON of a task (default: the paper encoders + scratch), the
cell's own run and checkpoint are taken from the JSON (`run`, `checkpoint`),
the task's scorer is called directly inside `units_context(..., keep_raw=False)`
(so bench/results/_raw audit files are not rewritten either), and the freshly
recomputed primary metric is compared with the stored one. A difference is
logged (and the sidecar still written, flagged by its own self-check and by
`python -m bench.bootstrap --check`); the JSON is never rewritten.

    python -m bench.rescore_units --task har_mhealth                 # all finished paper cells of a task
    python -m bench.rescore_units --task pd_pads --cell mae_5src__probe__s42
    python -m bench.rescore_units --task har_mhealth --result-json bench/results/_superseded_legacy_run/har_mhealth__mae_5src__probe__s42.json
    python -m bench.rescore_units --write-slurm                      # bench/generated/slurm/units_rescore_<task>.sh

Resumable: a cell whose sidecar already matches the JSON's run / checkpoint and
passed its self-check is skipped (use --force to redo it).
Log: bench/results/_units/<task>/_rescore_log.jsonl (one JSON row per attempt).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import importlib
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from bench.registry_io import REPO, load_tasks
from bench.score.common import run_dir_of, units_context, units_path, units_root

PAPER_TASKS = ["arm_pose_nymeria", "full_body_pose_nymeria", "odometry_nymeria_local", "microgesture_openwatch",
               "language_humanml3d", "pd_pads", "har_hhar", "har_mhealth", "har_prism", "har_userwatch",
               "microgesture_openwatch_releasesplit"]
PAPER_ENCODERS = ["mae_5src", "limubert_5src", "simmtm_5src", "focal_5src", "relcon_5src",
                  "sslwear_5src", "dino_5src", "jepa_5src", "mae_5src_noaltattn", "mae_5src_acconly"]
PAPER_CELLS = [f"{e}__{p}" for e in PAPER_ENCODERS for p in ("probe", "finetune")] + ["scratch__scratch"]

SLURM_DIR = os.path.join(REPO, "bench", "generated", "slurm")


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _log(task: str, row: Dict[str, Any]) -> None:
    d = os.path.join(units_root(), task)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "_rescore_log.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


def _cells_for_task(task: str, all_cells: bool, only: Optional[List[str]]) -> List[str]:
    tdir = os.path.join(REPO, "bench", "results", task)
    out = []
    for p in sorted(glob.glob(os.path.join(tdir, "*.json"))):
        cell = os.path.basename(p)[:-5]
        enc_proto = cell.rsplit("__s", 1)[0]
        if only and cell not in only and enc_proto not in only:
            continue
        if not only and not all_cells and enc_proto not in PAPER_CELLS:
            continue
        out.append(p)
    return out


def _sidecar_up_to_date(sc_path: str, js: Dict[str, Any]) -> bool:
    if not os.path.exists(sc_path):
        return False
    try:
        from bench.bootstrap import load_sidecar
        m = load_sidecar(sc_path).meta
    except Exception:
        return False
    stored = (js.get("metrics") or {}).get(js.get("primary_metric"))
    return (m.get("run") == js.get("run") and m.get("checkpoint") == js.get("checkpoint")
            and bool((m.get("self_check") or {}).get("ok")) and stored is not None
            and m.get("metric_reported") is not None
            and abs(float(m["metric_reported"]) - float(stored)) <= float((m.get("self_check") or {}).get("tol", 0)))


def rescore_cell(task_id: str, result_json: str, tasks=None, force: bool = False,
                 max_windows: int = 0, dry_run: bool = False) -> Dict[str, Any]:
    tasks = tasks or load_tasks()
    task = tasks["tasks"][task_id]
    js = json.load(open(result_json))
    enc, proto, seed = js["encoder"], js["protocol"], int(js["seed"])
    primary = task["metrics"]["primary"]
    row: Dict[str, Any] = {"date": _now(), "task": task_id, "cell": f"{enc}__{proto}__s{seed}",
                           "result_json": os.path.relpath(result_json, REPO), "run": js.get("run"),
                           "checkpoint": js.get("checkpoint"), "primary_metric": primary,
                           "stored": (js.get("metrics") or {}).get(primary),
                           "slurm_job": os.environ.get("SLURM_JOB_ID")}
    if js.get("protocol") == "external" or not js.get("run"):
        row["status"] = "skip_external"
        return row
    run_dir = run_dir_of(js["run"])
    ckpt_path = os.path.join(run_dir, "checkpoints", js.get("checkpoint") or "best.pt")
    if not os.path.isdir(run_dir):
        row["status"] = "missing_run_dir"
        return row
    if not os.path.exists(ckpt_path):
        row["status"] = "missing_checkpoint"
        return row
    sc_path = units_path(task_id, enc, proto, seed)
    if not force and _sidecar_up_to_date(sc_path, js):
        row["status"] = "up_to_date"
        return row
    if dry_run:
        row["status"] = "would_rescore"
        return row

    mod = importlib.import_module(task["scorer"])
    t0 = time.time()
    kw = {"max_windows": max_windows} if (max_windows and task["scorer"].endswith("odometry_local")) else {}
    with units_context(task_id, enc, proto, seed, run=js["run"], keep_raw=False):
        metrics, info = mod.score(run_dir, ckpt_path, task, **kw)
        hook = getattr(mod, "write_units_after_save", None)
        if hook is not None:
            hook(run_dir, ckpt_path, task, dict(metrics))
    row["elapsed_s"] = round(time.time() - t0, 1)
    row["recomputed"] = metrics.get(primary)
    if kw:
        row["max_windows"] = max_windows
    try:
        from bench.bootstrap import check_sidecar, load_sidecar
        chk = check_sidecar(load_sidecar(sc_path), result_json=result_json)
        row.update({"sidecar": os.path.relpath(sc_path, REPO), "from_units": chk.get("from_units"),
                    "units_vs_json": chk.get("status"), "units_abs_diff": chk.get("abs_diff"),
                    "tol": chk.get("tol")})
    except Exception as e:
        row["sidecar_error"] = repr(e)
    st, rc = row.get("stored"), row.get("recomputed")
    if st is not None and rc is not None:
        row["scorer_abs_diff"] = abs(float(st) - float(rc))
    ok = row.get("units_vs_json") == "ok"
    row["status"] = "ok" if ok else ("RECOMPUTED_DIFFERS" if row.get("units_vs_json") == "MISMATCH"
                                     else (row.get("units_vs_json") or "no_sidecar"))
    if not ok:
        print(f"[rescore] !! {task_id}/{row['cell']}: stored {primary}={st} recomputed={rc} "
              f"from_units={row.get('from_units')} -> {row['status']} (result JSON left untouched)")
    return row


def rescore_task(task_id: str, all_cells: bool = False, only: Optional[List[str]] = None,
                 result_jsons: Optional[List[str]] = None, force: bool = False, max_windows: int = 0,
                 dry_run: bool = False) -> List[Dict[str, Any]]:
    tasks = load_tasks()
    paths = result_jsons or _cells_for_task(task_id, all_cells, only)
    rows = []
    print(f"[rescore] {task_id}: {len(paths)} cell(s)")
    for p in paths:
        try:
            row = rescore_cell(task_id, p, tasks=tasks, force=force, max_windows=max_windows, dry_run=dry_run)
        except Exception as e:
            traceback.print_exc()
            row = {"date": _now(), "task": task_id, "result_json": os.path.relpath(p, REPO),
                   "status": "error", "error": repr(e), "slurm_job": os.environ.get("SLURM_JOB_ID")}
        print(f"[rescore] {task_id} {row.get('cell', os.path.basename(p))}: {row['status']} "
              f"stored={row.get('stored')} from_units={row.get('from_units')}")
        if not dry_run:
            _log(task_id, row)
        rows.append(row)
    return rows


SBATCH = """#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --exclude=qgpu3006,qgpu3013,qgpu2014,qgpu2006
#SBATCH --job-name=units_rescore_{task}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Re-score the finished {task} cells to write per-unit sidecars (bench/results/_units/{task}/)
# for bootstrap CIs. No training; result JSONs are never rewritten (see bench/rescore_units.py).
# Resumable: resubmitting skips cells whose sidecar is already up to date.
set -euo pipefail
cd {repo}
source TransfHAR_env/bin/activate
mkdir -p logs
export BENCH_TMP={repo}/bench/generated/tmp
mkdir -p "$BENCH_TMP"
# CLIP text tower (language task) lives in the project HF cache; offline everywhere.
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export WANDB_MODE=disabled
echo "[units] job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
python -m bench.rescore_units --task {task} || echo "[units] some cells errored or differ (see bench/results/_units/{task}/_rescore_log.jsonl)"
python -m bench.bootstrap --check --task {task} || echo "[units] check reported mismatches (see above)"
echo "[units] done $(date -Is)"
"""


def write_slurm(tasks: List[str]) -> List[str]:
    os.makedirs(SLURM_DIR, exist_ok=True)
    out = []
    for t in tasks:
        p = os.path.join(SLURM_DIR, f"units_rescore_{t}.sh")
        with open(p, "w") as f:
            f.write(SBATCH.format(task=t, repo=REPO))
        os.chmod(p, 0o755)
        out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", default=None)
    ap.add_argument("--cell", action="append", default=None, help="enc__proto[__sSEED]; repeatable")
    ap.add_argument("--result-json", action="append", default=None,
                    help="rescore exactly this result JSON (e.g. a superseded one); needs one --task")
    ap.add_argument("--all-cells", action="store_true", help="every result JSON of the task, not just the paper grid")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-windows", type=int, default=0, help="odometry only: random subset (smoke tests)")
    ap.add_argument("--write-slurm", action="store_true")
    args = ap.parse_args(argv)

    if args.write_slurm:
        for p in write_slurm(args.task or PAPER_TASKS):
            print(p)
        return 0
    if not args.task:
        ap.error("--task is required")
    if args.result_json and len(args.task) != 1:
        ap.error("--result-json needs exactly one --task")
    bad = 0
    for t in args.task:
        rows = rescore_task(t, all_cells=args.all_cells, only=args.cell, result_jsons=args.result_json,
                            force=args.force, max_windows=args.max_windows, dry_run=args.dry_run)
        bad += sum(r["status"] in ("error", "RECOMPUTED_DIFFERS") for r in rows)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
