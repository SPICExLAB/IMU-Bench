#!/usr/bin/env python3
"""Submit benchmark cells to SLURM (train -> score in one job), or score-only jobs.

    python -m bench.launch --cell arm_pose_nymeria focal finetune [--seed 42] [--dry-run]
    python -m bench.launch --score-only --cell pd_pads mae_11src probe --run vit1d_pd_probe_pads_mae
    python -m bench.launch --fill-missing [--tasks t1 t2] [--seeds 42] [--dry-run]
    python -m bench.launch --harvest-external          # write result JSONs for external baseline rows

--fill-missing decides per grid cell:
   result JSON exists (status ok)             -> skip
   job with that name in squeue               -> skip
   legacy_runs mapping in tasks.yaml          -> score-only job on the legacy run  (provenance legacy†)
   bench run dir has a checkpoint             -> score-only job
   else                                        -> train+score job
Every submission is appended to bench/generated/submissions.tsv (date, jobid, kind, task, encoder,
protocol, seed, run, script) so nothing is lost.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from bench.make_config import write_config
from bench.registry_io import REPO, grid_cells, load_encoders, load_tasks, run_name, task_ids
from bench.schema import RESULTS_ROOT, Result, save_result, stamp_provenance
from bench.score.common import resolve_ckpt

SLURM_DIR = os.path.join(REPO, "bench", "generated", "slurm")
SUBMISSIONS = os.path.join(REPO, "bench", "generated", "submissions.tsv")
# Default = general GPU queue; `--buyin` (or BENCH_PARTITION=b53017) targets the group's
# buy-in node qgpu2015 (4x A100, account b53017 -> QOS buyin; cannot be combined with gengpu).
ACCOUNT = os.environ.get("BENCH_ACCOUNT", "p32139")
PARTITION = os.environ.get("BENCH_PARTITION", "gengpu")
VENV = "TransfHAR_env/bin/activate"

_HEADER = """#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --time={time}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task={cpus} --mem={mem}
#SBATCH --job-name={job}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd {repo}
source {venv}
mkdir -p logs
export BENCH_TMP={repo}/bench/generated/tmp
mkdir -p "$BENCH_TMP"
# CLIP text tower (language task) lives in the project HF cache; offline everywhere.
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export WANDB_MODE=${{WANDB_MODE:-online}}
echo "[bench] job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
"""


def _write(path: str, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)
    os.chmod(path, 0o755)


def _sbatch(path: str, dry: bool, dep: Optional[str] = None) -> Optional[int]:
    cmd = ["sbatch"]
    if dep:
        cmd += [f"--dependency={dep}"]
    cmd.append(path)
    if dry:
        print("[dry-run]", " ".join(cmd))
        return None
    out = subprocess.check_output(cmd, text=True, cwd=REPO).strip()
    jid = int(out.split()[-1])
    print(f"[sbatch] {jid}  {os.path.basename(path)}")
    return jid


def _log_submission(jid, kind, task, enc, proto, seed, run, script):
    if jid is None:          # dry-run
        return
    new = not os.path.exists(SUBMISSIONS)
    with open(SUBMISSIONS, "a") as f:
        if new:
            f.write("date\tjob_id\tkind\ttask\tencoder\tprotocol\tseed\trun\tscript\n")
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')}\t{jid}\t{kind}\t{task}\t{enc}\t{proto}\t{seed}\t{run}\t{script}\n")


def train_script(task_id, enc, proto, seed, tasks, encoders) -> Tuple[str, str]:
    name, cfg_path = write_config(task_id, enc, proto, seed, tasks=tasks, encoders=encoders)
    task = tasks["tasks"][task_id]
    sl = task.get("slurm", {})
    body = _HEADER.format(account=ACCOUNT, partition=PARTITION, gres=sl.get("gres", "gpu:1"),
                          time=sl.get("time", "06:00:00"), cpus=sl.get("cpus", 8), mem=sl.get("mem", "64G"),
                          job=name, repo=REPO, venv=VENV)
    body += f"""
# ---- train ----------------------------------------------------------------
# twin guard: the same cell may be queued on two partitions; the second copy exits if the
# first already produced the result or is RUNNING right now.
if [ -f "{RESULTS_ROOT}/{task_id}/{enc}__{proto}__s{seed}.json" ]; then echo "[bench] result exists -> skip"; exit 0; fi
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then echo "[bench] twin RUNNING elsewhere -> skip"; exit 0; fi
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \\
    --config configs/base.yaml \\
    --model-config {os.path.relpath(cfg_path, REPO)} \\
    --run-name {name}

# ---- score ----------------------------------------------------------------
python -m bench.score --task {task_id} --encoder {enc} --protocol {proto} --seed {seed}
python -m bench.leaderboard && python -m bench.leaderboard_html || true
echo "[bench] done $(date -Is)"
"""
    path = os.path.join(SLURM_DIR, name + ".sh")
    _write(path, body)
    return name, path


def score_script(task_id, enc, proto, seed, run, tasks, legacy: bool) -> Tuple[str, str]:
    job = f"bscore_{task_id}__{enc}__{proto}" + (f"__s{seed}" if seed != 42 else "")
    task = tasks["tasks"][task_id]
    sl = task.get("slurm", {})
    body = _HEADER.format(account=ACCOUNT, partition=PARTITION, gres="gpu:1", time="02:00:00",
                          cpus=sl.get("cpus", 8), mem=sl.get("mem", "64G"), job=job, repo=REPO, venv=VENV)
    body += f"""
python -m bench.score --task {task_id} --encoder {enc} --protocol {proto} --seed {seed} --run {run} {'--legacy' if legacy else ''}
python -m bench.leaderboard && python -m bench.leaderboard_html || true
echo "[bench] done $(date -Is)"
"""
    path = os.path.join(SLURM_DIR, job + ".sh")
    _write(path, body)
    return job, path


def squeue_names() -> Dict[str, str]:
    try:
        out = subprocess.check_output(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j|%T"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        return {}
    d = {}
    for line in out.splitlines():
        if "|" in line:
            n, s = line.rsplit("|", 1)
            d[n.strip()] = s.strip()
    return d


def result_ok(task_id, enc, proto, seed) -> bool:
    p = os.path.join(RESULTS_ROOT, task_id, f"{enc}__{proto}__s{seed}.json")
    if not os.path.exists(p):
        return False
    import json
    with open(p) as f:
        d = json.load(f)
    return d.get("status") == "ok"


def fill_missing(task_filter: List[str], seeds: List[int], dry: bool, tasks, encoders, allow_legacy=True, no_train=False):
    q = squeue_names()
    os.makedirs(SLURM_DIR, exist_ok=True)
    n_sub = 0
    for (t, e, p) in grid_cells(tasks):
        if task_filter and t not in task_filter:
            continue
        for seed in seeds:
            if result_ok(t, e, p, seed):
                continue
            rn = run_name(t, e, p, seed)
            if rn in q or f"bscore_{t}__{e}__{p}" + (f"__s{seed}" if seed != 42 else "") in q:
                print(f"[queue] {rn} already {q.get(rn) or 'queued'}")
                continue
            legacy = (tasks["tasks"][t].get("legacy_runs") or {}).get(f"{e}/{p}") if (seed == 42 and allow_legacy) else None
            if legacy and os.path.isdir(os.path.join(REPO, "runs", legacy)) and resolve_ckpt(os.path.join(REPO, "runs", legacy), tasks["tasks"][t].get("checkpoint", "best")):
                job, path = score_script(t, e, p, seed, legacy, tasks, legacy=True)
                jid = _sbatch(path, dry)
                _log_submission(jid, "score-legacy", t, e, p, seed, legacy, path)
                n_sub += 1
                continue
            rdir = os.path.join(REPO, "runs", rn)
            if os.path.isdir(rdir) and resolve_ckpt(rdir, tasks["tasks"][t].get("checkpoint", "best")):
                job, path = score_script(t, e, p, seed, rn, tasks, legacy=False)
                jid = _sbatch(path, dry)
                _log_submission(jid, "score", t, e, p, seed, rn, path)
                n_sub += 1
                continue
            if no_train:
                print(f"[skip-train] {rn}")
                continue
            name, path = train_script(t, e, p, seed, tasks, encoders)
            jid = _sbatch(path, dry)
            _log_submission(jid, "train+score", t, e, p, seed, name, path)
            n_sub += 1
    print(f"[fill-missing] submitted {n_sub} jobs" + (" (dry-run)" if dry else ""))


def harvest_external(tasks, encoders):
    n = 0
    for t in task_ids(tasks):
        task = tasks["tasks"][t]
        for e, spec in (task.get("external") or {}).items():
            if spec.get("scored"):      # produced by its own scorer script; never overwrite from the registry
                continue
            r = Result(task=t, encoder=e, protocol="external", seed=0, metrics=dict(spec["metrics"]),
                       primary_metric=task["metrics"]["primary"], eval_split=spec.get("eval_split", task.get("eval_split", "val")),
                       run=None, checkpoint=None, selection={"split": "n/a", "monitor": "n/a"},
                       provenance=stamp_provenance(f"harvest:{spec.get('source','')}"),
                       status=spec.get("status", "ok"), notes=spec.get("notes", ""))
            if r.primary_metric not in r.metrics:
                r.status = "partial"
                r.metrics[r.primary_metric] = None
                # schema requires primary present; mark partial with None
            save_result(r)
            n += 1
    print(f"[harvest-external] wrote {n} external results")


def harvest_pd_valloss(tasks, dry):
    """PD legacy runs used val_loss selection (not recipe-equivalent) -> score them as status=partial
    so the table is populated while the bench PD runs train."""
    t = "pd_pads"
    for key, run in (tasks["tasks"][t].get("legacy_runs_valloss") or {}).items():
        e, p = key.split("/")
        job, path = score_script(t, e, p, 42, run, tasks, legacy=True)
        # mark partial via a note; the scorer writes status ok — we post-process in a wrapper line
        with open(path, "a") as f:
            f.write(f"""python - <<'EOF'
import json,os
p=os.path.join("{RESULTS_ROOT}","{t}","{e}__{p}__s42.json")
d=json.load(open(p)); d["status"]="partial"; d["protocol"]="legacy_valloss"
d["notes"]=(d.get("notes","")+" LEGACY val_loss-selected checkpoint (bench recipe selects on val bal_acc); shown for reference only.").strip()
json.dump(d,open(os.path.join("{RESULTS_ROOT}","{t}","{e}__legacy_valloss_{p}__s42.json"),"w"),indent=2); os.remove(p)
EOF
python -m bench.leaderboard && python -m bench.leaderboard_html || true
""")
        jid = _sbatch(path, dry)
        _log_submission(jid, "score-legacy-partial", t, e, p, 42, run, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=3, metavar=("TASK", "ENCODER", "PROTOCOL"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", default=None, help="with --score-only: existing run to score")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--fill-missing", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    ap.add_argument("--no-legacy", action="store_true", help="fill-missing: ignore legacy_runs, always train")
    ap.add_argument("--no-train", action="store_true", help="fill-missing: only submit score-only jobs")
    ap.add_argument("--harvest-external", action="store_true")
    ap.add_argument("--harvest-pd-valloss", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--buyin", action="store_true", help="submit to the b53017 buy-in partition (account b53017)")
    ap.add_argument("--dependency", default=None, help="sbatch --dependency spec for --cell jobs, e.g. afterok:123")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--account", default=None)
    args = ap.parse_args()
    global ACCOUNT, PARTITION
    if args.buyin:
        ACCOUNT, PARTITION = "b53017", "b53017"
    if args.partition:
        PARTITION = args.partition
    if args.account:
        ACCOUNT = args.account
    tasks, encoders = load_tasks(), load_encoders()
    os.makedirs(SLURM_DIR, exist_ok=True)

    if args.harvest_external:
        harvest_external(tasks, encoders)
    if args.harvest_pd_valloss:
        harvest_pd_valloss(tasks, args.dry_run)
    if args.fill_missing:
        fill_missing(args.tasks, args.seeds, args.dry_run, tasks, encoders, allow_legacy=not args.no_legacy, no_train=args.no_train)
    if args.cell:
        t, e, p = args.cell
        if args.score_only:
            run = args.run or run_name(t, e, p, args.seed)
            job, path = score_script(t, e, p, args.seed, run, tasks, legacy=bool(args.run))
            jid = _sbatch(path, args.dry_run)
            _log_submission(jid, "score", t, e, p, args.seed, run, path)
        else:
            name, path = train_script(t, e, p, args.seed, tasks, encoders)
            jid = _sbatch(path, args.dry_run, dep=args.dependency)
            _log_submission(jid, "train+score", t, e, p, args.seed, name, path)


if __name__ == "__main__":
    main()
