"""Selection-free control: score every finished cell from its FINAL checkpoint instead of the
best-on-validation one, into a parallel results tree (bench/results_nosel/).

Eight of the ten tasks select the reported checkpoint on the same split they report on (only
microgesture and IMU--text hold out a separate test split), so a reviewer cannot tell how much of
each number is the encoder and how much is best-of-N selection on the reported data. Re-scoring the
same runs from ``latest.pt`` removes the best-of-N step entirely: what remains is the checkpoint the
run happened to end on. The difference between the two tables is an upper bound on what selecting on
the reported split bought. Early stopping (patience on val loss) still decides *when* a run ends, so
this is a bound on the selection effect, not a split-clean re-run -- that needs retraining on a
three-way split, which would invalidate the scaling grid and is left for the camera-ready.

    python -m bench.rescore_noselection --task har_prism               # one task, in-process
    python -m bench.rescore_noselection --dry-run                      # what would be re-scored
    python -m bench.rescore_noselection --write-slurm                  # one sbatch file per task
    python -m bench.rescore_noselection --write-slurm --submit         # ... and submit them

Nothing under bench/results/ is written. Resumable: a cell already scored from the same run and
checkpoint step is skipped unless --force.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import traceback
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_ROOT = os.path.join(REPO, "bench", "results")          # read-only here
# Redirect every write (result JSONs and per-unit sidecars) before bench.schema is imported.
os.environ.setdefault("BENCH_RESULTS_ROOT", os.path.join(REPO, "bench", "results_nosel"))
OUT_ROOT = os.environ["BENCH_RESULTS_ROOT"]

from bench.registry_io import load_encoders, load_tasks                      # noqa: E402
from bench.rescore_units import PAPER_CELLS, PAPER_TASKS, SLURM_DIR          # noqa: E402
from bench.score import score_cell                                           # noqa: E402
from bench.score.common import run_dir_of                                    # noqa: E402


def _targets(task: str, all_cells: bool, only: Optional[List[str]]) -> List[str]:
    out = []
    for p in sorted(glob.glob(os.path.join(MAIN_ROOT, task, "*.json"))):
        cell = os.path.basename(p)[:-5]
        enc_proto = cell.rsplit("__s", 1)[0]
        if only and cell not in only and enc_proto not in only:
            continue
        if not only and not all_cells and enc_proto not in PAPER_CELLS:
            continue
        out.append(p)
    return out


def rescore_cell(task_id: str, result_json: str, tasks, encoders, force: bool = False) -> Dict:
    js = json.load(open(result_json))
    enc, proto, seed = js["encoder"], js["protocol"], int(js["seed"])
    row = {"task": task_id, "cell": f"{enc}__{proto}__s{seed}", "run": js.get("run"),
           "selected": js.get("checkpoint"), "stored": (js.get("metrics") or {}).get(js.get("primary_metric"))}
    if js.get("protocol") == "external" or not js.get("run"):
        row["status"] = "skip_external"
        return row
    run_dir = run_dir_of(js["run"])
    if not os.path.exists(os.path.join(run_dir, "checkpoints", "latest.pt")):
        row["status"] = "no_latest_ckpt"
        return row
    dest = os.path.join(OUT_ROOT, task_id, f"{enc}__{proto}__s{seed}.json")
    if not force and os.path.exists(dest):
        try:
            if json.load(open(dest)).get("run") == js["run"]:
                row["status"] = "up_to_date"
                return row
        except Exception:
            pass
    r = score_cell(task_id, enc, proto, seed=seed, run=js["run"], ckpt="latest",
                   note="selection-free control: final checkpoint, no best-of-N selection "
                        "(bench/rescore_noselection.py)", tasks=tasks, encoders=encoders)
    row["status"] = "ok"
    row["latest"] = r.metrics.get(r.primary_metric)
    row["checkpoint_step"] = r.checkpoint_step
    return row


def rescore_task(task_id: str, all_cells=False, only=None, force=False, dry_run=False) -> List[Dict]:
    tasks, encoders = load_tasks(), load_encoders()
    rows = []
    for p in _targets(task_id, all_cells, only):
        if dry_run:
            rows.append({"task": task_id, "cell": os.path.basename(p)[:-5], "status": "would_rescore"})
            print(f"[nosel] {task_id} {rows[-1]['cell']}: would_rescore")
            continue
        try:
            row = rescore_cell(task_id, p, tasks, encoders, force=force)
        except Exception as e:
            traceback.print_exc()
            row = {"task": task_id, "cell": os.path.basename(p)[:-5], "status": "error", "error": repr(e)}
        d = row.get("stored")
        l = row.get("latest")
        delta = f" best={d:.4f} latest={l:.4f} delta={l - d:+.4f}" if isinstance(d, float) and isinstance(l, float) else ""
        print(f"[nosel] {task_id} {row['cell']}: {row['status']}{delta}")
        rows.append(row)
    return rows


# The scorers only run inference, so the control can go to the CPU partition (--cpu) instead of
# queueing behind the training jobs on gengpu.
SBATCH = """#!/bin/bash
#SBATCH --account={account}
#SBATCH --partition={partition}
{gres}#SBATCH --time={time}
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task={cpus} --mem=64G
#SBATCH --job-name=nosel_{task}
#SBATCH --nice=2000
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Selection-free control for {task}: score the finished cells from latest.pt into
# bench/results_nosel/{task}/. No training, and bench/results/ is not touched.
set -euo pipefail
cd {repo}
source TransfHAR_env/bin/activate
mkdir -p logs
export BENCH_TMP={repo}/bench/generated/tmp
mkdir -p "$BENCH_TMP"
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export WANDB_MODE=disabled
export BENCH_RESULTS_ROOT={repo}/bench/results_nosel
echo "[nosel] job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"
python -m bench.rescore_noselection --task {task}
echo "[nosel] done $(date -Is)"
"""


def write_slurm(task_list: List[str], account: str, partition: str, cpu: bool = False) -> List[str]:
    os.makedirs(SLURM_DIR, exist_ok=True)
    out = []
    for t in task_list:
        p = os.path.join(SLURM_DIR, f"nosel_{t}.sh")
        with open(p, "w") as f:
            f.write(SBATCH.format(task=t, repo=REPO, account=account,
                                  partition=("normal" if cpu else partition),
                                  gres="" if cpu else "#SBATCH --gres=gpu:1\n#SBATCH --exclude=qgpu3006,qgpu3013,qgpu2014,qgpu2006\n",
                                  time="12:00:00" if cpu else "04:00:00", cpus=16 if cpu else 8))
        os.chmod(p, 0o755)
        out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", action="append", default=None)
    ap.add_argument("--cell", action="append", default=None, help="enc__proto[__sSEED]; repeatable")
    ap.add_argument("--all-cells", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-slurm", action="store_true")
    ap.add_argument("--submit", action="store_true", help="with --write-slurm: sbatch each file")
    ap.add_argument("--account", default="p32139")
    ap.add_argument("--partition", default="gengpu")
    ap.add_argument("--cpu", action="store_true", help="CPU partition (inference only); no GPU queueing")
    args = ap.parse_args(argv)

    if args.write_slurm:
        for p in write_slurm(args.task or PAPER_TASKS, args.account, args.partition, cpu=args.cpu):
            print(p if not args.submit else
                  subprocess.run(["sbatch", p], capture_output=True, text=True, cwd=REPO).stdout.strip())
        return 0
    bad = 0
    for t in (args.task or PAPER_TASKS):
        rows = rescore_task(t, all_cells=args.all_cells, only=args.cell, force=args.force, dry_run=args.dry_run)
        bad += sum(r["status"] == "error" for r in rows)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
