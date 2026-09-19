"""Result-JSON contract for IMU-Bench.

Every number that appears on the leaderboard MUST come from one of these files.
No hand-typed numbers. A scorer produces exactly one Result per
(task, encoder, protocol, seed) cell and writes it with `save_result`.

    bench/results/<task>/<encoder>__<protocol>__s<seed>.json

Fields
------
schema_version   int, bump on incompatible change
task             task id from registry/tasks.yaml
encoder          encoder id from registry/encoders.yaml
protocol         probe | finetune | scratch | external
seed             int (trainer seed)
run              run-dir name under runs/ (None for external baselines)
job_ids          {"train": ..., "score": ...} SLURM ids when known
checkpoint       which checkpoint file was scored (best.pt / best_mpjpe.pt / ...)
checkpoint_step  step of that checkpoint if known
selection        {"split": "val", "monitor": "val_loss"|"mpjpe_mm"|"macro_f1"...}
eval_split       split the metrics below were computed on ("val" or "test")
metrics          flat {metric_key: float}; must contain the task's primary metric
primary_metric   key into metrics (copied from the task card for self-containment)
provenance       {"source": "scorer:<module>@<git-sha>" | "harvest:<file>",
                  "date": ISO-8601, "scorer_args": {...}}
status           "ok" | "partial" | "invalid" (invalid = keep for record, hide from ranking)
notes            free text (caveats that must travel with the number)
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BENCH_RESULTS_ROOT redirects every read and write to another results tree; used by the
# selection-free control (bench/rescore_noselection.py), which scores the same runs from
# latest.pt into bench/results_nosel without touching the reported cells.
RESULTS_ROOT = os.environ.get("BENCH_RESULTS_ROOT") or os.path.join(REPO, "bench", "results")


def git_sha(short: bool = True) -> str:
    try:
        out = subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short" if short else "HEAD", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.call(["git", "-C", REPO, "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Result:
    task: str
    encoder: str
    protocol: str
    seed: int
    metrics: Dict[str, float]
    primary_metric: str
    eval_split: str = "val"
    run: Optional[str] = None
    checkpoint: Optional[str] = None
    checkpoint_step: Optional[int] = None
    selection: Dict[str, Any] = field(default_factory=lambda: {"split": "val", "monitor": "val_loss"})
    job_ids: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    notes: str = ""
    schema_version: int = SCHEMA_VERSION

    # ---- helpers ---------------------------------------------------------
    @property
    def cell_id(self) -> str:
        return f"{self.encoder}__{self.protocol}__s{self.seed}"

    @property
    def path(self) -> str:
        return os.path.join(RESULTS_ROOT, self.task, self.cell_id + ".json")

    def validate(self) -> None:
        if self.primary_metric not in self.metrics:
            raise ValueError(f"{self.task}/{self.cell_id}: primary metric {self.primary_metric!r} "
                             f"missing from metrics {sorted(self.metrics)}")
        for k, v in self.metrics.items():
            if v is None:
                continue
            if not isinstance(v, (int, float)):
                raise TypeError(f"{self.task}/{self.cell_id}: metric {k}={v!r} is not numeric")
        if self.status not in ("ok", "partial", "invalid"):
            raise ValueError(f"bad status {self.status!r}")
        if self.protocol not in ("probe", "finetune", "scratch", "external", "legacy_valloss"):
            raise ValueError(f"bad protocol {self.protocol!r}")

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Result":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def stamp_provenance(source: str, **extra: Any) -> Dict[str, Any]:
    p = {"source": source, "date": now_iso(), "git": git_sha()}
    p.update(extra)
    return p


def save_result(r: Result, overwrite: bool = True) -> str:
    r.validate()
    os.makedirs(os.path.dirname(r.path), exist_ok=True)
    if os.path.exists(r.path) and not overwrite:
        raise FileExistsError(r.path)
    tmp = r.path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(r.to_dict(), f, indent=2, sort_keys=False)
    os.replace(tmp, r.path)
    return r.path


def load_result(path: str) -> Result:
    with open(path) as f:
        return Result.from_dict(json.load(f))


def iter_results(root: str = RESULTS_ROOT):
    for task in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        tdir = os.path.join(root, task)
        if not os.path.isdir(tdir) or task.startswith("_"):
            continue
        for fn in sorted(os.listdir(tdir)):
            if fn.endswith(".json"):
                try:
                    yield load_result(os.path.join(tdir, fn))
                except Exception as e:  # keep the leaderboard robust to one bad file
                    print(f"[schema] skipping {tdir}/{fn}: {e}")
