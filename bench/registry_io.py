"""Load the three registries and expose small helpers used across bench/."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Tuple

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG_DIR = os.path.join(REPO, "bench", "registry")
RUNS_ROOT = os.path.join(REPO, "runs")


def _load(name: str) -> Dict[str, Any]:
    with open(os.path.join(REG_DIR, name)) as f:
        return yaml.safe_load(f)


def load_tasks() -> Dict[str, Any]:
    from imu_lm.utils.data_root import remap_tree      # IMUBENCH_DATA_ROOT -> dataset paths
    return remap_tree(_load("tasks.yaml"))


def load_encoders() -> Dict[str, Any]:
    """encoders.yaml plus one card per file in encoders.d/ (how a submission registers its encoder:
    a file `encoders.d/<id>.yaml` holding `{<id>: {...card...}}`). A card id may appear only once."""
    encs = _load("encoders.yaml")["encoders"]
    extra_dir = os.path.join(REG_DIR, "encoders.d")
    if os.path.isdir(extra_dir):
        for fn in sorted(os.listdir(extra_dir)):
            if not fn.endswith((".yaml", ".yml")):
                continue
            with open(os.path.join(extra_dir, fn)) as f:
                cards = yaml.safe_load(f) or {}
            for k, v in cards.items():
                if k in encs:
                    raise ValueError(f"encoder id {k!r} in encoders.d/{fn} is already registered")
                encs[k] = v
    return encs


def load_protocols() -> Dict[str, Any]:
    return _load("protocols.yaml")


def task_ids(tasks: Dict[str, Any] | None = None) -> List[str]:
    tasks = tasks or load_tasks()
    # a task card with `enabled: false` is kept for the record but excluded everywhere
    return [k for k, v in sorted(tasks["tasks"].items(), key=lambda kv: kv[1].get("order", 999))
            if v.get("enabled", True)]


def grid_cells(tasks: Dict[str, Any] | None = None) -> List[Tuple[str, str, str]]:
    """All (task, encoder, protocol) cells of the controlled grid."""
    tasks = tasks or load_tasks()
    g = tasks["grid"]
    cells = []
    for t in task_ids(tasks):
        # `in_grid: false` cards are eval-condition variants of a base task
        # (e.g. the same task fed in a rotated coordinate frame). They are
        # scored like any other cell, but they are NOT part of the controlled
        # grid -- `--fill-missing` must not try to run all seven grid encoders
        # on them.
        if tasks["tasks"][t].get("in_grid") is False:
            continue
        for e in g["encoders"]:
            for p in g["protocols"]:
                cells.append((t, e, p))
        cells.append((t, g["scratch_encoder"], "scratch"))
    return cells


def run_name(task: str, encoder: str, protocol: str, seed: int) -> str:
    return f"bench_{task}__{encoder}__{protocol}__s{seed}"


def parse_run_name(name: str):
    if not name.startswith("bench_"):
        return None
    body = name[len("bench_"):]
    parts = body.split("__")
    if len(parts) != 4 or not parts[3].startswith("s"):
        return None
    return parts[0], parts[1], parts[2], int(parts[3][1:])


# ---- dotted-key helpers ---------------------------------------------------
def dget(d: Dict[str, Any], dotted: str, default=None):
    cur = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def dset(d: Dict[str, Any], dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = copy.deepcopy(value)


def ddel(d: Dict[str, Any], dotted: str) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return
        cur = cur[k]
    if isinstance(cur, dict):
        cur.pop(keys[-1], None)


def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base
