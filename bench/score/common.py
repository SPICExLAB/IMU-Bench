"""Shared helpers for the per-task scorers."""
from __future__ import annotations

import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from bench.registry_io import REPO, RUNS_ROOT

_LOGS_DIR = os.path.join(REPO, "logs")


def run_dir_of(run: str) -> str:
    return run if os.path.isdir(run) else os.path.join(RUNS_ROOT, run)


def resolve_ckpt(run_dir: str, prefer: str = "best") -> Optional[str]:
    """Return the checkpoint path to score.

    prefer="best"       -> best.pt, else best_mpjpe.pt (legacy pose runs), else None
    prefer="latest"     -> latest.pt
    prefer="best_mpjpe" -> best_mpjpe.pt
    """
    cdir = os.path.join(run_dir, "checkpoints")
    order = {
        "best": ["best.pt", "best_mpjpe.pt"],
        "latest": ["latest.pt"],
        "best_mpjpe": ["best_mpjpe.pt", "best.pt"],
    }[prefer]
    for fn in order:
        p = os.path.join(cdir, fn)
        if os.path.exists(p):
            return p
    return None


def load_ckpt_meta(ckpt_path: str) -> Dict[str, Any]:
    """Cheap-ish: torch.load on CPU and return {step, cfg, best_val_loss, best_val_mpjpe}."""
    import torch
    try:    # mmap: tensors stay on disk until touched -> reading `step`/`cfg` is cheap for metric-only scorers
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "step": int(obj.get("step", -1)),
        "cfg": obj.get("cfg", {}),
        "best_val_loss": obj.get("best_val_loss"),
        "best_val_mpjpe": obj.get("best_val_mpjpe"),
        "_obj": obj,
    }


_KV = re.compile(r"(\S+?)=(\S+)")


def parse_metrics_txt(run_dir: str, split: str = "val") -> List[Dict[str, Any]]:
    """Parse logs/metrics.txt into a list of dict rows for the given split."""
    p = os.path.join(run_dir, "logs", "metrics.txt")
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        for line in f:
            if f"split={split}" not in line:
                continue
            row: Dict[str, Any] = {}
            for k, v in _KV.findall(line.strip()):
                try:
                    row[k] = float(v) if k not in ("split",) else v
                except ValueError:
                    row[k] = v
            if "step" in row:
                row["step"] = int(row["step"])
            rows.append(row)
    return rows


def val_row_at_step(rows: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    """Val row logged at exactly `step`; falls back to the closest earlier row."""
    exact = [r for r in rows if r.get("step") == step]
    if exact:
        return exact[-1]
    earlier = [r for r in rows if r.get("step", -1) <= step]
    return earlier[-1] if earlier else None


def slurm_job_id_for_run(run: str) -> Optional[int]:
    """Newest logs/<run>_<jobid>.out -> jobid (train job)."""
    cands = glob.glob(os.path.join(_LOGS_DIR, f"{run}_*.out"))
    best: Tuple[float, Optional[int]] = (-1.0, None)
    pat = re.compile(r"^" + re.escape(run) + r"_(\d+)\.out$")
    for c in cands:
        m = pat.match(os.path.basename(c))
        if not m:
            continue
        mt = os.path.getmtime(c)
        if mt > best[0]:
            best = (mt, int(m.group(1)))
    return best[1]


def acc_only_slice(x, cfg):
    """Mirror Trainer._acc_only_slice for scorers that bypass the trainer."""
    from imu_lm.utils.helpers import cfg_get
    if bool(cfg_get(cfg, ["data", "acc_only"], False)):
        n = len(cfg_get(cfg, ["data", "sensor_columns"], []) or []) or 3
        if x.shape[1] > n:
            return x[:, :n]
    return x
