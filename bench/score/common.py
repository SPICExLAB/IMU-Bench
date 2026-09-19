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


# ---- per-unit sidecars (bootstrap CIs) --------------------------------------------------------
# Every scorer hands its per-evaluation-unit predictions / errors to `save_units`, which writes
#     bench/results/_units/<task>/<encoder>__<protocol>__s<seed>.npz
# for bench/bootstrap.py. The cell is taken from `units_context` (set by bench.score.score_cell and
# bench/rescore_units.py); a direct scorer call on a bench_<task>__<enc>__<proto>__s<seed> run falls
# back to parsing the run name; anything else writes no sidecar. Sidecar problems never fail a
# scoring run: every error is caught and printed. BENCH_UNITS=0 disables sidecars entirely;
# BENCH_UNITS_DIR redirects them (tests).
import contextlib as _contextlib

UNITS_ROOT = os.path.join(os.environ.get("BENCH_RESULTS_ROOT") or os.path.join(REPO, "bench", "results"), "_units")
_UNITS_CTX: Dict[str, Any] = {}


@_contextlib.contextmanager
def units_context(task_id: str, encoder: str, protocol: str, seed: int, **extra: Any):
    """Tell the scorers which cell they are scoring (so they can name the sidecar).
    extra: run=..., keep_raw=False (rescoring: don't overwrite bench/results/_raw audit files)."""
    global _UNITS_CTX
    prev = _UNITS_CTX
    _UNITS_CTX = dict(task=task_id, encoder=encoder, protocol=protocol, seed=int(seed), **extra)
    try:
        yield _UNITS_CTX
    finally:
        _UNITS_CTX = prev


def nymeria_participant(session_file: str) -> str:
    """Nymeria session file '<date>_s<k>_<first>_<last>_act<k>_<hash>_...pkl' -> '<first>_<last>'."""
    m = re.match(r"^\d+_s\d+_(.+?)_act\d+_", os.path.basename(session_file))
    return m.group(1) if m else os.path.basename(session_file)


def classification_kind(primary: str) -> str:
    """bench/bootstrap.py metric kind for a classification primary metric
    (macro_f1 / subject_macro_f1 / clip_macro_f1 -> macro_f1, *bal_acc -> bal_acc, *acc -> acc)."""
    if primary.endswith("macro_f1"):
        return "macro_f1"
    if primary.endswith("bal_acc"):
        return "bal_acc"
    if primary.endswith("acc"):
        return "acc"
    raise ValueError(f"no sidecar metric kind for primary metric {primary!r}")


def units_safely(fn, *args, **kwargs):
    """Call a scorer's sidecar builder; any exception is printed, never raised (scoring must not
    fail because of a sidecar)."""
    if not units_enabled():
        return None
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        import traceback
        print(f"[units] WARNING: sidecar builder {getattr(fn, '__name__', fn)} failed: {e!r}")
        traceback.print_exc()
        return None


def units_enabled() -> bool:
    return os.environ.get("BENCH_UNITS", "1") != "0"


def keep_raw_outputs() -> bool:
    """False while rescoring for sidecars: scorers then leave bench/results/_raw untouched."""
    return bool(_UNITS_CTX.get("keep_raw", True))


def units_root() -> str:
    return os.environ.get("BENCH_UNITS_DIR") or UNITS_ROOT


def units_path(task_id: str, encoder: str, protocol: str, seed: int, root: Optional[str] = None) -> str:
    return os.path.join(root or units_root(), task_id, f"{encoder}__{protocol}__s{int(seed)}.npz")


def _units_cell(run_dir: str) -> Optional[Dict[str, Any]]:
    if _UNITS_CTX:
        return dict(_UNITS_CTX)
    from bench.registry_io import parse_run_name
    p = parse_run_name(os.path.basename(run_dir.rstrip("/")))
    if p is None:
        return None
    return {"task": p[0], "encoder": p[1], "protocol": p[2], "seed": p[3]}


def save_units(run_dir: str, ckpt_path: str, task: Dict[str, Any], *, kind: str,
               arrays: Dict[str, Any], groups: Dict[str, Any], metric_reported: Optional[float],
               metric_scale: float = 1.0, checkpoint_step: Optional[int] = None,
               notes: str = "", extra_meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Write the per-unit sidecar for the cell being scored. Returns its path (or None).

    kind           metric kind in bench/bootstrap.py (macro_f1 | mean | r1_two_dirs | ...)
                   that recomputes the task's PRIMARY metric from `arrays`
    arrays         per-unit arrays, all of equal length; must include `unit_key` (a stable id of
                   the evaluation unit, identical across encoders on the same task) plus the
                   inputs of `kind` (y_true/y_pred, value, hit/direction); extra per-unit arrays
                   are stored under `x__<name>`
    groups         {level: per-unit group ids}, ordered coarse -> fine (e.g. subject, session)
    metric_reported the primary metric this scorer is reporting, for the self-check
    metric_scale   multiply the kind's value by this to get the reported units (e.g. 100 for %)
    """
    if not units_enabled():
        return None
    try:
        import datetime as _dt
        import json as _json
        import numpy as np
        from bench import bootstrap as _bs

        cell = _units_cell(run_dir)
        if cell is None:
            print(f"[units] no cell context for {run_dir}; sidecar not written")
            return None
        primary = task["metrics"]["primary"]
        core = {"unit_key", "y_true", "y_pred", "value", "hit", "direction"}
        out: Dict[str, Any] = {}
        n = None
        for k, v in arrays.items():
            a = np.asarray(v)
            if a.dtype == object:
                a = a.astype(str)
            n = len(a) if n is None else n
            if len(a) != n:
                raise ValueError(f"array {k} has length {len(a)} != {n}")
            out[k if k in core else f"x__{k}"] = a
        if "unit_key" not in out:
            raise ValueError("arrays must include unit_key")
        for lvl, g in groups.items():
            if g is None:
                continue
            a = np.asarray(g).astype(str)
            if len(a) != n:
                raise ValueError(f"group {lvl} has length {len(a)} != {n}")
            out[f"group__{lvl}"] = a
        meta = {
            "schema": 1, **{k: cell[k] for k in ("task", "encoder", "protocol", "seed")},
            "run": os.path.basename(run_dir.rstrip("/")), "checkpoint": os.path.basename(ckpt_path),
            "checkpoint_step": checkpoint_step, "scorer": task.get("scorer"),
            "primary_metric": primary,
            "higher_is_better": (task.get("metrics", {}).get(primary) or {}).get("higher_is_better"),
            "eval_split": task.get("eval_split", "val"),
            "metric_kind": kind, "metric_scale": float(metric_scale),
            "group_levels": [lvl for lvl, g in groups.items() if g is not None],
            "n_units": int(n), "metric_reported": (None if metric_reported is None else float(metric_reported)),
            "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "slurm_job": os.environ.get("SLURM_JOB_ID"), "notes": notes,
        }
        meta.update(extra_meta or {})
        sc = _bs.Sidecar(path="", arrays=dict(out), meta=meta)
        meta["metric_from_units"] = sc.metric()
        meta["metric_as_reported_from_units"] = _bs.reported_style_value(sc)
        if metric_reported is not None:
            tol = _bs.TOLERANCE.get(primary, _bs.DEFAULT_TOL)
            d = abs(meta["metric_as_reported_from_units"] - float(metric_reported))
            meta["self_check"] = {"abs_diff": d, "tol": tol, "ok": bool(d <= tol)}
            if d > tol:
                print(f"[units] WARNING {cell['task']}/{cell['encoder']}__{cell['protocol']}: {primary} from units "
                      f"{meta['metric_as_reported_from_units']:.6f} != reported {float(metric_reported):.6f} (tol {tol})")
        out["meta_json"] = np.asarray(_json.dumps(meta))
        path = units_path(cell["task"], cell["encoder"], cell["protocol"], cell["seed"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path[:-4] + f".tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **out)
        os.replace(tmp, path)
        print(f"[units] wrote {os.path.relpath(path, REPO)} (n={n}, {primary} from units="
              f"{meta['metric_as_reported_from_units']:.6f})")
        return path
    except Exception as e:     # a sidecar must never break scoring
        import traceback
        print(f"[units] WARNING: sidecar not written for {run_dir}: {e!r}")
        traceback.print_exc()
        return None
