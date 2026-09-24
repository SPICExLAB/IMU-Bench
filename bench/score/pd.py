"""PADS Parkinson's scorer: window-level AND subject-level (mean-prob over all
of a subject's windows) balanced accuracy / macro-F1 on the val subjects.
Wraps scripts/eval_pads_subject_level.py; the run's own resolved config
(from the checkpoint) is used to rebuild the identical split.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import shutil
import tempfile
from typing import Any, Dict, Tuple

import yaml

from bench.registry_io import REPO
from bench.score.common import (classification_kind, keep_raw_outputs, load_ckpt_meta, save_units, units_enabled,
                                 units_safely)


def score(run_dir: str, ckpt_path: str, task: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    meta = load_ckpt_meta(ckpt_path)
    cfg = meta["cfg"]
    run_name = os.path.basename(run_dir.rstrip("/"))
    os.makedirs(os.environ.get('BENCH_TMP') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bench', 'results', '_tmp'), exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="bench_pd_", dir=os.environ.get("BENCH_TMP") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench", "results", "_tmp"))
    cfg_path = os.path.join(tmpdir, f"{run_name}_resolved.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    out_json = os.path.join(tmpdir, f"{run_name}.json")
    out_npz = os.path.join(tmpdir, f"{run_name}_units.npz")
    cmd = [sys.executable, os.path.join(REPO, "scripts", "eval_pads_subject_level.py"),
           "--config", cfg_path, "--model-config", cfg_path, "--run", run_name,
           "--ckpt", ckpt_path, "--out", out_json]
    if units_enabled():
        cmd += ["--dump-npz", out_npz]      # per-window / per-subject predictions for the sidecar
    subprocess.run(cmd, check=True, cwd=REPO)
    with open(out_json) as f:
        js = json.load(f)
    metrics = {
        "subject_bal_acc": float(js["subject_mean_prob"]["bal_acc"]),
        "subject_macro_f1": float(js["subject_mean_prob"]["macro_f1"]),
        "subject_acc": float(js["subject_mean_prob"]["acc"]),
        "subject_vote_bal_acc": float(js["subject_vote"]["bal_acc"]),
        "window_bal_acc": float(js["window"]["bal_acc"]),
        "window_macro_f1": float(js["window"]["macro_f1"]),
        "window_acc": float(js["window"]["acc"]),
    }
    conf = js.get("subject_confusion_mean_prob")
    if conf:   # sanity flag: how many classes does the subject-level classifier ever predict?
        metrics["n_classes_predicted"] = float(sum(1 for j in range(len(conf)) if sum(row[j] for row in conf) > 0))
    info = {"checkpoint_step": meta["step"], "n_windows": js.get("n_windows"), "n_subjects": js.get("n_subjects"),
            "subject_confusion_mean_prob": js.get("subject_confusion_mean_prob")}
    if keep_raw_outputs():
        keep_dir = os.path.join(REPO, "bench", "results", "_raw", "pd_pads")
        os.makedirs(keep_dir, exist_ok=True)
        shutil.move(out_json, os.path.join(keep_dir, f"{run_name}__{os.path.basename(ckpt_path)[:-3]}.json"))
    if os.path.exists(out_npz):
        units_safely(_save_pd_units, run_dir, ckpt_path, task, out_npz, metrics, meta["step"])
        os.remove(out_npz)
    if not keep_raw_outputs():              # rescoring: leave nothing behind in the tmp dir
        shutil.rmtree(tmpdir, ignore_errors=True)
    return metrics, info


def _save_pd_units(run_dir, ckpt_path, task, npz_path, metrics, step):
    """Sidecar at the unit of the task's PRIMARY metric, grouped by subject.

    pd_pads ranked on `subject_macro_f1` until 2026-09-22 and on `window_macro_f1` after it. The
    sidecar used to store the 94 subject units unconditionally while labelling them with whatever
    the primary metric was, so once the primary went window-level the bootstrap recomputed a WINDOW
    metric from SUBJECT predictions and its `--check` pass disagreed with the reported value. The
    unit now follows the primary metric; the grouping stays the subject either way, so a resample
    still draws whole participants.
    """
    import numpy as np
    z = np.load(npz_path)
    primary = task["metrics"]["primary"]
    kind = classification_kind(primary)
    if primary.startswith("window_"):
        win_subj = z["win_subject"].astype(str)
        save_units(run_dir, ckpt_path, task, kind=kind,
                   arrays={"unit_key": np.arange(len(win_subj)).astype(str),
                           "y_true": z["win_true"].astype(np.int64),
                           "y_pred": z["win_pred"].astype(np.int64)},
                   groups={"subject": win_subj},
                   metric_reported=metrics.get(primary), checkpoint_step=step,
                   notes="unit = val window; group = subject, so a resample draws whole participants")
        return
    subjects = z["subjects"].astype(str)
    win_subj = z["win_subject"].astype(str)
    n_win = np.array([int(np.sum(win_subj == s)) for s in subjects])
    save_units(run_dir, ckpt_path, task, kind=kind,
               arrays={"unit_key": subjects, "y_true": z["subj_true"].astype(np.int64),
                       "y_pred": z["subj_pred_mean"].astype(np.int64),
                       "pred_vote": z["subj_pred_vote"].astype(np.int64), "n_windows": n_win},
               groups={"subject": subjects},
               metric_reported=metrics.get(primary), checkpoint_step=step,
               notes="unit = val subject (mean-softmax over all of the subject\'s windows, argmax); "
                     "group = subject (units are already subjects)")
