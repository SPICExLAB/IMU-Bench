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
from bench.score.common import load_ckpt_meta


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
    cmd = [sys.executable, os.path.join(REPO, "scripts", "eval_pads_subject_level.py"),
           "--config", cfg_path, "--model-config", cfg_path, "--run", run_name,
           "--ckpt", ckpt_path, "--out", out_json]
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
    keep_dir = os.path.join(REPO, "bench", "results", "_raw", "pd_pads")
    os.makedirs(keep_dir, exist_ok=True)
    shutil.move(out_json, os.path.join(keep_dir, f"{run_name}__{os.path.basename(ckpt_path)[:-3]}.json"))
    return metrics, info
