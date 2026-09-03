"""Microgesture scorer: OpenWatch paper protocol (1 s windows, stride 20,
6 windows/clip, k=3 clip voting) on the contamination-safe posture-stratified
split; val AND test are scored, test is the leaderboard number.
Wraps scripts/eval_openwatch_windowed_protocol.py::eval_run unchanged.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

from bench.registry_io import REPO
from bench.score.common import load_ckpt_meta

sys.path.insert(0, REPO)


def score(run_dir: str, ckpt_path: str, task: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    from scripts.eval_openwatch_windowed_protocol import eval_run   # noqa: E402
    from imu_lm.utils.helpers import cfg_get

    if os.path.basename(ckpt_path) != "best.pt":
        raise RuntimeError("microgesture scorer scores checkpoints/best.pt (val macro-F1 selection); "
                           f"got {ckpt_path}")
    meta = load_ckpt_meta(ckpt_path)
    cfg = meta["cfg"]
    obj_key = cfg_get(cfg, ["objective", "type"], "openwatch_gesture_probe")
    run_name = os.path.basename(run_dir.rstrip("/"))

    out: Dict[str, float] = {}
    info: Dict[str, Any] = {"checkpoint_step": meta["step"]}
    for split in ("val", "test"):
        res = eval_run(run_name, run_name, obj_key, split=split)
        if res is None:
            raise RuntimeError(f"eval_run returned None for {run_name} / {split}")
        prefix = "" if split == "test" else "val_"
        out[f"{prefix}clip_macro_f1"] = float(res["clip"]["macro_f1"])
        out[f"{prefix}clip_acc"] = float(res["clip"]["acc"])
        out[f"{prefix}window_macro_f1"] = float(res["window"]["macro_f1"])
        out[f"{prefix}window_acc"] = float(res["window"]["acc"])
        info[f"n_clips_{split}"] = int(res["n_clips"])
        info[f"n_windows_{split}"] = int(res["n_windows"])
    return out, info
