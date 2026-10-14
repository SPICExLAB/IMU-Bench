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
from bench.score.common import classification_kind, load_ckpt_meta, save_units, units_safely

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
    per_split: Dict[str, Any] = {}
    for split in ("val", "test"):
        res = eval_run(run_name, run_name, obj_key, split=split, return_units=True)
        if res is None:
            raise RuntimeError(f"eval_run returned None for {run_name} / {split}")
        per_split[split] = res
        prefix = "" if split == "test" else "val_"
        out[f"{prefix}clip_macro_f1"] = float(res["clip"]["macro_f1"])
        out[f"{prefix}clip_acc"] = float(res["clip"]["acc"])
        out[f"{prefix}window_macro_f1"] = float(res["window"]["macro_f1"])
        out[f"{prefix}window_acc"] = float(res["window"]["acc"])
        info[f"n_clips_{split}"] = int(res["n_clips"])
        info[f"n_windows_{split}"] = int(res["n_windows"])
    units_safely(_save_microgesture_units, run_dir, ckpt_path, task, per_split, out, meta["step"])
    return out, info


def _save_microgesture_units(run_dir, ckpt_path, task, per_split, out, step):
    """Per-CLIP sidecar on the reported split (eval_split = test): true / voted label, subject.
    The val clips (a different length) ride along in the meta as `val_clips`.
    Group = the subject that defines the split: the canonical identity under the
    contamination-safe split, the raw release id otherwise (release-split task)."""
    from imu_lm.data.openwatch_loader import canonical_subject
    split = task.get("eval_split", "test")
    res = per_split[split]
    pc = res["per_clip"]
    safe = bool(res.get("contamination_safe", False))
    raw = [int(v) for v in pc["subject_id"]]
    canon = [int(canonical_subject(v)) for v in raw]
    subj = [f"id_{c if safe else r}" for c, r in zip(canon, raw)]
    keys = [os.path.relpath(p, task.get("dataset", {}).get("root", "/")) for p in pc["path"]]
    val = per_split.get("val", {}).get("per_clip") if split != "val" else None
    save_units(run_dir, ckpt_path, task, kind=classification_kind(task["metrics"]["primary"]),
               arrays={"unit_key": keys, "y_true": pc["label"], "y_pred": pc["pred"],
                       "subject_raw": [f"id_{r}" for r in raw], "subject_canonical": [f"id_{c}" for c in canon],
                       "position": pc["position"], "gesture": pc["gesture"], "n_windows": pc["n_windows"]},
               groups={"subject": subj},
               metric_reported=out.get(task["metrics"]["primary"]), checkpoint_step=step,
               notes=f"unit = {split} clip (k=3 window vote); subject = "
                     f"{'canonical identity (contamination-safe split)' if safe else 'raw release id'}",
               extra_meta={"units_split": split, "contamination_safe": safe,
                           "val_clips": ({"subject_raw": val["subject_id"], "y_true": val["label"],
                                          "y_pred": val["pred"]} if val else None)})
