"""Language-alignment scorer: bidirectional retrieval (R@1/5/10, MRR) at a
fixed 100-item gallery and the full gallery, first-caption protocol, on the
HumanML3D TEST split (val also scored for reference).
Runs scripts/eval_retrieval_alignment.py as a subprocess (it is a CLI script
with its own arg parsing) and flattens its JSON.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import shutil
import tempfile
from typing import Any, Dict, Tuple

from bench.registry_io import REPO
from bench.score.common import keep_raw_outputs, save_units, units_enabled, units_safely


def _flatten(js: Dict[str, Any], prefix: str) -> Dict[str, float]:
    p = js["protocols"]["first"] if "first" in js["protocols"] else js["protocols"]["all"]
    g100 = p["fixed"]["100"]
    full = p["full"]
    out = {
        f"{prefix}r1_i2t_g100": float(g100["imu2text"]["R@1"]),
        f"{prefix}r1_t2i_g100": float(g100["text2imu"]["R@1"]),
        f"{prefix}r5_i2t_g100": float(g100["imu2text"]["R@5"]),
        f"{prefix}r5_t2i_g100": float(g100["text2imu"]["R@5"]),
        f"{prefix}r10_i2t_g100": float(g100["imu2text"]["R@10"]),
        f"{prefix}r10_t2i_g100": float(g100["text2imu"]["R@10"]),
        f"{prefix}mrr_i2t_g100": float(g100["imu2text"]["MRR"]),
        f"{prefix}mrr_t2i_g100": float(g100["text2imu"]["MRR"]),
        f"{prefix}r1_i2t_full": float(full["imu2text"]["R@1"]),
        f"{prefix}r1_t2i_full": float(full["text2imu"]["R@1"]),
        f"{prefix}mrr_i2t_full": float(full["imu2text"]["MRR"]),
        f"{prefix}mrr_t2i_full": float(full["text2imu"]["MRR"]),
        f"{prefix}medr_i2t_full": float(full["imu2text"]["MedR"]),
        f"{prefix}medr_t2i_full": float(full["text2imu"]["MedR"]),
    }
    out[f"{prefix}r1_g100"] = 0.5 * (out[f"{prefix}r1_i2t_g100"] + out[f"{prefix}r1_t2i_g100"])
    return out


def score(run_dir: str, ckpt_path: str, task: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    ckpt_name = os.path.basename(ckpt_path)[:-3]
    run_name = os.path.basename(run_dir.rstrip("/"))
    out: Dict[str, float] = {}
    info: Dict[str, Any] = {}
    os.makedirs(os.environ.get('BENCH_TMP') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'bench', 'results', '_tmp'), exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="bench_lang_", dir=os.environ.get("BENCH_TMP") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench", "results", "_tmp"))
    units_npz = None
    for split in ("test", "val"):
        oj = os.path.join(tmpdir, f"{run_name}__{split}.json")
        cmd = [sys.executable, os.path.join(REPO, "scripts", "eval_retrieval_alignment.py"),
               "--model", "ours", "--run", run_dir, "--ckpt", ckpt_name, "--split", split,
               "--gallery-sizes", "100", "--n-subsets", "200", "--seed", "42",
               "--label", f"bench {run_name} {split}", "--out", oj]
        if split == task.get("eval_split", "test") and units_enabled():
            units_npz = os.path.join(tmpdir, f"{run_name}__{split}_units.npz")
            cmd += ["--dump-units", units_npz]     # per-query ranks for the bootstrap sidecar
        subprocess.run(cmd, check=True, cwd=REPO)
        with open(oj) as f:
            js = json.load(f)
        out.update(_flatten(js, "" if split == "test" else "val_"))
        info[f"n_items_{split}"] = js.get("n_items")
        info["checkpoint_step"] = js.get("meta", {}).get("ckpt_step")
        # keep the raw evaluator JSON next to the bench results for auditability
        if keep_raw_outputs():
            keep_dir = os.path.join(REPO, "bench", "results", "_raw", "language_humanml3d")
            os.makedirs(keep_dir, exist_ok=True)
            shutil.move(oj, os.path.join(keep_dir, f"{run_name}__{ckpt_name}__{split}.json"))
    if units_npz and os.path.exists(units_npz):
        units_safely(_save_language_units, run_dir, ckpt_path, task, units_npz, out, info.get("checkpoint_step"))
        os.remove(units_npz)
    if not keep_raw_outputs():              # rescoring: leave nothing behind in the tmp dir
        shutil.rmtree(tmpdir, ignore_errors=True)
    return out, info


def _save_language_units(run_dir, ckpt_path, task, npz_path, out, step, G: int = 100, mode: str = "first"):
    """Per-QUERY sidecar for the primary metric r1_g100 (mean of i2t and t2i R@1 pooled over the
    200 random G=100 galleries, first-caption protocol). Units: every i2t query (item x gallery)
    and every t2i query (unique caption x gallery); hit = rank 1. Group = the caption: an i2t
    query belongs to its item's first caption, a t2i query to its caption, so items that share a
    caption (and the caption's own queries) are resampled together."""
    import numpy as np
    z = np.load(npz_path)
    item_key = z["item_key"].astype(str)
    item_cap = z[f"{mode}__item_cap"]
    p = f"{mode}__g{G}__"
    i_item, i_rank, i_sub = z[p + "i2t_item"], z[p + "i2t_rank"], z[p + "i2t_subset"]
    t_cap, t_rank, t_sub = z[p + "t2i_cap"], z[p + "t2i_rank"], z[p + "t2i_subset"]
    keys = np.concatenate([np.char.add(np.char.add(np.char.add("i2t|s", i_sub.astype(str)), "|"), item_key[i_item]),
                           np.char.add(np.char.add(np.char.add("t2i|s", t_sub.astype(str)), "|cap"), t_cap.astype(str))])
    groups = np.concatenate([np.char.add("cap", item_cap[i_item].astype(str)), np.char.add("cap", t_cap.astype(str))])
    return save_units(run_dir, ckpt_path, task, kind="r1_two_dirs", metric_scale=100.0,
               arrays={"unit_key": keys,
                       "hit": np.concatenate([i_rank <= 1, t_rank <= 1]).astype(np.int8),
                       "direction": np.concatenate([np.zeros(len(i_rank), np.int8), np.ones(len(t_rank), np.int8)]),
                       "rank": np.concatenate([i_rank, t_rank]),
                       "subset": np.concatenate([i_sub, t_sub])},
               groups={"caption": groups},
               metric_reported=out.get(task["metrics"]["primary"]), checkpoint_step=step,
               notes=f"unit = retrieval query in one of 200 random G={G} galleries ({mode}-caption protocol); "
                     "direction 0 = IMU->text, 1 = text->IMU; group = first caption (shared by duplicate-caption items)",
               extra_meta={"n_items": int(len(item_key)), "n_captions": int(z[f"{mode}__n_captions"]),
                           "gallery": G, "caption_mode": mode})
