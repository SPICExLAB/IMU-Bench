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
    for split in ("test", "val"):
        oj = os.path.join(tmpdir, f"{run_name}__{split}.json")
        cmd = [sys.executable, os.path.join(REPO, "scripts", "eval_retrieval_alignment.py"),
               "--model", "ours", "--run", run_dir, "--ckpt", ckpt_name, "--split", split,
               "--gallery-sizes", "100", "--n-subsets", "200", "--seed", "42",
               "--label", f"bench {run_name} {split}", "--out", oj]
        subprocess.run(cmd, check=True, cwd=REPO)
        with open(oj) as f:
            js = json.load(f)
        out.update(_flatten(js, "" if split == "test" else "val_"))
        info[f"n_items_{split}"] = js.get("n_items")
        info["checkpoint_step"] = js.get("meta", {}).get("ckpt_step")
        # keep the raw evaluator JSON next to the bench results for auditability
        keep_dir = os.path.join(REPO, "bench", "results", "_raw", "language_humanml3d")
        os.makedirs(keep_dir, exist_ok=True)
        shutil.move(oj, os.path.join(keep_dir, f"{run_name}__{ckpt_name}__{split}.json"))
    return out, info
