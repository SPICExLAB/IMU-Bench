"""Full-body pose scorer.

The primary metric is the root-relative MPJPE (mm) over SMPL joints 1..23 that
the trainer computes on the full validation set at every eval
(imu_lm/objectives/pose_1d.py::_compute_val_mpjpe). We read it from
logs/metrics.txt at the step of the selected checkpoint (best.pt under the
bench recipe = min val mpjpe; legacy runs: best_mpjpe.pt), so the number is
exactly the trainer's, not a re-implementation.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from bench.score.common import load_ckpt_meta, parse_metrics_txt, val_row_at_step


def score(run_dir: str, ckpt_path: str, task: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    meta = load_ckpt_meta(ckpt_path)
    step = meta["step"]
    rows = parse_metrics_txt(run_dir, "val")
    if not rows:
        raise RuntimeError(f"no val rows in {run_dir}/logs/metrics.txt")
    row = val_row_at_step(rows, step)
    if row is None or "mpjpe_mm" not in row:
        raise RuntimeError(f"no mpjpe_mm val row at step {step} in {run_dir}")
    metrics = {"mpjpe_mm": float(row["mpjpe_mm"])}
    if "pose_mse" in row:
        metrics["val_pose_mse"] = float(row["pose_mse"])
    # sanity: the min over the whole curve (what best_mpjpe.pt tracks)
    min_row = min((r for r in rows if "mpjpe_mm" in r), key=lambda r: r["mpjpe_mm"])
    info = {"checkpoint_step": step, "min_val_mpjpe_mm": float(min_row["mpjpe_mm"]),
            "min_val_mpjpe_step": int(min_row["step"]), "n_val_evals": len(rows)}
    return metrics, info
