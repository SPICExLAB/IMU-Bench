"""Scoring for TLIO / Air-IO under the benchmark's WRIST-FRAME odometry protocol.

Both networks are made to output wrist-frame (FLU) pelvis velocity on the benchmark's grid -- 8 patches x 4
sub-steps per 2.56 s window -- and are then scored by bench/score/odometry_local.py::reconstruct_batch itself:
the predicted velocities are rotated to the world frame with the ground-truth orientation at the window start
propagated by integrating the raw gyroscope, summed into a trajectory and compared by ATE. Same function, same
73,515 windows, same metric definitions as every other row of the odometry table.
"""
import os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from bench.score.odometry_local import reconstruct_batch      # noqa: E402


def score(v_flu, va, ps=16, chunk=4096):
    """v_flu [N, 8, 4, 3] predicted wrist-frame velocity; va: dict with Y, GY, R0, POS."""
    ates, ferrs, plens = [], [], []
    for i in range(0, len(v_flu), chunk):
        a, f, p = reconstruct_batch(v_flu[i:i + chunk], va["GY"][i:i + chunk], va["R0"][i:i + chunk], va["POS"][i:i + chunk], ps)
        ates.append(a); ferrs.append(f); plens.append(p)
    ates, ferrs, plens = map(np.concatenate, (ates, ferrs, plens))
    moving = plens > 0.1
    return {"ate_m": float(np.mean(ates)),
            "drift_pct": float(np.mean(ferrs[moving] / plens[moving]) * 100.0) if moving.any() else float("nan"),
            "vel_mse": float(np.mean((v_flu - va["Y"]) ** 2))}


def write_result(encoder, metrics, info, notes):
    from bench.schema import Result, save_result, stamp_provenance
    r = Result(task="odometry_nymeria_local", encoder=encoder, protocol="external", seed=0, metrics=metrics,
               primary_metric="ate_m", eval_split="val", selection={"split": "val", "monitor": "ate_m", "mode": "min"},
               job_ids={"train": os.environ.get("SLURM_JOB_ID")},
               provenance=stamp_provenance(f"scorer:baselines/odometry/{info['script']}", info=info), status="ok", notes=notes)
    print("  ->", save_result(r), flush=True)
