#!/usr/bin/env python3
"""Score the Scratch odometry checkpoint on the canonical world-frame target.

Sibling to vis_odom_scratch.py, which reconstructs trajectories via gyro
integration for the wrist-FLU objective (odom_nymeria_real_wrist_flu). This
script instead scores a checkpoint trained on the world-frame objective
(odom_nymeria_real_wrist_worldframe_flu, see
imu_lm/data/nymeria_real_wrist_odom_worldframe_loader_FLU.py): each window's
velocity target is expressed in a per-window canonical frame (heading fixed
from net displacement), so reconstruction is a plain cumulative sum -- no
gyro/orientation tracking needed. Reuses eval_odom_nymeria_worldframe.py's
build_windows/reconstruct_and_score, and vis_odom_scratch.py's model-loading
and prediction code unchanged (architecture-agnostic to the target frame).

Usage
-----
    python scripts/vis_odom_scratch_worldframe.py \\
        --run vit1d_odom_worldframe_supervised_nymeria_altattn_scratch \\
        --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \\
        --out-dir data_vis/nymeria_odom_worldframe_probe_multidata --label Scratch
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_odom_nymeria_worldframe as wf
from vis_odom_scratch import load_scratch_model, predict_all


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="vit1d_odom_worldframe_supervised_nymeria_altattn_scratch")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--val-dir", default="/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-val-files", type=int, default=101)
    p.add_argument("--add-gravity", action="store_true", default=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-windows", type=int, default=3000,
                    help="Cap on validation windows scored (reconstruct_and_score has an "
                         "unvectorized per-timestep Python loop)")
    p.add_argument("--out-dir", default="data_vis/nymeria_odom_worldframe_probe_multidata")
    p.add_argument("--label", default="Scratch")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    enc, decoder, cfg, T, ps = load_scratch_model(
        os.path.join(args.runs_root, args.run), args.step, device
    )
    hop = T // 2

    va_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]
    va_sess = [r for f in va_files
               if (r := wf.load_session_local_vel_flu(f, target_hz=int(wf.FS),
                                                        add_gravity=args.add_gravity)) is not None]
    print(f"  sessions val={len(va_sess)}")

    keys = ["X", "Y", "POS"]
    va = dict(zip(keys, wf.build_windows(va_sess, cfg, T, ps, hop, int(wf.FS))))
    n_total = len(va["X"])
    print(f"  windows val={n_total}")

    if n_total > args.max_windows:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(n_total, size=args.max_windows, replace=False))
    else:
        idx = np.arange(n_total)

    preds = predict_all(enc, decoder, va["X"][idx], device, args.batch_size)

    ates, ferrs, plens = [], [], []
    for i in range(len(preds)):
        ate, ferr, plen = wf.reconstruct_and_score(preds[i], va["POS"][idx[i]], ps, target_hz=int(wf.FS))
        ates.append(ate); ferrs.append(ferr); plens.append(plen)
    ates, ferrs, plens = np.asarray(ates), np.asarray(ferrs), np.asarray(plens)
    moving = plens > 0.1
    drift = float(np.mean(ferrs[moving] / plens[moving]) * 100.0) if moving.any() else float("nan")

    vel_mse = float(np.mean((preds - va["Y"][idx]) ** 2))
    ate_mean = float(np.mean(ates))
    print(f"\n{args.run}: vel_mse={vel_mse:.5f} ATE={ate_mean:.4f}m drift={drift:.2f}%")

    slug = args.label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    out_csv = os.path.join(args.out_dir, f"odom_probe_{slug}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "vel_mse", "ate", "drift"])
        w.writeheader()
        w.writerow({"run": args.run, "vel_mse": vel_mse, "ate": ate_mean, "drift": drift})
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
