#!/usr/bin/env python3
"""Build a shared, model-agnostic Nymeria world-frame odometry window cache
for the TLIO and Air-IO baselines.

Reuses the exact same session loading / preprocessing / heading-canonicalization
math as the repo's own "Ours" world-frame odometry pipeline
(imu_lm/data/nymeria_real_wrist_odom_worldframe_loader_FLU.py,
scripts/eval_odom_nymeria_worldframe.py) so both baselines see the identical
input distribution and window/frame convention as "Ours"/Scratch/MobilePoser.

Per window we cache:
  X        : [6, T]     float32  -- preprocessed (impute/filter/normalize) wrist
                                     IMU, FLU frame: acc_F,acc_L,acc_U,gyro_F,gyro_L,gyro_U
  POS_full : [T+1, 3]   float32  -- pelvis position at every native 50Hz sample
                                     within the window, in this window's own
                                     heading-canonicalized world frame (Y-up,
                                     metres, POS_full[0] == origin)

Caching at full per-sample resolution (rather than only the 33-point patch
grid "Ours" itself trains against) lets each baseline's native temporal
output resolution be evaluated on the SAME 2.56s window / same canonical
frame, then scored on a common grid via common_worldframe_eval.py -- see that
module's docstring for why.

Usage:
    python prepare_nymeria_windows.py --split train --out-dir data_cache
    python prepare_nymeria_windows.py --split val   --out-dir data_cache
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import yaml
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, _REPO_ROOT)

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import load_session_local_vel_flu
from imu_lm.data.nymeria_real_wrist_odom_worldframe_loader_FLU import _heading_rotation
from imu_lm.utils.helpers import deep_update


def build_cfg():
    base = yaml.safe_load(open(os.path.join(_REPO_ROOT, "configs", "base.yaml")))
    model = yaml.safe_load(open(os.path.join(
        _REPO_ROOT, "configs", "vit1d_odom_worldframe_supervised_nymeria_altattn_scratch.yaml"
    )))
    return deep_update(base, model)


def build_windows(sessions, cfg, T, hop, target_hz=50):
    X_out, POS_out = [], []
    stats = PreprocessStats()
    for X_flu, pos_root, _R_X_fw in sessions:
        N = X_flu.shape[0]
        for s in range(0, N - T - 1, hop):
            Xw = X_flu[s:s + T]
            proc = preprocess_window(Xw, cfg, stats)
            if proc is None:
                continue
            R_canon = _heading_rotation(
                pos_root[s + T].astype(np.float64) - pos_root[s].astype(np.float64)
            )
            idxs = list(range(s, s + T + 1))
            pos_full = (R_canon @ (pos_root[idxs].astype(np.float64) - pos_root[s].astype(np.float64)).T).T
            X_out.append(proc.astype(np.float32))
            POS_out.append(pos_full.astype(np.float32))
    return np.stack(X_out), np.stack(POS_out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--train-dir", default="/projects/b53017/vbc1591/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir", default="/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=867)
    p.add_argument("--max-val-files", type=int, default=101)
    p.add_argument("--window-seconds", type=float, default=2.56)
    p.add_argument("--target-hz", type=int, default=50)
    p.add_argument("--add-gravity", action="store_true", default=True)
    p.add_argument("--out-dir", default=os.path.join(_THIS_DIR, "data_cache"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = build_cfg()

    if args.split == "train":
        files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))[:args.max_train_files]
    else:
        files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]

    print(f"[{args.split}] loading {len(files)} sessions ...")
    sessions = []
    for f in tqdm(files, desc=f"loading {args.split} sessions"):
        r = load_session_local_vel_flu(f, target_hz=args.target_hz, add_gravity=args.add_gravity)
        if r is not None:
            sessions.append(r)
    print(f"[{args.split}] loaded {len(sessions)}/{len(files)} sessions")

    T = int(round(args.window_seconds * args.target_hz))
    hop = T // 2
    X, POS = build_windows(sessions, cfg, T, hop, args.target_hz)
    print(f"[{args.split}] windows={len(X)}  X.shape={X.shape}  POS.shape={POS.shape}")

    out_path = os.path.join(args.out_dir, f"nymeria_worldframe_{args.split}.npz")
    np.savez_compressed(out_path, X=X, POS=POS, window_seconds=args.window_seconds, target_hz=args.target_hz)
    print(f"[{args.split}] saved -> {out_path}")


if __name__ == "__main__":
    main()
