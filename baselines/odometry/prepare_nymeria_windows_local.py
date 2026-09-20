#!/usr/bin/env python3
"""Window cache for the WRIST-FRAME odometry task, built with the benchmark's own window builder.

The world-frame cache (prepare_nymeria_windows.py) cannot be reused: it has 73,542 evaluation windows where
the benchmark's wrist-frame scorer evaluates 73,515, and its targets are heading-canonical world positions.
This one calls scripts/eval_odom_nymeria_altattn.py::build_windows with the config stored in the benchmark's
own from-scratch odometry run, so TLIO and Air-IO see exactly the inputs, windows and targets of every other
row of the odometry table:
    X   [N, 6, 128]   z-scored acc+gyro window (same preprocessing as the benchmark encoders)
    Y   [N, 8, 4, 3]  pelvis velocity in the wrist (FLU) frame, 4 sub-steps per 0.32 s patch
    GY  [N, 128, 3]   raw gyroscope, used only by the scorer's orientation integration
    R0  [N, 3, 3]     world<-FLU at the window start (ground truth)
    POS [N, 33, 3]    world pelvis positions relative to the window start

    python prepare_nymeria_windows_local.py --split val
"""
import argparse, glob, os, sys
import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import eval_odom_nymeria_altattn as lf      # noqa: E402

REF_RUN = os.path.join(REPO, "runs", "bench_odometry_nymeria_local__scratch__scratch__s42", "checkpoints", "best.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--out-dir", default=os.path.join(_THIS, "data_cache"))
    p.add_argument("--max-files", type=int, default=0)
    a = p.parse_args()
    cfg = torch.load(REF_RUN, map_location="cpu", weights_only=False)["cfg"]
    obj = cfg["objective"][cfg["objective"]["type"]]
    d = obj["train_pkl_root"] if a.split == "train" else obj["val_pkl_root"]
    T, ps = int(round(float(cfg["windowing"]["window_seconds"]) * lf.FS)), 16
    files = sorted(glob.glob(os.path.join(d, "*.pkl")))
    if a.max_files:
        files = files[:a.max_files]
    sess = [r for f in files if (r := lf.load_session_local_vel_flu(f, target_hz=int(lf.FS), add_gravity=True)) is not None]
    print(f"[{a.split}] {len(sess)}/{len(files)} sessions", flush=True)
    X, Y, GY, R0, POS = lf.build_windows(sess, cfg, T, ps, T // 2)
    print(f"[{a.split}] X{X.shape} Y{Y.shape} GY{GY.shape} R0{R0.shape} POS{POS.shape}", flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"nymeria_local_{a.split}.npz")
    np.savez(out, X=X.astype(np.float32), Y=Y.astype(np.float32), GY=GY.astype(np.float32),
             R0=R0.astype(np.float32), POS=POS.astype(np.float32), T=T, ps=ps)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
