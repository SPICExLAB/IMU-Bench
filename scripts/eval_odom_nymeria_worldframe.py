#!/usr/bin/env python3
"""Probe a frozen encoder on Nymeria pelvis odometry, canonical-world-frame target.

Complementary to eval_odom_nymeria_altattn.py: that script predicts velocity
in the wrist's own continuously-rotating FLU frame, so reconstructing a
trajectory needs real gyro integration at every timestep. This script instead
predicts velocity in a **per-window canonicalized world frame** (see
imu_lm/data/nymeria_real_wrist_odom_worldframe_loader_FLU.py): each window is
rotated once, about the vertical axis only, so its own net displacement points
along +X (heading derived from movement direction -- pelvis body orientation
isn't available, only position). Reconstruction is then a plain velocity
cumulative sum -- no gyro loop needed, since the frame doesn't rotate within a
window.

Reuses RegisterOdomHead/load_encoder/features from eval_odom_nymeria_altattn.py
unchanged (identical probe architecture; only the target/reconstruction math
differs).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import load_session_local_vel_flu
from imu_lm.data.nymeria_real_wrist_odom_worldframe_loader_FLU import _heading_rotation
from eval_odom_nymeria_altattn import load_encoder, features, RegisterOdomHead, FS


def build_windows(sessions, cfg, T, ps, hop, target_hz=50):
    """Like eval_odom_nymeria_altattn.build_windows, but labels/reconstruction
    targets are in the per-window canonical world frame instead of the
    wrist's continuously-rotating FLU frame."""
    Pt, Q = T // ps, ps // 4
    Ns, dtQ = Pt * 4, Q / target_hz
    X, Y, POS = [], [], []
    stats = PreprocessStats()
    for X_flu, pos_root, _R_X_fw in sessions:
        N = X_flu.shape[0]
        for s in range(0, N - T - ps, hop):
            Xw = X_flu[s:s + T]
            proc = preprocess_window(Xw, cfg, stats)
            if proc is None:
                continue
            R_canon = _heading_rotation(
                pos_root[s + T].astype(np.float64) - pos_root[s].astype(np.float64)
            )
            vel = np.empty((Pt, 4, 3), dtype=np.float32)
            for p in range(Pt):
                tp = s + p * ps
                for k in range(4):
                    ts, te = tp + k * Q, tp + k * Q + Q
                    vw = (pos_root[te].astype(np.float64) - pos_root[ts].astype(np.float64)) / dtQ
                    vel[p, k] = (R_canon @ vw).astype(np.float32)
            idxs = [s + i * Q for i in range(Ns + 1)]
            pos_canon = (R_canon @ (pos_root[idxs].astype(np.float64) - pos_root[s].astype(np.float64)).T).T
            X.append(proc.astype(np.float32))
            Y.append(vel)
            POS.append(pos_canon.astype(np.float32))
    return np.stack(X), np.stack(Y), np.stack(POS)


def reconstruct_and_score(v_pred, pos_gt, ps, target_hz=50):
    """Straight cumulative sum -- no gyro/orientation tracking needed, since
    the canonical frame is fixed for the whole window."""
    Pt, Q = v_pred.shape[0], ps // 4
    Ns, dt = Pt * 4, 1.0 / target_hz
    pos = np.zeros(3)
    traj = [pos.copy()]
    for vf in v_pred.reshape(Ns, 3).astype(np.float64):
        pos = pos + vf * (Q * dt)
        traj.append(pos.copy())
    traj = np.asarray(traj)
    ate = float(np.sqrt(np.mean(np.sum((traj - pos_gt) ** 2, axis=1))))
    ferr = float(np.linalg.norm(traj[-1] - pos_gt[-1]))
    plen = float(np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)))
    return ate, ferr, plen


@torch.no_grad()
def eval_metrics(enc, head, va, device, ps, batch_size=256, max_windows=2000):
    enc.eval(); head.eval()
    X, Y = torch.from_numpy(va["X"]), torch.from_numpy(va["Y"])
    mses, preds = [], []
    n = min(len(X), max_windows)
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        pred = head(features(enc, xb))
        mses.append(F.mse_loss(pred, yb).item())
        if i < n:
            preds.append(pred[:max(0, min(batch_size, n - i))].cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    ates, final_errs, path_lens = [], [], []
    for i in range(len(preds)):
        ate, ferr, plen = reconstruct_and_score(preds[i], va["POS"][i], ps)
        ates.append(ate); final_errs.append(ferr); path_lens.append(plen)
    path_lens = np.asarray(path_lens); final_errs = np.asarray(final_errs)
    moving = path_lens > 0.1
    drift = float(np.mean(final_errs[moving] / path_lens[moving]) * 100.0) if moving.any() else float("nan")
    return {"vel_mse": float(np.mean(mses)), "ate": float(np.mean(ates)), "drift": drift}


def train_probe(enc, tr, va, device, ps, epochs, patience, lr, batch_size):
    Pt = tr["X"].shape[-1] // ps
    head = RegisterOdomHead(enc.embed_dim, Pt).to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler()
    dl = DataLoader(TensorDataset(torch.from_numpy(tr["X"]), torch.from_numpy(tr["Y"])),
                    batch_size=batch_size, shuffle=True, drop_last=True)
    best, best_state, no_imp = float("inf"), None, 0
    for ep in range(epochs):
        head.train(); enc.eval()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                with torch.no_grad():
                    feat = features(enc, xb)
                loss = F.mse_loss(head(feat), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        m = eval_metrics(enc, head, va, device, ps, max_windows=500)
        if m["vel_mse"] < best - 1e-6:
            best, best_state, no_imp = m["vel_mse"], {k: v.cpu().clone() for k, v in head.state_dict().items()}, 0
        else:
            no_imp += 1
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1:3d} val_vel_mse={m['vel_mse']:.5f} ATE={m['ate']:.4f} best={best:.5f}")
        if patience > 0 and no_imp >= patience:
            print(f"    early stop @ epoch {ep+1}")
            break
    if best_state:
        head.load_state_dict(best_state)
    return head, eval_metrics(enc, head, va, device, ps)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--train-dir", default="/projects/b53017/vbc1591/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir", default="/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=867)
    p.add_argument("--max-val-files", type=int, default=101)
    p.add_argument("--window-seconds", type=float, default=2.56)
    p.add_argument("--add-gravity", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", default="data_vis/nymeria_odom_worldframe_probe")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    enc, cfg = load_encoder(os.path.join(args.runs_root, args.run), args.step, device)
    cfg = {**cfg, "windowing": {**cfg.get("windowing", {}), "window_seconds": args.window_seconds}}
    T = int(round(args.window_seconds * FS))
    from imu_lm.utils.helpers import cfg_get
    ps = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))
    hop = T // 2
    tr_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))[:args.max_train_files]
    va_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]

    def load(files):
        return [r for f in files if (r := load_session_local_vel_flu(f, target_hz=int(FS), add_gravity=args.add_gravity)) is not None]

    tr_sess, va_sess = load(tr_files), load(va_files)
    print(f"  sessions train={len(tr_sess)} val={len(va_sess)}")
    keys = ["X", "Y", "POS"]
    tr = dict(zip(keys, build_windows(tr_sess, cfg, T, ps, hop, int(FS))))
    va = dict(zip(keys, build_windows(va_sess, cfg, T, ps, hop, int(FS))))
    print(f"  windows train={len(tr['X'])} val={len(va['X'])}")

    head, m = train_probe(enc, tr, va, device, ps, args.epochs, args.patience, args.lr, args.batch_size)
    print(f"\n{args.run}: vel_mse={m['vel_mse']:.5f} ATE={m['ate']:.4f}m drift={m['drift']:.2f}%")
    with open(os.path.join(args.out_dir, "odom_probe.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "vel_mse", "ate", "drift"])
        w.writeheader(); w.writerow({"run": args.run, **m})

    Pt = tr["X"].shape[-1] // ps
    head_path = os.path.join(args.out_dir, "odom_head.pt")
    torch.save({
        "head_state_dict": head.state_dict(),
        "embed_dim": enc.embed_dim,
        "num_patches": Pt,
        "patch_size": ps,
        "run": args.run,
        "metrics": m,
    }, head_path)
    print(f"saved trained probe head to {head_path}")


if __name__ == "__main__":
    main()
