#!/usr/bin/env python3
"""Probe run_ssl_baseline.py checkpoints on Nymeria real-wrist odometry."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window
from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import load_session_local_vel_flu
from imu_lm.models.baselines.ssl_baselines import build_ssl_model
from imu_lm.utils.helpers import cfg_get

FS = 50.0


def exp_so3(w):
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def build_windows(sessions, cfg, T, ps, hop, target_hz=50):
    Pt, Q = T // ps, ps // 4
    Ns, dtQ = Pt * 4, Q / target_hz
    X, Y, GY, R0s, POS = [], [], [], [], []
    stats = PreprocessStats()
    for X_flu, pos_root, R_X_fw in sessions:
        N = X_flu.shape[0]
        for s in range(0, N - T - ps, hop):
            Xw = X_flu[s:s + T]
            proc = preprocess_window(Xw, cfg, stats)
            if proc is None:
                continue
            vel = np.empty((Pt, 4, 3), dtype=np.float32)
            for p in range(Pt):
                tp = s + p * ps
                for k in range(4):
                    ts, te, tm = tp + k * Q, tp + k * Q + Q, tp + k * Q + Q // 2
                    vw = (pos_root[te].astype(np.float64) - pos_root[ts].astype(np.float64)) / dtQ
                    vel[p, k] = (R_X_fw[tm].astype(np.float64) @ vw).astype(np.float32)
            idxs = [s + i * Q for i in range(Ns + 1)]
            X.append(proc.astype(np.float32)); Y.append(vel)
            GY.append(Xw[:, 3:6].astype(np.float32))
            R0s.append(R_X_fw[s].T.astype(np.float32))
            POS.append((pos_root[idxs] - pos_root[s]).astype(np.float32))
    return (np.stack(X), np.stack(Y), np.stack(GY), np.stack(R0s), np.stack(POS))


class WindowOdomHead(nn.Module):
    def __init__(self, embed_dim, num_patches, hidden=384, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.proj = nn.Linear(embed_dim, hidden)
        self.time_query = nn.Embedding(num_patches, hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.block = nn.TransformerEncoder(layer, num_layers=1)
        self.norm2 = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 12)

    def forward(self, emb):
        B = emb.shape[0]
        memory = self.proj(emb).unsqueeze(1)
        t = torch.arange(self.num_patches, device=emb.device)
        q = self.time_query(t).unsqueeze(0).expand(B, -1, -1)
        h, _ = self.cross_attn(q, memory, memory)
        h = self.norm1(h + q)
        return self.head(self.norm2(self.block(h))).reshape(B, self.num_patches, 4, 3)


def load_model(model_name, run_dir, step, device):
    ckpt = os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt") if step is not None else os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "checkpoints", "latest.pt")
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = obj["cfg"]
    model = build_ssl_model(model_name, cfg_get(cfg, ["model"], {}) or {}).to(device)
    model.load_state_dict(obj["model"], strict=True)
    model.eval()
    print(f"  loaded {model_name} from {ckpt} step={obj.get('step', '?')}")
    return model, cfg


@torch.no_grad()
def encode(model, x_ct):
    return model.encode(x_ct.float().transpose(1, 2).contiguous())


def reconstruct_and_score(v_flu_pred, gyro_raw, R0, pos_gt, ps, target_hz=50):
    Pt, Q = v_flu_pred.shape[0], ps // 4
    Ns, dt = Pt * 4, 1.0 / target_hz
    R = R0.copy().astype(np.float64)
    Rs = [R.copy()]
    for t in range(gyro_raw.shape[0]):
        R = R @ exp_so3(gyro_raw[t].astype(np.float64) * dt)
        Rs.append(R.copy())
    pos = np.zeros(3); traj = [pos.copy()]
    for i, vf in enumerate(v_flu_pred.reshape(Ns, 3).astype(np.float64)):
        pos = pos + (Rs[i * Q + Q // 2] @ vf) * (Q * dt)
        traj.append(pos.copy())
    traj = np.asarray(traj)
    ate = float(np.sqrt(np.mean(np.sum((traj - pos_gt) ** 2, axis=1))))
    ferr = float(np.linalg.norm(traj[-1] - pos_gt[-1]))
    plen = float(np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)))
    return ate, ferr, plen


@torch.no_grad()
def eval_metrics(model, head, va, device, ps, batch_size=256, max_windows=2000):
    model.eval(); head.eval()
    X, Y = torch.from_numpy(va["X"]), torch.from_numpy(va["Y"])
    mses, preds = [], []
    n = min(len(X), max_windows)
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size].to(device), Y[i:i + batch_size].to(device)
        pred = head(encode(model, xb))
        mses.append(F.mse_loss(pred, yb).item())
        if i < n:
            preds.append(pred[:max(0, min(batch_size, n - i))].cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    ates, final_errs, path_lens = [], [], []
    for i in range(len(preds)):
        ate, ferr, plen = reconstruct_and_score(preds[i], va["GY"][i], va["R0"][i], va["POS"][i], ps)
        ates.append(ate); final_errs.append(ferr); path_lens.append(plen)
    path_lens = np.asarray(path_lens); final_errs = np.asarray(final_errs)
    moving = path_lens > 0.1
    drift = float(np.mean(final_errs[moving] / path_lens[moving]) * 100.0) if moving.any() else float("nan")
    return {"vel_mse": float(np.mean(mses)), "ate": float(np.mean(ates)), "drift": drift}


def train_probe(model, tr, va, device, ps, epochs, patience, lr, batch_size):
    Pt = tr["X"].shape[-1] // ps
    with torch.no_grad():
        D = int(encode(model, torch.from_numpy(va["X"][:1]).to(device)).shape[-1])
    head = WindowOdomHead(D, Pt).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler()
    dl = DataLoader(TensorDataset(torch.from_numpy(tr["X"]), torch.from_numpy(tr["Y"])),
                    batch_size=batch_size, shuffle=True, drop_last=True)
    best, best_state, no_imp = float("inf"), None, 0
    for ep in range(epochs):
        head.train(); model.eval()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                with torch.no_grad():
                    emb = encode(model, xb)
                loss = F.mse_loss(head(emb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sched.step()
        m = eval_metrics(model, head, va, device, ps, max_windows=500)
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
    return eval_metrics(model, head, va, device, ps)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["simmtm", "limu_bert", "focal", "moment"])
    p.add_argument("--run", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--mode", choices=["probe"], default="probe")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--train-dir", default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir", default="/mnt/nas2/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=867)
    p.add_argument("--max-val-files", type=int, default=101)
    p.add_argument("--window-seconds", type=float, default=2.56)
    p.add_argument("--add-gravity", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out-dir", default="data_vis/nymeria_odom_baseline_probe")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    model, cfg = load_model(args.model, os.path.join(args.runs_root, args.run), args.step, device)
    cfg = {**cfg, "windowing": {**cfg.get("windowing", {}), "window_seconds": args.window_seconds}}
    T, ps, hop = int(round(args.window_seconds * FS)), 16, int(round(args.window_seconds * FS)) // 2
    tr_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))[:args.max_train_files]
    va_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]
    def load(files):
        return [r for f in files if (r := load_session_local_vel_flu(f, target_hz=int(FS), add_gravity=args.add_gravity)) is not None]
    tr_sess, va_sess = load(tr_files), load(va_files)
    print(f"  sessions train={len(tr_sess)} val={len(va_sess)}")
    keys = ["X", "Y", "GY", "R0", "POS"]
    tr = dict(zip(keys, build_windows(tr_sess, cfg, T, ps, hop, int(FS))))
    va = dict(zip(keys, build_windows(va_sess, cfg, T, ps, hop, int(FS))))
    print(f"  windows train={len(tr['X'])} val={len(va['X'])}")
    m = train_probe(model, tr, va, device, ps, args.epochs, args.patience, args.lr, args.batch_size)
    print(f"\n{args.run}: vel_mse={m['vel_mse']:.5f} ATE={m['ate']:.4f}m drift={m['drift']:.2f}%")
    out_csv = os.path.join(args.out_dir, f"odom_probe_{args.model}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "run", "vel_mse", "ate", "drift"])
        w.writeheader(); w.writerow({"model": args.model, "run": args.run, **m})


if __name__ == "__main__":
    main()
