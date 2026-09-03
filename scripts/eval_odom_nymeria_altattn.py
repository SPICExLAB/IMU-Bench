#!/usr/bin/env python3
"""Probe an alt-attention/register MAE encoder on Nymeria real-wrist odometry."""
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
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn
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
    # Acc-only ablation: the encoder's own input (X, appended below) is
    # truncated to its acc-only prefix -- matching Trainer._acc_only_slice
    # during training -- but GY (raw gyro) is kept full-width regardless: it
    # feeds the classical exp_so3 gyro-integration step below (ground-truth
    # rotation, not a model input), which every row in this table relies on
    # to reconstruct a trajectory from predicted local velocities.
    acc_only = bool(cfg_get(cfg, ["data", "acc_only"], False))
    n_acc = len(cfg_get(cfg, ["data", "sensor_columns"], []) or []) if acc_only else None
    for X_flu, pos_root, R_X_fw in sessions:
        N = X_flu.shape[0]
        for s in range(0, N - T - ps, hop):
            Xw = X_flu[s:s + T]
            proc = preprocess_window(Xw, cfg, stats)
            if proc is None:
                continue
            if acc_only:
                proc = proc[:n_acc, :]   # proc is [C, T] (preprocess_window transposes to channel-first)
            vel = np.empty((Pt, 4, 3), dtype=np.float32)
            for p in range(Pt):
                tp = s + p * ps
                for k in range(4):
                    ts, te, tm = tp + k * Q, tp + k * Q + Q, tp + k * Q + Q // 2
                    vw = (pos_root[te].astype(np.float64) - pos_root[ts].astype(np.float64)) / dtQ
                    vel[p, k] = (R_X_fw[tm].astype(np.float64) @ vw).astype(np.float32)
            idxs = [s + i * Q for i in range(Ns + 1)]
            X.append(proc.astype(np.float32))
            Y.append(vel)
            GY.append(Xw[:, 3:6].astype(np.float32))
            R0s.append(R_X_fw[s].T.astype(np.float32))
            POS.append((pos_root[idxs] - pos_root[s]).astype(np.float32))
    return (np.stack(X), np.stack(Y), np.stack(GY), np.stack(R0s), np.stack(POS))


class RegisterOdomHead(nn.Module):
    def __init__(self, embed_dim, num_patches, hidden=192, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(embed_dim, hidden)
        self.time_pos = nn.Embedding(num_patches, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.block = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 12)

    def forward(self, reg):
        t = torch.arange(reg.shape[1], device=reg.device)
        x = self.proj(reg) + self.time_pos(t).unsqueeze(0)
        return self.head(self.norm(self.block(x))).reshape(reg.shape[0], reg.shape[1], 4, 3)


def load_encoder(run_dir, step, device):
    ckpt = os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt") if step is not None else os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "checkpoints", "latest.pt")
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = obj["cfg"]
    # Build whichever encoder architecture the checkpoint was actually trained
    # with: alt-attn for our MAE model, plain ViT1DEncoder for the FOCAL/SimMTM
    # baselines and from-scratch runs (features() below handles both).
    from imu_lm.models.baselines.inertia1_encoder import build_encoder as _build_encoder
    enc = _build_encoder(cfg).to(device)   # also handles the Inertia-1 external baseline (cfg["inertia1"])
    enc.load_state_dict(obj["model"], strict=True)
    print(f"  loaded {os.path.basename(run_dir)} from {ckpt} step={obj.get('step', '?')}")
    return enc, cfg


def features(enc, x):
    if bool(getattr(enc, "_use_register", False)):
        return enc.forward_registers(x.float())[0]
    toks, Pt = enc.forward_tokens(x.float())
    B, C, D = x.shape[0], enc.in_channels, enc.embed_dim
    return toks.reshape(B, C, Pt, D).mean(dim=1)


def reconstruct_and_score(v_flu_pred, gyro_raw, R0, pos_gt, ps, target_hz=50):
    Pt, Q = v_flu_pred.shape[0], ps // 4
    Ns, dt = Pt * 4, 1.0 / target_hz
    R = R0.copy().astype(np.float64)
    Rs = [R.copy()]
    for t in range(gyro_raw.shape[0]):
        R = R @ exp_so3(gyro_raw[t].astype(np.float64) * dt)
        Rs.append(R.copy())
    pos = np.zeros(3)
    traj = [pos.copy()]
    for i, vf in enumerate(v_flu_pred.reshape(Ns, 3).astype(np.float64)):
        pos = pos + (Rs[i * Q + Q // 2] @ vf) * (Q * dt)
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
        ate, ferr, plen = reconstruct_and_score(preds[i], va["GY"][i], va["R0"][i], va["POS"][i], ps)
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
    p.add_argument("--out-dir", default="data_vis/nymeria_odom_altattn_probe")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42,
                    help="seeds random/np.random/torch (+cuda) for probe-head init and data shuffling "
                         "-- previously unseeded (non-deterministic run-to-run); default 42 makes single "
                         "runs reproducible, override for multi-seed variance studies")
    return p.parse_args()


def main():
    args = parse_args()
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    enc, cfg = load_encoder(os.path.join(args.runs_root, args.run), args.step, device)
    cfg = {**cfg, "windowing": {**cfg.get("windowing", {}), "window_seconds": args.window_seconds}}
    T = int(round(args.window_seconds * FS))
    ps = int((cfg_get(cfg, ["vit1d"], {}) or {}).get("patch_size", 16))
    hop = T // 2
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
    head, m = train_probe(enc, tr, va, device, ps, args.epochs, args.patience, args.lr, args.batch_size)
    print(f"\n{args.run}: vel_mse={m['vel_mse']:.5f} ATE={m['ate']:.4f}m drift={m['drift']:.2f}% seed={args.seed}")
    csv_name = "odom_probe.csv" if args.seed == 42 else f"odom_probe_seed{args.seed}.csv"
    with open(os.path.join(args.out_dir, csv_name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "seed", "vel_mse", "ate", "drift"])
        w.writeheader(); w.writerow({"run": args.run, "seed": args.seed, **m})

    # Persist the trained probe head so future analysis/visualization can just
    # load + run inference instead of re-running the full ~50-epoch training
    # (this was previously thrown away -- only the CSV summary survived).
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
