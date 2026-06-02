#!/usr/bin/env python3
"""
Odometry evaluation for SSL-pretrained ViT1D encoders on Nymeria real-wrist FLU.

Pipeline (matches the "predict body-frame velocity + gyro-integrated rotation"
approach):
  1. Network predicts pelvis velocity in the wrist FLU body frame, per sub-patch:
        encoder tokens → 1-transformer-block head → [B, Pt, 4, 3]  (32 sub-patch v_FLU)
  2. Integrate the raw FLU gyro to estimate orientation R_world_from_FLU(t),
        anchored at the GT initial orientation.
  3. Rotate predicted v_FLU into world frame and integrate → predicted trajectory.
  4. Compare to the GT root trajectory (real pelvis world positions).

Modes:
  --mode probe     : freeze encoder, train only the 1-block velocity head.
  --mode finetune  : train encoder + head jointly.

Metrics (on val windows, 2.56 s each):
  vel_mse  : MSE of FLU-frame velocity (m/s)²
  ATE      : abs. trajectory error (RMSE position, metres)
  drift    : final-position error / GT path length (%)

Usage:
    python scripts/eval_odom_nymeria_ssl.py \
        --runs vit1d_ssl_mae vit1d_ssl_limu_bert vit1d_ssl_simmtm vit1d_ssl_focal \
        --mode probe --out-dir data_vis/nymeria_odom_probe_ssl/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import load_session_local_vel_flu
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.utils.helpers import cfg_get

FS = 50.0


# ── SO(3) ─────────────────────────────────────────────────────────────────────

def exp_so3(w):
    """Rodrigues: axis-angle vector [3] → rotation matrix [3,3]."""
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# ── Window builder (gives raw gyro + GT positions for trajectory eval) ────────

def build_windows(sessions, cfg, T, ps, hop, target_hz=50, add_gravity=True):
    """Return arrays: X[M,6,T] (preproc), Y[M,Pt,4,3] FLU vel,
    GYRO[M,T,3] raw FLU gyro, R0[M,3,3] R_world_from_X at start,
    POSGT[M,Ns+1,3] GT world positions at sub-patch boundaries (start-relative)."""
    Pt = T // ps
    Q  = ps // 4
    Ns = Pt * 4
    dtQ = Q / target_hz
    X, Y, GY, R0s, POS = [], [], [], [], []

    class _Stats:  # preprocess_window needs a stats holder
        pass
    from imu_lm.data.augmentations.preprocess import PreprocessStats
    stats = PreprocessStats()

    for X_flu, pos_root, R_X_fw in sessions:
        N = X_flu.shape[0]
        for s in range(0, N - T - ps, hop):
            Xw = X_flu[s:s + T]                              # [T,6] raw FLU
            proc = preprocess_window(Xw, cfg, stats)
            if proc is None:
                continue
            # FLU velocity label [Pt,4,3]
            vel = np.empty((Pt, 4, 3), dtype=np.float32)
            for p in range(Pt):
                tp = s + p * ps
                for k in range(4):
                    ts, te, tm = tp + k * Q, tp + k * Q + Q, tp + k * Q + Q // 2
                    vw = (pos_root[te].astype(np.float64) - pos_root[ts].astype(np.float64)) / dtQ
                    vel[p, k] = (R_X_fw[tm].astype(np.float64) @ vw).astype(np.float32)
            # GT world positions at sub-patch boundaries, start-relative
            idxs = [s + i * Q for i in range(Ns + 1)]
            pos_gt = (pos_root[idxs] - pos_root[s]).astype(np.float32)   # [Ns+1,3]

            X.append(proc.astype(np.float32))                # [6,T]
            Y.append(vel)
            GY.append(Xw[:, 3:6].astype(np.float32))         # raw FLU gyro [T,3]
            R0s.append(R_X_fw[s].T.astype(np.float32))       # R_world_from_X(start)
            POS.append(pos_gt)
    return (np.stack(X), np.stack(Y), np.stack(GY),
            np.stack(R0s), np.stack(POS))


# ── One-transformer-block velocity head ───────────────────────────────────────

class OneBlockOdomHead(nn.Module):
    def __init__(self, embed_dim, in_channels, num_patches, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_patches = num_patches
        self.time_pos = nn.Embedding(num_patches, embed_dim)
        self.block = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 4 * 3)            # 4 sub-patches × 3D velocity

    def forward(self, tokens):
        B = tokens.shape[0]
        x = tokens.reshape(B, self.in_channels, self.num_patches, -1).mean(dim=1)
        t = torch.arange(self.num_patches, device=tokens.device)
        x = self.norm(self.block(x + self.time_pos(t).unsqueeze(0)))
        return self.head(x).reshape(B, self.num_patches, 4, 3)   # [B, Pt, 4, 3]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_encoder(run_dir, step, device):
    ckpt_path = (os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt")
                 if step is not None else os.path.join(run_dir, "checkpoints", "best.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    enc = ViT1DEncoder(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    short = os.path.basename(run_dir).replace("vit1d_ssl_", "").replace("vit1d_", "")
    print(f"  {short}  step={ckpt.get('step','?')}")
    return enc, cfg, short


# ── Trajectory reconstruction + ATE ───────────────────────────────────────────

def reconstruct_and_score(v_flu_pred, gyro_raw, R0, pos_gt, ps, target_hz=50):
    """v_flu_pred [Pt,4,3], gyro_raw [T,3], R0 [3,3], pos_gt [Ns+1,3]."""
    Pt = v_flu_pred.shape[0]
    Q  = ps // 4
    Ns = Pt * 4
    T  = gyro_raw.shape[0]
    dt = 1.0 / target_hz

    # Integrate gyro → R_world_from_FLU at every frame (anchor GT initial orient.)
    R = R0.copy().astype(np.float64)
    Rs = [R.copy()]
    for t in range(T):
        R = R @ exp_so3(gyro_raw[t].astype(np.float64) * dt)
        Rs.append(R.copy())

    v_flat = v_flu_pred.reshape(Ns, 3).astype(np.float64)
    pos = np.zeros(3)
    traj = [pos.copy()]
    for i in range(Ns):
        mid = i * Q + Q // 2
        v_world = Rs[mid] @ v_flat[i]
        pos = pos + v_world * (Q * dt)
        traj.append(pos.copy())
    traj = np.array(traj)                                  # [Ns+1, 3]

    ate       = float(np.sqrt(np.mean(np.sum((traj - pos_gt) ** 2, axis=1))))
    final_err = float(np.linalg.norm(traj[-1] - pos_gt[-1]))
    path_len  = float(np.sum(np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)))
    return ate, final_err, path_len


# ── Train / eval ──────────────────────────────────────────────────────────────

def run_one(enc, ps, tr, va, device, mode, lr, epochs, patience, batch_size=128):
    X_tr, Y_tr = tr["X"], tr["Y"]
    Pt = X_tr.shape[-1] // ps
    head = OneBlockOdomHead(enc.embed_dim, enc.in_channels, Pt).to(device)

    if mode == "probe":
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
        params = list(head.parameters())
    else:
        for p in enc.parameters():
            p.requires_grad_(True)
        params = list(enc.parameters()) + list(head.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler()

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    best, best_state, no_imp = float("inf"), None, 0
    for epoch in range(epochs):
        head.train()
        if mode == "finetune":
            enc.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with autocast():
                if mode == "probe":
                    with torch.no_grad():
                        toks, _ = enc.forward_tokens(xb.float())
                else:
                    toks, _ = enc.forward_tokens(xb.float())
                pred = head(toks)
                loss = F.mse_loss(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
        sched.step()

        vmse = vel_mse(enc, head, va, device)
        if vmse < best - 1e-6:
            best = vmse
            best_state = ({k: v.cpu().clone() for k, v in head.state_dict().items()},
                          {k: v.cpu().clone() for k, v in enc.state_dict().items()} if mode == "finetune" else None)
            no_imp = 0
        else:
            no_imp += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch+1:3d}  val_vel_mse={vmse:.5f}  best={best:.5f}")
        if no_imp >= patience:
            print(f"    early stop @ epoch {epoch+1}")
            break

    head.load_state_dict(best_state[0])
    if mode == "finetune" and best_state[1] is not None:
        enc.load_state_dict(best_state[1])

    # Final metrics incl. trajectory ATE / drift
    vmse = vel_mse(enc, head, va, device)
    ate, drift = trajectory_metrics(enc, head, va, device, ps)
    return {"vel_mse": vmse, "ate": ate, "drift": drift}


@torch.no_grad()
def vel_mse(enc, head, va, device, batch_size=256):
    head.eval(); enc.eval()
    X, Y = torch.from_numpy(va["X"]), torch.from_numpy(va["Y"])
    errs = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = Y[i:i + batch_size].to(device)
        toks, _ = enc.forward_tokens(xb.float())
        errs.append(F.mse_loss(head(toks), yb).item())
    return float(np.mean(errs))


@torch.no_grad()
def trajectory_metrics(enc, head, va, device, ps, batch_size=256, max_windows=2000):
    head.eval(); enc.eval()
    X = torch.from_numpy(va["X"])
    n = min(len(X), max_windows)
    preds = []
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size].to(device)
        toks, _ = enc.forward_tokens(xb.float())
        preds.append(head(toks).cpu().numpy())
    preds = np.concatenate(preds, axis=0)                 # [n, Pt, 4, 3]
    ates, final_errs, path_lens = [], [], []
    for i in range(n):
        ate, ferr, plen = reconstruct_and_score(
            preds[i], va["GY"][i], va["R0"][i], va["POS"][i], ps)
        ates.append(ate); final_errs.append(ferr); path_lens.append(plen)
    ates = np.array(ates); final_errs = np.array(final_errs); path_lens = np.array(path_lens)
    # Drift % only over windows with meaningful motion (path > 0.1 m) to avoid
    # divide-by-~0 on near-stationary windows.
    moving = path_lens > 0.1
    drift = float(np.mean(final_errs[moving] / path_lens[moving]) * 100.0) if moving.any() else float("nan")
    return float(ates.mean()), drift


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(results, shorts, mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    metrics = [("vel_mse", "Velocity MSE (m/s)²"), ("ate", "ATE (m)"), ("drift", "Drift (%)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, label) in zip(axes, metrics):
        vals = [results[s][key] for s in shorts]
        bars = ax.bar(shorts, vals, color=[colors[i % len(colors)] for i in range(len(shorts))],
                      alpha=0.85, width=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, rotation=15)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Nymeria Odometry — {mode} (lower = better)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"odom_{mode}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_dir}/odom_{mode}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["vit1d_ssl_mae", "vit1d_ssl_limu_bert",
                            "vit1d_ssl_simmtm", "vit1d_ssl_focal"])
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--train-dir", default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir",   default="/mnt/nas2/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=80)
    p.add_argument("--max-val-files",   type=int, default=20)
    p.add_argument("--add-gravity", action="store_true", default=True)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else (3e-4 if args.mode == "probe" else 1e-4)
    out_dir = args.out_dir or f"data_vis/nymeria_odom_{args.mode}_ssl"
    os.makedirs(out_dir, exist_ok=True)

    # Use first run's cfg for windowing/preproc (identical across runs)
    _, base_cfg, _ = load_encoder(os.path.join(args.runs_root, args.runs[0]), args.step, device)
    T = int(round(float(cfg_get(base_cfg, ["windowing", "window_seconds"], 2.56)) * FS))
    ps = int((cfg_get(base_cfg, ["vit1d"], {}) or {}).get("patch_size", 16))
    hop = max(1, T // 2)

    print("\nLoading Nymeria sessions (raw, for gyro + GT trajectory) ...")
    tr_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))[:args.max_train_files]
    va_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]

    def _load(files):
        out = []
        for f in files:
            r = load_session_local_vel_flu(f, target_hz=int(FS), add_gravity=args.add_gravity)
            if r is not None:
                out.append(r)
        return out

    tr_sess, va_sess = _load(tr_files), _load(va_files)
    print(f"  train sessions: {len(tr_sess)}  val sessions: {len(va_sess)}")
    tr = dict(zip(["X", "Y", "GY", "R0", "POS"], build_windows(tr_sess, base_cfg, T, ps, hop, int(FS), args.add_gravity)))
    va = dict(zip(["X", "Y", "GY", "R0", "POS"], build_windows(va_sess, base_cfg, T, ps, hop, int(FS), args.add_gravity)))
    print(f"  train windows: {len(tr['X'])}  val windows: {len(va['X'])}")

    results, shorts = {}, []
    for run in args.runs:
        print(f"\n{'='*55}\n{run}  [{args.mode}]\n{'='*55}")
        enc, _, short = load_encoder(os.path.join(args.runs_root, run), args.step, device)
        m = run_one(enc, ps, tr, va, device, args.mode, lr, args.epochs, args.patience)
        results[short] = m
        shorts.append(short)
        print(f"  → {short}: vel_mse={m['vel_mse']:.5f}  ATE={m['ate']:.4f}m  drift={m['drift']:.2f}%")

    print(f"\n{'='*60}\nNymeria Odometry — {args.mode}")
    print(f"{'Model':<14} {'vel_mse':>10} {'ATE (m)':>10} {'drift %':>10}")
    print("-" * 60)
    for s in shorts:
        print(f"{s:<14} {results[s]['vel_mse']:>10.5f} {results[s]['ate']:>10.4f} {results[s]['drift']:>10.2f}")
    print("=" * 60)

    pd.DataFrame([{"model": s, **results[s]} for s in shorts]).to_csv(
        os.path.join(out_dir, f"odom_{args.mode}.csv"), index=False)
    plot(results, shorts, args.mode, out_dir)


if __name__ == "__main__":
    main()
