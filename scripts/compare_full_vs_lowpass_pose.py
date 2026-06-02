#!/usr/bin/env python3
"""
Supervised full-body pose estimation on IMUPoser — full 60 Hz vs lowpass filtered.

Trains the same 1D-CNN regressor twice:
  - Model A: full 60 Hz signal
  - Model B: 0-<cutoff> Hz lowpass filtered signal (FFT zeroing)

Input : acc (5 IMUs × 3 = 15 channels) +
        angular velocity derived from orientation matrices (5 × 3 = 15 channels)
        → 30 channels total

Target: SMPL pose as 6D rotation (24 joints × 6 = 144 dims) per window

Metric: mean per-joint rotation error in degrees (lower = better)

Split : participants P1–P8 train, P9–P10 test (clean cross-subject split)

Usage:
    python scripts/compare_full_vs_lowpass_pose.py
    python scripts/compare_full_vs_lowpass_pose.py --cutoff 4
    python scripts/compare_full_vs_lowpass_pose.py --data-dir /mnt/nas2/imuposer_dataset
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

FS       = 60.0
WIN      = 64       # 64 frames @ 60 Hz ≈ 1.07 s
HOP      = 16       # 16 frames ≈ 0.27 s
N_IMU    = 5        # left wrist(0), right wrist(1), left pocket(2), right pocket(3), head(4)
N_CH     = 6        # right wrist only: acc(3) + ang_vel(3)
N_JOINTS = 24
OUT_DIM  = N_JOINTS * 6   # 6D rotation per joint = 144

TRAIN_SUBJ = ["P1","P2","P3","P4","P5","P6","P7","P8"]
TEST_SUBJ  = ["P9","P10"]


# ── Rotation utilities ────────────────────────────────────────────────────────

def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """R [..., 3, 3] → 6D [..., 6] (first two columns)."""
    return R[..., :, :2].transpose(*range(R.ndim - 2), R.ndim - 1, R.ndim - 2).reshape(
        *R.shape[:-2], 6)


def rotation_error_deg(pred_6d: torch.Tensor, true_6d: torch.Tensor) -> float:
    """Mean per-joint rotation error in degrees.

    pred_6d, true_6d: [N, 144]  (24 joints × 6D)
    """
    B = pred_6d.shape[0]
    p = pred_6d.view(B, N_JOINTS, 6)
    t = true_6d.view(B, N_JOINTS, 6)

    def to_rotmat(x6d):
        # Gram-Schmidt orthonormalisation
        a1 = x6d[..., :3]
        a2 = x6d[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)   # [..., 3, 3]

    R_pred = to_rotmat(p)   # [B, 24, 3, 3]
    R_true = to_rotmat(t)
    R_diff = R_pred.transpose(-1, -2) @ R_true
    trace  = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos    = ((trace - 1) / 2).clamp(-1, 1)
    err_rad = torch.acos(cos)   # [B, 24]
    return err_rad.mean().item() * 180.0 / np.pi


# ── Signal processing ─────────────────────────────────────────────────────────

def lowpass_fft(windows: np.ndarray, cutoff_hz: float, fs: float = FS) -> np.ndarray:
    """windows [N, C, T] → lowpass filtered [N, C, T]."""
    freqs = np.fft.rfftfreq(windows.shape[2], d=1.0 / fs)
    F_    = np.fft.rfft(windows, axis=2)
    F_[:, :, freqs >= cutoff_hz] = 0
    return np.fft.irfft(F_, n=windows.shape[2], axis=2).astype(np.float32)


def znorm(windows: np.ndarray) -> np.ndarray:
    mu  = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True) + 1e-8
    return (windows - mu) / std


def derive_angular_velocity(ori: np.ndarray, fps: float) -> np.ndarray:
    """ori [T, 3, 3] → ang_vel [T, 3] via finite differences (single IMU)."""
    T  = ori.shape[0]
    av = np.zeros((T, 3), dtype=np.float32)
    for i in range(T - 1):
        dR = ori[i+1] @ ori[i].T          # [3, 3]
        av[i] = np.array([dR[2,1]-dR[1,2],
                           dR[0,2]-dR[2,0],
                           dR[1,0]-dR[0,1]]) * 0.5 * fps
    av[-1] = av[-2]
    return av


# ── Data loading ──────────────────────────────────────────────────────────────

def load_subject(subj_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load all sequences for one subject → (imu_signal [T_total, 30], pose_6d [T_total, 144])."""
    pkl_files = sorted(glob.glob(os.path.join(subj_dir, "*.pkl")))
    if not pkl_files:
        return None

    all_imu, all_pose = [], []
    for fpath in pkl_files:
        try:
            d = pickle.load(open(fpath, "rb"))
        except Exception:
            continue

        imu_raw  = d["imu"].numpy() if hasattr(d["imu"], "numpy") else np.array(d["imu"])
        pose_raw = d["pose"].numpy() if hasattr(d["pose"], "numpy") else np.array(d["pose"])
        T = imu_raw.shape[0]
        if T < WIN:
            continue

        # Right wrist = IMU index 1
        acc = imu_raw[:, 3:6].astype(np.float32)              # [T, 3]
        ori = imu_raw[:, 24:33].reshape(T, 3, 3).astype(np.float64)  # [T, 3, 3]
        av  = derive_angular_velocity(ori, FS)                # [T, 3]

        imu_signal = np.concatenate([acc, av], axis=1)        # [T, 6]

        # Pose: axis-angle [T, 72] → rotation matrix [T, 24, 3, 3] → 6D [T, 24, 6] → [T, 144]
        pose_aa = pose_raw.reshape(T, N_JOINTS, 3)
        # Rodrigues formula: axis-angle → rotation matrix
        angles = np.linalg.norm(pose_aa, axis=-1, keepdims=True) + 1e-8  # [T, 24, 1]
        axes   = pose_aa / angles                                          # [T, 24, 3]
        angles = angles.squeeze(-1)                                        # [T, 24]
        c, s   = np.cos(angles), np.sin(angles)
        t_     = 1 - c
        x, y, z = axes[..., 0], axes[..., 1], axes[..., 2]
        R = np.stack([
            c + t_*x*x,   t_*x*y - s*z, t_*x*z + s*y,
            t_*x*y + s*z, c + t_*y*y,   t_*y*z - s*x,
            t_*x*z - s*y, t_*y*z + s*x, c + t_*z*z,
        ], axis=-1).reshape(T, N_JOINTS, 3, 3)
        pose_6d = R[..., :2].transpose(0, 1, 3, 2).reshape(T, N_JOINTS * 6)  # [T, 144]

        all_imu.append(imu_signal.astype(np.float32))
        all_pose.append(pose_6d.astype(np.float32))

    if not all_imu:
        return None
    return np.concatenate(all_imu), np.concatenate(all_pose)


def make_windows(imu: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window → X [N, 30, WIN], y [N, 144] (center frame pose)."""
    T  = imu.shape[0]
    cx = WIN // 2
    X_list, y_list = [], []
    for start in range(0, T - WIN + 1, HOP):
        X_list.append(imu[start : start + WIN].T)          # [30, WIN]
        y_list.append(pose[start + cx])                     # [144]
    if not X_list:
        return np.empty((0, N_CH, WIN), np.float32), np.empty((0, OUT_DIM), np.float32)
    return np.stack(X_list), np.stack(y_list)


def load_split(data_dir: str, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        if not os.path.isdir(subj_dir):
            print(f"  [warn] {subj_dir} not found, skipping")
            continue
        result = load_subject(subj_dir)
        if result is None:
            continue
        imu, pose = result
        X, y = make_windows(imu, pose)
        all_X.append(X); all_y.append(y)
        print(f"  {subj}: {len(X)} windows  (imu {imu.shape}, pose {pose.shape})")
    if not all_X:
        return np.empty((0, N_CH, WIN), np.float32), np.empty((0, OUT_DIM), np.float32)
    return np.concatenate(all_X), np.concatenate(all_y)


# ── Dataset / Model ───────────────────────────────────────────────────────────

class PoseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class Conv1DPoseNet(nn.Module):
    """Same 3-layer 1D-CNN as HAR script, with regression head instead of classifier."""
    def __init__(self, n_ch: int = N_CH, out_dim: int = OUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 64,  kernel_size=8, padding=4), nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(64,   128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128,  256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).squeeze(-1))


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, device):
    model.train()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        loss = F.mse_loss(model(X), y)
        loss.backward(); opt.step()
        total_loss += loss.item() * len(y); n += len(y)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_err, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        total_err += rotation_error_deg(pred, y) * len(y)
        n += len(y)
    return total_err / n   # mean rotation error in degrees


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(label, train_X, train_y, test_X, test_y, cutoff_hz, args):
    device = torch.device(args.device)

    # Apply lowpass then z-norm
    X_tr = lowpass_fft(train_X.copy(), cutoff_hz) if cutoff_hz else train_X.copy()
    X_te = lowpass_fft(test_X.copy(),  cutoff_hz) if cutoff_hz else test_X.copy()
    X_tr, X_te = znorm(X_tr), znorm(X_te)

    # Val split: last 10% of train windows
    n_val = max(1, int(len(X_tr) * 0.1))
    X_val, y_val   = X_tr[-n_val:], train_y[-n_val:]
    X_tr2, y_tr2   = X_tr[:-n_val], train_y[:-n_val]

    train_loader = DataLoader(PoseDataset(X_tr2, y_tr2), batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(PoseDataset(X_val,  y_val),  batch_size=512, shuffle=False)
    test_loader  = DataLoader(PoseDataset(X_te,   test_y), batch_size=512, shuffle=False)

    model = Conv1DPoseNet().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_val, best_state = 1e9, None
    log_every = max(1, args.epochs // 10)

    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"  Input: {'full 60 Hz' if not cutoff_hz else f'0–{cutoff_hz} Hz lowpass'}")
    print(f"{'─'*62}")
    print(f"  {'Ep':>4}  {'train_loss':>10}  {'val_err_deg':>12}")

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, device)
        verr = evaluate(model, val_loader, device)
        sched.step()
        if verr < best_val:
            best_val   = verr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % log_every == 0 or ep == 1:
            print(f"  {ep:>4}  {loss:>10.6f}  {verr:>12.2f}°")

    model.load_state_dict(best_state)
    test_err = evaluate(model, test_loader, device)
    print(f"\n  Test rotation error : {test_err:.2f}°  (lower = better)")
    return {"label": label, "test_err_deg": test_err, "best_val_deg": best_val}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",   default="/mnt/nas2/imuposer_dataset")
    p.add_argument("--cutoff",     type=float, default=20.0,
                   help="Lowpass cutoff Hz for Model B (default: 20)")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device  : {args.device}")
    print(f"Epochs  : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"Cutoff  : {args.cutoff} Hz")
    print(f"Input   : right wrist acc(3) + ang_vel(3) = {N_CH} channels")
    print(f"Output  : {N_JOINTS} joints × 6D rotation = {OUT_DIM} dims")

    print(f"\nLoading train subjects {TRAIN_SUBJ} ...")
    train_X, train_y = load_split(args.data_dir, TRAIN_SUBJ)
    print(f"  Train windows: {len(train_X)}")

    print(f"\nLoading test subjects {TEST_SUBJ} ...")
    test_X, test_y = load_split(args.data_dir, TEST_SUBJ)
    print(f"  Test  windows: {len(test_X)}")

    if len(train_X) == 0 or len(test_X) == 0:
        raise RuntimeError("No data loaded — check --data-dir")

    results = []
    results.append(run_experiment("Model A — full 60 Hz",
                                  train_X, train_y, test_X, test_y,
                                  cutoff_hz=None, args=args))
    results.append(run_experiment(f"Model B — 0–{args.cutoff} Hz lowpass",
                                  train_X, train_y, test_X, test_y,
                                  cutoff_hz=args.cutoff, args=args))

    print(f"\n{'='*62}")
    print(f"  SUMMARY — IMUPoser test (P9, P10)")
    print(f"{'='*62}")
    print(f"  {'Model':<38}  {'rot_err_deg':>11}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['label']:<38}  {r['test_err_deg']:>10.2f}°")
    print(f"{'='*62}")

    delta = results[1]["test_err_deg"] - results[0]["test_err_deg"]
    if abs(delta) < 0.5:
        verdict = "negligible difference: high-freq content not important for pose"
    elif delta > 0:
        verdict = f"full signal better by {delta:.2f}°: high-freq helps pose estimation"
    else:
        verdict = f"lowpass better by {-delta:.2f}°: high-freq is mainly noise for pose"
    print(f"\n  → {verdict}\n")


if __name__ == "__main__":
    main()
