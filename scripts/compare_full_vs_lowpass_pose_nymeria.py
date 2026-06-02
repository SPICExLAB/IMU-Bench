#!/usr/bin/env python3
"""
Supervised full-body pose estimation on Nymeria real wrist IMU.

Trains the same 1D-CNN regressor twice:
  - Model A: full 50 Hz signal
  - Model B: 0-<cutoff> Hz lowpass filtered signal (FFT zeroing)

Input : real right-wrist acc(3) + gyro(3) = 6 channels @ 50 Hz
Target: 23-joint xsens_pose as 6D rotation (23 × 6 = 138 dims)
Metric: mean per-joint rotation error in degrees (lower = better)
Split : random 80% subjects train / 20% subjects test

Usage:
    python scripts/compare_full_vs_lowpass_pose_nymeria.py
    python scripts/compare_full_vs_lowpass_pose_nymeria.py --cutoff 8
    python scripts/compare_full_vs_lowpass_pose_nymeria.py --cutoff 4
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

FS       = 50.0
WIN      = 128      # 128 frames @ 50 Hz = 2.56 s
HOP      = 32       # 0.64 s hop
N_CH     = 6        # acc(3) + gyro(3)
N_JOINTS = 23       # Nymeria uses 23 xsens joints
OUT_DIM  = N_JOINTS * 6   # 6D rotation = 138


# ── Rotation utilities ────────────────────────────────────────────────────────

def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """R [..., 3, 3] → 6D [..., 6] (first two columns, row-major)."""
    return R[..., :3, :2].transpose(*range(R.ndim - 2), R.ndim - 1, R.ndim - 2
                                    ).reshape(*R.shape[:-2], 6)


@torch.no_grad()
def rotation_error_deg(pred_6d: torch.Tensor, true_6d: torch.Tensor) -> float:
    """Mean per-joint rotation error in degrees. [N, 138]."""
    B = pred_6d.shape[0]
    p = pred_6d.view(B, N_JOINTS, 6)
    t = true_6d.view(B, N_JOINTS, 6)

    def to_rotmat(x):
        a1, a2 = x[..., :3], x[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    R_diff = to_rotmat(p).transpose(-1, -2) @ to_rotmat(t)
    trace  = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    err    = torch.acos(((trace - 1) / 2).clamp(-1, 1))
    return err.mean().item() * 180.0 / np.pi


# ── Signal processing ─────────────────────────────────────────────────────────

def lowpass_fft(windows: np.ndarray, cutoff_hz: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(windows.shape[2], d=1.0 / FS)
    F_    = np.fft.rfft(windows, axis=2)
    F_[:, :, freqs >= cutoff_hz] = 0
    return np.fft.irfft(F_, n=windows.shape[2], axis=2).astype(np.float32)


def znorm(windows: np.ndarray) -> np.ndarray:
    mu  = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True) + 1e-8
    return (windows - mu) / std


# ── Data loading ──────────────────────────────────────────────────────────────

def subject_from_path(fpath: str) -> str:
    """Extract subject id (e.g. 's0_james_johnson') from filename."""
    parts = os.path.basename(fpath).split("_")
    # format: date_sX_firstname_lastname_actY_...
    return "_".join(parts[1:4])   # e.g. s0_james_johnson


def load_file(fpath: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one PKL → (imu [T,6], pose_6d [T,138]) or None."""
    try:
        d = pickle.load(open(fpath, "rb"))
    except Exception:
        return None

    if not d.get("has_wrist_imu", False):
        return None

    acc  = np.array(d["wrist_imu_data"]["accel"], dtype=np.float32)  # [T, 3]
    gyro = np.array(d["wrist_imu_data"]["gyro"],  dtype=np.float32)  # [T, 3]
    imu  = np.concatenate([acc, gyro], axis=1)                        # [T, 6]

    xpose = np.array(d["gt_data"]["xsens_pose"], dtype=np.float32)   # [T, 23, 4, 4]
    R     = xpose[:, :, :3, :3]                                       # [T, 23, 3, 3]
    # 6D: take first two columns of each rotation matrix
    pose_6d = R[:, :, :3, :2].transpose(0, 1, 3, 2).reshape(-1, N_JOINTS * 6)  # [T, 138]

    T_imu  = imu.shape[0]
    T_pose = pose_6d.shape[0]
    T      = min(T_imu, T_pose)
    if T < WIN:
        return None

    return imu[:T], pose_6d[:T]


def make_windows(imu: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window → X [N, 6, WIN], y [N, 138] (centre-frame pose)."""
    T, cx = imu.shape[0], WIN // 2
    X_list, y_list = [], []
    for start in range(0, T - WIN + 1, HOP):
        X_list.append(imu[start : start + WIN].T)    # [6, WIN]
        y_list.append(pose[start + cx])               # [138]
    if not X_list:
        return (np.empty((0, N_CH, WIN), np.float32),
                np.empty((0, OUT_DIM),   np.float32))
    return np.stack(X_list), np.stack(y_list)


def load_split(files: list[str], desc: str) -> tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []
    for fpath in files:
        result = load_file(fpath)
        if result is None:
            continue
        X, y = make_windows(*result)
        if len(X) == 0:
            continue
        all_X.append(X); all_y.append(y)
    if not all_X:
        return (np.empty((0, N_CH, WIN), np.float32),
                np.empty((0, OUT_DIM),   np.float32))
    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    print(f"  {desc}: {len(files)} files → {len(X)} windows")
    return X, y


# ── Dataset / Model ───────────────────────────────────────────────────────────

class PoseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class Conv1DPoseNet(nn.Module):
    """Same 3-layer 1D-CNN as HAR script, with regression head."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_CH, 64,  kernel_size=8, padding=4), nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(64,   128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128,  256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(256, OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x).squeeze(-1))


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, device):
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        loss = F.mse_loss(model(X), y)
        loss.backward(); opt.step()
        total += loss.item() * len(y); n += len(y)
    return total / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_err, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        total_err += rotation_error_deg(model(X), y) * len(y)
        n += len(y)
    return total_err / n


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(label, train_X, train_y, test_X, test_y, cutoff_hz, args):
    device = torch.device(args.device)

    X_tr = lowpass_fft(train_X.copy(), cutoff_hz) if cutoff_hz else train_X.copy()
    X_te = lowpass_fft(test_X.copy(),  cutoff_hz) if cutoff_hz else test_X.copy()
    X_tr, X_te = znorm(X_tr), znorm(X_te)

    n_val        = max(1, int(len(X_tr) * 0.1))
    X_val, y_val = X_tr[-n_val:], train_y[-n_val:]
    X_tr2, y_tr2 = X_tr[:-n_val], train_y[:-n_val]

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
    print(f"  Input: {'full 50 Hz' if not cutoff_hz else f'0–{cutoff_hz} Hz lowpass'}")
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
    return {"label": label, "test_err_deg": test_err}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",   default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--cutoff",     type=float, default=20.0)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--max-files",  type=int,   default=300,
                   help="Max PKL files to load (default 300, use -1 for all 867)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device  : {args.device}")
    print(f"Epochs  : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"Cutoff  : {args.cutoff} Hz")
    print(f"Input   : real wrist acc(3) + gyro(3) = {N_CH} channels @ {FS} Hz")
    print(f"Output  : {N_JOINTS} joints × 6D rotation = {OUT_DIM} dims")

    # Collect and shuffle files
    all_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pkl")))
    rng = np.random.default_rng(42)
    rng.shuffle(all_files)
    if args.max_files > 0:
        all_files = all_files[:args.max_files]

    # Split by subject (80/20)
    subjects   = list(dict.fromkeys(subject_from_path(f) for f in all_files))
    n_test     = max(1, int(len(subjects) * 0.2))
    test_subjs = set(subjects[-n_test:])
    train_subjs= set(subjects[:-n_test])

    train_files = [f for f in all_files if subject_from_path(f) in train_subjs]
    test_files  = [f for f in all_files if subject_from_path(f) in test_subjs]

    print(f"\nSubjects: {len(subjects)} total  "
          f"({len(train_subjs)} train / {len(test_subjs)} test)")

    print("\nLoading train ...")
    train_X, train_y = load_split(train_files, "train")

    print("Loading test ...")
    test_X, test_y = load_split(test_files, "test")

    if len(train_X) == 0 or len(test_X) == 0:
        raise RuntimeError("No data loaded — check --data-dir")

    results = []
    results.append(run_experiment("Model A — full 50 Hz",
                                  train_X, train_y, test_X, test_y,
                                  cutoff_hz=None, args=args))
    results.append(run_experiment(f"Model B — 0–{args.cutoff} Hz lowpass",
                                  train_X, train_y, test_X, test_y,
                                  cutoff_hz=args.cutoff, args=args))

    print(f"\n{'='*62}")
    print(f"  SUMMARY — Nymeria real wrist → full body pose")
    print(f"{'='*62}")
    print(f"  {'Model':<38}  {'rot_err_deg':>11}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['label']:<38}  {r['test_err_deg']:>10.2f}°")
    print(f"{'='*62}")

    delta = results[1]["test_err_deg"] - results[0]["test_err_deg"]
    if abs(delta) < 0.5:
        verdict = "negligible: high-freq not important for pose"
    elif delta > 0:
        verdict = f"full signal better by {delta:.2f}°: high-freq helps"
    else:
        verdict = f"lowpass better by {-delta:.2f}°: high-freq is noise"
    print(f"\n  → {verdict}\n")


if __name__ == "__main__":
    main()
