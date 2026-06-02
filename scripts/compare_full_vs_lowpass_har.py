#!/usr/bin/env python3
"""
Supervised HAR classification — full 50 Hz vs lowpass filtered signal.

Trains the same 1D-CNN twice on any parquet dataset:
  - Model A: full 50 Hz signal
  - Model B: 0-<cutoff> Hz lowpass filtered signal (FFT zeroing)

Split strategy:
  - >= 10 sessions: split by session (no cross-session leakage)
  - <  10 sessions: split by time within each session (70/15/15)

Usage:
    python scripts/compare_full_vs_lowpass_har.py --dataset utd_mhad --cutoff 8
    python scripts/compare_full_vs_lowpass_har.py --dataset pamap2   --cutoff 8
    python scripts/compare_full_vs_lowpass_har.py --dataset samosa   --cutoff 4
"""
from __future__ import annotations

import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

WIN      = 128
FS       = 50.0
IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ── Signal processing ─────────────────────────────────────────────────────────

def lowpass_fft(windows: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """windows [N, C, T] → lowpass filtered [N, C, T] via FFT zero-out."""
    freqs = np.fft.rfftfreq(windows.shape[2], d=1.0 / FS)
    F     = np.fft.rfft(windows, axis=2)
    F[:, :, freqs >= cutoff_hz] = 0
    return np.fft.irfft(F, n=windows.shape[2], axis=2).astype(np.float32)


def znorm(windows: np.ndarray) -> np.ndarray:
    mu  = windows.mean(axis=2, keepdims=True)
    std = windows.std(axis=2, keepdims=True) + 1e-8
    return (windows - mu) / std


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(parquet_path: str, dataset: str, hop: int, rng) -> tuple:
    needed = ["dataset", "dataset_activity_label", "session_id", "timestamp_ns"] + IMU_COLS
    df = pd.read_parquet(parquet_path, columns=needed,
                         filters=[("dataset", "==", dataset)])
    if df.empty:
        raise ValueError(f"Dataset '{dataset}' not found in parquet.")
    df = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)
    df["dataset_activity_label"] = df["dataset_activity_label"].fillna("unknown")

    le = LabelEncoder()
    df["label_int"] = le.fit_transform(df["dataset_activity_label"])
    n_classes     = len(le.classes_)
    unique_sess   = df["session_id"].unique()
    n_sess        = len(unique_sess)

    print(f"  Dataset   : {dataset}")
    print(f"  Sessions  : {n_sess}")
    print(f"  Classes   : {n_classes}  {list(le.classes_[:6])}{'...' if n_classes>6 else ''}")
    print(f"  Rows      : {len(df)}")

    splits_X  = {"train": [], "val": [], "test": []}
    splits_y  = {"train": [], "val": [], "test": []}

    # Use session split only when sessions outnumber classes enough that each
    # split subset is likely to contain all activity classes (i.e. sessions are
    # subjects, not per-activity recordings).  When n_sess ≈ n_classes the
    # sessions are probably activity recordings and splitting by session gives
    # disjoint class sets in train/val → balanced_accuracy = 0.
    use_session_split = n_sess >= 10 and n_sess > n_classes * 2

    if use_session_split:
        # Split by session ID
        rng.shuffle(unique_sess)
        t1 = int(n_sess * 0.70)
        t2 = int(n_sess * 0.85)
        sess_split = {}
        for s in unique_sess[:t1]:  sess_split[s] = "train"
        for s in unique_sess[t1:t2]: sess_split[s] = "val"
        for s in unique_sess[t2:]:  sess_split[s] = "test"
        print(f"  Split     : by session (train={t1}, val={t2-t1}, test={n_sess-t2})")
    else:
        print(f"  Split     : by time within session (70/15/15%)")

    for sess_id, grp in df.groupby("session_id"):
        grp  = grp.sort_values("timestamp_ns")
        vals = grp[IMU_COLS].values.T.astype(np.float32)   # [6, L]
        labs = grp["label_int"].values
        L    = vals.shape[1]
        if L < WIN:
            continue

        # Generate windows for this session
        sess_X, sess_y = [], []
        for start in range(0, L - WIN + 1, hop):
            wlabs  = labs[start : start + WIN]
            counts = np.bincount(wlabs, minlength=n_classes)
            if counts.max() < WIN * 0.6:
                continue
            sess_X.append(vals[:, start : start + WIN])
            sess_y.append(int(counts.argmax()))

        if not sess_X:
            continue
        sess_X = np.stack(sess_X)
        sess_y = np.array(sess_y, dtype=np.int64)
        n_win  = len(sess_X)

        if use_session_split:
            split = sess_split.get(sess_id, "train")
            splits_X[split].append(sess_X)
            splits_y[split].append(sess_y)
        else:
            # Time-based split within session
            t1 = int(n_win * 0.70)
            t2 = int(n_win * 0.85)
            splits_X["train"].append(sess_X[:t1])
            splits_X["val"].append(sess_X[t1:t2])
            splits_X["test"].append(sess_X[t2:])
            splits_y["train"].append(sess_y[:t1])
            splits_y["val"].append(sess_y[t1:t2])
            splits_y["test"].append(sess_y[t2:])

    splits = {}
    for split in ("train", "val", "test"):
        if splits_X[split]:
            X = np.concatenate(splits_X[split])
            y = np.concatenate(splits_y[split])
        else:
            X = np.empty((0, 6, WIN), dtype=np.float32)
            y = np.empty((0,), dtype=np.int64)
        splits[split] = (X, y)

    print(f"  Windows   : train={len(splits['train'][0])}  "
          f"val={len(splits['val'][0])}  test={len(splits['test'][0])}")
    return splits, n_classes, le.classes_


# ── Dataset / Model ───────────────────────────────────────────────────────────

class IMUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class Conv1DClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(6,   64,  kernel_size=8, padding=4), nn.BatchNorm1d(64),  nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(64,  128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1))


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        out  = model(X)
        loss = F.cross_entropy(out, y)
        loss.backward(); opt.step()
        loss_sum += loss.item() * len(y)
        correct  += (out.argmax(1) == y).sum().item()
        n        += len(y)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        preds.append(model(X.to(device)).argmax(1).cpu().numpy())
        trues.append(y.numpy())
    pred = np.concatenate(preds); true = np.concatenate(trues)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bal = balanced_accuracy_score(true, pred)
    return (pred == true).mean(), bal


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(label, splits, n_classes, cutoff_hz, args):
    device = torch.device(args.device)

    data = {}
    for split, (X, y) in splits.items():
        Xp = lowpass_fft(X.copy(), cutoff_hz) if cutoff_hz else X.copy()
        data[split] = (znorm(Xp), y)

    train_loader = DataLoader(IMUDataset(*data["train"]), batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(IMUDataset(*data["val"]),   batch_size=512, shuffle=False)
    test_loader  = DataLoader(IMUDataset(*data["test"]),  batch_size=512, shuffle=False)

    model = Conv1DClassifier(n_classes).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_val, best_state = -1.0, None
    log_every = max(1, args.epochs // 10)

    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"  Input: {'full 50 Hz' if not cutoff_hz else f'0–{cutoff_hz} Hz lowpass'}")
    print(f"{'─'*62}")
    print(f"  {'Ep':>4}  {'loss':>8}  {'train_acc':>10}  {'val_bal_acc':>12}")

    for ep in range(1, args.epochs + 1):
        loss, tacc = train_epoch(model, train_loader, opt, device)
        _, vbal    = evaluate(model, val_loader, device)
        sched.step()
        if vbal > best_val:
            best_val   = vbal
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if ep % log_every == 0 or ep == 1:
            print(f"  {ep:>4}  {loss:>8.4f}  {tacc:>10.3f}  {vbal:>12.3f}")

    model.load_state_dict(best_state)
    test_acc, test_bal = evaluate(model, test_loader, device)
    print(f"\n  Accuracy         : {test_acc:.4f}")
    print(f"  Balanced accuracy: {test_bal:.4f}")
    return {"label": label, "test_acc": test_acc, "test_bal": test_bal}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet",    default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--dataset",    default="utd_mhad",
                   help="Dataset name in parquet (default: utd_mhad)")
    p.add_argument("--cutoff",     type=float, default=8.0,
                   help="Lowpass cutoff Hz for Model B (default: 8)")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hop",        type=int,   default=32,
                   help="Sliding window hop samples (default: 32 = 0.64s)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Device : {args.device}")
    print(f"Epochs : {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"Cutoff : {args.cutoff} Hz\n")

    rng = np.random.default_rng(42)
    splits, n_classes, classes = load_dataset(
        args.parquet, args.dataset, hop=args.hop, rng=rng)

    results = []
    results.append(run_experiment("Model A — full 50 Hz",
                                  splits, n_classes, cutoff_hz=None, args=args))
    results.append(run_experiment(f"Model B — 0–{args.cutoff} Hz lowpass",
                                  splits, n_classes, cutoff_hz=args.cutoff, args=args))

    print(f"\n{'='*62}")
    print(f"  SUMMARY — {args.dataset} test set")
    print(f"{'='*62}")
    print(f"  {'Model':<38}  {'acc':>6}  {'bal_acc':>8}")
    print(f"  {'-'*56}")
    for r in results:
        print(f"  {r['label']:<38}  {r['test_acc']:>6.4f}  {r['test_bal']:>8.4f}")
    print(f"{'='*62}")

    delta = results[0]["test_bal"] - results[1]["test_bal"]
    if abs(delta) < 0.02:
        verdict = "negligible difference: that frequency band not important"
    elif delta > 0:
        verdict = f"full signal better by {delta:.3f}: high-freq carries useful info"
    else:
        verdict = f"lowpass better by {-delta:.3f}: high-freq is mainly noise"
    print(f"\n  → {verdict}\n")


if __name__ == "__main__":
    main()
