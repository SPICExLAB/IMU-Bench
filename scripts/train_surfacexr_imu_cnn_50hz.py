#!/usr/bin/env python3
"""Train a lightweight SurfaceXR IMU-only CNN at 50 Hz.

Architecture follows the IMU-only model described in the prompt:
raw 6-channel smartwatch IMU -> 4 frequency components per channel ->
three Conv2D(5x1)+GroupNorm+MaxPool+ReLU blocks -> dense heads for
gesture classification and touch/contact detection.

The original SurfaceXR model uses 200 Hz and a 32+ Hz band. At 50 Hz the
Nyquist frequency is 25 Hz, so this 50 Hz version uses:
  - 0.22-8 Hz
  - 8-20 Hz
  - 20-24 Hz
  - original signal
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import sosfiltfilt, butter
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.surfaceXR_dataloader import (  # noqa: E402
    ALL_USERS,
    RAW_IMU_PKL_DEFAULT,
    _collect_windows_50hz,
    _load_pkl,
)


@dataclass
class Metrics:
    loss: float
    gesture_acc: float
    touch_acc: float


class ChannelDropout2D(nn.Module):
    """Drop sensor/frequency columns from [B, C, T, F] feature maps."""

    def __init__(self, p: float = 0.2):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        keep = 1.0 - self.p
        mask = torch.empty(x.shape[0], 1, 1, x.shape[-1], device=x.device).bernoulli_(keep)
        return x * mask / keep


class SurfaceXRIMUCNN(nn.Module):
    def __init__(
        self,
        num_gesture_classes: int = 9,
        channel_dropout: float = 0.2,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.channel_dropout = ChannelDropout2D(channel_dropout)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.GroupNorm(4, 16),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.GroupNorm(8, 32),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.GroupNorm(8, 64),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.gesture_head = nn.Linear(embedding_dim, num_gesture_classes)
        self.touch_head = nn.Linear(embedding_dim, 2)

    def forward(self, x: torch.Tensor):
        # x: [B, T, 6, 4] -> [B, 1, T, 24]
        b, t, c, f = x.shape
        x = x.reshape(b, t, c * f).unsqueeze(1)
        x = self.channel_dropout(x)
        z = self.project(self.encoder(x))
        return self.gesture_head(z), self.touch_head(z)


def _safe_sos_filter(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Filter one [T, C] window; fall back to zeros if too short/degenerate."""
    try:
        return sosfiltfilt(sos, x, axis=0).astype(np.float32)
    except ValueError:
        return np.zeros_like(x, dtype=np.float32)


def decompose_frequency(x: np.ndarray, sample_rate: int = 50) -> np.ndarray:
    """x [N, 6, T] -> [N, T, 6, 4]."""
    assert sample_rate == 50, "this trainer intentionally uses only 50 Hz data"
    # Normalize each raw window to roughly [-1, 1].
    xt = np.transpose(x, (0, 2, 1)).astype(np.float32)  # [N,T,6]
    scale = np.percentile(np.abs(xt), 99, axis=(1, 2), keepdims=True)
    xt = np.clip(xt / np.maximum(scale, 1.0e-6), -1.0, 1.0)

    nyq = sample_rate / 2.0
    sos_low = butter(4, [0.22 / nyq, 8.0 / nyq], btype="bandpass", output="sos")
    sos_mid = butter(4, [8.0 / nyq, 20.0 / nyq], btype="bandpass", output="sos")
    sos_high = butter(4, 20.0 / nyq, btype="highpass", output="sos")
    out = np.empty((len(xt), xt.shape[1], xt.shape[2], 4), dtype=np.float32)
    for i, win in enumerate(xt):
        out[i, :, :, 0] = _safe_sos_filter(win, sos_low)
        out[i, :, :, 1] = _safe_sos_filter(win, sos_mid)
        out[i, :, :, 2] = _safe_sos_filter(win, sos_high)
        out[i, :, :, 3] = win
    return out


def load_surfacexr_50hz(
    pkl_path: str,
    split: str,
    test_user: str,
    window_samples: int,
    hop_samples: int,
    apply_flu: bool,
    seed: int,
):
    data = _load_pkl(pkl_path)
    if split == "lopo":
        if test_user not in ALL_USERS:
            raise ValueError(f"--test-user must be one of {ALL_USERS}, got {test_user}")
        train_users = [u for u in ALL_USERS if u != test_user]
        val_users = [test_user]
        xtr, ygtr = _collect_windows_50hz(data, train_users, "gesture", apply_flu, window_samples, hop_samples)
        _, yttr = _collect_windows_50hz(data, train_users, "touch", apply_flu, window_samples, hop_samples)
        xva, ygva = _collect_windows_50hz(data, val_users, "gesture", apply_flu, window_samples, hop_samples)
        _, ytva = _collect_windows_50hz(data, val_users, "touch", apply_flu, window_samples, hop_samples)
        return (xtr, ygtr, yttr), (xva, ygva, ytva)

    x, yg = _collect_windows_50hz(data, ALL_USERS, "gesture", apply_flu, window_samples, hop_samples)
    _, yt = _collect_windows_50hz(data, ALL_USERS, "touch", apply_flu, window_samples, hop_samples)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    x, yg, yt = x[idx], yg[idx], yt[idx]
    n_train = int(0.8 * len(x))
    n_val = int(0.1 * len(x))
    return (
        x[:n_train], yg[:n_train], yt[:n_train],
    ), (
        x[n_train:n_train + n_val],
        yg[n_train:n_train + n_val],
        yt[n_train:n_train + n_val],
    )


def make_loader(x, yg, yt, batch_size, shuffle):
    xf = decompose_frequency(x)
    ds = TensorDataset(
        torch.from_numpy(xf).float(),
        torch.from_numpy(yg).long(),
        torch.from_numpy(yt).long(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def run_epoch(model, loader, device, optimizer=None, touch_weight=0.5) -> Metrics:
    train = optimizer is not None
    model.train(train)
    total_loss = total_g = total_t = total_n = 0.0
    for xb, yg, yt in loader:
        xb, yg, yt = xb.to(device), yg.to(device), yt.to(device)
        if train:
            optimizer.zero_grad()
        logits_g, logits_t = model(xb)
        loss_g = F.cross_entropy(logits_g, yg)
        loss_t = F.cross_entropy(logits_t, yt)
        loss = loss_g + touch_weight * loss_t
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        n = xb.shape[0]
        total_loss += float(loss.detach()) * n
        total_g += float((logits_g.argmax(1) == yg).float().sum())
        total_t += float((logits_t.argmax(1) == yt).float().sum())
        total_n += n
    return Metrics(total_loss / total_n, total_g / total_n, total_t / total_n)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pkl-path", default=RAW_IMU_PKL_DEFAULT)
    p.add_argument("--split", choices=["lopo", "random"], default="lopo")
    p.add_argument("--test-user", default="Timmy")
    p.add_argument("--window-seconds", type=float, default=1.0)
    p.add_argument("--hop-seconds", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--touch-weight", type=float, default=0.5)
    p.add_argument("--channel-dropout", type=float, default=0.2)
    p.add_argument("--apply-flu", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default="surfacexr_imu_cnn_50hz")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    window_samples = int(round(args.window_seconds * 50))
    hop_samples = int(round(args.hop_seconds * 50))
    run_dir = os.path.join(args.runs_root, args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    print(f"Loading SurfaceXR 50 Hz windows from {args.pkl_path}")
    print(f"  window={window_samples} samples ({args.window_seconds:.2f}s), hop={hop_samples}")
    train, val = load_surfacexr_50hz(
        args.pkl_path, args.split, args.test_user,
        window_samples, hop_samples, args.apply_flu, args.seed)
    print(f"  train={len(train[0])} val={len(val[0])}")
    train_loader = make_loader(*train, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(*val, batch_size=args.batch_size, shuffle=False)

    model = SurfaceXRIMUCNN(channel_dropout=args.channel_dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,} ({n_params/1e3:.1f}K)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1.0
    rows = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, opt, args.touch_weight)
        with torch.no_grad():
            va = run_epoch(model, val_loader, device, None, args.touch_weight)
        score = va.gesture_acc
        rows.append({"epoch": epoch, **{f"train_{k}": v for k, v in asdict(tr).items()},
                     **{f"val_{k}": v for k, v in asdict(va).items()}})
        if score > best:
            best = score
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "num_params": n_params,
                "val": asdict(va),
            }, os.path.join(ckpt_dir, "best.pt"))
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch {epoch:03d} | "
                f"train gesture={tr.gesture_acc:.3f} touch={tr.touch_acc:.3f} | "
                f"val gesture={va.gesture_acc:.3f} touch={va.touch_acc:.3f}"
            )
    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": args.epochs},
               os.path.join(ckpt_dir, "latest.pt"))
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(run_dir, "logs", "metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done. best val gesture acc={best:.3f} -> {run_dir}")


if __name__ == "__main__":
    main()
