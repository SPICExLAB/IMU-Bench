#!/usr/bin/env python3
"""
Supervised odometry on Nymeria real wrist IMU — full 50 Hz vs lowpass.

Uses the actual ViT1D encoder + ViT1DOdomDecoder from the codebase.
Input : real right-wrist FLU acc(3) + gyro(3) = 6 channels @ 50 Hz
Target: per-patch pelvis FLU-frame velocity [B, P_t, 4, 3] → flattened [B, P_t, 12]
Metric: velocity MAE in m/s (lower = better)

Usage:
    python scripts/compare_full_vs_lowpass_odom_nymeria.py --config configs/vit1d_odom_supervised_nymeria_real_wrist_flu.yaml
    python scripts/compare_full_vs_lowpass_odom_nymeria.py --cutoff 8
    python scripts/compare_full_vs_lowpass_odom_nymeria.py --cutoff 4
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DOdomDecoder
from imu_lm.data.nymeria_real_wrist_odom_loader_FLU import make_nymeria_real_wrist_odom_loaders_flu
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.helpers import cfg_get

FS = 50.0


# ── Lowpass filter ────────────────────────────────────────────────────────────

def lowpass_fft_batch(x: torch.Tensor, cutoff_hz: float, fs: float = FS) -> torch.Tensor:
    """x: [B, C, T] → lowpass filtered [B, C, T] on the same device."""
    T = x.shape[2]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs, device=x.device)
    X = torch.fft.rfft(x.float(), dim=2)
    X[:, :, freqs >= cutoff_hz] = 0
    return torch.fft.irfft(X, n=T, dim=2).to(x.dtype)


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_epoch(encoder, odom_decoder, loader, opt, device, cutoff_hz):
    encoder.train(); odom_decoder.train()
    total, n = 0.0, 0
    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x, y = x.to(device), y.to(device)
        if cutoff_hz:
            x = lowpass_fft_batch(x, cutoff_hz)
        y_flat = y.reshape(y.shape[0], y.shape[1], -1)   # [B, Pt, 12]

        tokens, _ = encoder.forward_features(x)
        pred = odom_decoder(tokens)                        # [B, Pt, 12]

        opt.zero_grad()
        loss = F.mse_loss(pred, y_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(odom_decoder.parameters()), 1.0)
        opt.step()
        total += loss.item() * len(x); n += len(x)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(encoder, odom_decoder, loader, device, cutoff_hz):
    """Mean absolute error in m/s across all velocity predictions."""
    encoder.eval(); odom_decoder.eval()
    total_mae, n = 0.0, 0
    for batch in loader:
        if batch is None:
            continue
        x, y = batch
        x, y = x.to(device), y.to(device)
        if cutoff_hz:
            x = lowpass_fft_batch(x, cutoff_hz)
        y_flat = y.reshape(y.shape[0], y.shape[1], -1)

        tokens, _ = encoder.forward_features(x)
        pred = odom_decoder(tokens)

        mae = (pred - y_flat).abs().mean().item()
        total_mae += mae * len(x); n += len(x)
    return total_mae / max(n, 1)


# ── Experiment ────────────────────────────────────────────────────────────────

def run_experiment(label, cutoff_hz, loaders, cfg, args, device):
    T, _   = compute_T_and_hop(cfg)
    vit_cfg = cfg_get(cfg, ["vit1d"], {}) or {}
    patch_size = int(vit_cfg.get("patch_size", 16))
    num_patches_per_channel = T // patch_size

    encoder     = ViT1DEncoder(cfg).to(device)
    odom_decoder = ViT1DOdomDecoder(
        embed_dim=encoder.embed_dim,
        num_patches_per_channel=num_patches_per_channel,
        in_channels=encoder.in_channels,
        cfg=cfg,
        out_dim=12,   # 4 sub-patches × 3D FLU velocity
    ).to(device)

    all_params = list(encoder.parameters()) + list(odom_decoder.parameters())
    opt   = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_val, best_enc_state, best_dec_state = 1e9, None, None
    log_every = max(1, args.epochs // 10)

    train_loader = loaders["train_loader"]
    val_loader   = loaders.get("val_loader") or loaders.get("test_loader")

    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"  Input: {'full 50 Hz' if not cutoff_hz else f'0–{cutoff_hz} Hz lowpass'}")
    print(f"  Encoder: {encoder.embed_dim}D, {len(encoder.blocks)} layers  |  "
          f"Decoder: {num_patches_per_channel} patches, out_dim=12")
    print(f"{'─'*62}")
    print(f"  {'Ep':>4}  {'train_mse':>10}  {'val_mae_ms':>12}")

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(encoder, odom_decoder, train_loader, opt, device, cutoff_hz)
        vmae = evaluate(encoder, odom_decoder, val_loader, device, cutoff_hz)
        sched.step()
        if vmae < best_val:
            best_val       = vmae
            best_enc_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
            best_dec_state = {k: v.cpu().clone() for k, v in odom_decoder.state_dict().items()}
        if ep % log_every == 0 or ep == 1:
            print(f"  {ep:>4}  {loss:>10.6f}  {vmae:>11.4f} m/s")

    encoder.load_state_dict(best_enc_state)
    odom_decoder.load_state_dict(best_dec_state)

    test_loader = loaders.get("test_loader") or val_loader
    test_mae = evaluate(encoder, odom_decoder, test_loader, device, cutoff_hz)
    print(f"\n  Test velocity MAE: {test_mae:.4f} m/s  (lower = better)")
    return {"label": label, "test_mae_ms": test_mae}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",    default="configs/vit1d_odom_supervised_nymeria_real_wrist_flu.yaml")
    p.add_argument("--cutoff",    type=float, default=20.0)
    p.add_argument("--epochs",    type=int,   default=50)
    p.add_argument("--lr",        type=float, default=1e-4)
    p.add_argument("--max-files", type=int,   default=200,
                   help="Max train PKL files (default 200, -1 = all)")
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Patch config for faster standalone run
    cfg.setdefault("data", {})
    cfg["data"]["batch_size"]      = 256
    cfg["data"]["eval_batch_size"] = 512
    cfg["data"]["num_workers"]     = 4
    cfg["data"]["pin_memory"]      = True

    odom_cfg = cfg_get(cfg, ["objective", "odom_nymeria_real_wrist_flu"], {}) or {}
    train_root = odom_cfg.get("train_pkl_root", "/mnt/nas2/processed_data_body_Ty_rwrist")
    val_root   = odom_cfg.get("val_pkl_root",   train_root)

    train_files = sorted(glob.glob(os.path.join(train_root, "*.pkl")))
    val_files   = sorted(glob.glob(os.path.join(val_root,   "*.pkl")))

    if args.max_files > 0:
        rng = np.random.default_rng(42)
        rng.shuffle(train_files)
        train_files = train_files[:args.max_files]

    print(f"Device     : {device}")
    print(f"Epochs     : {args.epochs}  |  LR: {args.lr}")
    print(f"Cutoff     : {args.cutoff} Hz")
    print(f"Train files: {len(train_files)}  |  Val files: {len(val_files)}")
    print(f"Config     : {args.config}")

    loaders = make_nymeria_real_wrist_odom_loaders_flu(
        cfg=cfg,
        pkl_root="/",
        train_files=train_files,
        val_files=val_files,
        add_gravity=bool(odom_cfg.get("add_gravity", True)),
    )

    n_train = len(loaders["train_loader"].dataset)
    n_val   = len(loaders["val_loader"].dataset)
    print(f"Windows    : train={n_train}  val={n_val}")

    results = []
    results.append(run_experiment("Model A — full 50 Hz",
                                  cutoff_hz=None, loaders=loaders,
                                  cfg=cfg, args=args, device=device))
    results.append(run_experiment(f"Model B — 0–{args.cutoff} Hz lowpass",
                                  cutoff_hz=args.cutoff, loaders=loaders,
                                  cfg=cfg, args=args, device=device))

    print(f"\n{'='*62}")
    print(f"  SUMMARY — Nymeria real wrist → odometry (FLU velocity)")
    print(f"{'='*62}")
    print(f"  {'Model':<38}  {'vel_MAE_m/s':>11}")
    print(f"  {'-'*52}")
    for r in results:
        print(f"  {r['label']:<38}  {r['test_mae_ms']:>10.4f} m/s")
    print(f"{'='*62}")

    delta = results[1]["test_mae_ms"] - results[0]["test_mae_ms"]
    if abs(delta) < 0.002:
        verdict = "negligible: high-freq not important for odometry"
    elif delta > 0:
        verdict = f"full signal better by {delta:.4f} m/s: high-freq helps"
    else:
        verdict = f"lowpass better by {-delta:.4f} m/s: high-freq is noise"
    print(f"\n  → {verdict}\n")


if __name__ == "__main__":
    main()
