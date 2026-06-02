#!/usr/bin/env python3
"""Evaluate MAE reconstruction loss on the OCDetect dataset.

Replicates the exact random 80/10/10 window-level split used during training
(seed=42) so the reported val/test loss is directly comparable to training logs.

Usage
-----
    python scripts/eval_mae_ocdetect.py \
        --ckpt    runs/vit1d_mae_ocdetect_flu/checkpoints/best.pt \
        [--data-dir /mnt/nas2/yiquan_data/preprocessed] \
        [--batch-size 256]
        [--seed 42]
        [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import random as pyrandom
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.OCDetect_dataloader import _collect_windows, _list_files
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoder
from imu_lm.objectives.masked_1d import generate_patch_mask

WINDOW_LEN = 128


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval().to(device)

    out_ch = len(cfg.get("data", {}).get("sensor_columns", [""] * 6))
    decoder = ViT1DDecoder(
        embed_dim    = encoder.embed_dim,
        out_channels = out_ch,
        target_T     = WINDOW_LEN,
        cfg          = cfg,
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval().to(device)

    obj_cfg       = cfg.get("objective", {})
    mask_ratio    = float(obj_cfg.get("mask_ratio", 0.75))
    mask_strategy = str(obj_cfg.get("mask_strategy", "composite"))
    # Read split config from checkpoint
    mae_cfg = obj_cfg.get(obj_cfg.get("type", "mae_ocdetect_flu"), {}) or {}
    splits  = mae_cfg.get("splits", [0.8, 0.1, 0.1])

    return encoder, decoder, cfg, mask_ratio, mask_strategy, splits


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_windows(windows: torch.Tensor, encoder, decoder,
                 mask_ratio, mask_strategy, batch_size, seed, device) -> dict:
    """windows: [M, C, T] float32 tensor, already preprocessed."""
    torch.manual_seed(seed)
    pyrandom.seed(seed)

    loader     = DataLoader(TensorDataset(windows), batch_size=batch_size,
                            shuffle=False, num_workers=0)
    patch_size = encoder.patch_size
    losses     = []

    for (x,) in loader:
        x = x.to(device)
        B, C, T = x.shape
        Pt = T // patch_size

        mask = generate_patch_mask(B, Pt, C, mask_ratio,
                                   strategy=mask_strategy, device=device)
        vis_tok, vis_pad = encoder(x, mask=mask, return_keep_padding_mask=True)
        recon = decoder(vis_tok, mask, Pt, C, visible_padding_mask=vis_pad)

        sm = mask.view(B, C, Pt).unsqueeze(-1).expand(-1, -1, -1, patch_size).reshape(B, C, T)
        losses.append(F.mse_loss(recon[sm], x[sm]).item())

    return {
        "loss":      float(np.mean(losses)),
        "loss_std":  float(np.std(losses)),
        "n_windows": len(windows),
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_raw_windows(X_raw: np.ndarray, cfg: dict) -> torch.Tensor:
    """X_raw: [M, 6, 128] float32 (FLU, raw) → [M, 6, 128] preprocessed tensor."""
    out = []
    for i in range(len(X_raw)):
        Xw = X_raw[i].T  # [128, 6]
        Xp = preprocess_window(Xw, cfg)
        if Xp is None:
            # fallback: plain zscore
            Xp = (Xw - Xw.mean(0, keepdims=True)) / (Xw.std(0, keepdims=True) + 1e-6)
            Xp = Xp.T.astype(np.float32)
        out.append(torch.from_numpy(Xp).float())
    return torch.stack(out, dim=0)  # [M, 6, 128]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--data-dir",   default="/mnt/nas2/yiquan_data/preprocessed")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed",       type=int, default=42,
                   help="RNG seed matching the training split (default 42)")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    encoder, decoder, cfg, mask_ratio, mask_strategy, splits = load_model(args.ckpt, device)
    print(f"  mask_ratio={mask_ratio}  strategy={mask_strategy}  splits={splits}")

    print(f"\nCollecting windows from: {args.data_dir}")
    all_files = _list_files(args.data_dir)
    print(f"  {len(all_files)} files found")
    X_raw, _ = _collect_windows(all_files, window_samples=WINDOW_LEN, hop_samples=WINDOW_LEN // 2)
    print(f"  {len(X_raw)} total windows collected")

    # Replicate the exact training split
    rng  = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X_raw))
    X_raw = X_raw[perm]

    n       = len(X_raw)
    n_train = int(splits[0] * n)
    n_val   = int(splits[1] * n)

    split_data = {
        "val":  X_raw[n_train : n_train + n_val],
        "test": X_raw[n_train + n_val :],
    }

    results = {}
    for name, X_split in split_data.items():
        print(f"\n[{name}] {len(X_split)} windows — preprocessing … ", end="", flush=True)
        windows = preprocess_raw_windows(X_split, cfg)
        print(f"done. Evaluating … ", end="", flush=True)
        res = eval_windows(windows, encoder, decoder,
                           mask_ratio, mask_strategy, args.batch_size, args.seed, device)
        results[name] = res
        print(f"loss={res['loss']:.4f} ± {res['loss_std']:.4f}")

    print("\n" + "=" * 52)
    print(f"{'Split':<8}  {'MSE loss':>10}  {'±std':>8}  {'n':>8}")
    print("-" * 52)
    for name, r in results.items():
        print(f"  {name:<6}  {r['loss']:>10.4f}  {r['loss_std']:>8.4f}  {r['n_windows']:>8}")
    print("=" * 52)


if __name__ == "__main__":
    main()
