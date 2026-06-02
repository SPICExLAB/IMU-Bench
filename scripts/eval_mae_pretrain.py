#!/usr/bin/env python3
"""Evaluate MAE reconstruction loss per dataset for a pretrained checkpoint.

Loads the encoder + decoder from a checkpoint, iterates over windows from
each training dataset in the parquet, and reports the average MSE loss
(masked patches only) per dataset — identical to the training objective.

Usage
-----
    python scripts/eval_mae_pretrain.py \
        --ckpt    runs/vit1d_realdata_pretrain/checkpoints/best.pt \
        --parquet /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
        [--n-windows 200]   # windows per dataset (default 200)
        [--batch-size 64]
        [--seed 0]
        [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoder
from imu_lm.objectives.masked_1d import generate_patch_mask

WINDOW_LEN     = 128
PARQUET_COLS   = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]



# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device):
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg     = ckpt["cfg"]

    def _remap(sd):
        out = {}
        for k, v in sd.items():
            k = k.replace(".self_attn.", ".attn.")
            k = k.replace(".linear1.", ".ff.0.")
            k = k.replace(".linear2.", ".ff.3.")
            out[k] = v
        return out

    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(_remap(ckpt["model"]))
    encoder.eval().to(device)

    out_channels = len(cfg.get("data", {}).get("sensor_columns", PARQUET_COLS))
    decoder = ViT1DDecoder(
        embed_dim    = encoder.embed_dim,
        out_channels = out_channels,
        target_T     = WINDOW_LEN,
        cfg          = cfg,
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval().to(device)

    mask_ratio = float(cfg.get("objective", {}).get("mask_ratio", 0.75))
    mask_strategy = str(cfg.get("objective", {}).get("mask_strategy", "composite"))
    train_datasets = cfg.get("data", {}).get("train_on_datasets", None)
    return encoder, decoder, cfg, mask_ratio, mask_strategy, train_datasets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_dataset_from_windows(
    windows: list,
    encoder: ViT1DEncoder,
    decoder: ViT1DDecoder,
    mask_ratio: float,
    mask_strategy: str,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    """MSE loss (masked patches only) over a list of pre-processed window tensors."""
    import random as pyrandom
    torch.manual_seed(seed)
    pyrandom.seed(seed)   # seeds composite strategy selection in generate_patch_mask

    loader = DataLoader(windows, batch_size=batch_size, shuffle=False, num_workers=0)
    patch_size = encoder.patch_size
    losses = []

    for x in loader:
        x = x.to(device)
        B, C, T = x.shape
        Pt = T // patch_size

        patch_mask = generate_patch_mask(
            B, Pt, C, mask_ratio, strategy=mask_strategy, device=device,
        )
        visible_tokens, vis_pad = encoder(x, mask=patch_mask,
                                          return_keep_padding_mask=True)
        recon = decoder(visible_tokens, patch_mask, Pt, C,
                        visible_padding_mask=vis_pad)

        pm_2d       = patch_mask.view(B, C, Pt)
        signal_mask = pm_2d.unsqueeze(-1).expand(-1, -1, -1, patch_size).reshape(B, C, T)
        losses.append(F.mse_loss(recon[signal_mask], x[signal_mask]).item())

    return {
        "loss":      float(np.mean(losses)),
        "loss_std":  float(np.std(losses)),
        "n_windows": len(windows),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--parquet",    required=True)
    p.add_argument("--n-windows",  type=int, default=0,
                   help="Windows to evaluate per dataset (0 = all)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    encoder, decoder, cfg, mask_ratio, mask_strategy, train_datasets = \
        load_model(args.ckpt, device)

    print(f"  mask_ratio={mask_ratio}  strategy={mask_strategy}")
    print(f"  Training datasets: {train_datasets}\n")

    results = {}

    for dataset in train_datasets:
        print(f"[{dataset}] loading all sessions …", end=" ", flush=True)

        # Load ALL sessions for this dataset (pyarrow row-group filter)
        df = pd.read_parquet(
            args.parquet,
            columns=["session_id", "timestamp_ns"] + PARQUET_COLS,
            filters=[("dataset", "==", dataset)],
        )
        if df.empty:
            print("NOT FOUND — skipping")
            results[dataset] = {"loss": float("nan"), "n_windows": 0}
            continue

        # Window each session with the same hop as training (50% overlap)
        hop       = WINDOW_LEN // 2   # matches windowing.window_hop_ratio: 0.5
        sessions  = df["session_id"].unique()
        all_wins: list[torch.Tensor] = []
        for sess in sessions:
            sess_df = df[df["session_id"] == sess].sort_values("timestamp_ns")
            imu     = sess_df[PARQUET_COLS].values.astype(np.float32)
            start   = 0
            while start + WINDOW_LEN <= len(imu):
                raw = imu[start : start + WINDOW_LEN]
                w   = preprocess_window(raw, cfg)
                if w is not None:
                    all_wins.append(torch.from_numpy(w).float())
                start += hop
                if args.n_windows > 0 and len(all_wins) >= args.n_windows:
                    break
            if args.n_windows > 0 and len(all_wins) >= args.n_windows:
                break

        n_total = len(all_wins)
        print(f"{len(sessions)} sessions, {n_total} windows …", end=" ", flush=True)

        if n_total == 0:
            print("no valid windows — skipping")
            results[dataset] = {"loss": float("nan"), "loss_std": float("nan"), "n_windows": 0}
            continue

        res = eval_dataset_from_windows(
            all_wins, encoder, decoder,
            mask_ratio, mask_strategy,
            args.batch_size, args.seed, device,
        )
        results[dataset] = res
        print(f"loss={res['loss']:.4f} ± {res['loss_std']:.4f}  "
              f"(n={res['n_windows']})")

    # Summary table
    print("\n" + "=" * 52)
    print(f"{'Dataset':<18}  {'MSE loss':>10}  {'±std':>8}  {'n':>6}")
    print("-" * 52)
    valid = {k: v for k, v in results.items() if not np.isnan(v["loss"])}
    for ds, r in results.items():
        if np.isnan(r["loss"]):
            print(f"  {ds:<16}  {'N/A':>10}")
        else:
            print(f"  {ds:<16}  {r['loss']:>10.4f}  {r['loss_std']:>8.4f}  {r['n_windows']:>6}")
    if valid:
        losses = np.array([v["loss"] for v in valid.values()])
        ns     = np.array([v["n_windows"] for v in valid.values()])
        avg_unweighted = float(np.mean(losses))
        avg_weighted   = float(np.sum(losses * ns) / np.sum(ns))
        print("-" * 52)
        print(f"  {'MEAN (unweighted)':<16}  {avg_unweighted:>10.4f}")
        print(f"  {'MEAN (weighted)':<16}  {avg_weighted:>10.4f}  (by n_windows)")
    print("=" * 52)


if __name__ == "__main__":
    main()
