#!/usr/bin/env python3
"""Visualize MAE reconstruction for vit1d_mae_ocdetect_flu_no_norm_aug.

Loads the trained checkpoint, runs MAE on a chosen OCDetect window,
and saves a side-by-side GT vs reconstructed plot with masked patches shaded.

mask_ratio and mask_strategy are read from --model-config YAML (merged with
--config base), exactly as during training.  The checkpoint config is only
used as a fallback when no YAML is supplied.

Usage
-----
    python scripts/vis_mae_ocdetect_no_norm.py \
        --config       configs/base.yaml \
        --model-config configs/vit1d_mae_ocdetect_flu_no_norm_aug.yaml \
        --ckpt         runs/vit1d_mae_ocdetect_flu_no_norm_aug/checkpoints/best.pt \
        --csv          /mnt/nas2/yiquan_data/preprocessed/OCDetect_01_recording_00_55ef074f-147f-424f-b9c6-58c0cf9dfee0.csv \
        --window-idx   20 \
        --out          data_vis/mae_recon_no_norm_aug.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoder
from imu_lm.objectives.masked_1d import generate_patch_mask
from imu_lm.utils.helpers import deep_update, load_yaml

WINDOW_LEN = 128          # 2.56 s @ 50 Hz
IMU_COLS   = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
CH_NAMES   = ["acc_F (m/s²)", "acc_L (m/s²)", "acc_U (m/s²)",
               "gyro_F (rad/s)", "gyro_L (rad/s)", "gyro_U (rad/s)"]
SAMPLE_HZ  = 50


# ── model ────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device, yaml_cfg: dict | None = None):
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg  = ckpt["cfg"]

    encoder = ViT1DEncoder(ckpt_cfg)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval().to(device)

    decoder = ViT1DDecoder(
        embed_dim    = encoder.embed_dim,
        out_channels = 6,
        target_T     = WINDOW_LEN,
        cfg          = ckpt_cfg,
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval().to(device)

    # mask_ratio / strategy: prefer the live YAML config over the frozen ckpt config
    src_cfg    = yaml_cfg if yaml_cfg is not None else ckpt_cfg
    obj_cfg    = src_cfg.get("objective", {})
    mask_ratio = float(obj_cfg.get("mask_ratio", 0.75))
    strategy   = str(obj_cfg.get("mask_strategy", "composite"))

    source = "YAML config" if yaml_cfg is not None else "checkpoint config"
    print(f"  mask_ratio={mask_ratio}  strategy={strategy}  (from {source})")
    return encoder, decoder, ckpt_cfg, mask_ratio, strategy


# ── data ─────────────────────────────────────────────────────────────────────

def load_window(csv_path: str, cfg: dict, window_idx: int):
    """Load one clean window from an OCDetect CSV, apply FLU transform + preprocess."""
    df     = pd.read_csv(csv_path, usecols=["ignore"] + IMU_COLS, low_memory=False)
    imu    = df[IMU_COLS].values.astype(np.float32)
    ignore = df["ignore"].values.astype(np.int32)

    # Android → FLU: F = +Y_android, L = -X_android, U = +Z_android
    acc_x, acc_y   = imu[:, 0].copy(), imu[:, 1].copy()
    gyro_x, gyro_y = imu[:, 3].copy(), imu[:, 4].copy()
    imu[:, 0] =  acc_y;  imu[:, 1] = -acc_x
    imu[:, 3] =  gyro_y; imu[:, 4] = -gyro_x

    # Collect clean windows (ignore == 0), pick window_idx-th one
    usable = (ignore == 0)
    changes = np.diff(usable.astype(np.int8), prepend=0, append=0)
    seg_starts = np.where(changes ==  1)[0]
    seg_ends   = np.where(changes == -1)[0]

    windows = []
    for s, e in zip(seg_starts, seg_ends):
        seg_len = int(e - s)
        if seg_len < WINDOW_LEN:
            continue
        n_wins = (seg_len - WINDOW_LEN) // (WINDOW_LEN // 2) + 1
        for i in range(n_wins):
            start = s + i * (WINDOW_LEN // 2)
            windows.append(imu[start : start + WINDOW_LEN])   # [128, 6]

    if not windows:
        raise RuntimeError("No usable windows found in this CSV.")

    idx = window_idx % len(windows)
    raw = windows[idx]   # [128, 6]

    processed = preprocess_window(raw, cfg)  # [6, 128] or None
    if processed is None:
        # fallback: no normalization (matches no_norm config)
        processed = raw.T.astype(np.float32)  # [6, 128]

    print(f"  Loaded window {idx}/{len(windows)-1}  "
          f"(acc_F mean={raw[:,0].mean():.2f}  gyro_F mean={raw[:,3].mean():.3f})")
    return raw.T, processed   # raw [6,128], processed [6,128]


# ── reconstruction ────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct(window_proc: np.ndarray, encoder, decoder,
                mask_ratio: float, strategy: str, device: torch.device):
    """window_proc: [6, 128].  Returns recon [6,128], patch_mask [C*Pt], losses."""
    x = torch.from_numpy(window_proc).unsqueeze(0).float().to(device)   # [1, 6, 128]
    B, C, T = x.shape
    Pt = T // encoder.patch_size

    patch_mask = generate_patch_mask(B, Pt, C, mask_ratio,
                                     strategy=strategy, device=device)
    vis_tok, vis_pad = encoder(x, mask=patch_mask, return_keep_padding_mask=True)
    recon = decoder(vis_tok, patch_mask, Pt, C, visible_padding_mask=vis_pad)  # [1,6,128]

    # Expand patch mask → signal mask [1, 6, 128]
    sm = (patch_mask.view(B, C, Pt)
                    .unsqueeze(-1)
                    .expand(-1, -1, -1, encoder.patch_size)
                    .reshape(B, C, T))

    loss_total = torch.nn.functional.mse_loss(recon[sm], x[sm]).item()

    recon_np = recon.squeeze(0).cpu().numpy()    # [6, 128]
    x_np     = x.squeeze(0).cpu().numpy()        # [6, 128]
    sm_np    = sm.squeeze(0).cpu().numpy()        # [6, 128]

    loss_per_ch = [
        float(np.mean((recon_np[c][sm_np[c]] - x_np[c][sm_np[c]]) ** 2))
        if sm_np[c].any() else float("nan")
        for c in range(C)
    ]

    return recon_np, patch_mask.squeeze(0).cpu().numpy(), loss_total, loss_per_ch


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(raw: np.ndarray, processed: np.ndarray, recon: np.ndarray,
         patch_mask: np.ndarray, patch_size: int, Pt: int,
         loss_total: float, loss_per_ch: list, out_path: str, mask_ratio: float):
    """
    Two columns:
      left  — raw FLU signal (no normalization)
      right — processed input to model + reconstruction
    """
    C  = raw.shape[0]
    t  = np.arange(WINDOW_LEN) / float(SAMPLE_HZ)

    fig, axes = plt.subplots(C, 2, figsize=(16, 11), sharex=True)
    fig.suptitle(
        f"MAE Reconstruction  |  mask {int(mask_ratio*100)}%"
        f"  |  MSE (masked) = {loss_total:.4f}",
        fontsize=13, fontweight="bold",
    )

    col_titles = ["Raw FLU input (no z-norm)", "Model input vs Reconstruction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, pad=4)

    for c in range(C):
        # ── left: raw signal ──
        ax_raw = axes[c, 0]
        ax_raw.plot(t, raw[c], color="steelblue", lw=1.2)
        ax_raw.set_ylabel(CH_NAMES[c], fontsize=8)
        ax_raw.tick_params(labelsize=7)
        ax_raw.grid(axis="y", lw=0.4, alpha=0.5)

        # ── right: processed input + reconstruction + masked patches ──
        ax_rec = axes[c, 1]
        for p in range(Pt):
            if patch_mask[c * Pt + p]:
                ax_rec.axvspan(p * patch_size / SAMPLE_HZ,
                               (p + 1) * patch_size / SAMPLE_HZ,
                               color="lightgrey", alpha=0.65, zorder=0)

        ax_rec.plot(t, processed[c], color="steelblue", lw=1.2,
                    label="Input", zorder=2)
        ax_rec.plot(t, recon[c],     color="tomato",    lw=1.2,
                    linestyle="--", label="Recon", zorder=3)

        ch_loss = loss_per_ch[c]
        loss_str = f"MSE={ch_loss:.4f}" if not np.isnan(ch_loss) else "MSE=n/a"
        ax_rec.set_ylabel(loss_str, fontsize=7)
        ax_rec.tick_params(labelsize=7)
        ax_rec.grid(axis="y", lw=0.4, alpha=0.5)

        for p in range(1, Pt):
            ax_rec.axvline(p * patch_size / SAMPLE_HZ,
                           color="grey", lw=0.4, alpha=0.4, zorder=1)

    axes[-1, 0].set_xlabel("Time (s)", fontsize=9)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=9)

    handles = [
        plt.Line2D([0], [0], color="steelblue", lw=1.5, label="Input"),
        plt.Line2D([0], [0], color="tomato",    lw=1.5, linestyle="--", label="Reconstructed"),
        mpatches.Patch(color="lightgrey", alpha=0.65, label="Masked patch"),
    ]
    axes[0, 1].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/base.yaml",
        help="Base config YAML (same as run_pretrain.py --config)")
    p.add_argument("--model-config", default="configs/vit1d_mae_ocdetect_flu_no_norm_aug.yaml",
        help="Model config YAML (same as run_pretrain.py --model-config)")
    p.add_argument("--ckpt",
        default="runs/vit1d_mae_ocdetect_flu_no_norm_aug/checkpoints/best.pt")
    p.add_argument("--csv",
        default="/mnt/nas2/yiquan_data/preprocessed/"
                "OCDetect_01_recording_00_55ef074f-147f-424f-b9c6-58c0cf9dfee0.csv")
    p.add_argument("--window-idx", type=int, default=20,
        help="Which clean window to visualise (0-indexed, wraps around)")
    p.add_argument("--out",
        default="data_vis/mae_recon_no_norm_aug.png")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and merge YAMLs exactly as run_pretrain.py does
    yaml_cfg = deep_update(load_yaml(args.config), load_yaml(args.model_config))
    print(f"Config: {args.config} + {args.model_config}")

    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.ckpt}")
    encoder, decoder, cfg, mask_ratio, strategy = load_model(args.ckpt, device, yaml_cfg)
    Pt = WINDOW_LEN // encoder.patch_size
    print(f"  patch_size={encoder.patch_size}")

    print(f"Loading CSV: {args.csv}  window_idx={args.window_idx}")
    raw, processed = load_window(args.csv, cfg, args.window_idx)

    print("Running reconstruction …")
    recon, patch_mask, loss_total, loss_per_ch = reconstruct(
        processed, encoder, decoder, mask_ratio, strategy, device)

    n_masked = int(patch_mask.sum())
    print(f"  Masked patches: {n_masked}/{len(patch_mask)}  "
          f"({n_masked/len(patch_mask)*100:.0f}%)")
    print(f"  MSE (masked): {loss_total:.4f}")
    for i, (name, loss) in enumerate(zip(CH_NAMES, loss_per_ch)):
        print(f"    {name:<20}: {loss:.4f}")

    plot(raw, processed, recon, patch_mask, encoder.patch_size, Pt,
         loss_total, loss_per_ch, args.out, mask_ratio)


if __name__ == "__main__":
    main()
