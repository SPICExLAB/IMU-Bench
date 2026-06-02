#!/usr/bin/env python3
"""Visualise RVQ tokenizer reconstruction quality vs number of codebook levels.

For each window, plots the original 6-channel IMU signal overlaid with four
reconstructions: using cumulative codes from 1, 2, 3, and 4 RVQ levels.

Usage
-----
    python scripts/vis_rvq_reconstruction.py \
        --run-dir runs/vit1d_rvq_tokenizer_ocdetect_anyrot \
        --out     data_vis/rvq_reconstruction.png \
        --n-windows 4
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives.rvq_tokenizer import (
    ResidualVectorQuantizer,
    PatchDecoder,
    _temporal_preds_to_signal,
    _patches_to_signal,
)
from imu_lm.utils.helpers import cfg_get

CH_NAMES = ["acc_F", "acc_L", "acc_U", "gyro_F", "gyro_L", "gyro_U"]

def _level_colors(n: int) -> list:
    """Generate n visually distinct colors for RVQ levels."""
    import matplotlib.cm as cm
    return [cm.tab10(i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model(run_dir: str, device: torch.device):
    ckpt_path = os.path.join(run_dir, "checkpoints", "latest.pt")
    if not os.path.exists(ckpt_path):
        # fall back to any step checkpoint
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        ckpts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
        ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)

    cfg = state["cfg"]

    # ── Encoder ──────────────────────────────────────────────────────────────
    encoder = ViT1DEncoder(cfg).to(device)
    encoder.load_state_dict(state["model"], strict=True)
    encoder.eval()

    # ── RVQ ──────────────────────────────────────────────────────────────────
    rvq_cfg   = cfg_get(cfg, ["objective", "rvq_tokenizer_ocdetect_anyrot"], {}) or {}
    num_levels        = int(rvq_cfg.get("num_levels",         4))
    num_codes         = int(rvq_cfg.get("num_codes",          512))
    code_dim          = int(rvq_cfg.get("code_dim",           256))
    ema_decay         = float(rvq_cfg.get("ema_decay",        0.99))
    commitment_weight = float(rvq_cfg.get("commitment_weight",0.25))
    decoder_hidden    = int(rvq_cfg.get("decoder_hidden_dim", 512))
    pool_channels     = bool(rvq_cfg.get("pool_channels",     True))
    num_channels_in   = int(rvq_cfg.get("num_channels",       6))

    dec_out_dim = (num_channels_in * encoder.patch_size) if pool_channels else encoder.patch_size

    rvq = ResidualVectorQuantizer(
        num_levels=num_levels,
        num_codes=num_codes,
        code_dim=code_dim,
        ema_decay=ema_decay,
        commitment_weight=commitment_weight,
    ).to(device)
    rvq.load_state_dict(state["rvq"], strict=True)
    rvq.eval()

    decoders = nn.ModuleList([
        PatchDecoder(code_dim, dec_out_dim, decoder_hidden)
        for _ in range(num_levels)
    ]).to(device)
    decoders.load_state_dict(state["decoders"], strict=True)
    decoders.eval()

    # ── Optional projection ───────────────────────────────────────────────────
    proj = None
    if encoder.embed_dim != code_dim and "rvq_proj" in state:
        proj = nn.Linear(encoder.embed_dim, code_dim).to(device)
        proj.load_state_dict(state["rvq_proj"], strict=True)
        proj.eval()

    return encoder, rvq, decoders, proj, cfg, pool_channels


# ---------------------------------------------------------------------------
# Forward: get per-level reconstructions
# ---------------------------------------------------------------------------

@torch.no_grad()
def reconstruct_all_levels(
    x: torch.Tensor,        # [B, C, T]
    encoder,
    rvq,
    decoders,
    proj,
    pool_channels: bool,
) -> list[torch.Tensor]:
    """Return list of K reconstructed signals [B, C, T], one per cumulative RVQ level."""
    B, C, T = x.shape
    tokens, Pt = encoder.forward_tokens(x)          # [B, C*Pt, D]
    D_enc = tokens.shape[-1]

    if pool_channels:
        tokens_q = tokens.view(B, C, Pt, D_enc).mean(dim=1)   # [B, Pt, D]
    else:
        tokens_q = tokens

    z = proj(tokens_q) if proj is not None else tokens_q
    z = F.normalize(z, dim=-1)   # must match forward_loss in rvq_tokenizer.py

    # Run RVQ manually to get per-level cumulative quantized vectors
    residual     = z
    cumulative_q = torch.zeros_like(z)
    recon_signals = []

    for k, (vq, dec) in enumerate(zip(rvq.quantizers, decoders)):
        q, _ = vq(residual)
        cumulative_q = cumulative_q + q
        residual     = residual - q.detach()

        pred_flat = dec(cumulative_q)   # [B, N_q, out_dim]
        if pool_channels:
            sig = _temporal_preds_to_signal(pred_flat, B, C, Pt, encoder.patch_size)
        else:
            sig = _patches_to_signal(pred_flat, B, C, Pt, encoder.patch_size)

        recon_signals.append(sig[:, :, :T])   # trim to input length

    return recon_signals   # list of K tensors [B, C, T]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_reconstructions(
    x_np:       np.ndarray,          # [W, 6, T]
    recons_np:  list[np.ndarray],    # K × [W, 6, T]
    out_path:   str,
    hz:         float = 50.0,
):
    W   = x_np.shape[0]
    K   = len(recons_np)
    T   = x_np.shape[2]
    t   = np.arange(T) / hz
    LEVEL_COLORS = _level_colors(K)

    fig = plt.figure(figsize=(4 * W, 2.2 * 6))
    fig.suptitle(
        "RVQ Reconstruction: original vs 1–4 cumulative codebook levels\n"
        "Grey = original   Red=L1   Orange=L1+L2   Green=L1+2+3   Blue=all 4",
        fontsize=11, fontweight="bold",
    )

    outer = gridspec.GridSpec(6, W, figure=fig, hspace=0.55, wspace=0.25)

    for ch in range(6):
        for wi in range(W):
            ax = fig.add_subplot(outer[ch, wi])
            # Original
            ax.plot(t, x_np[wi, ch], color="0.55", lw=1.0, label="orig", zorder=5)

            for k in range(K):
                rmse = float(np.sqrt(np.mean((recons_np[k][wi, ch] - x_np[wi, ch]) ** 2)))
                ax.plot(t, recons_np[k][wi, ch],
                        color=LEVEL_COLORS[k], lw=0.9,
                        label=f"L1-{k+1}  RMSE={rmse:.3f}",
                        alpha=0.85, zorder=4 - k)

            if wi == 0:
                ax.set_ylabel(CH_NAMES[ch], fontsize=8)
            if ch == 0:
                ax.set_title(f"Window {wi+1}", fontsize=9)
            if ch == 5:
                ax.set_xlabel("s", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(axis="y", lw=0.3, alpha=0.4)

            if ch == 0 and wi == W - 1:
                ax.legend(loc="upper right", fontsize=6, framealpha=0.7)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def plot_rmse_summary(
    x_np:      np.ndarray,       # [W, 6, T]
    recons_np: list[np.ndarray], # K × [W, 6, T]
    out_path:  str,
):
    """Bar chart: per-level per-channel mean RMSE across all windows."""
    K = len(recons_np)
    W = x_np.shape[0]
    LEVEL_COLORS = _level_colors(K)

    # [K, 6] mean RMSE
    rmse = np.array([
        [float(np.sqrt(np.mean((recons_np[k][:, ch] - x_np[:, ch]) ** 2)))
         for ch in range(6)]
        for k in range(K)
    ])

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(6)
    width = 0.18
    offsets = np.linspace(-(K - 1) / 2, (K - 1) / 2, K) * width
    labels = [f"L1-{k+1}" for k in range(K)]

    for k in range(K):
        ax.bar(x_pos + offsets[k], rmse[k], width,
               color=LEVEL_COLORS[k], label=labels[k], alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(CH_NAMES, fontsize=9)
    ax.set_ylabel("Mean RMSE across windows", fontsize=10)
    ax.set_title("Reconstruction RMSE per channel vs number of RVQ levels", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Data: grab N windows by reading just a few CSV/parquet files directly
# ---------------------------------------------------------------------------

def get_windows(cfg: dict, n_windows: int, device: torch.device):
    """Load N windows from the first available OCDetect file — no full dataset scan."""
    import glob
    from imu_lm.data.OCDetect_dataloader import (
        DATA_DIR_DEFAULT,
        WINDOW_SAMPLES,
        _load_recording,
    )
    from imu_lm.data.augmentations.preprocess import preprocess_window
    from imu_lm.data.windowing import compute_T_and_hop

    rvq_cfg      = cfg_get(cfg, ["objective", "rvq_tokenizer_ocdetect_anyrot"], {}) or {}
    data_dir     = rvq_cfg.get("data_dir", DATA_DIR_DEFAULT)
    apply_flu    = bool(rvq_cfg.get("apply_flu_transform", False))
    win_T, _     = compute_T_and_hop(cfg)

    # Find files — prefer parquet for speed
    files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.csv")))
    if not files:
        raise FileNotFoundError(f"No OCDetect files found in {data_dir}")

    raw_windows = []
    for path in files:
        if len(raw_windows) >= n_windows:
            break
        try:
            imu, _, ignore = _load_recording(path, apply_flu_transform=apply_flu)
        except Exception as e:
            print(f"  skip {os.path.basename(path)}: {e}")
            continue

        usable = np.where(ignore == 0)[0]
        # Collect non-overlapping windows from usable segments
        i = 0
        while i + win_T <= len(imu) and len(raw_windows) < n_windows:
            seg = usable[(usable >= i) & (usable < i + win_T)]
            if len(seg) == win_T:               # fully usable window
                raw_windows.append(imu[i:i + win_T].astype(np.float32))  # [T, C]
            i += win_T                           # non-overlapping for variety

    if not raw_windows:
        raise RuntimeError("Could not extract any usable windows from OCDetect files.")

    # Apply same per-window z-score normalization used during training.
    # preprocess_window: [T, C] → [C, T]
    windows_norm = []
    for w in raw_windows[:n_windows]:
        w_norm = preprocess_window(w, cfg)
        if w_norm is not None:
            windows_norm.append(w_norm)   # [C, T] = [6, T]

    if not windows_norm:
        raise RuntimeError("All windows were dropped by preprocess_window.")

    x = torch.from_numpy(np.stack(windows_norm)).to(device)   # [W, 6, T]
    return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualise RVQ reconstruction per codebook level.")
    p.add_argument("--run-dir", required=True,
                   help="Path to the training run directory.")
    p.add_argument("--out", default="data_vis/rvq_reconstruction.png",
                   help="Output path for the signal panel plot.")
    p.add_argument("--n-windows", type=int, default=4,
                   help="Number of example windows to visualise.")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args  = parse_args()
    dev   = torch.device(args.device)

    encoder, rvq, decoders, proj, cfg, pool_channels = load_model(args.run_dir, dev)

    print(f"Fetching {args.n_windows} validation windows …")
    x = get_windows(cfg, args.n_windows, dev)   # [W, 6, T]
    print(f"Input shape: {tuple(x.shape)}")

    print("Running reconstruction …")
    recons = reconstruct_all_levels(x, encoder, rvq, decoders, proj, pool_channels)
    # recons: list of 4 tensors [W, 6, T]

    x_np     = x.cpu().numpy()
    recons_np = [r.cpu().numpy() for r in recons]

    # Signal panel
    plot_reconstructions(x_np, recons_np, args.out)

    # RMSE bar chart
    summary_path = args.out.replace(".png", "_rmse.png")
    plot_rmse_summary(x_np, recons_np, summary_path)

    # Console summary
    K = len(recons_np)
    print(f"\n{'Level':<10} {'Overall RMSE':>14}")
    for k in range(K):
        rmse = float(np.sqrt(np.mean((recons_np[k] - x_np) ** 2)))
        print(f"  L1-{k+1:<6}  {rmse:>14.4f}")


if __name__ == "__main__":
    main()
