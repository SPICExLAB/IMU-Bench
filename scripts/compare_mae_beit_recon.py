#!/usr/bin/env python3
"""Compare MAE vs BEiT vs RVQ oracle reconstruction on the same windows.

Evaluates three models on 65% composite-masked windows sampled identically from
each training dataset, reporting Z-norm MSE (masked patches only).

Z-norm MSE: both original and reconstruction are normalized by their own
per-channel per-window statistics independently, so the metric is scale-invariant
and comparable across models with different input preprocessing.
MSE approx 2  -> uncorrelated reconstruction
MSE < 2  -> model captures signal structure

Models
------
  MAE encoder+decoder  pretrained on realdata with z-norm (vit1d_realdata_pretrain)
  BEiT encoder+heads   pretrained on realdata without z-norm; decodes via RVQ codebook
  RVQ oracle           tokenizer run on the UNMASKED signal -- reconstruction ceiling

Usage
-----
    MPLBACKEND=Agg python scripts/compare_mae_beit_recon.py \\
        [--mae-ckpt  runs/vit1d_realdata_pretrain/checkpoints/best.pt] \\
        [--beit-ckpt runs/vit1d_beit_realdata_nopooling_8lvl/checkpoints/latest.pt] \\
        [--rvq-ckpt  runs/vit1d_rvq_tokenizer_realdata_nopooling_8lvl/checkpoints/latest.pt] \\
        [--parquet   /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet] \\
        [--n-windows 200] [--batch-size 64] [--seed 42] [--device cuda] \\
        [--out data_vis/compare_mae_beit_recon.png]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoder
from imu_lm.objectives.masked_1d import generate_patch_mask
from imu_lm.objectives.rvq_tokenizer import ResidualVectorQuantizer, PatchDecoder
from imu_lm.data.augmentations.preprocess import preprocess_window

WIN       = 128
ACC_COLS  = ["acc_x", "acc_y", "acc_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]
IMU_COLS  = ACC_COLS + GYRO_COLS
DATASETS  = ["pamap2", "recofit", "samosa", "shoaib", "ut_watch", "utd_mhad", "wisdm"]
MASK_RATIO    = 0.65
MASK_STRATEGY = "composite"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _remap_mae_keys(sd: dict) -> dict:
    """Translate old nn.TransformerEncoderLayer key names to current naming."""
    out = {}
    for k, v in sd.items():
        k = k.replace(".self_attn.", ".attn.")
        k = k.replace(".linear1.",  ".ff.0.")
        k = k.replace(".linear2.",  ".ff.3.")
        out[k] = v
    return out


def load_mae(ckpt_path: str, device: torch.device):
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg     = ckpt["cfg"]
    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(_remap_mae_keys(ckpt["model"]))
    encoder.eval().to(device)

    out_channels = len(cfg.get("data", {}).get("sensor_columns", IMU_COLS))
    decoder = ViT1DDecoder(
        embed_dim    = encoder.embed_dim,
        out_channels = out_channels,
        target_T     = WIN,
        cfg          = cfg,
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval().to(device)

    return encoder, decoder, cfg


def load_beit(ckpt_path: str, device: torch.device):
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg     = ckpt["cfg"]
    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval().to(device)

    beit_cfg   = cfg["objective"][cfg["objective"]["type"]]
    num_levels = beit_cfg["num_levels"]
    num_codes  = beit_cfg["num_codes"]

    heads = nn.ModuleList([
        nn.Linear(encoder.embed_dim, num_codes) for _ in range(num_levels)
    ])
    heads.load_state_dict(ckpt["beit_heads"], strict=True)
    heads.eval().to(device)

    return encoder, heads, num_levels, num_codes


def load_rvq(ckpt_path: str, device: torch.device):
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg     = ckpt["cfg"]
    cfg_key = cfg["objective"]["type"]
    obj_cfg = cfg["objective"][cfg_key]

    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval().to(device)

    num_levels = obj_cfg["num_levels"]
    num_codes  = obj_cfg["num_codes"]
    code_dim   = obj_cfg["code_dim"]
    hidden_dim = obj_cfg.get("decoder_hidden_dim", 512)
    patch_size = cfg["vit1d"]["patch_size"]

    rvq = ResidualVectorQuantizer(
        num_levels=num_levels, num_codes=num_codes, code_dim=code_dim,
        ema_decay=obj_cfg["ema_decay"],
        commitment_weight=obj_cfg["commitment_weight"],
    )
    rvq.load_state_dict(ckpt["rvq"], strict=True)
    rvq.eval().to(device)

    decoders = nn.ModuleList([
        PatchDecoder(code_dim, patch_size, hidden_dim) for _ in range(num_levels)
    ])
    decoders.load_state_dict(ckpt["decoders"], strict=True)
    decoders.eval().to(device)

    return encoder, rvq, decoders, patch_size


# ---------------------------------------------------------------------------
# Z-norm helper
# ---------------------------------------------------------------------------

def _to_znorm_patches(
    patches: torch.Tensor,
    B: int, C: int, Pt: int, patch_size: int,
) -> torch.Tensor:
    """Normalize [B, N, patch_size] (channel-major) by per-channel per-window stats."""
    sig = patches.reshape(B, C, Pt * patch_size)   # [B, C, T]
    mu  = sig.mean(dim=2, keepdim=True)             # [B, C, 1]
    sd  = sig.std(dim=2,  keepdim=True) + 1e-8
    mu_p = mu.repeat(1, 1, Pt).reshape(B, C * Pt, 1).expand_as(patches)
    sd_p = sd.repeat(1, 1, Pt).reshape(B, C * Pt, 1).expand_as(patches)
    return (patches - mu_p) / sd_p


def znorm_mse_masked(
    x_patches: torch.Tensor,
    pred_patches: torch.Tensor,
    mask: torch.Tensor,
    B: int, C: int, Pt: int, patch_size: int,
) -> float:
    """Z-norm MSE on masked patches with independent normalization per signal."""
    x_z    = _to_znorm_patches(x_patches,    B, C, Pt, patch_size)
    pred_z = _to_znorm_patches(pred_patches, B, C, Pt, patch_size)
    return ((x_z - pred_z) ** 2)[mask].mean().item()


# ---------------------------------------------------------------------------
# Per-model reconstruction
# ---------------------------------------------------------------------------

@torch.no_grad()
def mae_reconstruct(
    x_raw_np: np.ndarray,
    mask: torch.Tensor,
    encoder, decoder,
    mae_cfg: dict,
    device: torch.device,
) -> float:
    B, C, T = x_raw_np.shape
    Pt = T // encoder.patch_size
    patch_size = encoder.patch_size

    # Apply MAE's own preprocessing (z-norm per its saved cfg)
    x_list = []
    for i in range(B):
        raw = x_raw_np[i].T.astype(np.float32)   # [T, C]
        w   = preprocess_window(raw, mae_cfg)
        if w is None:
            w = x_raw_np[i]
        x_list.append(torch.from_numpy(w).float())
    x_mae = torch.stack(x_list, dim=0).to(device)

    visible_tokens, vis_pad = encoder(x_mae, mask=mask, return_keep_padding_mask=True)
    recon = decoder(visible_tokens, mask, Pt, C, visible_padding_mask=vis_pad)

    N = C * Pt
    x_patches     = x_mae.view(B, C, Pt, patch_size).reshape(B, N, patch_size)
    recon_patches = recon.view(B, C, Pt, patch_size).reshape(B, N, patch_size)
    return znorm_mse_masked(x_patches, recon_patches, mask, B, C, Pt, patch_size)


@torch.no_grad()
def beit_reconstruct(
    x_raw: torch.Tensor,
    mask: torch.Tensor,
    beit_encoder, beit_heads,
    rvq, decoders,
    device: torch.device,
) -> float:
    B, C, T = x_raw.shape
    Pt = T // beit_encoder.patch_size
    patch_size = beit_encoder.patch_size
    N  = C * Pt
    code_dim = rvq.quantizers[0].codebook.shape[1]

    x = x_raw.to(device)
    _, patch_out = beit_encoder.forward_dino(x, mask)   # [B, N, D]

    cumul_emb = torch.zeros(B, N, code_dim, device=device)
    for head, vq in zip(beit_heads, rvq.quantizers):
        logits    = head(patch_out)              # [B, N, num_codes]
        pred_idx  = logits.argmax(dim=-1)        # [B, N]
        emb_k     = vq.codebook[pred_idx]        # [B, N, code_dim]
        cumul_emb = cumul_emb + emb_k

    pred_patches = decoders[-1](cumul_emb)        # [B, N, patch_size]
    x_patches    = x.view(B, C, Pt, patch_size).reshape(B, N, patch_size)
    return znorm_mse_masked(x_patches, pred_patches, mask, B, C, Pt, patch_size)


@torch.no_grad()
def rvq_oracle_reconstruct(
    x_raw: torch.Tensor,
    mask: torch.Tensor,
    tok_encoder, rvq, decoders,
    device: torch.device,
) -> float:
    """RVQ tokenizer on UNMASKED signal; MSE measured at masked positions."""
    B, C, T = x_raw.shape
    Pt = T // tok_encoder.patch_size
    patch_size = tok_encoder.patch_size
    N  = C * Pt

    x = x_raw.to(device)
    tokens, _     = tok_encoder.forward_tokens(x)        # [B, N, D]
    z             = F.normalize(tokens, dim=-1)
    quantized_list, _, _ = rvq(z)
    pred_patches  = decoders[-1](quantized_list[-1])     # [B, N, patch_size]

    x_patches = x.view(B, C, Pt, patch_size).reshape(B, N, patch_size)
    return znorm_mse_masked(x_patches, pred_patches, mask, B, C, Pt, patch_size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mae-ckpt",
                   default="runs/vit1d_realdata_pretrain/checkpoints/best.pt")
    p.add_argument("--beit-ckpt",
                   default="runs/vit1d_beit_realdata_nopooling_8lvl/checkpoints/latest.pt")
    p.add_argument("--rvq-ckpt",
                   default="runs/vit1d_rvq_tokenizer_realdata_nopooling_8lvl/checkpoints/latest.pt")
    p.add_argument("--parquet",
                   default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--n-windows",  type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="data_vis/compare_mae_beit_recon.png")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    rng    = np.random.default_rng(args.seed)

    print("Loading models ...")
    mae_enc, mae_dec, mae_cfg = load_mae(args.mae_ckpt, device)
    beit_enc, beit_heads, num_levels, num_codes = load_beit(args.beit_ckpt, device)
    tok_enc, rvq, decoders, rvq_patch_size = load_rvq(args.rvq_ckpt, device)

    print(f"  MAE:  embed={mae_enc.embed_dim} patch={mae_enc.patch_size}"
          f"  normalize={mae_cfg.get('preprocess',{}).get('normalize',{}).get('enabled',False)}")
    print(f"  BEiT: embed={beit_enc.embed_dim} patch={beit_enc.patch_size}"
          f"  levels={num_levels} codes={num_codes}")
    print(f"  RVQ:  embed={tok_enc.embed_dim} patch={rvq_patch_size}"
          f"  levels={len(rvq.quantizers)}")
    print()

    patch_size = beit_enc.patch_size
    assert tok_enc.patch_size == patch_size, "RVQ and BEiT patch sizes must match"
    C  = len(IMU_COLS)
    Pt = WIN // patch_size

    needed_cols = ["dataset", "session_id", "timestamp_ns"] + IMU_COLS
    results = {}

    for ds in args.datasets:
        print(f"[{ds}] loading ...", end=" ", flush=True)
        df = pd.read_parquet(args.parquet, columns=needed_cols,
                             filters=[("dataset", "==", ds)])
        if df.empty:
            print("NOT FOUND -- skip")
            continue
        df = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)

        sessions = df["session_id"].unique()
        raw_wins: list = []
        for sess in sessions:
            seg  = df[df["session_id"] == sess]
            vals = seg[IMU_COLS].values.T.astype(np.float32)   # [C, L]
            L    = vals.shape[1]
            if L < WIN:
                continue
            n_from = max(1, min(args.n_windows // max(len(sessions), 1), L - WIN + 1))
            starts = rng.choice(L - WIN + 1, size=n_from, replace=False)
            for s in starts:
                raw_wins.append(vals[:, s : s + WIN])
            if len(raw_wins) >= args.n_windows:
                break
        if not raw_wins:
            print("no windows -- skip")
            continue
        raw_wins = raw_wins[:args.n_windows]
        print(f"{len(raw_wins)} windows", end=" ", flush=True)

        import random as pyrandom
        torch.manual_seed(args.seed)
        pyrandom.seed(args.seed)

        mae_losses, beit_losses, rvq_losses = [], [], []
        bs = args.batch_size

        for start in range(0, len(raw_wins), bs):
            batch_raw = np.stack(raw_wins[start : start + bs], axis=0)  # [B, C, T]
            B_cur = batch_raw.shape[0]
            x_raw = torch.from_numpy(batch_raw).float()

            mask = generate_patch_mask(
                B_cur, Pt, C, MASK_RATIO, MASK_STRATEGY, device=device,
            )

            mae_losses.append(mae_reconstruct(
                batch_raw, mask, mae_enc, mae_dec, mae_cfg, device,
            ))
            beit_losses.append(beit_reconstruct(
                x_raw, mask, beit_enc, beit_heads, rvq, decoders, device,
            ))
            rvq_losses.append(rvq_oracle_reconstruct(
                x_raw, mask, tok_enc, rvq, decoders, device,
            ))

        r = {
            "mae":  float(np.mean(mae_losses)),
            "beit": float(np.mean(beit_losses)),
            "rvq":  float(np.mean(rvq_losses)),
            "n":    len(raw_wins),
        }
        results[ds] = r
        print(f"  MAE={r['mae']:.4f}  BEiT={r['beit']:.4f}  RVQ={r['rvq']:.4f}")

    if not results:
        print("No results -- check checkpoint paths and parquet.")
        return

    # -- Text table --
    print("\n" + "=" * 66)
    print("Z-norm MSE (masked patches, 65% composite mask)")
    print("MSE ~ 2 -> uncorrelated  |  MSE < 2 -> structure captured")
    print("-" * 66)
    print(f"{'dataset':>12}  {'n_win':>5}  {'MAE':>8}  {'BEiT':>8}  {'RVQ oracle':>10}")
    print("-" * 66)
    ds_names = list(results.keys())
    for ds in ds_names:
        r = results[ds]
        print(f"{ds:>12}  {r['n']:>5}  {r['mae']:>8.4f}  {r['beit']:>8.4f}  {r['rvq']:>10.4f}")
    print("-" * 66)
    mae_avg  = np.mean([results[d]["mae"]  for d in ds_names])
    beit_avg = np.mean([results[d]["beit"] for d in ds_names])
    rvq_avg  = np.mean([results[d]["rvq"]  for d in ds_names])
    print(f"{'MEAN':>12}  {'':>5}  {mae_avg:>8.4f}  {beit_avg:>8.4f}  {rvq_avg:>10.4f}")
    print("=" * 66)

    # -- Plot --
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    mae_vals  = [results[d]["mae"]  for d in ds_names]
    beit_vals = [results[d]["beit"] for d in ds_names]
    rvq_vals  = [results[d]["rvq"]  for d in ds_names]

    x_pos = np.arange(len(ds_names))
    w     = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "MAE vs BEiT vs RVQ oracle -- Z-norm MSE (masked patches, 65% composite)\n"
        "Same windows and same mask applied to all three models",
        fontsize=11, fontweight="bold",
    )

    # Left: grouped bar chart
    ax = axes[0]
    bars_mae  = ax.bar(x_pos - w, mae_vals,  width=w, label="MAE encoder+decoder",
                       color="#3498db", edgecolor="k", lw=0.5)
    bars_beit = ax.bar(x_pos,     beit_vals, width=w, label="BEiT encoder+heads",
                       color="#e74c3c", edgecolor="k", lw=0.5)
    bars_rvq  = ax.bar(x_pos + w, rvq_vals,  width=w, label="RVQ oracle (ceiling)",
                       color="#2ecc71", edgecolor="k", lw=0.5, alpha=0.85)
    ax.axhline(2.0, color="gray", lw=1.0, ls="--", alpha=0.6, label="uncorrelated (~2)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ds_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Z-norm MSE  (masked patches only)", fontsize=9)
    ax.set_title("Per-dataset comparison", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.tick_params(axis="y", labelsize=8)
    for bars, vals in [(bars_mae, mae_vals), (bars_beit, beit_vals), (bars_rvq, rvq_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.grid(axis="y", alpha=0.3)

    # Right: mean across datasets (horizontal bars)
    ax2 = axes[1]
    models = ["MAE\nencoder+decoder", "BEiT\nencoder+heads", "RVQ oracle\n(ceiling)"]
    means  = [mae_avg, beit_avg, rvq_avg]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    y_pos2 = np.arange(len(models))
    ax2.barh(y_pos2, means, color=colors, edgecolor="k", lw=0.5, height=0.5)
    ax2.axvline(2.0, color="gray", lw=1.2, ls="--", alpha=0.7, label="uncorrelated (~2)")
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(models, fontsize=9)
    ax2.set_xlabel("Mean Z-norm MSE  (across datasets)", fontsize=9)
    ax2.set_title("Mean across all datasets", fontsize=10, fontweight="bold")
    for i, v in enumerate(means):
        ax2.text(v + 0.02, i, f"{v:.4f}", va="center", fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.8)
    ax2.set_xlim(0, max(means) * 1.25)
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
