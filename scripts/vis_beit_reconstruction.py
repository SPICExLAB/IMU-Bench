"""Visualise BEiT masked reconstruction for vit1d_beit_ocdetect_anyrot_nopooling_8lvl.

For each example window:
  - Grey background: the masked patches (65% of patches)
  - Black line   : original clean signal
  - Blue  line   : oracle   — decode ground-truth RVQ token indices for masked patches
  - Red   line   : BEiT     — decode BEiT-predicted   token indices for masked patches
  - (Visible patches: all three lines coincide with the original by construction)

Usage
-----
    python scripts/vis_beit_reconstruction.py \
        --beit-run  runs/vit1d_beit_ocdetect_anyrot_nopooling_8lvl \
        --out       data_vis/beit_reconstruction.png \
        --n-windows 4
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives.rvq_tokenizer import (
    ResidualVectorQuantizer,
    PatchDecoder,
    _patches_to_signal,
)
from imu_lm.objectives.masked_1d import generate_patch_mask
from imu_lm.objectives.beit_1d import load_frozen_tokenizer
from imu_lm.data.OCDetect_dataloader import DATA_DIR_DEFAULT, _load_recording
from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.helpers import cfg_get

CH_NAMES = ["acc_F", "acc_L", "acc_U", "gyro_F", "gyro_L", "gyro_U"]
FS = 50.0


# ---------------------------------------------------------------------------
# Load BEiT encoder + heads
# ---------------------------------------------------------------------------

def load_beit(run_dir: str, device: torch.device):
    ckpt_path = os.path.join(run_dir, "checkpoints", "latest.pt")
    if not os.path.exists(ckpt_path):
        ckpts = sorted(f for f in os.listdir(os.path.join(run_dir, "checkpoints"))
                       if f.endswith(".pt"))
        ckpt_path = os.path.join(run_dir, "checkpoints", ckpts[-1])

    print(f"Loading BEiT checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    cfg   = state["cfg"]
    step  = state.get("step", "?")

    encoder = ViT1DEncoder(cfg).to(device)
    encoder.load_state_dict(state["model"], strict=True)
    encoder.eval()

    beit_cfg    = cfg_get(cfg, ["objective", "beit_ocdetect_anyrot_nopooling_8lvl"], {}) or {}
    num_levels  = int(beit_cfg.get("num_levels", 8))
    num_codes   = int(beit_cfg.get("num_codes",  1024))

    heads = nn.ModuleList([
        nn.Linear(encoder.embed_dim, num_codes)
        for _ in range(num_levels)
    ]).to(device)
    heads.load_state_dict(state["beit_heads"], strict=True)
    heads.eval()

    print(f"  step={step}  embed_dim={encoder.embed_dim}  "
          f"num_levels={num_levels}  num_codes={num_codes}")
    return encoder, heads, cfg, step


# ---------------------------------------------------------------------------
# Load RVQ tokenizer + decoders (needed for decoding indices → signal)
# ---------------------------------------------------------------------------

def load_tokenizer_with_decoders(tok_run: str, device: torch.device):
    ckpt_path = os.path.join(tok_run, "checkpoints", "latest.pt")
    state = torch.load(ckpt_path, map_location=device)
    tok_cfg = state["cfg"]

    rvq_cfg    = cfg_get(tok_cfg, ["objective", "rvq_tokenizer_ocdetect_anyrot"], {}) or {}
    num_levels = int(rvq_cfg.get("num_levels",  8))
    num_codes  = int(rvq_cfg.get("num_codes",   1024))
    code_dim   = int(rvq_cfg.get("code_dim",    256))
    ema_decay  = float(rvq_cfg.get("ema_decay", 0.95))
    cmt_w      = float(rvq_cfg.get("commitment_weight", 0.05))
    dec_hidden = int(rvq_cfg.get("decoder_hidden_dim", 512))
    patch_size = int(cfg_get(tok_cfg, ["vit1d", "patch_size"], 16))

    tok_encoder = ViT1DEncoder(tok_cfg).to(device)
    tok_encoder.load_state_dict(state["model"], strict=True)
    tok_encoder.eval()

    rvq = ResidualVectorQuantizer(num_levels, num_codes, code_dim, ema_decay, cmt_w).to(device)
    rvq.load_state_dict(state["rvq"], strict=True)
    rvq.eval()

    dec_out_dim = patch_size  # pool_channels=False
    decoders = nn.ModuleList([
        PatchDecoder(code_dim, dec_out_dim, dec_hidden) for _ in range(num_levels)
    ]).to(device)
    decoders.load_state_dict(state["decoders"], strict=True)
    decoders.eval()

    proj = None
    if tok_encoder.embed_dim != code_dim and "rvq_proj" in state:
        proj = nn.Linear(tok_encoder.embed_dim, code_dim).to(device)
        proj.load_state_dict(state["rvq_proj"], strict=True)
        proj.eval()

    return tok_encoder, rvq, decoders, proj, patch_size, num_levels


# ---------------------------------------------------------------------------
# Decode a set of indices → signal  [B, C, T]
# ---------------------------------------------------------------------------

@torch.no_grad()
def indices_to_signal(
    indices:    torch.Tensor,   # [B, N_q, K]  int64
    rvq:        ResidualVectorQuantizer,
    decoders:   nn.ModuleList,
    B: int, C: int, Pt: int, patch_size: int,
) -> torch.Tensor:
    """Look up code vectors for each level, accumulate, apply final decoder."""
    K = indices.shape[-1]
    cumq = torch.zeros(B, C * Pt, rvq.quantizers[0].code_dim, device=indices.device)
    for k in range(K):
        idx_k  = indices[:, :, k]                          # [B, N_q]
        code_k = rvq.quantizers[k].codebook[idx_k]         # [B, N_q, code_dim]
        cumq   = cumq + code_k

    pred_patches = decoders[-1](cumq)                       # [B, N_q, patch_size]
    return _patches_to_signal(pred_patches, B, C, Pt, patch_size)   # [B, C, T]


# ---------------------------------------------------------------------------
# Build signal-space mask:  [B, C, T] bool,  True = masked patch
# ---------------------------------------------------------------------------

def patch_mask_to_signal(
    patch_mask: torch.Tensor,  # [B, N_q] bool
    B: int, C: int, Pt: int, patch_size: int,
) -> torch.Tensor:
    sig_mask = torch.zeros(B, C, Pt * patch_size, dtype=torch.bool, device=patch_mask.device)
    for n in range(C * Pt):
        c      = n // Pt
        t      = n % Pt
        sig_mask[:, c, t * patch_size:(t + 1) * patch_size] = patch_mask[:, n].unsqueeze(-1)
    return sig_mask


# ---------------------------------------------------------------------------
# Load data windows
# ---------------------------------------------------------------------------

def get_windows(cfg, n_windows: int, device: torch.device) -> torch.Tensor:
    rvq_cfg  = cfg_get(cfg, ["objective", "rvq_tokenizer_ocdetect_anyrot"], {}) or {}
    data_dir = rvq_cfg.get("data_dir", DATA_DIR_DEFAULT)
    win_T, _ = compute_T_and_hop(cfg)

    files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.csv")))

    raw = []
    for path in files:
        if len(raw) >= n_windows: break
        try:
            imu, _, ignore = _load_recording(path, apply_flu_transform=False)
            usable = np.where(ignore == 0)[0]
            i = 0
            while i + win_T <= len(imu) and len(raw) < n_windows:
                if len(usable[(usable >= i) & (usable < i + win_T)]) == win_T:
                    w = preprocess_window(imu[i:i + win_T].astype(np.float32), cfg)
                    if w is not None:
                        raw.append(w)
                i += win_T
        except Exception:
            continue

    if not raw:
        raise RuntimeError(f"No windows loaded from {data_dir}")
    return torch.from_numpy(np.stack(raw)).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--beit-run",  default="runs/vit1d_beit_ocdetect_anyrot_nopooling_8lvl")
    p.add_argument("--tok-run",
                   default="runs/vit1d_rvq_tokenizer_ocdetect_anyrot_nopooling_8lvl")
    p.add_argument("--out",       default="data_vis/beit_reconstruction.png")
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--mask-ratio", type=float, default=0.65)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # ── Load models ────────────────────────────────────────────────────────────
    beit_encoder, beit_heads, beit_cfg, step = load_beit(args.beit_run, device)
    tok_encoder, rvq, decoders, proj, patch_size, K = load_tokenizer_with_decoders(
        args.tok_run, device
    )
    print(f"patch_size={patch_size}  K={K}")

    # ── Data ───────────────────────────────────────────────────────────────────
    # Use tokenizer's config for data loading (knows the correct data_dir)
    tok_state  = torch.load(os.path.join(args.tok_run, "checkpoints", "latest.pt"),
                            map_location="cpu")
    tok_cfg    = tok_state["cfg"]

    print(f"Loading {args.n_windows} windows …")
    x = get_windows(tok_cfg, args.n_windows, device)   # [W, 6, T]
    W, C, T = x.shape
    Pt = T // patch_size
    print(f"Input: {tuple(x.shape)}  Pt={Pt}")

    # ── Patch mask ─────────────────────────────────────────────────────────────
    patch_mask = generate_patch_mask(
        W, Pt, C, args.mask_ratio, "composite", device
    )   # [W, C*Pt] bool

    # ── Ground-truth RVQ indices (oracle) ─────────────────────────────────────
    @torch.no_grad()
    def get_gt_indices(x_in):
        tokens, _ = tok_encoder.forward_tokens(x_in)
        z = proj(tokens) if proj is not None else tokens
        z = F.normalize(z, dim=-1)
        res, idxs = z, []
        for vq in rvq.quantizers:
            q, idx = vq(res)
            idxs.append(idx)
            res = res - q.detach()
        return torch.stack(idxs, dim=-1)   # [W, C*Pt, K]

    gt_indices = get_gt_indices(x)

    # ── BEiT forward → predicted indices ──────────────────────────────────────
    @torch.no_grad()
    def get_pred_indices(x_in, mask):
        _, patch_out = beit_encoder.forward_dino(x_in, mask)   # [W, C*Pt, D]
        pred_idxs = []
        for head in beit_heads:
            logits = head(patch_out)                # [W, C*Pt, num_codes]
            pred_idxs.append(logits.argmax(-1))     # [W, C*Pt]
        return torch.stack(pred_idxs, dim=-1)       # [W, C*Pt, K]

    pred_indices = get_pred_indices(x, patch_mask)

    # ── Decode indices → signals ───────────────────────────────────────────────
    oracle_sig = indices_to_signal(gt_indices,   rvq, decoders, W, C, Pt, patch_size)
    beit_sig   = indices_to_signal(pred_indices, rvq, decoders, W, C, Pt, patch_size)

    # Signal-space mask: True = masked patch
    sig_mask = patch_mask_to_signal(patch_mask, W, C, Pt, patch_size)  # [W, C, T]

    # Inpaint visible patches with original signal
    oracle_inpainted = torch.where(sig_mask, oracle_sig, x)
    beit_inpainted   = torch.where(sig_mask,   beit_sig, x)

    x_np       = x.cpu().numpy()
    oracle_np  = oracle_inpainted.cpu().numpy()
    beit_np    = beit_inpainted.cpu().numpy()
    mask_np    = sig_mask.cpu().numpy()

    t_axis = np.arange(T) / FS

    # ── Per-window masked-region RMSE ─────────────────────────────────────────
    def masked_rmse(pred, target, mask):
        diff = (pred - target) ** 2
        diff[~mask] = 0.0
        n = mask.sum()
        return float(np.sqrt(diff.sum() / max(n, 1)))

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        6, W,
        figsize=(4.5 * W, 2.0 * 6),
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    fig.suptitle(
        f"BEiT Masked Reconstruction  —  {os.path.basename(args.beit_run)}\n"
        f"step={step}   mask_ratio={args.mask_ratio:.0%}   "
        f"Grey=masked   Black=original   Blue=oracle   Red=BEiT",
        fontsize=11, fontweight="bold",
    )

    for wi in range(W):
        o_rmse = masked_rmse(oracle_np[wi], x_np[wi], mask_np[wi])
        b_rmse = masked_rmse(beit_np[wi],  x_np[wi], mask_np[wi])

        for ch in range(6):
            ax = axes[ch, wi]

            # Shade masked patches
            in_masked = False
            t_start = 0
            for t in range(T + 1):
                currently = mask_np[wi, ch, t] if t < T else False
                if currently and not in_masked:
                    t_start, in_masked = t, True
                elif not currently and in_masked:
                    ax.axvspan(t_axis[t_start], t_axis[t - 1], alpha=0.15,
                               color="grey", linewidth=0)
                    in_masked = False

            ax.plot(t_axis, x_np[wi, ch],      color="black", lw=0.9, zorder=5, label="original")
            ax.plot(t_axis, oracle_np[wi, ch],  color="#2196F3", lw=1.1, zorder=4,
                    alpha=0.85, label=f"oracle  RMSE={o_rmse:.3f}")
            ax.plot(t_axis, beit_np[wi, ch],    color="#F44336", lw=1.1, zorder=3,
                    alpha=0.85, label=f"BEiT    RMSE={b_rmse:.3f}")

            if wi == 0:
                ax.set_ylabel(CH_NAMES[ch], fontsize=8)
            if ch == 0:
                ax.set_title(f"Window {wi + 1}", fontsize=9)
            if ch == 5:
                ax.set_xlabel("s", fontsize=8)

            ax.tick_params(labelsize=6)
            ax.grid(axis="y", lw=0.3, alpha=0.4)

        # Legend on top-right of first channel
        axes[0, wi].legend(loc="upper right", fontsize=6, framealpha=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {args.out}")

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'Window':<10} {'Oracle RMSE':>12} {'BEiT RMSE':>12}")
    for wi in range(W):
        o = masked_rmse(oracle_np[wi], x_np[wi], mask_np[wi])
        b = masked_rmse(beit_np[wi],  x_np[wi], mask_np[wi])
        print(f"  {wi+1:<8}  {o:>12.4f}  {b:>12.4f}")


if __name__ == "__main__":
    main()
