#!/usr/bin/env python3
"""
Visualize MAE reconstruction quality of vit1d_mae_dino_ocdetect_anyrot.

For each sampled sequence:
  1. Preprocess the window (no augmentation — clean signal)
  2. Apply a random mask (same ratio as training)
  3. Student encoder forward_dino (BERT-style [MASK] tokens)
  4. recon_head(masked patch tokens) → predicted signal values
  5. Compare predicted vs original at masked patches

Output files (saved to --out):
  mae_dino_recon_panels.png   — per-sequence signal overlay panels
  mae_dino_recon_summary.png  — aggregate MSE distribution + per-channel quality

Usage:
    python scripts/vis_mae_dino_recon.py \\
        --ckpt  runs/vit1d_mae_dino_ocdetect_anyrot/checkpoints/latest.pt \\
        --data  /mnt/nas2/yiquan_data/preprocessed \\
        --out   data_vis/mae_dino_recon \\
        --n     8
"""

from __future__ import annotations
import argparse, os, random, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.models.ViT1D.model            import ViT1DEncoder
from imu_lm.objectives.masked_1d          import generate_patch_mask

# ── constants ──────────────────────────────────────────────────────────────
WINDOW_LEN    = 128
N_CHANNELS    = 6
N_PATCHES     = WINDOW_LEN // 16   # 8 per channel
N_TOKENS      = N_CHANNELS * N_PATCHES
OCDETECT_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
CH_NAMES      = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
CH_COLORS     = ["#e74c3c", "#e67e22", "#f1c40f", "#2980b9", "#27ae60", "#8e44ad"]


# ── model loading ──────────────────────────────────────────────────────────

class PatchReconHead(nn.Module):
    def __init__(self, embed_dim, patch_size):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, patch_size)
    def forward(self, x):
        return self.proj(self.norm(x))


def _remap(sd):
    return {k.replace(".self_attn.", ".attn.").replace(".linear1.", ".ff.0.")
             .replace(".linear2.", ".ff.3."): v for k, v in sd.items()}


def load_model(ckpt_path):
    state      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg        = state["cfg"]
    patch_size = cfg.get("vit1d", {}).get("patch_size", 16)
    embed_dim  = cfg.get("vit1d", {}).get("encoder", {}).get("hidden_size", 384)

    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(_remap(state["model"]), strict=True)
    encoder.eval()

    recon_head = PatchReconHead(embed_dim, patch_size)
    recon_head.load_state_dict(state["recon_head"])
    recon_head.eval()

    return encoder, recon_head, cfg, patch_size


# ── data loading ───────────────────────────────────────────────────────────

def sample_windows(data_dir, n, cfg, seed=42):
    import pandas as pd
    csvs  = sorted(f for f in os.listdir(data_dir)
                   if f.endswith(".csv") and "metadata" not in f)
    rng   = random.Random(seed)
    chosen = rng.sample(csvs, min(n * 4, len(csvs)))
    windows, labels = [], []
    for fname in chosen:
        try:
            df = pd.read_csv(os.path.join(data_dir, fname),
                             usecols=OCDETECT_COLS).dropna()
            offset = 20 * (WINDOW_LEN // 2)
            if len(df) < offset + WINDOW_LEN:
                continue
            chunk = df.iloc[offset: offset + WINDOW_LEN][OCDETECT_COLS].values.astype(np.float32)
            w = preprocess_window(chunk, cfg)
            if w is None or w.shape != (N_CHANNELS, WINDOW_LEN):
                continue
            windows.append(w)
            labels.append(fname[:26])
        except Exception:
            continue
        if len(windows) == n:
            break
    return windows, labels


# ── reconstruction ─────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct(encoder, recon_head, windows, mask_ratio=0.50, seed=42):
    """
    Returns:
        recon   : [N, C, T]  full signal with predictions inserted at masked patches
        patch_mask : [N, N_TOKENS] bool
        mse_map : [N, C, Pt] per-patch MSE
    """
    torch.manual_seed(seed)
    x = torch.from_numpy(np.stack(windows))   # [N, C, T]
    B, C, T = x.shape
    Pt = T // encoder.patch_size

    patch_mask = generate_patch_mask(B, Pt, C, mask_ratio, "random,temporal", x.device)

    # Student forward (masked input)
    _, s_patches = encoder.forward_dino(x, patch_mask=patch_mask)
    # s_patches: [B, N_TOKENS, D]

    # Reconstruct at masked positions
    masked_tokens = s_patches[patch_mask]           # [N_msk, D]
    pred_vals     = recon_head(masked_tokens)        # [N_msk, patch_size]

    # Build full reconstruction grid (original everywhere, predicted at masked)
    x_recon = x.clone()                             # [B, C, T]
    x_patches = x.reshape(B, C, Pt, encoder.patch_size)      # [B, C, Pt, ps]
    mask_2d   = patch_mask.view(B, C, Pt)                     # [B, C, Pt]

    # Insert predictions back
    x_recon_patches = x_patches.clone()
    x_recon_patches[mask_2d] = pred_vals
    x_recon = x_recon_patches.reshape(B, C, T)

    # Per-patch MSE: [B, C, Pt]
    err = (x_patches - x_recon_patches) ** 2        # [B, C, Pt, ps]
    mse_map = err.mean(dim=-1)                       # [B, C, Pt]

    return x_recon.numpy(), patch_mask.numpy(), mse_map.numpy(), x.numpy()


# ── Figure 1: per-sequence signal panels ──────────────────────────────────

def plot_panels(windows_orig, windows_recon, patch_mask, mse_map,
                labels, patch_size, out_path):
    """N rows × 6 columns — one subplot per (sequence, channel).

    Each subplot:
      grey thin line  = original signal (all timesteps)
      coloured line   = reconstructed values at masked patches
      red shading     = masked patch regions
    """
    N  = len(windows_orig)
    Pt = N_PATCHES
    t  = np.arange(WINDOW_LEN) / 50.0

    fig, axes = plt.subplots(
        N, N_CHANNELS,
        figsize=(N_CHANNELS * 2.8, N * 1.9),
        sharex=True,
    )
    # ensure axes is always 2D
    if N == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "MAE+DINO signal reconstruction\n"
        "grey = original  |  colour = predicted at masked patches  |  shading = masked region",
        fontsize=10, y=1.01,
    )

    for i in range(N):
        mask_2d = patch_mask[i].reshape(N_CHANNELS, Pt)   # [C, Pt]
        msk_flat = mask_2d.flatten()
        mse_flat = mse_map[i].flatten()
        m_mse = mse_flat[msk_flat].mean() if msk_flat.any() else 0.0

        for c in range(N_CHANNELS):
            ax = axes[i, c]

            # original signal in grey
            ax.plot(t, windows_orig[i][c], color="#aaaaaa", lw=0.9, zorder=2)

            # reconstruction overlaid at masked patches only
            for p in range(Pt):
                t0, t1 = p * patch_size, (p + 1) * patch_size
                if mask_2d[c, p]:
                    # shade background
                    ax.axvspan(t[t0], t[t1 - 1], alpha=0.18, color="#e74c3c", zorder=1)
                    # predicted signal in channel colour
                    ax.plot(t[t0:t1], windows_recon[i][c, t0:t1],
                            color=CH_COLORS[c], lw=1.6, zorder=3)
                    # connect original→pred at patch boundary with faint vertical tick
                    for boundary in [t0, t1 - 1]:
                        ax.plot([t[boundary], t[boundary]],
                                [windows_orig[i][c, boundary],
                                 windows_recon[i][c, boundary]],
                                color=CH_COLORS[c], lw=0.7, ls=":", alpha=0.6, zorder=4)

            # patch grid lines
            for p in range(1, Pt):
                ax.axvline(t[p * patch_size], color="#cccccc", lw=0.4, zorder=0)

            ax.margins(x=0)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(CH_NAMES[c], fontsize=8, color=CH_COLORS[c], pad=3)
            if c == 0:
                ax.set_ylabel(labels[i][:16], fontsize=6, rotation=0,
                              ha="right", va="center", labelpad=52)
                ax.text(-0.48, 0.5, f"MSE={m_mse:.4f}",
                        transform=ax.transAxes, fontsize=5.5,
                        ha="right", va="center", color="#555555")
            if i == N - 1:
                ax.set_xlabel("s", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── Figure 2: aggregate summary ───────────────────────────────────────────

def plot_summary(windows_orig, windows_recon, patch_mask, mse_map,
                 labels, patch_size, out_path):
    N  = len(windows_orig)
    Pt = N_PATCHES

    all_vis_mse, all_msk_mse = [], []
    per_seq_vis, per_seq_msk = [], []
    ch_vis_mse  = [[] for _ in range(N_CHANNELS)]
    ch_msk_mse  = [[] for _ in range(N_CHANNELS)]

    for i in range(N):
        mask_2d  = patch_mask[i].reshape(N_CHANNELS, Pt)
        mse_flat = mse_map[i].flatten()
        msk_flat = mask_2d.flatten()

        all_vis_mse.extend(mse_flat[~msk_flat].tolist())
        all_msk_mse.extend(mse_flat[ msk_flat].tolist())
        per_seq_vis.append(mse_flat[~msk_flat].mean() if (~msk_flat).any() else 0)
        per_seq_msk.append(mse_flat[ msk_flat].mean() if  msk_flat.any()  else 0)

        for c in range(N_CHANNELS):
            m_c = mask_2d[c]
            ch_vis_mse[c].extend(mse_map[i, c][~m_c].tolist())
            ch_msk_mse[c].extend(mse_map[i, c][ m_c].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # ── left: overall MSE distribution ────────────────────────────────────
    bins = np.linspace(0, np.percentile(all_msk_mse + all_vis_mse, 98) + 1e-8, 50)
    axes[0].hist(all_vis_mse, bins=bins, alpha=0.6, color="#2ecc71", density=True,
                 label=f"Visible  μ={np.mean(all_vis_mse):.4f}")
    axes[0].hist(all_msk_mse, bins=bins, alpha=0.6, color="#e74c3c", density=True,
                 label=f"Masked   μ={np.mean(all_msk_mse):.4f}")
    axes[0].axvline(np.mean(all_vis_mse), color="#27ae60", lw=1.5, ls="--")
    axes[0].axvline(np.mean(all_msk_mse), color="#c0392b", lw=1.5, ls="--")
    axes[0].set_xlabel("Patch MSE", fontsize=10)
    axes[0].set_ylabel("Density", fontsize=10)
    axes[0].set_title("All patches: visible vs masked\nMSE distribution", fontsize=10)
    axes[0].legend(fontsize=8)
    ratio = np.mean(all_msk_mse) / (np.mean(all_vis_mse) + 1e-8)
    axes[0].text(0.97, 0.97,
                 f"masked / visible ratio\n= {ratio:.2f}×\n\n"
                 f"(1.0 = no better than visible\n <2× = good reconstruction)",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=8,
                 bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=4))

    # ── middle: per-channel mean MSE ──────────────────────────────────────
    x = np.arange(N_CHANNELS)
    w = 0.35
    vis_means = [np.mean(ch_vis_mse[c]) if ch_vis_mse[c] else 0 for c in range(N_CHANNELS)]
    msk_means = [np.mean(ch_msk_mse[c]) if ch_msk_mse[c] else 0 for c in range(N_CHANNELS)]
    axes[1].bar(x - w/2, vis_means, width=w, color="#2ecc71", label="Visible", alpha=0.85)
    axes[1].bar(x + w/2, msk_means, width=w, color="#e74c3c", label="Masked",  alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(CH_NAMES, rotation=25, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean MSE", fontsize=10)
    axes[1].set_title("Per-channel reconstruction MSE\n(visible vs masked)", fontsize=10)
    axes[1].legend(fontsize=8)

    # ── right: per-sequence scatter ────────────────────────────────────────
    axes[2].scatter(per_seq_vis, per_seq_msk, s=60, alpha=0.85,
                    c=np.arange(N), cmap="tab20")
    for i, lbl in enumerate(labels):
        axes[2].annotate(lbl[:14], (per_seq_vis[i], per_seq_msk[i]),
                         fontsize=5, alpha=0.7)
    lim_max = max(max(per_seq_vis), max(per_seq_msk)) * 1.15
    axes[2].plot([0, lim_max], [0, lim_max], "k--", lw=0.8, label="y=x (equal)")
    axes[2].set_xlim(0, lim_max); axes[2].set_ylim(0, lim_max)
    axes[2].set_xlabel("Visible mean MSE", fontsize=10)
    axes[2].set_ylabel("Masked mean MSE", fontsize=10)
    axes[2].set_title("Per-sequence MSE\n(below diagonal = masked worse than visible)", fontsize=9)
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"MAE+DINO reconstruction quality  (N={N} sequences, mask_ratio=50%)\n"
        f"Overall masked MSE={np.mean(all_msk_mse):.4f}  visible MSE={np.mean(all_vis_mse):.4f}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")

    return np.mean(all_vis_mse), np.mean(all_msk_mse)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt",        required=True)
    ap.add_argument("--data",        required=True)
    ap.add_argument("--out",         default="data_vis/mae_dino_recon")
    ap.add_argument("--n",           type=int, default=8)
    ap.add_argument("--mask-ratio",  type=float, default=0.50)
    ap.add_argument("--seed",        type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading model from {args.ckpt} ...")
    encoder, recon_head, cfg, patch_size = load_model(args.ckpt)
    print(f"  patch_size={patch_size}  embed_dim={encoder.embed_dim}")

    print(f"Sampling {args.n} windows ...")
    windows, labels = sample_windows(args.data, args.n, cfg, seed=args.seed)
    N = len(windows)
    print(f"  loaded {N} windows")

    print(f"Running reconstruction (mask_ratio={args.mask_ratio}) ...")
    windows_recon, patch_mask, mse_map, windows_orig = reconstruct(
        encoder, recon_head, windows,
        mask_ratio=args.mask_ratio, seed=args.seed,
    )

    print("Figure 1: per-sequence panels ...")
    plot_panels(windows_orig, windows_recon, patch_mask, mse_map, labels, patch_size,
                os.path.join(args.out, "mae_dino_recon_panels.png"))

    print("Figure 2: aggregate summary ...")
    vis_mse, msk_mse = plot_summary(windows_orig, windows_recon, patch_mask, mse_map,
                                     labels, patch_size,
                                     os.path.join(args.out, "mae_dino_recon_summary.png"))

    print(f"\nReconstruction quality (N={N}, mask={args.mask_ratio:.0%}):")
    print(f"  Visible  patches mean MSE : {vis_mse:.6f}")
    print(f"  Masked   patches mean MSE : {msk_mse:.6f}")
    print(f"  Ratio masked/visible      : {msk_mse / (vis_mse + 1e-8):.2f}×")
    print()
    print("  Interpretation:")
    print("    ratio < 2×  → model reconstructs masked patches nearly as well as visible")
    print("    ratio 2–5×  → reasonable early training reconstruction")
    print("    ratio > 10× → model has not yet learned to predict masked content")


if __name__ == "__main__":
    main()
