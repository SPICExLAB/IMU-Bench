"""Evaluate BEiT masked reconstruction quality per dataset.

Pipeline:
  1. Mask 65% of patches (same as BEiT training)
  2. BEiT encoder predicts RVQ token indices for masked positions
  3. Look up predicted embeddings in RVQ codebook (sum all K levels)
  4. Decode via RVQ PatchDecoder → reconstructed signal at masked patches
  5. Compute MSE between original and reconstruction on masked patches only

Outputs: data_vis/beit_recon_mse.png
  - Left  : per-dataset MSE bar chart (masked patches only)
  - Right : per-dataset heatmap (signal channel × masked-patch MSE)
  + example qualitative panel: original vs reconstruction for 3 actions

Usage (from repo root):
    MPLBACKEND=Agg python scripts/eval_beit_recon.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── paths ─────────────────────────────────────────────────────────────────────
BEIT_CKPT = ("runs/vit1d_beit_realdata_nopooling_8lvl/"
             "checkpoints/latest.pt")
TOK_CKPT  = ("runs/vit1d_rvq_tokenizer_realdata_nopooling_8lvl/"
             "checkpoints/latest.pt")
PARQUET   = "/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet"
OUT_FILE  = "data_vis/beit_recon_mse.png"

DATASETS  = ["pamap2", "recofit", "samosa", "shoaib", "ut_watch", "utd_mhad", "wisdm"]
N_WINDOWS = 200     # windows per dataset
WIN       = 128     # 2.56 s @ 50 Hz
MASK_RATIO    = 0.65
MASK_STRATEGY = "composite"   # same as training: randomly picks random/temporal/signal per sample
ACC_COLS  = ["acc_x", "acc_y", "acc_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]

# ── load tokenizer (codebooks + decoders) ─────────────────────────────────────
print("Loading RVQ tokenizer …")
tok_state = torch.load(TOK_CKPT, map_location="cpu", weights_only=False)
tok_cfg   = tok_state["cfg"]

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives.rvq_tokenizer import ResidualVectorQuantizer, PatchDecoder

tok_cfg_key = tok_cfg["objective"]["type"]
tok_obj_cfg = tok_cfg["objective"][tok_cfg_key]
num_levels  = tok_obj_cfg["num_levels"]   # 8
num_codes   = tok_obj_cfg["num_codes"]    # 1024
code_dim    = tok_obj_cfg["code_dim"]     # 256
hidden_dim  = tok_obj_cfg.get("decoder_hidden_dim", 512)
patch_size  = tok_cfg["vit1d"]["patch_size"]   # 16

rvq = ResidualVectorQuantizer(
    num_levels=num_levels, num_codes=num_codes, code_dim=code_dim,
    ema_decay=tok_obj_cfg["ema_decay"],
    commitment_weight=tok_obj_cfg["commitment_weight"],
)
rvq.load_state_dict(tok_state["rvq"], strict=True)
rvq.eval()

decoders = nn.ModuleList([
    PatchDecoder(code_dim, patch_size, hidden_dim) for _ in range(num_levels)
])
decoders.load_state_dict(tok_state["decoders"], strict=True)
decoders.eval()
print(f"  Tokenizer: {num_levels} lvls × {num_codes} codes × {code_dim}D")

# ── load BEiT encoder + prediction heads ──────────────────────────────────────
print("Loading BEiT encoder …")
beit_state = torch.load(BEIT_CKPT, map_location="cpu", weights_only=False)
beit_cfg   = beit_state["cfg"]

beit_encoder = ViT1DEncoder(beit_cfg)
beit_encoder.load_state_dict(beit_state["model"], strict=True)
beit_encoder.eval()

beit_heads = nn.ModuleList([
    nn.Linear(beit_encoder.embed_dim, num_codes) for _ in range(num_levels)
])
beit_heads.load_state_dict(beit_state["beit_heads"], strict=True)
beit_heads.eval()
print(f"  BEiT encoder: embed_dim={beit_encoder.embed_dim}  step={beit_state.get('step','?')}")


# ── reconstruction helper ──────────────────────────────────────────────────────
@torch.no_grad()
def reconstruct(x_np: np.ndarray, seed: int = 0):
    """x_np [B, 6, WIN] → (mse_raw, mse_znorm, recon [B,6,WIN], mask [B,C*Pt], ch_mse, ch_mse_znorm).

    Model always receives raw signal (it was trained without Z-norm).
    Z-norm MSE is computed by normalizing BOTH original and reconstruction with
    per-window per-channel stats from the original signal, so the scale is
    comparable across datasets (MSE ~ 1 means error ≈ signal variance).
    """
    x  = torch.from_numpy(x_np)   # [B, 6, WIN]
    B, C, T = x.shape
    Pt = T // patch_size           # 8
    N  = C * Pt                    # 48 (channel-major)

    # ── mask (same composite strategy as training) ────────────────────────────
    torch.manual_seed(seed)
    from imu_lm.objectives.masked_1d import generate_patch_mask
    mask = generate_patch_mask(B, Pt, C, MASK_RATIO, MASK_STRATEGY, device=x.device)

    # ── BEiT encoder ─────────────────────────────────────────────────────────
    _, patch_out = beit_encoder.forward_dino(x, mask)  # [B, N, D]

    # ── predict token indices for all positions, decode masked ones ───────────
    # Accumulate codebook embeddings across levels for each (b, n)
    cumul_emb = torch.zeros(B, N, code_dim)   # [B, N, code_dim]
    for k, (head, vq) in enumerate(zip(beit_heads, rvq.quantizers)):
        logits   = head(patch_out)             # [B, N, num_codes]
        pred_idx = logits.argmax(dim=-1)       # [B, N]
        emb_k    = vq.codebook[pred_idx]       # [B, N, code_dim]
        cumul_emb = cumul_emb + emb_k

    # ── decode cumulative embedding → patch signal ────────────────────────────
    # Use the final (K-th) decoder which decodes the fully-refined embedding
    pred_patches = decoders[-1](cumul_emb)     # [B, N, patch_size]

    # ── assemble full signal: masked→predicted, unmasked→original ─────────────
    # Reshape original to patches [B, C, Pt, patch_size] then to [B, N, patch_size]
    x_patches = x.view(B, C, Pt, patch_size)   \
                  .permute(0, 1, 2, 3)          \
                  .reshape(B, N, patch_size)

    recon_patches = torch.where(
        mask.unsqueeze(-1),   # [B, N, 1]
        pred_patches,
        x_patches,
    )   # [B, N, patch_size]

    # Reshape back to [B, C, T]
    recon = recon_patches.view(B, C, Pt, patch_size).reshape(B, C, T)

    # ── MSE on masked patches only ────────────────────────────────────────────
    diff_sq = (x_patches - pred_patches) ** 2   # [B, N, patch_size]
    mse_masked = diff_sq[mask].mean().item()

    # ── Z-norm MSE: each signal normalized by its own mean/std ───────────────
    # x_z     = (x     - mean(x))     / std(x)      per channel per window
    # recon_z = (recon - mean(recon)) / std(recon)   per channel per window
    # MSE_znorm = mean( (x_z - recon_z)² )  on masked patches only
    def _to_znorm_patches(patches_BCPt, B, C, Pt, patch_size):
        """patches_BCPt: [B, N, patch_size] channel-major → z-normed [B, N, patch_size]."""
        # Reconstruct [B, C, T] to compute per-channel stats over full window
        sig = patches_BCPt.reshape(B, C, Pt * patch_size)   # [B, C, T]
        mu  = sig.mean(dim=2, keepdim=True)                  # [B, C, 1]
        sd  = sig.std(dim=2,  keepdim=True) + 1e-8
        # Expand stats to patch layout [B, N, patch_size]
        mu_p = mu.repeat(1, 1, Pt).reshape(B, C * Pt, 1).expand_as(patches_BCPt)
        sd_p = sd.repeat(1, 1, Pt).reshape(B, C * Pt, 1).expand_as(patches_BCPt)
        return (patches_BCPt - mu_p) / sd_p

    x_patches_z    = _to_znorm_patches(x_patches,    B, C, Pt, patch_size)
    pred_patches_z = _to_znorm_patches(pred_patches,  B, C, Pt, patch_size)
    diff_sq_z  = (x_patches_z - pred_patches_z) ** 2
    mse_masked_z = diff_sq_z[mask].mean().item()

    # Per-channel MSE (raw and Z-norm) on masked patches
    ch_mse, ch_mse_z = [], []
    for c in range(C):
        ch_mask = mask[:, c * Pt : (c + 1) * Pt]    # [B, Pt]
        s = c * Pt;  e = (c + 1) * Pt
        ch_diff   = (x_patches[:,   s:e, :] - pred_patches[:,   s:e, :]) ** 2
        ch_diff_z = (x_patches_z[:, s:e, :] - pred_patches_z[:, s:e, :]) ** 2
        if ch_mask.any():
            ch_mse.append(ch_diff[ch_mask].mean().item())
            ch_mse_z.append(ch_diff_z[ch_mask].mean().item())
        else:
            ch_mse.append(float("nan"))
            ch_mse_z.append(float("nan"))

    return (mse_masked, mse_masked_z,
            recon.numpy(), mask.numpy(),
            np.array(ch_mse), np.array(ch_mse_z))


# ── per-dataset evaluation ────────────────────────────────────────────────────
needed_cols = ["dataset", "session_id", "timestamp_ns"] + ACC_COLS + GYRO_COLS
rng = np.random.default_rng(42)

results: dict[str, dict] = {}

for ds in DATASETS:
    print(f"  {ds} …", end=" ", flush=True)
    part = pd.read_parquet(PARQUET, columns=needed_cols,
                           filters=[("dataset", "==", ds)])
    if part.empty:
        print("not found, skip")
        continue
    part = part.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)

    # Sample windows
    sessions = part["session_id"].unique()
    windows  = []
    for sess in sessions:
        seg  = part[part["session_id"] == sess]
        vals = seg[ACC_COLS + GYRO_COLS].values.T.astype(np.float32)
        L    = vals.shape[1]
        if L < WIN:
            continue
        n_from = max(1, min(N_WINDOWS // max(len(sessions), 1), L - WIN + 1))
        starts = rng.choice(L - WIN + 1, size=n_from, replace=False)
        for s in starts:
            windows.append(vals[:, s : s + WIN])
    if not windows:
        print("no windows, skip")
        continue
    windows = np.stack(windows[:N_WINDOWS], axis=0)   # [N, 6, WIN]

    mse, mse_z, _, _, ch_mse, ch_mse_z = reconstruct(windows, seed=42)
    results[ds] = {
        "mse": mse, "ch_mse": ch_mse,
        "mse_norm": mse_z, "ch_mse_norm": ch_mse_z,
        "n": len(windows),
    }
    print(f"n={len(windows):3d}  MSE={mse:.4f}  RMSE={mse**0.5:.4f}"
          f"  MSE_znorm={mse_z:.4f}  RMSE_znorm={mse_z**0.5:.4f}")

if not results:
    raise RuntimeError("No results — check checkpoint paths and parquet.")

# ── qualitative panel: 1 example window per first 3 datasets ─────────────────
VIS_DATASETS = list(results.keys())[:3]
VIS_ACTIONS  = {  # pick first available action in each dataset
    "utd_mhad": "clap",
    "wisdm": "walking",
    "pamap2": "running",
    "samosa": None,
    "recofit": None,
    "shoaib": None,
    "ut_watch": None,
}
LABEL_COL = "dataset_activity_label"
vis_samples = {}
for ds in VIS_DATASETS:
    needed = ["dataset", LABEL_COL, "session_id", "timestamp_ns"] + ACC_COLS + GYRO_COLS
    part   = pd.read_parquet(PARQUET, columns=needed,
                             filters=[("dataset", "==", ds)])
    part   = part.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)
    target_act = VIS_ACTIONS.get(ds)
    if target_act and target_act in part[LABEL_COL].values:
        seg = part[part[LABEL_COL] == target_act]
    else:
        seg = part
    sess = seg["session_id"].iloc[0]
    seg  = seg[seg["session_id"] == sess]
    vals = seg[ACC_COLS + GYRO_COLS].values.T.astype(np.float32)
    if vals.shape[1] >= WIN:
        x_np = vals[:, :WIN][None]   # [1, 6, WIN]
        _, _, recon, mask, _, _ = reconstruct(x_np, seed=0)
        vis_samples[ds] = {
            "x":     x_np[0],       # [6, WIN]
            "recon": recon[0],      # [6, WIN]
            "mask":  mask[0],       # [N]
            "label": target_act or "sample",
        }

# ── plot ───────────────────────────────────────────────────────────────────────
ACC_C  = ["#e74c3c", "#2ecc71", "#3498db"]
GYRO_C = ["#e67e22", "#1abc9c", "#9b59b6"]
CH_LABELS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

ds_names      = list(results.keys())
mse_vals      = [results[d]["mse"]      for d in ds_names]
mse_norm_vals = [results[d]["mse_norm"] for d in ds_names]
rmse_vals      = [v ** 0.5 for v in mse_vals]
rmse_norm_vals = [v ** 0.5 for v in mse_norm_vals]
ch_mse_mat      = np.stack([results[d]["ch_mse"]      for d in ds_names], axis=0)
ch_mse_norm_mat = np.stack([results[d]["ch_mse_norm"] for d in ds_names], axis=0)

Pt = WIN // patch_size
t  = np.arange(WIN) / 50.0

n_vis = len(vis_samples)
n_rows_qual = n_vis

fig = plt.figure(figsize=(16, 4 + 4 * n_rows_qual))
fig.suptitle(
    f"BEiT masked reconstruction quality — vit1d_beit_realdata_nopooling_8lvl\n"
    f"mask_ratio={MASK_RATIO}  |  {N_WINDOWS} windows/dataset  |  "
    f"MSE on masked patches only",
    fontsize=11, fontweight="bold",
)

outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.55,
                          height_ratios=[2.2, max(1, n_rows_qual) * 4])

# ── top: MSE bars + Z-norm heatmap ───────────────────────────────────────────
top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                       wspace=0.35, width_ratios=[1.4, 1.6])

# grouped bar chart: raw vs Z-norm
ax_bar = fig.add_subplot(top[0])
x_pos  = np.arange(len(ds_names))
w      = 0.38
colors = plt.cm.tab10(np.linspace(0, 1, len(ds_names)))
bars1  = ax_bar.bar(x_pos - w/2, rmse_vals,      width=w, label="raw signal",
                    color=[(*c[:3], 0.85) for c in colors], edgecolor="k", lw=0.5)
bars2  = ax_bar.bar(x_pos + w/2, rmse_norm_vals, width=w, label="Z-norm",
                    color=[(*c[:3], 0.45) for c in colors], edgecolor="k", lw=0.5)
ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels(ds_names, rotation=30, ha="right", fontsize=8)
ax_bar.set_ylabel("RMSE  (masked patches)", fontsize=9)
ax_bar.set_title("Per-dataset reconstruction RMSE\nraw signal vs Z-norm",
                 fontsize=9, fontweight="bold")
ax_bar.legend(fontsize=8, framealpha=0.7)
ax_bar.tick_params(axis="y", labelsize=8)
for bar, v in zip(bars1, rmse_vals):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6)
for bar, v in zip(bars2, rmse_norm_vals):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=6)
ax_bar.grid(axis="y", alpha=0.3)

# Z-norm channel × dataset heatmap
ax_heat = fig.add_subplot(top[1])
im = ax_heat.imshow(ch_mse_norm_mat ** 0.5, aspect="auto", cmap="YlOrRd", origin="upper")
ax_heat.set_xticks(range(6))
ax_heat.set_xticklabels(CH_LABELS, fontsize=8, rotation=30)
ax_heat.set_yticks(range(len(ds_names)))
ax_heat.set_yticklabels(ds_names, fontsize=8)
ax_heat.set_title("Z-norm RMSE per channel  (dataset × channel)",
                  fontsize=9, fontweight="bold")
for i in range(len(ds_names)):
    for j in range(6):
        v = ch_mse_norm_mat[i, j] ** 0.5
        ax_heat.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.5)
plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)

# ── bottom: qualitative reconstructions ──────────────────────────────────────
if vis_samples:
    bot_outer = gridspec.GridSpecFromSubplotSpec(
        n_rows_qual, 1, subplot_spec=outer[1], hspace=0.6
    )
    for row_i, (ds, data) in enumerate(vis_samples.items()):
        x      = data["x"]      # [6, WIN]
        recon  = data["recon"]  # [6, WIN]
        mask_n = data["mask"]   # [48] bool
        label  = data["label"]

        # Expand patch mask to sample level for shading
        # mask_n is channel-major [C*Pt]: patch n is masked?
        gs_row = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=bot_outer[row_i],
            wspace=0.3, hspace=0.2, width_ratios=[2.5, 2.5],
        )
        ax_a = fig.add_subplot(gs_row[0, 0])
        ax_g = fig.add_subplot(gs_row[1, 0], sharex=ax_a)
        ax_ra = fig.add_subplot(gs_row[0, 1])
        ax_rg = fig.add_subplot(gs_row[1, 1], sharex=ax_ra)

        for c in range(3):
            ax_a.plot(t, x[c],    color=ACC_C[c],  lw=1.0, label=f"acc {'xyz'[c]}")
            ax_g.plot(t, x[c+3], color=GYRO_C[c], lw=1.0, label=f"gyro {'xyz'[c]}")
            ax_ra.plot(t, x[c],    color=ACC_C[c],  lw=1.0, alpha=0.3)
            ax_rg.plot(t, x[c+3], color=GYRO_C[c], lw=1.0, alpha=0.3)
            ax_ra.plot(t, recon[c],    color=ACC_C[c],  lw=1.0, ls="--")
            ax_rg.plot(t, recon[c+3], color=GYRO_C[c], lw=1.0, ls="--")

        # Shade masked patches for acc (ch0 = first Pt patches)
        for p in range(Pt):
            if mask_n[p]:   # ch0 mask (acc_x)
                t0, t1 = p * patch_size / 50.0, (p + 1) * patch_size / 50.0
                for ax in (ax_ra, ax_rg):
                    ax.axvspan(t0, t1, alpha=0.15, color="gray")

        ax_a.set_title(f"[{ds.upper()} / {label}]  Original acc",
                       fontsize=8, fontweight="bold")
        ax_g.set_title("Original gyro", fontsize=8)
        ax_ra.set_title("BEiT reconstruction  (grey = masked)", fontsize=8, fontweight="bold")
        ax_rg.set_title("Gyro reconstruction", fontsize=8)
        for ax in (ax_a, ax_g, ax_ra, ax_rg):
            ax.axhline(0, color="k", lw=0.3, ls="--", alpha=0.4)
            ax.tick_params(labelsize=7)
        ax_a.legend(fontsize=6, ncol=3, loc="upper right", framealpha=0.5)
        ax_g.legend(fontsize=6, ncol=3, loc="upper right", framealpha=0.5)
        legend_els = [
            Line2D([0], [0], color="gray", lw=1.0, alpha=0.3, label="original"),
            Line2D([0], [0], color="gray", lw=1.0, ls="--",   label="BEiT recon"),
        ]
        ax_ra.legend(handles=legend_els, fontsize=7, loc="upper right", framealpha=0.5)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FILE}")

# ── text summary ───────────────────────────────────────────────────────────────
print("\n── Per-dataset masked-patch reconstruction (raw signal) ──")
print(f"{'dataset':>12}  {'n_win':>5}  {'MSE':>8}  {'RMSE':>8}")
print("-" * 42)
for ds in ds_names:
    r = results[ds]
    print(f"{ds:>12}  {r['n']:>5}  {r['mse']:>8.4f}  {r['mse']**0.5:>8.4f}")

print("\n── Per-dataset masked-patch reconstruction (Z-norm) ──")
print(f"{'dataset':>12}  {'n_win':>5}  {'MSE':>8}  {'RMSE':>8}")
print("-" * 42)
for ds in ds_names:
    r = results[ds]
    print(f"{ds:>12}  {r['n']:>5}  {r['mse_norm']:>8.4f}  {r['mse_norm']**0.5:>8.4f}")
