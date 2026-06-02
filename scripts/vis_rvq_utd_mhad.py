"""Visualise the RVQ tokenizer (vit1d_rvq_tokenizer_ocdetect_anyrot_nopooling_8lvl)
applied to UTD-MHAD action samples.

For each of several action classes, shows:
  Left  : original IMU signal  (acc + gyro, Android frame)
  Centre: RVQ token-index heatmap  (48 patch positions × 8 levels)
  Right : cumulative signal reconstruction per RVQ level

Usage (from repo root):
    MPLBACKEND=Agg python scripts/vis_rvq_utd_mhad.py
"""

from __future__ import annotations
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── paths ────────────────────────────────────────────────────────────────────
CKPT_PATH  = ("runs/vit1d_rvq_tokenizer_ocdetect_anyrot_nopooling_8lvl/"
              "checkpoints/latest.pt")
PARQUET    = "/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet"
OUT_FILE   = "data_vis/rvq_utd_mhad.png"

# Actions to visualise (one 2.56 s window each)
SHOW_ACTIONS = ["clap", "throw", "arm_curl", "bowling", "wave", "boxing"]

LABEL_COL = "dataset_activity_label"
ACC_COLS  = ["acc_x", "acc_y", "acc_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]
FS        = 50
PATCH     = 16
WIN       = 128                      # 8 temporal patches per channel
N_PATCHES = 6 * (WIN // PATCH)      # 48  (channel-major layout)

# ── load checkpoint ──────────────────────────────────────────────────────────
print("Loading RVQ tokenizer checkpoint …")
ckpt    = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
tok_cfg = ckpt["cfg"]

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives.rvq_tokenizer import ResidualVectorQuantizer, PatchDecoder

encoder = ViT1DEncoder(tok_cfg)
encoder.load_state_dict(ckpt["model"], strict=True)
encoder.eval()

obj_key    = tok_cfg["objective"]["type"]
obj_cfg    = tok_cfg["objective"][obj_key]
num_levels = obj_cfg["num_levels"]            # 8
num_codes  = obj_cfg["num_codes"]             # 1024
code_dim   = obj_cfg["code_dim"]              # 256
hidden_dim = obj_cfg.get("decoder_hidden_dim", 512)
patch_size = tok_cfg["vit1d"]["patch_size"]   # 16

rvq = ResidualVectorQuantizer(
    num_levels=num_levels, num_codes=num_codes,
    code_dim=code_dim, ema_decay=obj_cfg["ema_decay"],
    commitment_weight=obj_cfg["commitment_weight"],
)
rvq.load_state_dict(ckpt["rvq"], strict=True)
rvq.eval()

decoders = torch.nn.ModuleList([
    PatchDecoder(code_dim, patch_size, hidden_dim) for _ in range(num_levels)
])
decoders.load_state_dict(ckpt["decoders"], strict=True)
decoders.eval()

print(f"  Loaded: {num_levels} levels × {num_codes} codes × {code_dim}D  "
      f"| pool_channels={obj_cfg['pool_channels']}")

# ── load UTD-MHAD samples from master parquet ────────────────────────────────
print("Loading UTD-MHAD from master parquet …")
needed_cols = (["dataset", LABEL_COL, "subject_id", "session_id", "timestamp_ns"]
               + ACC_COLS + GYRO_COLS)
df = pd.read_parquet(PARQUET, columns=needed_cols,
                     filters=[("dataset", "==", "utd_mhad")])
df = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)
print(f"  UTD-MHAD timesteps: {len(df)}  "
      f"| actions: {sorted(df[LABEL_COL].unique())}")


def get_sample(action: str) -> np.ndarray | None:
    """Return [6, 128] float32 from the first session of this action."""
    rows = df[df[LABEL_COL] == action]
    if rows.empty:
        return None
    sess    = rows["session_id"].iloc[0]
    seg     = rows[rows["session_id"] == sess]
    vals    = seg[ACC_COLS + GYRO_COLS].values.T.astype(np.float32)  # [6, N]
    if vals.shape[1] < WIN:
        vals = np.pad(vals, ((0, 0), (0, WIN - vals.shape[1])))
    return vals[:, :WIN]


@torch.no_grad()
def tokenize(x_np: np.ndarray):
    """x_np [6,128] → indices [K, N],  recon_levels list[K] of [6,128]."""
    x = torch.from_numpy(x_np).unsqueeze(0)   # [1, 6, 128]
    B, C, T = x.shape
    Pt = T // patch_size

    tokens, _ = encoder.forward_tokens(x)     # [1, C*Pt, D]
    z = F.normalize(tokens, dim=-1)

    _, indices_list, _ = rvq(z)
    indices = torch.stack(indices_list, dim=0).squeeze(1)   # [K, N]

    residual     = z
    cumulative_q = torch.zeros_like(z)
    recon_levels = []
    for k, vq in enumerate(rvq.quantizers):
        q, _ = vq(residual)
        cumulative_q = cumulative_q + q
        pred = decoders[k](cumulative_q)                               # [1, N, patch_size]
        recon = pred.view(1, C, Pt, patch_size).reshape(1, C, T)
        recon_levels.append(recon.squeeze(0).numpy())
        residual = residual - q.detach()

    return indices.numpy(), recon_levels


# ── collect ───────────────────────────────────────────────────────────────────
samples: dict = {}
for action in SHOW_ACTIONS:
    x = get_sample(action)
    if x is None:
        print(f"  WARNING: '{action}' not found, skipping")
        continue
    idx, recon = tokenize(x)
    samples[action] = {"x": x, "indices": idx, "recon": recon}
    print(f"  {action:25s}  idx={idx.shape}  "
          f"recon[0]=[{recon[0].min():.2f}, {recon[0].max():.2f}]")

if not samples:
    raise RuntimeError("No samples collected.")

# ── plot ──────────────────────────────────────────────────────────────────────
ACC_C  = ["#e74c3c", "#2ecc71", "#3498db"]
GYRO_C = ["#e67e22", "#1abc9c", "#9b59b6"]

n_actions = len(samples)
t  = np.arange(WIN) / FS
Pt = WIN // patch_size

fig = plt.figure(figsize=(20, 4.5 * n_actions))
fig.suptitle(
    f"RVQ tokenizer ({num_levels} levels × {num_codes} codes) applied to UTD-MHAD\n"
    "Left: original IMU  |  Centre: token-index heatmap  |  Right: cumulative reconstruction",
    fontsize=12, fontweight="bold", y=1.002,
)

for row_i, (action, data) in enumerate(samples.items()):
    x     = data["x"]        # [6, 128]
    idx   = data["indices"]  # [8, 48]
    recon = data["recon"]    # list[8] of [6, 128]

    gs = gridspec.GridSpecFromSubplotSpec(
        2, 3,
        subplot_spec=gridspec.GridSpec(n_actions, 1, figure=fig, hspace=0.55)[row_i],
        wspace=0.35, hspace=0.15, width_ratios=[2.5, 1.8, 2.5],
    )

    # ── original signal ───────────────────────────────────────────────────────
    ax_acc  = fig.add_subplot(gs[0, 0])
    ax_gyro = fig.add_subplot(gs[1, 0], sharex=ax_acc)
    for c in range(3):
        ax_acc.plot(t, x[c],     color=ACC_C[c],  lw=1.0, label=f"acc {'xyz'[c]}")
        ax_gyro.plot(t, x[c+3], color=GYRO_C[c], lw=1.0, label=f"gyro {'xyz'[c]}")
    ax_acc.set_title(f"[{action.upper()}]  Original — Accelerometer",
                     fontsize=9, fontweight="bold")
    ax_gyro.set_title("Original — Gyroscope", fontsize=9)
    ax_acc.set_ylabel("m s⁻²", fontsize=8)
    ax_gyro.set_ylabel("rad s⁻¹", fontsize=8)
    ax_gyro.set_xlabel("Time (s)", fontsize=8)
    ax_acc.legend(fontsize=7, loc="upper right", ncol=3, framealpha=0.6)
    ax_gyro.legend(fontsize=7, loc="upper right", ncol=3, framealpha=0.6)
    for ax in (ax_acc, ax_gyro):
        ax.axhline(0, color="k", lw=0.3, ls="--", alpha=0.4)
        ax.tick_params(labelsize=7)

    # ── token-index heatmap ───────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[:, 1])
    im = ax_heat.imshow(idx, aspect="auto", interpolation="nearest",
                        cmap="tab20", vmin=0, vmax=num_codes - 1, origin="upper")
    ax_heat.set_title("RVQ token indices\n(level × patch position)",
                      fontsize=9, fontweight="bold")
    ax_heat.set_xlabel("Patch position (channel-major)", fontsize=8)
    ax_heat.set_ylabel("RVQ level  (coarse → fine)", fontsize=8)
    ax_heat.set_yticks(range(num_levels))
    ax_heat.set_yticklabels([f"L{k+1}" for k in range(num_levels)], fontsize=7)
    for ch in range(1, 6):
        ax_heat.axvline(ch * Pt - 0.5, color="white", lw=0.8, alpha=0.6)
    ax_heat.set_xticks([ch * Pt + Pt // 2 - 1 for ch in range(6)])
    ax_heat.set_xticklabels(["aX", "aY", "aZ", "gX", "gY", "gZ"], fontsize=7)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # ── reconstruction overlay ────────────────────────────────────────────────
    ax_racc  = fig.add_subplot(gs[0, 2])
    ax_rgyro = fig.add_subplot(gs[1, 2], sharex=ax_racc)
    for c in range(3):
        ax_racc.plot(t, x[c],     color=ACC_C[c],  lw=1.2, alpha=0.25)
        ax_rgyro.plot(t, x[c+3], color=GYRO_C[c], lw=1.2, alpha=0.25)
    for lvl, alpha, ls in zip([0, 3, 7], [0.5, 0.7, 1.0], [":", "--", "-"]):
        r = recon[lvl]
        for c in range(3):
            ax_racc.plot(t, r[c],     color=ACC_C[c],  lw=0.9, alpha=alpha, ls=ls)
            ax_rgyro.plot(t, r[c+3], color=GYRO_C[c], lw=0.9, alpha=alpha, ls=ls)
    ax_racc.set_title("Reconstruction  (L1=coarse … L8=fine)",
                      fontsize=9, fontweight="bold")
    ax_rgyro.set_title("Gyro reconstruction", fontsize=9)
    ax_racc.set_ylabel("m s⁻²", fontsize=8)
    ax_rgyro.set_ylabel("rad s⁻¹", fontsize=8)
    ax_rgyro.set_xlabel("Time (s)", fontsize=8)
    legend_elems = [
        Line2D([0], [0], color="gray", lw=1.2, alpha=0.25, ls="-",  label="original"),
        Line2D([0], [0], color="gray", lw=0.9, alpha=0.5,  ls=":",  label="L1 (≤1 Hz)"),
        Line2D([0], [0], color="gray", lw=0.9, alpha=0.7,  ls="--", label="L4"),
        Line2D([0], [0], color="gray", lw=0.9, alpha=1.0,  ls="-",  label="L8 (full)"),
    ]
    ax_racc.legend(handles=legend_elems, fontsize=7, loc="upper right", framealpha=0.6)
    for ax in (ax_racc, ax_rgyro):
        ax.axhline(0, color="k", lw=0.3, ls="--", alpha=0.4)
        ax.tick_params(labelsize=7)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FILE}")
