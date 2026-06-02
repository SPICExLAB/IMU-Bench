"""Evaluate RVQ tokenizer reconstruction RMSE per dataset and per level.

For each dataset in the master parquet, samples N random windows, runs them
through the trained RVQ tokenizer, and computes RMSE between the original
signal and the cumulative reconstruction at each of the K levels.

Output: data_vis/rvq_recon_rmse.png
  - Left : RMSE vs. RVQ level (coarse→fine), one line per dataset
  - Right: per-level RMSE heatmap (dataset × level)

Usage (from repo root):
    MPLBACKEND=Agg python scripts/eval_rvq_recon.py
"""

from __future__ import annotations
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── config ────────────────────────────────────────────────────────────────────
CKPT_PATH   = ("runs/vit1d_rvq_tokenizer_realdata_nopooling_8lvl/"
               "checkpoints/latest.pt")
PARQUET     = "/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet"
OUT_FILE    = "data_vis/rvq_recon_rmse.png"

DATASETS    = ["pamap2", "recofit", "samosa", "shoaib", "ut_watch", "utd_mhad", "wisdm"]
N_WINDOWS   = 200      # windows per dataset
WIN         = 128      # samples per window (2.56 s @ 50 Hz)
ACC_COLS    = ["acc_x", "acc_y", "acc_z"]
GYRO_COLS   = ["gyro_x", "gyro_y", "gyro_z"]
LABEL_COL   = "dataset_activity_label"

# ── load checkpoint ───────────────────────────────────────────────────────────
print("Loading checkpoint …")
ckpt    = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
tok_cfg = ckpt["cfg"]

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.objectives.rvq_tokenizer import ResidualVectorQuantizer, PatchDecoder

encoder = ViT1DEncoder(tok_cfg)
encoder.load_state_dict(ckpt["model"], strict=True)
encoder.eval()

obj_key    = tok_cfg["objective"]["type"]
obj_cfg    = tok_cfg["objective"][obj_key]
num_levels = obj_cfg["num_levels"]
num_codes  = obj_cfg["num_codes"]
code_dim   = obj_cfg["code_dim"]
hidden_dim = obj_cfg.get("decoder_hidden_dim", 512)
patch_size = tok_cfg["vit1d"]["patch_size"]

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

print(f"  {num_levels} levels × {num_codes} codes × {code_dim}D "
      f"patch_size={patch_size}")

# ── sample windows per dataset ────────────────────────────────────────────────
needed_cols = ["dataset", "session_id", "timestamp_ns"] + ACC_COLS + GYRO_COLS

rng = np.random.default_rng(42)


def sample_windows(df_ds: pd.DataFrame, n: int) -> np.ndarray | None:
    """Sample up to n random non-overlapping WIN-length windows. Returns [n, 6, WIN]."""
    sessions = df_ds["session_id"].unique()
    windows  = []
    for sess in sessions:
        seg = df_ds[df_ds["session_id"] == sess]
        vals = seg[ACC_COLS + GYRO_COLS].values.T.astype(np.float32)  # [6, N]
        L = vals.shape[1]
        if L < WIN:
            continue
        max_start = L - WIN
        n_from_sess = max(1, min(n // max(len(sessions), 1), max_start + 1))
        starts = rng.choice(max_start + 1, size=n_from_sess, replace=False)
        for s in starts:
            windows.append(vals[:, s : s + WIN])
    if not windows:
        return None
    arr = np.stack(windows[:n], axis=0)   # [n, 6, WIN]
    return arr


@torch.no_grad()
def compute_rmse_levels(windows: np.ndarray) -> np.ndarray:
    """windows [N, 6, WIN] → rmse [K] averaged over all windows."""
    x = torch.from_numpy(windows)          # [N, 6, WIN]
    B, C, T = x.shape
    Pt = T // patch_size

    tokens, _ = encoder.forward_tokens(x)  # [N, C*Pt, D]
    z = F.normalize(tokens, dim=-1)

    residual     = z.clone()
    cumulative_q = torch.zeros_like(z)
    rmse_per_level = []

    for k, vq in enumerate(rvq.quantizers):
        q, _ = vq(residual)
        cumulative_q = cumulative_q + q
        pred  = decoders[k](cumulative_q)                         # [N, C*Pt, patch_size]
        recon = pred.view(B, C, Pt, patch_size).reshape(B, C, T)  # [N, 6, WIN]
        mse   = ((x - recon) ** 2).mean().item()
        rmse_per_level.append(mse ** 0.5)
        residual = residual - q.detach()

    return np.array(rmse_per_level)   # [K]


# ── evaluate ──────────────────────────────────────────────────────────────────
results: dict[str, np.ndarray] = {}   # dataset → rmse [K]

for ds in DATASETS:
    print(f"  {ds} …", end=" ", flush=True)
    part = pd.read_parquet(PARQUET, columns=needed_cols,
                           filters=[("dataset", "==", ds)])
    if part.empty:
        print("NOT FOUND, skipping")
        continue
    part = part.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)

    windows = sample_windows(part, N_WINDOWS)
    if windows is None:
        print("no valid windows, skipping")
        continue

    rmse = compute_rmse_levels(windows)
    results[ds] = rmse
    print(f"n={len(windows):3d}  L1={rmse[0]:.3f}  L4={rmse[3]:.3f}  L8={rmse[7]:.3f}")

if not results:
    raise RuntimeError("No results — check checkpoint and parquet paths.")

# ── plot ──────────────────────────────────────────────────────────────────────
ds_names  = list(results.keys())
rmse_mat  = np.stack([results[d] for d in ds_names], axis=0)  # [D, K]
levels    = np.arange(1, num_levels + 1)

cmap   = plt.cm.get_cmap("tab10", len(ds_names))
colors = [cmap(i) for i in range(len(ds_names))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"RVQ reconstruction RMSE — vit1d_rvq_tokenizer_realdata_nopooling_8lvl\n"
    f"({num_levels} levels × {num_codes} codes, {N_WINDOWS} windows/dataset)",
    fontsize=11, fontweight="bold",
)

# ── left: RMSE vs level ───────────────────────────────────────────────────────
ax = axes[0]
for i, (ds, color) in enumerate(zip(ds_names, colors)):
    ax.plot(levels, results[ds], marker="o", ms=5, lw=1.8,
            color=color, label=ds)
ax.set_xlabel("RVQ level  (1=coarse, 8=fine)", fontsize=10)
ax.set_ylabel("RMSE  (raw signal units)", fontsize=10)
ax.set_title("Cumulative reconstruction RMSE per level", fontsize=10)
ax.set_xticks(levels)
ax.legend(fontsize=8, framealpha=0.7)
ax.grid(True, alpha=0.3)

# ── right: heatmap (dataset × level) ─────────────────────────────────────────
ax2 = axes[1]
im  = ax2.imshow(rmse_mat, aspect="auto", cmap="YlOrRd", origin="upper")
ax2.set_xticks(np.arange(num_levels))
ax2.set_xticklabels([f"L{k+1}" for k in range(num_levels)], fontsize=8)
ax2.set_yticks(np.arange(len(ds_names)))
ax2.set_yticklabels(ds_names, fontsize=9)
ax2.set_xlabel("RVQ level", fontsize=10)
ax2.set_title("RMSE heatmap  (dataset × level)", fontsize=10)
for i in range(len(ds_names)):
    for j in range(num_levels):
        ax2.text(j, i, f"{rmse_mat[i, j]:.2f}", ha="center", va="center",
                 fontsize=7, color="black")
plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02).ax.tick_params(labelsize=7)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FILE}")
