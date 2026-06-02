#!/usr/bin/env python3
"""Visualize MAE reconstruction: ground-truth vs reconstructed IMU signal.

Supports three data sources:
  - OCDetect CSV  (default, FLU transform applied)
  - IMULM parquet (--parquet flag, used by vit1d_realdata_pretrain)
  - Nymeria pkl   (--pkl flag, used by vit1d_mae_nymeria_real_wrist_flu)

mask_ratio and channel names are read from the checkpoint config automatically.

Usage
-----
    # OCDetect MAE
    python scripts/vis_mae_reconstruction.py \
        --ckpt runs/vit1d_mae_ocdetect_flu/checkpoints/best.pt \
        --csv  /mnt/nas2/yiquan_data/preprocessed/OCDetect_01_recording_00_...csv \
        --out  data_vis/mae_reconstruction.png

    # Realdata pretrain (IMULM parquet)
    python scripts/vis_mae_reconstruction.py \
        --ckpt    runs/vit1d_realdata_pretrain/checkpoints/best.pt \
        --parquet /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
        --dataset pamap2 \
        --out     data_vis/mae_reconstruction_realdata.png

    # Nymeria real wrist
    python scripts/vis_mae_reconstruction.py \
        --ckpt runs/vit1d_mae_nymeria_real_wrist_flu/checkpoints/best.pt \
        --pkl  /mnt/nas2/processed_data_body_Ty_rwrist/20230607_s0_james_johnson_act1_7xwm28_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
        --out  data_vis/mae_reconstruction_nymeria.png
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

WINDOW_LEN = 128   # samples @ 50 Hz = 2.56 s

OCDETECT_IMU_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
PARQUET_IMU_COLS  = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


# ---------------------------------------------------------------------------
# Data loading — OCDetect CSV
# ---------------------------------------------------------------------------
def load_window_csv(csv_path: str, cfg: dict, window_idx: int = 20):
    """Load one pre-processed window [6, 128] from an OCDetect CSV."""
    df  = pd.read_csv(csv_path, usecols=["ignore"] + OCDETECT_IMU_COLS, low_memory=False)
    imu = df[OCDETECT_IMU_COLS].values.astype(np.float32)
    ignore = df["ignore"].values.astype(np.int32)

    # Android → FLU
    imu[:, 0] *= -1   # acc_x  → acc_F
    imu[:, 1] *= -1   # acc_y  → acc_L
    imu[:, 3] *= -1   # gyro_x → gyro_F
    imu[:, 4] *= -1   # gyro_y → gyro_L

    # Walk through usable segments to find the n-th clean window
    usable = (ignore == 0)
    start, count = 0, 0
    for i, ok in enumerate(usable):
        if ok:
            count += 1
            if count == WINDOW_LEN:
                seg_start = i - WINDOW_LEN + 1
                if window_idx == 0:
                    start = seg_start
                    break
                window_idx -= 1
                count = 0
        else:
            count = 0

    raw = imu[start : start + WINDOW_LEN]          # [128, 6]
    return preprocess_window(raw, cfg), start       # [6, 128]


# ---------------------------------------------------------------------------
# Data loading — Nymeria pkl (IMU only, no SMPL needed)
# ---------------------------------------------------------------------------

_R_WRIST_FROM_FLU = np.array(
    [[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], dtype=np.float64
)


def load_window_pkl(pkl_path: str, cfg: dict, window_idx: int = 20,
                    add_gravity: bool = True):
    """Load one pre-processed window [6, 128] from a Nymeria pkl file."""
    import pickle
    with open(pkl_path, "rb") as fh:
        raw = pickle.load(fh)

    wrist_imu = raw.get("wrist_imu_data", {})
    wrist_gt  = raw.get("wrist_gt_data",  {})

    fps_body  = float(raw.get("imu_sampling_rate_hz") or
                      raw.get("pose_sampling_rate_hz") or 50)
    fps_wrist = float(raw.get("wrist_imu_sampling_rate_hz", fps_body))
    ds = max(1, round(fps_wrist / 50))

    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    acc_l  = _np(wrist_imu["accel"]).astype(np.float64)[::ds]
    gyro_l = _np(wrist_imu["gyro"]).astype(np.float64)[::ds]

    orient = wrist_gt["orientation"]
    try:
        import pypose as pp
        if isinstance(orient, pp.LieTensor):
            R_world = orient.matrix().detach().cpu().numpy().astype(np.float64)[::ds]
        else:
            raise TypeError
    except (ImportError, TypeError):
        from scipy.spatial.transform import Rotation as ScipyRot
        R_world = ScipyRot.from_quat(_np(orient).astype(np.float64)[::ds]).as_matrix()

    N = min(acc_l.shape[0], gyro_l.shape[0], R_world.shape[0])
    acc_l, gyro_l, R_world = acc_l[:N], gyro_l[:N], R_world[:N]

    # local CPF → world
    acc_w  = np.einsum("nij,nj->ni", R_world, acc_l)
    gyro_w = np.einsum("nij,nj->ni", R_world, gyro_l)
    if not add_gravity:
        acc_w += np.array([0., -9.81, 0.])

    # world → FLU
    R_wfx = np.einsum("nij,jk->nik", R_world, _R_WRIST_FROM_FLU)
    R_xfw = R_wfx.transpose(0, 2, 1)
    imu = np.concatenate([
        np.einsum("nij,nj->ni", R_xfw, acc_w),
        np.einsum("nij,nj->ni", R_xfw, gyro_w),
    ], axis=1).astype(np.float32)  # [N, 6]

    start = window_idx * WINDOW_LEN
    if start + WINDOW_LEN > len(imu):
        start = 0
    raw_win = imu[start : start + WINDOW_LEN]
    return preprocess_window(raw_win, cfg), start


# ---------------------------------------------------------------------------
# Data loading — IMULM parquet
# ---------------------------------------------------------------------------
def get_val_session_ids(parquet_path: str, cfg: dict, dataset: str) -> list[str]:
    """Replicate the training val split and return session_ids for `dataset`.

    Reads only dataset/subject_id/session_id columns (3 cols, no IMU data).
    Applies the same make_splits logic:
      - exclude probe_dataset (prism)
      - group by subject_id across ALL non-probe datasets
      - shuffle with RandomState(seed)
      - first val_ratio fraction → val
    """
    import pyarrow.dataset as arrow_ds

    splits_cfg  = cfg.get("splits", {}) or {}
    probe_ds    = splits_cfg.get("probe_dataset", None)
    group_key   = splits_cfg.get("group_key", "subject_id")
    val_ratio   = float(splits_cfg.get("val_ratio", 0.05))
    test_ratio  = float(splits_cfg.get("test_ratio", 0.0))
    shuffle     = bool(splits_cfg.get("shuffle", True))
    seed        = int(splits_cfg.get("seed", 0))

    # Stream 3 columns in batches, accumulate only unique triples into a set.
    # Never materialises the full 734 M-row table — result is O(#sessions) small.
    dset    = arrow_ds.dataset(parquet_path, format="parquet")
    scanner = dset.scanner(columns=["dataset", group_key, "session_id"],
                           batch_size=500_000)
    seen: set = set()
    for batch in scanner.to_batches():
        d = batch["dataset"].to_pylist()
        s = batch[group_key].to_pylist()
        e = batch["session_id"].to_pylist()
        seen.update(zip(d, s, e))

    idx = pd.DataFrame(list(seen), columns=["dataset", group_key, "session_id"])

    # Exclude probe dataset
    non_probe = idx[idx["dataset"] != probe_ds].copy()

    # Group by subject_id (across all datasets)
    groups = [g for _, g in non_probe.groupby(group_key)]
    rng = np.random.RandomState(seed)
    if shuffle:
        rng.shuffle(groups)

    # Same slice logic as make_splits
    test_count = int(round(len(groups) * test_ratio))
    val_count  = int(round(len(groups) * val_ratio))
    val_groups = groups[test_count : test_count + val_count]

    if not val_groups:
        raise RuntimeError("Val split is empty — check val_ratio in config.")

    val_df = pd.concat(val_groups, ignore_index=True)
    val_sessions = val_df[val_df["dataset"] == dataset]["session_id"].tolist()

    if not val_sessions:
        available = val_df["dataset"].unique().tolist()
        raise ValueError(
            f"No val sessions found for dataset='{dataset}'. "
            f"Val contains: {available}"
        )

    print(f"  Val split: {len(val_groups)} subject groups → "
          f"{len(val_sessions)} val sessions for '{dataset}'")
    return val_sessions


def load_window_parquet(parquet_path: str, cfg: dict, dataset: str = "pamap2",
                        window_idx: int = 20):
    """Load one pre-processed window [6, 128] from the parquet."""
    cols = ["session_id", "timestamp_ns"] + PARQUET_IMU_COLS
    df = pd.read_parquet(
        parquet_path,
        columns=cols,
        filters=[("dataset", "==", dataset)],
    )
    if df.empty:
        raise ValueError(f"Dataset '{dataset}' not found in parquet.")

    session = df["session_id"].iloc[0]
    print(f"  Using session: {session}")
    sess_df = df[df["session_id"] == session].sort_values("timestamp_ns")
    imu     = sess_df[PARQUET_IMU_COLS].values.astype(np.float32)

    start = window_idx * WINDOW_LEN
    if start + WINDOW_LEN > len(imu):
        start = 0
    raw = imu[start : start + WINDOW_LEN]    # [128, 6]
    return preprocess_window(raw, cfg), start # [6, 128]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _remap_encoder_keys(state_dict: dict) -> dict:
    """Remap keys from nn.TransformerEncoderLayer to TransformerLayerWithDropPath."""
    remapped = {}
    for k, v in state_dict.items():
        # encoder.layers.N.self_attn.* → encoder.layers.N.attn.*
        k = k.replace(".self_attn.", ".attn.")
        # encoder.layers.N.linear1.* → encoder.layers.N.ff.0.*
        k = k.replace(".linear1.", ".ff.0.")
        # encoder.layers.N.linear2.* → encoder.layers.N.ff.3.*
        k = k.replace(".linear2.", ".ff.3.")
        remapped[k] = v
    return remapped


def load_model(ckpt_path: str, device: torch.device):
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg     = ckpt["cfg"]
    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(_remap_encoder_keys(ckpt["model"]))
    encoder.eval().to(device)

    out_channels = len(cfg.get("data", {}).get("sensor_columns", PARQUET_IMU_COLS))
    decoder = ViT1DDecoder(
        embed_dim    = encoder.embed_dim,
        out_channels = out_channels,
        target_T     = WINDOW_LEN,
        cfg          = cfg,
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval().to(device)

    mask_ratio = float(cfg.get("objective", {}).get("mask_ratio", 0.75))
    return encoder, decoder, cfg, mask_ratio


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------
@torch.no_grad()
def reconstruct(window, encoder, decoder, mask_ratio, device):
    x = torch.from_numpy(window).unsqueeze(0).float().to(device)  # [1, 6, 128]
    B, C, T = x.shape
    Pt = T // encoder.patch_size
    patch_size = encoder.patch_size

    patch_mask = generate_patch_mask(B, Pt, C, mask_ratio,
                                     strategy="composite", device=device)
    visible_tokens, vis_pad = encoder(x, mask=patch_mask,
                                      return_keep_padding_mask=True)
    recon = decoder(visible_tokens, patch_mask, Pt, C,
                    visible_padding_mask=vis_pad)  # [1, 6, 128]

    # Expand patch_mask [1, C*Pt] → signal_mask [1, C, T] (same as training loss)
    pm_2d      = patch_mask.view(B, C, Pt)                               # [1, 6, 8]
    signal_mask = pm_2d.unsqueeze(-1).expand(-1, -1, -1, patch_size)     # [1, 6, 8, 16]
    signal_mask = signal_mask.reshape(B, C, T)                            # [1, 6, 128]

    # Overall MSE loss on masked patches only (matches training objective)
    loss_total = torch.nn.functional.mse_loss(
        recon[signal_mask], x[signal_mask]
    ).item()

    # Per-channel MSE on masked patches
    recon_np = recon.squeeze(0).cpu().numpy()   # [6, 128]
    x_np     = x.squeeze(0).cpu().numpy()       # [6, 128]
    sm_np    = signal_mask.squeeze(0).cpu().numpy()  # [6, 128]  bool
    loss_per_ch = []
    for c in range(C):
        if sm_np[c].any():
            loss_per_ch.append(float(np.mean((recon_np[c][sm_np[c]] - x_np[c][sm_np[c]]) ** 2)))
        else:
            loss_per_ch.append(float("nan"))

    return recon_np, patch_mask.squeeze(0).cpu().numpy(), loss_total, loss_per_ch


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot(window, recon, patch_mask, patch_size, Pt,
         ch_names, out_path, mask_ratio, normalized, loss_total, loss_per_ch,
         title_extra=""):
    C, T = window.shape
    t    = np.arange(T) / 50.0

    fig, axes = plt.subplots(C, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"MAE Reconstruction  —  mask ratio {int(mask_ratio*100)}%"
        f"  |  MSE (masked) = {loss_total:.4f}"
        f"  (grey = masked patches){title_extra}",
        fontsize=12, fontweight="bold",
    )

    gt_label = "GT (z-scored)" if normalized else "GT"

    for c, ax in enumerate(axes):
        for p in range(Pt):
            if patch_mask[c * Pt + p]:
                ax.axvspan(p * patch_size / 50.0, (p + 1) * patch_size / 50.0,
                           color="lightgrey", alpha=0.6, zorder=0)

        ax.plot(t, window[c], color="steelblue", lw=1.5, zorder=2)
        ax.plot(t, recon[c],  color="tomato",    lw=1.5, linestyle="--", zorder=3)

        # Per-channel MSE label on the right
        ch_loss = loss_per_ch[c]
        loss_str = f"MSE={ch_loss:.3f}" if not np.isnan(ch_loss) else "MSE=n/a"
        ax.set_ylabel(f"{ch_names[c]}\n{loss_str}", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", lw=0.4, alpha=0.5)

        for p in range(1, Pt):
            ax.axvline(p * patch_size / 50.0, color="grey", lw=0.5, alpha=0.4, zorder=1)

    axes[-1].set_xlabel("Time (s)", fontsize=10)

    handles = [
        plt.Line2D([0], [0], color="steelblue", lw=1.5, label=gt_label),
        plt.Line2D([0], [0], color="tomato",    lw=1.5, linestyle="--", label="Reconstructed"),
        mpatches.Patch(color="lightgrey", alpha=0.6, label="Masked patch"),
    ]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       default="runs/vit1d_mae_ocdetect_flu/checkpoints/best.pt")
    # OCDetect source
    p.add_argument("--csv",        default=(
        "/mnt/nas2/yiquan_data/preprocessed/"
        "OCDetect_01_recording_00_55ef074f-147f-424f-b9c6-58c0cf9dfee0.csv"))
    # Parquet source
    p.add_argument("--parquet",    default=None,
                   help="Path to IMULM master parquet; if set, uses parquet instead of CSV")
    p.add_argument("--dataset",    default="pamap2",
                   help="Which dataset to sample from the parquet")
    # Nymeria pkl source
    p.add_argument("--pkl",        default=None,
                   help="Path to a Nymeria .pkl file; if set, uses pkl instead of CSV/parquet")
    p.add_argument("--out",        default="data_vis/mae_reconstruction.png")
    p.add_argument("--window-idx", type=int, default=20)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    encoder, decoder, cfg, mask_ratio = load_model(args.ckpt, device)
    Pt = WINDOW_LEN // encoder.patch_size  # 8

    sensor_cols = cfg.get("data", {}).get("sensor_columns", PARQUET_IMU_COLS)
    ch_names    = [c.replace("_", " ") for c in sensor_cols]
    normalized  = cfg.get("preprocess", {}).get("normalize", {}).get("enabled", False)

    if args.pkl:
        obj_cfg     = cfg.get("objective", {})
        add_gravity = bool((obj_cfg.get(obj_cfg.get("type", ""), {}) or {}).get("add_gravity", True))
        print(f"Loading pkl: {args.pkl}  window={args.window_idx}  add_gravity={add_gravity}")
        window, start = load_window_pkl(args.pkl, cfg, window_idx=args.window_idx,
                                        add_gravity=add_gravity)
        title_extra = "  [Nymeria real wrist]"
    elif args.parquet:
        print(f"Loading parquet: {args.parquet}  dataset={args.dataset}  window={args.window_idx}")
        window, start = load_window_parquet(
            args.parquet, cfg, dataset=args.dataset, window_idx=args.window_idx)
        title_extra = f"  [{args.dataset}]"
    else:
        print(f"Loading CSV: {args.csv}  window={args.window_idx}")
        window, start = load_window_csv(args.csv, cfg, window_idx=args.window_idx)
        title_extra = "  [OCDetect]"

    print(f"Running MAE  mask_ratio={mask_ratio} …")
    recon, patch_mask, loss_total, loss_per_ch = reconstruct(
        window, encoder, decoder, mask_ratio, device)

    n_masked = int(patch_mask.sum())
    print(f"Masked {n_masked}/48 patches ({n_masked/48*100:.0f}%)")
    print(f"MSE loss (masked patches only): {loss_total:.4f}")

    plot(window, recon, patch_mask, encoder.patch_size, Pt,
         ch_names, args.out, mask_ratio, normalized, loss_total, loss_per_ch,
         title_extra)


if __name__ == "__main__":
    main()
