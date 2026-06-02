#!/usr/bin/env python3
"""Evaluate MAE reconstruction loss using SYNTHETIC wrist IMU in FLU frame.

Synthesizes right-wrist IMU from xsens_pose (joint 9) via finite differences,
rotates into FLU local frame, then evaluates a pretrained MAE model's
reconstruction loss (masked patches only).

Mirrors eval_mae_nymeria.py but uses synthetic rather than real wrist IMU.
Works with any pkl file that has gt_data['xsens_pose'] — no has_wrist_imu
requirement.

Usage
-----
    python scripts/eval_mae_nymeria_syn.py \
        --ckpt          runs/vit1d_mae_nymeria_syn_wrist_flu/checkpoints/best.pt \
        [--train-root   /mnt/nas2/processed_data_body_Ty_rwrist] \
        [--val-root     /mnt/nas2/processed_data_test_body_Ty_rwrist] \
        [--batch-size   64] \
        [--seed         0] \
        [--device       cuda]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.models.ViT1D.model import ViT1DDecoder, ViT1DEncoder
from imu_lm.objectives.masked_1d import generate_patch_mask

WINDOW_LEN = 128
TARGET_HZ  = 50

# FLU frame definition — must match nymeria_syn_wrist_loader_FLU.py
R_WRIST_FROM_FLU = np.array(
    [[-1., 0., 0.],
     [ 0., 0., 1.],
     [ 0., 1., 0.]], dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Synthetic FLU IMU from pkl
# ---------------------------------------------------------------------------

def load_imu_syn_flu(
    pkl_path: str,
    target_hz: int = TARGET_HZ,
    add_gravity: bool = True,
) -> np.ndarray | None:
    """Synthesize [N, 6] float32 FLU IMU from a Nymeria pkl, or None on failure.

    Requires gt_data['xsens_pose'] [N, 23, 4, 4] in the pkl.
    """
    from scipy.spatial.transform import Rotation as ScipyRot

    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        print(f"  [warn] cannot open {os.path.basename(pkl_path)}: {exc}")
        return None

    xsens_pose = raw.get("gt_data", {}).get("xsens_pose")
    if xsens_pose is None:
        return None

    if hasattr(xsens_pose, "detach"):
        xsens_pose = xsens_pose.detach().cpu().numpy()
    xsens_pose = np.asarray(xsens_pose, dtype=np.float64)   # [N, 23, 4, 4]

    fps_raw = float(
        raw.get("imu_sampling_rate_hz") or raw.get("pose_sampling_rate_hz") or target_hz
    )
    ds = max(1, round(fps_raw / target_hz))
    xsens_pose = xsens_pose[::ds]
    N = xsens_pose.shape[0]
    if N < WINDOW_LEN:
        return None
    fps = float(target_hz)

    j   = 9  # right wrist (Xsens joint index)
    pos = xsens_pose[:, j, :3, 3].copy()    # [N, 3] world positions
    rot = xsens_pose[:, j, :3, :3].copy()   # [N, 3, 3] R_world_from_wrist

    # Linear acceleration via central finite difference (world frame)
    acc_world = np.zeros((N, 3), dtype=np.float64)
    acc_world[1:-1] = (pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc_world[0]    = acc_world[1]
    acc_world[-1]   = acc_world[-2]

    if add_gravity:
        acc_world[:, 1] += 9.81   # Y-up bias to mimic real static IMU

    # Angular velocity in world frame: dR_world = R_{i+1} @ R_i^T
    dR_world = np.einsum("nij,nkj->nik", rot[1:], rot[:-1])   # [N-1, 3, 3]
    gyro_raw = ScipyRot.from_matrix(dR_world).as_rotvec().astype(np.float64) * fps
    gyro_world = np.zeros((N, 3), dtype=np.float64)
    gyro_world[:-1] = gyro_raw
    gyro_world[-1]  = gyro_raw[-1]

    # World → FLU: R_FLU_from_world = (rot @ R_WRIST_FROM_FLU)^T
    R_world_from_FLU = np.einsum("nij,jk->nik", rot, R_WRIST_FROM_FLU)   # [N, 3, 3]
    R_FLU_from_world = R_world_from_FLU.transpose(0, 2, 1)                # [N, 3, 3]

    acc_flu  = np.einsum("nij,nj->ni", R_FLU_from_world, acc_world)
    gyro_flu = np.einsum("nij,nj->ni", R_FLU_from_world, gyro_world)

    return np.concatenate([acc_flu, gyro_flu], axis=1).astype(np.float32)   # [N, 6]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["cfg"]

    encoder = ViT1DEncoder(cfg)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval().to(device)

    out_ch = len(cfg.get("data", {}).get("sensor_columns", [""] * 6))
    decoder = ViT1DDecoder(
        embed_dim    = encoder.embed_dim,
        out_channels = out_ch,
        target_T     = WINDOW_LEN,
        cfg          = cfg,
    )
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval().to(device)

    obj_cfg       = cfg.get("objective", {})
    mask_ratio    = float(obj_cfg.get("mask_ratio", 0.75))
    mask_strategy = str(obj_cfg.get("mask_strategy", "composite"))

    # Resolve add_gravity from the objective-specific sub-config
    obj_type   = obj_cfg.get("type", "")
    sub_cfg    = obj_cfg.get(obj_type, {}) or {}
    add_gravity = bool(sub_cfg.get("add_gravity", True))

    return encoder, decoder, cfg, mask_ratio, mask_strategy, add_gravity


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_windows(
    windows: list,
    encoder,
    decoder,
    mask_ratio: float,
    mask_strategy: str,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict:
    import random as _r
    torch.manual_seed(seed)
    _r.seed(seed)

    loader     = DataLoader(windows, batch_size=batch_size, shuffle=False, num_workers=0)
    patch_size = encoder.patch_size
    losses     = []

    for x in loader:
        x = x.to(device)
        B, C, T = x.shape
        Pt = T // patch_size

        mask = generate_patch_mask(B, Pt, C, mask_ratio,
                                   strategy=mask_strategy, device=device)
        vis_tok, vis_pad = encoder(x, mask=mask, return_keep_padding_mask=True)
        recon = decoder(vis_tok, mask, Pt, C, visible_padding_mask=vis_pad)

        sm = mask.view(B, C, Pt).unsqueeze(-1).expand(-1, -1, -1, patch_size).reshape(B, C, T)
        losses.append(F.mse_loss(recon[sm], x[sm]).item())

    return {
        "loss":      float(np.mean(losses)),
        "loss_std":  float(np.std(losses)),
        "n_windows": len(windows),
    }


def windows_from_pkl_list(pkl_paths, cfg, add_gravity, hop):
    """Load and synthesize FLU IMU from all pkl files; return windowed tensors."""
    wins = []
    for pkl_path in pkl_paths:
        name = os.path.basename(pkl_path)
        imu = load_imu_syn_flu(pkl_path, target_hz=TARGET_HZ, add_gravity=add_gravity)
        if imu is None:
            print(f"  [skip] {name}")
            continue
        start = 0
        n_win = 0
        while start + WINDOW_LEN <= len(imu):
            w = preprocess_window(imu[start : start + WINDOW_LEN], cfg)
            if w is not None:
                wins.append(torch.from_numpy(w).float())
                n_win += 1
            start += hop
    return wins


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate MAE loss on synthetic Nymeria wrist IMU (FLU frame)"
    )
    p.add_argument("--ckpt",        required=True,
                   help="Path to MAE checkpoint (.pt)")
    p.add_argument("--train-root",  default=None,
                   help="Directory of train pkl files (optional)")
    p.add_argument("--val-root",
                   default="/mnt/nas2/processed_data_test_body_Ty_rwrist",
                   help="Directory of val pkl files")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    encoder, decoder, cfg, mask_ratio, mask_strategy, add_gravity = \
        load_model(args.ckpt, device)
    print(f"  mask_ratio={mask_ratio}  strategy={mask_strategy}  "
          f"add_gravity={add_gravity}")

    hop = WINDOW_LEN // 2   # 50% overlap, matches training

    splits = {}
    for name, root in [("train", args.train_root), ("val", args.val_root)]:
        if root is None or not os.path.isdir(root):
            print(f"[{name}] root not found: {root} — skipping")
            continue
        pkl_paths = sorted(
            os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pkl")
        )
        print(f"\n[{name}] {len(pkl_paths)} pkl files from {root}")
        wins = windows_from_pkl_list(pkl_paths, cfg, add_gravity, hop)
        print(f"  {len(wins)} windows total … ", end="", flush=True)
        if not wins:
            print("no valid windows — skipping")
            continue
        res = eval_windows(wins, encoder, decoder,
                           mask_ratio, mask_strategy, args.batch_size,
                           args.seed, device)
        splits[name] = res
        print(f"loss={res['loss']:.4f} ± {res['loss_std']:.4f}")

    print("\n" + "=" * 52)
    print(f"{'Split':<10}  {'MSE loss':>10}  {'±std':>8}  {'n':>8}")
    print("-" * 52)
    for name, r in splits.items():
        print(f"  {name:<8}  {r['loss']:>10.4f}  {r['loss_std']:>8.4f}  "
              f"{r['n_windows']:>8}")
    if len(splits) == 2:
        losses = np.array([r["loss"] for r in splits.values()])
        ns     = np.array([r["n_windows"] for r in splits.values()])
        w_mean = float(np.sum(losses * ns) / np.sum(ns))
        u_mean = float(np.mean(losses))
        print("-" * 52)
        print(f"  {'combined':<8}  {u_mean:>10.4f}  {'(unweighted)':>18}")
        print(f"  {'combined':<8}  {w_mean:>10.4f}  {'(weighted by n)':>18}")
    print("=" * 52)


if __name__ == "__main__":
    main()
