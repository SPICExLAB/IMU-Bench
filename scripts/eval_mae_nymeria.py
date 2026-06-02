#!/usr/bin/env python3
"""Evaluate MAE reconstruction loss on Nymeria real-wrist pkl files.

Loads only the wrist IMU (FLU frame) — no SMPL IK needed for MAE eval.
Reports loss for train split, val split, and combined.

Usage
-----
    python scripts/eval_mae_nymeria.py \
        --ckpt          runs/vit1d_mae_nymeria_real_wrist_flu/checkpoints/best.pt \
        [--train-root   /mnt/nas2/processed_data_body_Ty_rwrist] \
        [--val-root     /mnt/nas2/processed_data_test_body_Ty_rwrist] \
        [--batch-size   64]
        [--seed         0]
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

R_WRIST_FROM_FLU = np.array(
    [[-1., 0., 0.],
     [ 0., 0., 1.],
     [ 0., 1., 0.]], dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Lightweight pkl loader — IMU only, no SMPL
# ---------------------------------------------------------------------------

def load_imu_flu(pkl_path: str, target_hz: int = 50,
                 add_gravity: bool = True) -> np.ndarray | None:
    """Return [N, 6] float32 FLU IMU array from a Nymeria pkl, or None on failure."""
    try:
        with open(pkl_path, "rb") as fh:
            raw = pickle.load(fh)
    except Exception as exc:
        print(f"  [warn] cannot open {os.path.basename(pkl_path)}: {exc}")
        return None

    if not raw.get("has_wrist_imu", False):
        return None

    wrist_imu = raw.get("wrist_imu_data", {})
    wrist_gt  = raw.get("wrist_gt_data",  {})
    if "accel" not in wrist_imu or "gyro" not in wrist_imu:
        return None
    if "orientation" not in wrist_gt:
        return None

    fps_body  = float(raw.get("imu_sampling_rate_hz") or
                      raw.get("pose_sampling_rate_hz") or target_hz)
    fps_wrist = float(raw.get("wrist_imu_sampling_rate_hz", fps_body))
    ds = max(1, round(fps_wrist / target_hz))

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
    if N < WINDOW_LEN:
        return None

    acc_l, gyro_l, R_world = acc_l[:N], gyro_l[:N], R_world[:N]

    # local CPF → world
    acc_w  = np.einsum("nij,nj->ni", R_world, acc_l)
    gyro_w = np.einsum("nij,nj->ni", R_world, gyro_l)
    if not add_gravity:
        acc_w += np.array([0., -9.81, 0.])

    # world → FLU
    R_wfx = np.einsum("nij,jk->nik", R_world, R_WRIST_FROM_FLU)
    R_xfw = R_wfx.transpose(0, 2, 1)
    acc_f  = np.einsum("nij,nj->ni", R_xfw, acc_w)
    gyro_f = np.einsum("nij,nj->ni", R_xfw, gyro_w)

    return np.concatenate([acc_f, gyro_f], axis=1).astype(np.float32)


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
    add_gravity   = bool((obj_cfg.get(obj_cfg.get("type", ""), {}) or {}).get("add_gravity", True))

    return encoder, decoder, cfg, mask_ratio, mask_strategy, add_gravity


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_windows(windows: list, encoder, decoder,
                 mask_ratio, mask_strategy, batch_size, seed, device) -> dict:
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
    """Load all pkl files and return a flat list of preprocessed window tensors."""
    wins = []
    for pkl_path in pkl_paths:
        name = os.path.basename(pkl_path)
        imu = load_imu_flu(pkl_path, target_hz=50, add_gravity=add_gravity)
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
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",        required=True)
    p.add_argument("--train-root",  default=None)
    p.add_argument("--val-root",    default="/mnt/nas2/processed_data_test_body_Ty_rwrist")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    encoder, decoder, cfg, mask_ratio, mask_strategy, add_gravity = load_model(args.ckpt, device)
    print(f"  mask_ratio={mask_ratio}  strategy={mask_strategy}  add_gravity={add_gravity}")

    hop = WINDOW_LEN // 2  # 50% overlap, matches training

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
                           mask_ratio, mask_strategy, args.batch_size, args.seed, device)
        splits[name] = res
        print(f"loss={res['loss']:.4f} ± {res['loss_std']:.4f}")

    print("\n" + "=" * 52)
    print(f"{'Split':<10}  {'MSE loss':>10}  {'±std':>8}  {'n':>8}")
    print("-" * 52)
    for name, r in splits.items():
        print(f"  {name:<8}  {r['loss']:>10.4f}  {r['loss_std']:>8.4f}  {r['n_windows']:>8}")
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
