#!/usr/bin/env python3
"""Visualize odometry trajectory reconstruction for the Scratch model.

Unlike the probe-based columns (Ours-MAE/FOCAL/SimMTM), the Scratch odometry
model (objective ``odom_nymeria_real_wrist_flu``) trains its own
``ViT1DOdomDecoder`` jointly with the encoder end-to-end, and the full
checkpoint (encoder + decoder) is already saved to disk
(``runs/<run>/checkpoints/latest.pt``). So this is pure inference: no probe
head to train, no retraining -- just load both halves of the checkpoint, run
forward passes, and reconstruct/plot trajectories with the exact same
ATE/drift math used for the other four columns (imu_lm/objectives/odom_1d.py's
target convention -- per-sub-patch wrist-FLU velocity -- matches
eval_odom_nymeria_altattn.py's build_windows exactly, so the reconstruction
code is directly reusable).

Usage
-----
    python scripts/vis_odom_scratch.py \\
        --run vit1d_odom_supervised_nymeria_altattn_scratch \\
        --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \\
        --out-dir vis --label Scratch
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_odom_nymeria_altattn as odom
from vis_odom_trajectories import plot_trajectories, plot_summary_hist
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn, ViT1DOdomDecoder
from imu_lm.utils.helpers import cfg_get


def load_scratch_model(run_dir, step, device):
    """Load the jointly-trained encoder + ViT1DOdomDecoder from a checkpoint
    produced by the odom_nymeria_real_wrist_flu objective (run.py)."""
    ckpt = os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt") if step is not None \
        else os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(run_dir, "checkpoints", "latest.pt")
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = obj["cfg"]

    from imu_lm.models.baselines.inertia1_encoder import build_encoder as _build_encoder
    enc = _build_encoder(cfg).to(device)   # ViT1D / alt-attn / Inertia-1 adapter per ckpt cfg
    enc.load_state_dict(obj["model"], strict=True)
    enc.eval()

    T = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * odom.FS))
    ps = int((cfg.get("vit1d", {}) or {}).get("patch_size", 16))
    Pt = T // ps

    decoder = ViT1DOdomDecoder(
        embed_dim=enc.embed_dim, num_patches_per_channel=Pt,
        in_channels=enc.in_channels, cfg=cfg, out_dim=12,
    ).to(device)
    decoder.load_state_dict(obj["odom_decoder"], strict=True)
    decoder.eval()

    print(f"  loaded {os.path.basename(run_dir)} from {ckpt} step={obj.get('step', '?')} "
          f"(T={T} ps={ps} Pt={Pt})")
    return enc, decoder, cfg, T, ps


@torch.no_grad()
def predict_all(enc, decoder, X, device, batch_size=256):
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
        tokens, _ = enc.forward_tokens(xb)
        pred = decoder(tokens)                         # [B, Pt, 12]
        B, Pt, _ = pred.shape
        preds.append(pred.reshape(B, Pt, 4, 3).cpu().numpy())
    return np.concatenate(preds, axis=0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="vit1d_odom_supervised_nymeria_altattn_scratch")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--val-dir", default="/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-val-files", type=int, default=101)
    p.add_argument("--add-gravity", action="store_true", default=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-examples", type=int, default=6)
    p.add_argument("--max-hist-windows", type=int, default=3000,
                    help="Cap on validation windows scored (reconstruct_and_score has an "
                         "unvectorized per-timestep Python loop)")
    p.add_argument("--out-dir", default="vis")
    p.add_argument("--label", default="Scratch")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    slug = args.label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    enc, decoder, cfg, T, ps = load_scratch_model(
        os.path.join(args.runs_root, args.run), args.step, device
    )
    hop = T // 2

    va_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]
    va_sess = [r for f in va_files
               if (r := odom.load_session_local_vel_flu(f, target_hz=int(odom.FS),
                                                          add_gravity=args.add_gravity)) is not None]
    print(f"sessions val={len(va_sess)}")

    keys = ["X", "Y", "GY", "R0", "POS"]
    va = dict(zip(keys, odom.build_windows(va_sess, cfg, T, ps, hop, int(odom.FS))))
    n_total = len(va["X"])
    print(f"windows val={n_total}")

    if n_total > args.max_hist_windows:
        rng = np.random.RandomState(0)
        idx = np.sort(rng.choice(n_total, size=args.max_hist_windows, replace=False))
    else:
        idx = np.arange(n_total)

    preds = predict_all(enc, decoder, va["X"][idx], device, args.batch_size)

    ates, ferrs, plens = [], [], []
    for i in range(len(preds)):
        ate, ferr, plen = odom.reconstruct_and_score(
            preds[i], va["GY"][idx[i]], va["R0"][idx[i]], va["POS"][idx[i]], ps
        )
        ates.append(ate); ferrs.append(ferr); plens.append(plen)
    ates, ferrs, plens = np.asarray(ates), np.asarray(ferrs), np.asarray(plens)
    moving = plens > 0.1
    drifts = np.where(moving, ferrs / np.maximum(plens, 1e-9) * 100.0, np.nan)

    vel_mse = float(np.mean((preds - va["Y"][idx]) ** 2))
    ate_mean = float(np.mean(ates))
    drift_mean = float(np.mean(drifts[np.isfinite(drifts)]))
    print(f"{args.run}: vel_mse={vel_mse:.5f} ATE={ate_mean:.4f}m drift={drift_mean:.2f}%")

    va_sub = {k: v[idx] for k, v in va.items()}
    plot_trajectories(preds, va_sub, ps, args.num_examples,
                       os.path.join(args.out_dir, f"odom_traj_{slug}.png"), args.label)
    plot_summary_hist(ates, drifts, os.path.join(args.out_dir, f"odom_hist_{slug}.png"), args.label)

    np.savez(
        os.path.join(args.out_dir, f"odom_raw_{slug}.npz"),
        preds=preds, ates=ates, ferrs=ferrs, plens=plens,
        vel_mse=vel_mse, ate_mean=ate_mean, drift_mean=drift_mean,
    )
    print(f"saved outputs to {args.out_dir}/*_{slug}.*")


if __name__ == "__main__":
    main()
