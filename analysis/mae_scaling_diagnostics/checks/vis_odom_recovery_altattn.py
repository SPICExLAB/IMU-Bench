#!/usr/bin/env python
"""Top-view GT-vs-predicted odometry trajectory for an alt-attn ViT1D model.

scripts/vis_nymeria_odom.py hardcodes ViT1DEncoder (the plain encoder) when
building the model from a checkpoint, so it cannot load a checkpoint trained
with alt_attention: true (e.g. anything warm-started from the two-stage MAE
curriculum) -- load_state_dict(strict=True) would fail on the alt-attn-only
parameter names. scripts/eval_odom_nymeria_altattn.py already solves this for
metric-only eval by picking the encoder class from the checkpoint's own saved
cfg; this script applies the same fix but keeps the trajectory/PNG output.

Reuses vis_nymeria_odom.py's data loading, inference, and plotting functions
as-is (they take the encoder as a plain object and only call
encoder.forward_tokens(x), which both encoder classes implement identically)
-- only the encoder construction at load time differs.

Usage:
    python analysis/mae_scaling_diagnostics/checks/vis_odom_recovery_altattn.py \
        --pkl /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist/<session>.pkl \
        --model-config configs/vit1d_odom_finetune_nymeria_mae_twostage_flu.yaml \
        --run vit1d_odom_finetune_nymeria_mae_twostage --ckpt best
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch

from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn, ViT1DOdomDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

from scripts.vis_nymeria_odom import (
    load_nymeria_odom_vis, run_inference, run_inference_local_vel, save_topview,
)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-fig", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    device = torch.device(a.device)
    base_cfg = load_yaml(a.config)
    model_cfg = load_yaml(a.model_config)
    cfg = deep_update(base_cfg, model_cfg)

    runs_root = cfg.get("paths", {}).get("runs_root", "runs")
    ckpt_path = (os.path.join(runs_root, a.run, "checkpoints", f"{a.ckpt}.pt")
                 if a.ckpt in ("best", "latest") else a.ckpt)
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_cfg = state["cfg"]

    alt_attn = bool(cfg_get(ckpt_cfg, ["vit1d", "encoder", "alt_attention"], False))
    encoder = (ViT1DEncoderAltAttn(ckpt_cfg) if alt_attn else ViT1DEncoder(ckpt_cfg)).to(device)
    encoder.load_state_dict(state["model"], strict=True)
    encoder.eval()
    logger.info("Encoder: %s", type(encoder).__name__)

    T_win, _ = compute_T_and_hop(ckpt_cfg)
    patch_size = int((cfg_get(ckpt_cfg, ["vit1d"], {}) or {}).get("patch_size", 16))

    out_dim = int(state["odom_decoder"]["head.weight"].shape[0])
    odom_decoder = ViT1DOdomDecoder(
        embed_dim=encoder.embed_dim, num_patches_per_channel=T_win // patch_size,
        in_channels=encoder.in_channels, cfg=ckpt_cfg, out_dim=out_dim,
    ).to(device)
    odom_decoder.load_state_dict(state["odom_decoder"])
    odom_decoder.eval()
    use_local_vel = (out_dim == 12)

    obj_type = cfg_get(ckpt_cfg, ["objective", "type"], "odom_nymeria")
    odom_cfg = (cfg_get(ckpt_cfg, ["objective", obj_type], {}) or {})
    target_hz = float(odom_cfg.get("target_hz", a.fps))
    add_gravity = bool(odom_cfg.get("add_gravity", True))

    logger.info("Loading: %s  (target_hz=%.0f  add_gravity=%s  out_dim=%d)",
                os.path.basename(a.pkl), target_hz, add_gravity, out_dim)
    X_flu, pos_root_gt, R_world_from_X = load_nymeria_odom_vis(
        pkl_path=a.pkl, target_hz=target_hz, add_gravity=add_gravity, max_frames=a.max_frames,
    )

    logger.info("Running inference …")
    if use_local_vel:
        pos_pred = run_inference_local_vel(
            X_flu, R_world_from_X, pos_root_gt, encoder, odom_decoder, ckpt_cfg, device, target_hz,
        )
    else:
        pred_disp = run_inference(X_flu, encoder, odom_decoder, ckpt_cfg, device)
        pos_pred = np.cumsum(pred_disp, axis=0)
        pos_pred = pos_pred - pos_pred[0] + pos_root_gt[0]

    err = np.linalg.norm(pos_pred - pos_root_gt, axis=-1)
    path_len = np.linalg.norm(np.diff(pos_root_gt, axis=0), axis=-1).sum()
    print(f"[odom] {a.run}/{a.ckpt}.pt step={state.get('step'):,}  "
          f"N={len(pos_root_gt)} frames ({len(pos_root_gt)/target_hz:.1f}s)  "
          f"GT path length={path_len:.2f} m")
    print(f"[odom] final drift={err[-1]:.3f} m  mean pos error={err.mean():.3f} m  "
          f"max pos error={err.max():.3f} m")

    fig_path = a.save_fig or (os.path.splitext(os.path.basename(a.pkl))[0] + "_topview.png")
    save_topview(pos_root_gt, pos_pred, fig_path, seq_name=os.path.basename(a.pkl))
    print(f"[odom] wrote {fig_path}")


if __name__ == "__main__":
    main()
