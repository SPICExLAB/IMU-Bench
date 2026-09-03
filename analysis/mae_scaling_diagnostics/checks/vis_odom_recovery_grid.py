#!/usr/bin/env python
"""Grid of GT-vs-predicted odometry trajectories across several Nymeria val
sessions, for the same alt-attn-aware model loading as
vis_odom_recovery_altattn.py (which see for why that's needed).

The single-sequence version picks one file more or less arbitrarily, and a
near-stationary ground-truth session (small true displacement) can make
ordinary dead-reckoning drift look dramatic purely because the denominator
is small. Spreading sessions evenly across the sorted val-file list (rather
than the first N alphabetically) gives a fairer cross-section of session
lengths and motion types instead of cherry-picking or accidentally showing
only one regime.

Usage:
    python analysis/mae_scaling_diagnostics/checks/vis_odom_recovery_grid.py \
        --model-config configs/vit1d_odom_finetune_nymeria_mae_multidata_flu.yaml \
        --run vit1d_odom_finetune_nymeria_mae_multidata --ckpt best \
        --val-root /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \
        --n-sessions 9 --out data_vis/odom_recovery/grid_original.png
"""
from __future__ import annotations

import argparse
import glob
import logging
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn, ViT1DOdomDecoder
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

from scripts.vis_nymeria_odom import load_nymeria_odom_vis, run_inference, run_inference_local_vel

logger = logging.getLogger(__name__)
C_GT, C_PRED = "#00AA44", "#DD2222"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", default="/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--n-sessions", type=int, default=9,
                     help="how many sessions to SCORE (evenly spread by name); "
                          "ignored if --scan-all is set")
    ap.add_argument("--scan-all", action="store_true",
                     help="score every val session instead of an evenly-spaced sample")
    ap.add_argument("--top-k", type=int, default=0,
                     help="after scoring, plot only the K lowest-drift%% sessions "
                          "(0 = plot everything scored, in scoring order)")
    ap.add_argument("--min-path-len", type=float, default=20.0,
                     help="drop near-stationary sessions before ranking -- with little "
                          "true displacement, drift%% is a noisy/trivial denominator "
                          "even when the model is doing fine")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="data_vis/odom_recovery/grid.png")
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
    target_hz = float(odom_cfg.get("target_hz", 50.0))
    add_gravity = bool(odom_cfg.get("add_gravity", True))

    all_files = sorted(glob.glob(os.path.join(a.val_root, "*.pkl")))
    if a.scan_all:
        files = all_files
    else:
        idx = np.linspace(0, len(all_files) - 1, min(a.n_sessions, len(all_files)), dtype=int)
        files = [all_files[i] for i in idx]
    logger.info("Scoring %d/%d val sessions%s", len(files), len(all_files),
                "" if a.scan_all else " (evenly spread by sorted name)")

    results = []
    for pkl in files:
        name = os.path.basename(pkl).split("_cpfbody")[0]
        try:
            X_flu, pos_gt, R_world_from_X = load_nymeria_odom_vis(
                pkl_path=pkl, target_hz=target_hz, add_gravity=add_gravity, max_frames=a.max_frames,
            )
        except Exception as e:
            logger.warning("skip %s: %s", name, e)
            continue
        with torch.no_grad():
            if use_local_vel:
                pos_pred = run_inference_local_vel(
                    X_flu, R_world_from_X, pos_gt, encoder, odom_decoder, ckpt_cfg, device, target_hz,
                )
            else:
                pred_disp = run_inference(X_flu, encoder, odom_decoder, ckpt_cfg, device)
                pos_pred = np.cumsum(pred_disp, axis=0)
                pos_pred = pos_pred - pos_pred[0] + pos_gt[0]
        err = np.linalg.norm(pos_pred - pos_gt, axis=-1)
        path_len = np.linalg.norm(np.diff(pos_gt, axis=0), axis=-1).sum()
        drift_pct = 100.0 * err[-1] / path_len if path_len > 0.1 else float("nan")
        results.append(dict(name=name, pos_gt=pos_gt, pos_pred=pos_pred, n=len(pos_gt),
                             path_len=path_len, final_err=err[-1], mean_err=err.mean(), drift_pct=drift_pct))
        print(f"[odom] {name:<45s} N={len(pos_gt):>6d} ({len(pos_gt)/target_hz:6.1f}s)  "
              f"path={path_len:6.2f}m  final_err={err[-1]:6.2f}m  drift={drift_pct:6.1f}%", flush=True)

    if a.top_k > 0:
        eligible = [r for r in results if r["path_len"] >= a.min_path_len]
        dropped = len(results) - len(eligible)
        if dropped:
            print(f"[odom] dropped {dropped} session(s) with path_len < {a.min_path_len}m "
                  f"(near-stationary -- drift%% denominator too small to be meaningful)")
        ranked = sorted(eligible, key=lambda r: r["drift_pct"])
        results = ranked[:a.top_k]
        print(f"[odom] selected top-{len(results)} lowest-drift%% sessions "
              f"(path_len >= {a.min_path_len}m): "
              + ", ".join(f"{r['name']}({r['drift_pct']:.1f}%)" for r in results))

    n = len(results)
    ncols = int(np.ceil(np.sqrt(n))); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.4 * nrows), facecolor="#fcfcfb")
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        ax.plot(r["pos_gt"][:, 0], -r["pos_gt"][:, 2], color=C_GT, lw=1.3, label="Ground truth")
        ax.plot(r["pos_pred"][:, 0], -r["pos_pred"][:, 2], color=C_PRED, lw=1.3, label="Predicted")
        ax.plot(r["pos_gt"][0, 0], -r["pos_gt"][0, 2], "o", color=C_GT, ms=5)
        ax.plot(r["pos_pred"][0, 0], -r["pos_pred"][0, 2], "o", color=C_PRED, ms=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"{r['name']}\npath={r['path_len']:.1f}m  drift={r['drift_pct']:.0f}%", fontsize=8.5)
        ax.tick_params(labelsize=7)
        ax.grid(True, linewidth=0.3, alpha=0.5)
    for ax in axes[len(results):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{a.run}/{a.ckpt}.pt  --  odometry recovery across {n} val sessions "
                 f"(full-session dead-reckoning, no re-anchoring)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    dp = np.array([r["drift_pct"] for r in results if np.isfinite(r["drift_pct"])])
    print(f"\n[odom] drift%% across {len(dp)} sessions: mean={dp.mean():.1f}%  "
          f"median={np.median(dp):.1f}%  min={dp.min():.1f}%  max={dp.max():.1f}%")
    print(f"[odom] wrote {a.out}")


if __name__ == "__main__":
    main()
