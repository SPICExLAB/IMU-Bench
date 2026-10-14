#!/usr/bin/env python
"""Website figure: ground-truth vs predicted full-body skeletons (front and side) from the fine-tuned InrtlWrist.

    python scripts/vis_site_pose.py --run bench_full_body_pose_nymeria__mae_5src__finetune__s42 \
        --model-config bench/generated/configs/bench_full_body_pose_nymeria__mae_5src__finetune__s42.yaml \
        --ckpt best_mpjpe --frames 6 --max-val-sessions 4 --out website/static/images/pose_frames.png

Adapted from analysis/mae_scaling_diagnostics/checks/vis_pose_recovery.py.
Inspect the recovered full-body pose of a trained Nymeria 23-joint model.

Answers two questions the headline MPJPE cannot:
  1. WHERE does the error live? Per-joint MPJPE, ranked. With only a single
     right-wrist IMU as input, error should concentrate in joints kinematically
     far from that sensor -- if it does not, something is wrong.
  2. What does the recovered skeleton actually look like? Renders GT vs
     predicted skeletons (root-relative, two orthogonal projections) at several
     frames sampled across the val set.

Uses the production val loader and the same FK path as the training objective
(articulate ParametricModel.forward_kinematics), with the GT root spliced into
the prediction exactly as _assemble_full_R does -- so the numbers here are
directly comparable to the reported mpjpe rather than a re-derivation.

Usage:
    python analysis/mae_scaling_diagnostics/checks/vis_pose_recovery.py \
        --run vit1d_full_body_pose_finetune_nymeria_mae_multidata_lr1e-3 \
        --model-config configs/vit1d_full_body_pose_finetune_nymeria_mae_multidata_flu.yaml \
        --ckpt best_mpjpe --batches 8
"""
from __future__ import annotations

import argparse, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT); sys.path.insert(0, os.path.join(REPO_ROOT, "analysis", "mae_scaling_diagnostics"))
from lib.common import load_merged_cfg  # noqa: E402

C_GT, C_PR, INK, MUTED, SURF = "#1d6f8f", "#c2662a", "#16202e", "#6b7788", "#ffffff"
SMPL_PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
NAMES = ["pelvis","L hip","R hip","spine1","L knee","R knee","spine2","L ankle","R ankle",
         "spine3","L foot","R foot","neck","L collar","R collar","head","L shoulder",
         "R shoulder","L elbow","R elbow","L wrist","R wrist","L hand","R hand"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--ckpt", default="best_mpjpe")
    ap.add_argument("--batches", type=int, default=0,
                     help="cap on val batches for a quick smoke run; 0 = full val set "
                          "(needed for the aggregate MPJPE to match the reported number -- "
                          "the val loader is unshuffled, so a small prefix is just the "
                          "first few of many sessions, not a representative sample)")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--out-dir", default="data_vis/pose_recovery")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-val-sessions", type=int, default=4)
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--session-indices", default="", help="comma-separated indices into the sorted val files (overrides --max-val-sessions)")
    ap.add_argument("--snaps-out", default="", help="dump the sampled (gt, pred, err) frames to this .npz instead of plotting")
    ap.add_argument("--compose", nargs="*", default=None, help="compose a figure from these .npz snapshot files and exit")
    a = ap.parse_args()

    import core.articulate as art
    from core.paths import Paths
    from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn, ViT1DPoseDecoder
    from imu_lm.objectives.pose_1d import _assemble_full_R
    from imu_lm.data.windowing import compute_T_and_hop

    if a.compose:
        snaps = []
        for f in a.compose:
            z = np.load(f); snaps += [(z["g"][i], z["p"][i], float(z["e"][i])) for i in range(len(z["e"]))]
        per = np.full(24, np.nan); draw(a, snaps, per); return
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_merged_cfg(a.model_config)
    pose_cfg = cfg["objective"].get("pose") or cfg["objective"].get("pose_probe") or {}
    J = int(pose_cfg.get("num_joints", 23))
    ps = int(cfg["vit1d"]["patch_size"])
    K = int((cfg["vit1d"].get("pose_decoder") or {}).get("sub_patches", 1))
    T_win, _ = compute_T_and_hop(cfg)

    st = torch.load(os.path.join(REPO_ROOT, "runs", a.run, "checkpoints", f"{a.ckpt}.pt"),
                    map_location=dev, weights_only=False)
    enc = ViT1DEncoderAltAttn(cfg).to(dev); enc.load_state_dict(st["model"]); enc.eval()
    dec = ViT1DPoseDecoder(embed_dim=enc.embed_dim, num_patches_per_channel=T_win // ps,
                           in_channels=enc.in_channels, num_joints=J, cfg=cfg,
                           sub_patches=K).to(dev)
    dec.load_state_dict(st["pose_decoder"]); dec.eval()
    print(f"[pose] {a.run}/{a.ckpt}.pt step={st.get('step'):,} "
          f"reported best_val_mpjpe={st.get('best_val_mpjpe'):.2f} mm", flush=True)

    bm = art.model.ParametricModel(Paths.SMPL_FILE, device=dev)

    # production val loader for this pose task -- returns (x, y_full[24-joint
    # 6D rot], gt_pos[24-joint root-relative positions]), see
    # NymeriaRealWristFullBodyDatasetFLU.__getitem__ / collate_skip_none.
    from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import (
        make_nymeria_real_wrist_full_body_loaders_flu as mk)
    import glob
    vf = sorted(glob.glob(os.path.join(pose_cfg["val_pkl_root"], "*.pkl")))
    if a.session_indices:
        vf = [vf[int(i)] for i in a.session_indices.split(",")]
    elif a.max_val_sessions > 0:
        vf = vf[:: max(1, len(vf) // a.max_val_sessions)][: a.max_val_sessions]
    print(f"[pose] val sessions: {[os.path.basename(f)[:28] for f in vf]}", flush=True)
    loaders = mk(cfg=cfg, pkl_root="/", train_files=vf[:1], val_files=vf,
                 add_gravity=bool(pose_cfg.get("add_gravity", True)))
    val = loaders["val_loader"]
    n_batches = len(val)
    cap = n_batches if a.batches <= 0 else min(a.batches, n_batches)
    # val loader is unshuffled (session order), so evenly-spaced batch indices
    # across the FULL range give session-diverse snapshots instead of all 5
    # coming from whichever session happens to load first.
    snap_batches = set(np.linspace(0, cap - 1, min(a.frames, cap), dtype=int).tolist())
    print(f"[pose] scoring {cap:,}/{n_batches:,} val batches "
          f"({'full val set' if cap == n_batches else 'SUBSET -- pass --batches 0 for the full set'})",
          flush=True)

    se = np.zeros(24); cnt = 0; snaps = []
    with torch.no_grad():
        for bi, batch in enumerate(val):
            if bi >= cap: break
            x, y_full, gt_pos = batch[0].to(dev), batch[1].to(dev), batch[2].to(dev)
            tok, _ = enc.forward_tokens(x)
            pred = dec(tok)                                      # [B,Pt,K*J*6]
            B, Pt, _ = pred.shape
            Jl = y_full.shape[-1] // (K * 6)                      # 24
            full_R = _assemble_full_R(pred, y_full, B, Pt, K, J, Jl)  # GT root spliced in
            _, pp = bm.forward_kinematics(full_R.reshape(B * Pt * K, -1))
            pp = (pp - pp[:, 0:1]).cpu().numpy()                  # [N,24,3] root-relative
            gp = gt_pos.reshape(B, Pt, K, Jl, 3).reshape(-1, Jl, 3).cpu().numpy()  # already root-relative
            d = np.linalg.norm(pp - gp, axis=-1) * 1000.0         # [N,24] mm
            se += d.sum(0); cnt += len(d)
            if bi in snap_batches:
                mid = len(pp) // 2
                snaps.append((gp[mid], pp[mid], float(d[mid, 1:].mean())))
    per = se / max(1, cnt)
    print(f"\n[pose] per-joint MPJPE over {cnt:,} frames (root excluded from the mean):")
    order = np.argsort(-per[1:]) + 1
    for j in order:
        print(f"    {NAMES[j]:<12s} {per[j]:7.1f} mm")
    print(f"\n  MEAN over joints 1-23 = {per[1:].mean():.2f} mm   (root joint 0 = {per[0]:.3f} mm, zero by construction)")

    if a.snaps_out:
        np.savez(a.snaps_out, g=np.stack([x[0] for x in snaps]), p=np.stack([x[1] for x in snaps]), e=np.array([x[2] for x in snaps]))
        print(f"[pose] wrote {len(snaps)} snapshots to {a.snaps_out}"); return
    draw(a, snaps, per)


def draw(a, snaps, per):
    os.makedirs(a.out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, len(snaps), figsize=(3.0*len(snaps), 6.2), facecolor=SURF, squeeze=False)
    for c, (g, p, e) in enumerate(snaps):
        for r, (h, v, lbl) in enumerate([(0, 1, "front (x-y)"), (2, 1, "side (z-y)")]):
            ax = axes[r][c]; ax.set_facecolor(SURF)
            for pts, col, nm in [(g, C_GT, "ground truth"), (p, C_PR, "InrtlWrist, fine-tuned")]:
                for j, par in enumerate(SMPL_PARENTS):
                    if par < 0: continue
                    ax.plot([pts[j, h], pts[par, h]], [pts[j, v], pts[par, v]],
                            color=col, lw=2.0, solid_capstyle="round",
                            label=nm if (j == 1 and r == 0 and c == 0) else None, zorder=2)
                ax.scatter(pts[:, h], pts[:, v], s=9, color=col, zorder=3)
            ax.set_aspect("equal"); ax.axis("off")
            if r == 0: ax.set_title(f"frame {c+1}\n{e:.0f} mm", color=MUTED, fontsize=9)
            if c == 0: ax.text(-0.05, 0.5, lbl, transform=ax.transAxes, rotation=90,
                               va="center", ha="right", color=MUTED, fontsize=9)
    if not a.no_title:
        fig.suptitle(f"Full-body pose from one wrist IMU (mean {np.nanmean(per[1:]):.0f} mm over 23 joints on these sessions)", color=INK, fontsize=11, y=0.99)
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=10, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0.02, 0.04, 1, 0.99 if a.no_title else 0.95])
    out = a.out or os.path.join(a.out_dir, f"{a.run}_{a.ckpt}.png")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURF); plt.close(fig)
    print(f"\n[pose] wrote {out}")


if __name__ == "__main__":
    main()
