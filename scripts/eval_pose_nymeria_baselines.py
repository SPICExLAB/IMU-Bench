#!/usr/bin/env python3
"""
Nymeria real-wrist 23-joint full-body pose probing/finetuning for the ORIGINAL
SSL baseline models (SimMTM TFC-CNN / LIMU-BERT param-shared Tx / FOCAL).

These models have different architectures than ViT1D (no per-section token grid),
so a per-model adapter extracts a temporal feature and adaptive-pools it to the
8 sections of a 2.56 s window. A shared temporal pose head then predicts
per-section 6D joint rotations. Same losses/metrics as the ViT1D pose probe:
    loss   = rot_MSE + joint_pos_weight·FK_pos + jitter_weight·jitter
    metric = geodesic error (deg) + MPJPE (mm)

Modes: --mode probe (freeze encoder) / finetune (encoder + head).

Usage:
    python scripts/eval_pose_nymeria_baselines.py \
        --models simmtm limu_bert focal \
        --runs   ssl_simmtm_realdata_ocdetect ssl_limubert_realdata_ocdetect ssl_focal_realdata_ocdetect \
        --mode probe --out-dir data_vis/nymeria_pose_baselines_probe/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

try:
    import wandb
except ImportError:
    wandb = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.baselines.ssl_baselines import build_ssl_model
from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import (
    make_nymeria_real_wrist_full_body_loaders_flu,
)
from imu_lm.objectives.pose_1d import _compute_fk_pos_loss, _compute_val_mpjpe
# Reuse pose helpers + the temporal head from the alt-attn script
from scripts.eval_pose_nymeria_altattn import (
    rot6d_to_matrix, geodesic_error_deg, slice_joints_1_23, load_body_model,
    TemporalPoseHead, NUM_JOINTS, ROT6D_DIM, SMPL_JOINT_NAMES, FS,
)
from imu_lm.utils.helpers import cfg_get


# ── Per-model section feature extractor ───────────────────────────────────────

def extract_sections(wrapper, name, x, Pt):
    """x [B, 6, 128] (channel-first) → per-section feature [B, Pt, D_model]."""
    B = x.shape[0]
    if name == "limu_bert":
        feat = wrapper.model.transformer(x.transpose(1, 2))      # [B, T, H]
    elif name == "simmtm":
        feat = wrapper.model._conv(x).transpose(1, 2)            # [B, L', C']
    elif name == "focal":
        feats = wrapper.backbone(x.transpose(1, 2), proj_head=False)   # {mod:[B*S,H]}
        S = wrapper.backbone.seq_len
        feat = torch.cat([feats[m].reshape(B, S, -1) for m in wrapper.backbone.modalities], dim=-1)
    else:
        raise ValueError(name)
    # adaptive-pool the time/section axis to exactly Pt sections
    return F.adaptive_avg_pool1d(feat.transpose(1, 2), Pt).transpose(1, 2)   # [B, Pt, D]


def feat_dim(wrapper, name, device):
    with torch.no_grad():
        return extract_sections(wrapper, name, torch.zeros(1, 6, 128, device=device), 8).shape[-1]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_baseline(name, run_dir, step, device):
    ckpt_path = (os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt")
                 if step is not None else os.path.join(run_dir, "checkpoints", "best.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    wrapper = build_ssl_model(name, cfg["model"]).to(device)
    wrapper.load_state_dict(ckpt["model"], strict=True)
    print(f"  {name}  step={ckpt.get('step','?')}  params={sum(p.numel() for p in wrapper.parameters())/1e6:.2f}M")
    return wrapper, cfg


# ── Loss ──────────────────────────────────────────────────────────────────────

def pose_loss(pred, y_full, gt_pos, body_model, jpw, jw):
    loss = F.mse_loss(pred, slice_joints_1_23(y_full))
    if body_model is not None and gt_pos is not None and jpw > 0:
        loss = loss + jpw * _compute_fk_pos_loss(pred, y_full, gt_pos, 1, NUM_JOINTS, body_model)
    if jw > 0 and pred.shape[1] > 1:
        loss = loss + jw * ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()
    return loss


# ── Train / eval ──────────────────────────────────────────────────────────────

def run_one(wrapper, name, train_loader, val_loader, device, mode, Pt, lr, wd,
            epochs, patience, body_model, jpw, jw, dropout, wb_run=None):
    D = feat_dim(wrapper, name, device)
    head = TemporalPoseHead(D, Pt, dropout=dropout).to(device)

    if mode == "probe":
        for p in wrapper.parameters():
            p.requires_grad_(False)
        wrapper.eval()
        params = list(head.parameters())
    else:
        for p in wrapper.parameters():
            p.requires_grad_(True)
        params = list(wrapper.parameters()) + list(head.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler()

    best_err, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        head.train()
        if mode == "finetune":
            wrapper.train()
        losses, train_mpjpe = [], None
        for batch in train_loader:
            if batch is None:
                continue
            x = batch[0].to(device); y_full = batch[1].to(device)
            gt_pos = batch[2].to(device) if len(batch) > 2 else None
            opt.zero_grad()
            with autocast():
                if mode == "probe":
                    with torch.no_grad():
                        feat = extract_sections(wrapper, name, x.float(), Pt)
                else:
                    feat = extract_sections(wrapper, name, x.float(), Pt)
                pred = head(feat)
            loss = pose_loss(pred.float(), y_full, gt_pos, body_model, jpw, jw)
            losses.append(float(loss.detach()))
            if train_mpjpe is None and body_model is not None and gt_pos is not None:
                train_mpjpe = _compute_val_mpjpe(pred.detach().float(), y_full, gt_pos, 1, NUM_JOINTS, body_model)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
        sched.step()

        vm = evaluate(wrapper, name, head, val_loader, device, Pt, body_model)
        val_err = vm["geo_err_deg"]
        if val_err < best_err - 1e-4:
            best_err = val_err
            best_state = ({k: v.cpu().clone() for k, v in head.state_dict().items()},
                          {k: v.cpu().clone() for k, v in wrapper.state_dict().items()} if mode == "finetune" else None)
            no_improve = 0
        else:
            no_improve += 1
        if wb_run is not None:
            payload = {"epoch": epoch + 1, "train/loss": float(np.mean(losses)),
                       "val/geo_err_deg": val_err, "lr": sched.get_last_lr()[0]}
            if train_mpjpe is not None:
                payload["train/mpjpe_mm"] = train_mpjpe
            if vm.get("mpjpe_mm") is not None:
                payload["val/mpjpe_mm"] = vm["mpjpe_mm"]
            wb_run.log(payload, step=epoch + 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            mt = (f"  val_mpjpe={vm['mpjpe_mm']:.1f}mm" if vm.get("mpjpe_mm") is not None else "")
            print(f"    epoch {epoch+1:3d}  val_geo={val_err:.3f}°{mt}  best={best_err:.3f}°")
        if patience > 0 and no_improve >= patience:
            print(f"    early stop @ epoch {epoch+1}")
            break

    if best_state is not None:
        head.load_state_dict(best_state[0])
        if mode == "finetune" and best_state[1] is not None:
            wrapper.load_state_dict(best_state[1])
    final = evaluate(wrapper, name, head, val_loader, device, Pt, body_model)
    per_j = per_joint_error(wrapper, name, head, val_loader, device, Pt)
    return final, per_j


@torch.no_grad()
def evaluate(wrapper, name, head, loader, device, Pt, body_model=None):
    head.eval(); wrapper.eval()
    errs, mpjpes = [], []
    for batch in loader:
        if batch is None:
            continue
        x = batch[0].to(device); y_full = batch[1].to(device)
        gt_pos = batch[2].to(device) if len(batch) > 2 else None
        pred = head(extract_sections(wrapper, name, x.float(), Pt))
        y = slice_joints_1_23(y_full)
        errs.append(geodesic_error_deg(pred.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM),
                                       y.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)).item())
        if body_model is not None and gt_pos is not None:
            mpjpes.append(_compute_val_mpjpe(pred.float(), y_full, gt_pos, 1, NUM_JOINTS, body_model))
    return {"geo_err_deg": float(np.mean(errs)),
            "mpjpe_mm": float(np.mean(mpjpes)) if mpjpes else None}


@torch.no_grad()
def per_joint_error(wrapper, name, head, loader, device, Pt):
    head.eval(); wrapper.eval()
    acc = [[] for _ in range(NUM_JOINTS)]
    for batch in loader:
        x = batch[0].to(device)
        y = slice_joints_1_23(batch[1].to(device))
        pred = head(extract_sections(wrapper, name, x.float(), Pt)).reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
        gt = y.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
        R = rot6d_to_matrix(pred) @ rot6d_to_matrix(gt).transpose(-1, -2)
        tr = R.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1 + 1e-6, 3 - 1e-6)
        ang = torch.acos((tr - 1.0) / 2.0) * (180.0 / torch.pi)
        for j in range(NUM_JOINTS):
            acc[j].append(ang[:, :, j].mean().item())
    return np.array([float(np.mean(a)) for a in acc])


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["simmtm", "limu_bert", "focal"])
    p.add_argument("--runs", nargs="+",
                   default=["ssl_simmtm_realdata_ocdetect",
                            "ssl_limubert_realdata_ocdetect",
                            "ssl_focal_realdata_ocdetect"])
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--train-dir", default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir",   default="/mnt/nas2/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=200)
    p.add_argument("--max-val-files",   type=int, default=101)
    p.add_argument("--window-seconds",  type=float, default=2.56)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--joint-pos-weight", type=float, default=10.0)
    p.add_argument("--jitter-weight", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default="TransfHAR_project")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name-prefix", default="nymeria_pose_baseline")
    return p.parse_args()


def main():
    args = parse_args()
    assert len(args.models) == len(args.runs), "--models and --runs must align"
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else (3e-4 if args.mode == "probe" else 1e-4)
    out_dir = args.out_dir or f"data_vis/nymeria_pose_baselines_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)
    T = int(round(args.window_seconds * FS)); Pt = T // 16

    # Build Nymeria loaders once (use first model's cfg windowing, overridden to window_seconds)
    _, base_cfg = load_baseline(args.models[0], os.path.join(args.runs_root, args.runs[0]), args.step, device)
    base_cfg = dict(base_cfg)
    base_cfg["windowing"] = {**base_cfg.get("windowing", {}), "window_seconds": args.window_seconds}
    print(f"\nLoading Nymeria data (window {args.window_seconds}s → {Pt} sections) ...")
    tr_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))[:args.max_train_files]
    va_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]
    print(f"  train files: {len(tr_files)}  val files: {len(va_files)}")
    loaders = make_nymeria_real_wrist_full_body_loaders_flu(
        cfg=base_cfg, pkl_root=args.train_dir, train_files=tr_files, val_files=va_files, add_gravity=False)
    train_loader, val_loader = loaders["train_loader"], loaders["val_loader"]

    body_model = load_body_model(device)
    use_wandb = (wandb is not None) and (not args.no_wandb)
    if use_wandb:
        try:
            wandb.login()
        except Exception as exc:
            print(f"  W&B login failed ({exc})"); use_wandb = False
    patience = 0 if args.no_early_stop else args.patience

    results = {}
    for name, run in zip(args.models, args.runs):
        print(f"\n{'='*55}\n{name}  ({run})  [{args.mode}]\n{'='*55}")
        wrapper, _ = load_baseline(name, os.path.join(args.runs_root, run), args.step, device)
        wb_run = None
        if use_wandb:
            wb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                name=f"{args.wandb_name_prefix}_{args.mode}_{name}",
                                config=vars(args), reinit=True)
        final, per_j = run_one(wrapper, name, train_loader, val_loader, device, args.mode, Pt,
                               lr, args.weight_decay, args.epochs, patience, body_model,
                               args.joint_pos_weight, args.jitter_weight, args.dropout, wb_run)
        results[name] = {"geo": final["geo_err_deg"], "mpjpe": final.get("mpjpe_mm"), "per_joint": per_j}
        if wb_run is not None:
            wb_run.log({"best/val_geo_err_deg": final["geo_err_deg"],
                        "best/val_mpjpe_mm": final.get("mpjpe_mm")})
            wb_run.finish()
        mt = f"  MPJPE={final['mpjpe_mm']:.1f}mm" if final.get("mpjpe_mm") is not None else ""
        print(f"  → {name}: geodesic={final['geo_err_deg']:.3f}°{mt}")

    # Summary + CSV + plot
    print(f"\n{'='*55}\nNymeria 23-Joint Pose — baselines [{args.mode}]")
    print(f"{'Model':<12} {'Geo (°)':>9} {'MPJPE (mm)':>12}")
    print("-" * 40)
    for n in args.models:
        mp = results[n]["mpjpe"]
        print(f"{n:<12} {results[n]['geo']:>9.3f} {('' if mp is None else f'{mp:.1f}'):>12}")
    print("=" * 55)
    pd.DataFrame([{"model": n, "geo_err_deg": results[n]["geo"], "mpjpe_mm": results[n]["mpjpe"]}
                  for n in args.models]).to_csv(os.path.join(out_dir, f"pose_{args.mode}_overall.csv"), index=False)
    pd.DataFrame([{"joint": SMPL_JOINT_NAMES[j], **{n: results[n]["per_joint"][j] for n in args.models}}
                  for j in range(NUM_JOINTS)]).to_csv(os.path.join(out_dir, f"pose_{args.mode}_per_joint.csv"), index=False)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [results[n]["geo"] for n in args.models]
    ax.bar(args.models, vals, color=colors[:len(args.models)], alpha=0.85, width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}°", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Geodesic Error (°)")
    ax.set_title(f"Nymeria 23-Joint Pose — baselines [{args.mode}]", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pose_{args.mode}_overall.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    main()
