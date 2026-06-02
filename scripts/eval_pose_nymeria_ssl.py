#!/usr/bin/env python3
"""
Evaluate SSL-pretrained ViT1D encoders on Nymeria real-wrist 23-joint full-body
pose estimation, with a single-transformer-block pose head.

Two modes:
  --mode probe     : freeze the encoder, train only the 1-block pose head.
  --mode finetune  : train the encoder + 1-block head jointly (full parameter).

Pose head (one transformer block):
    encoder tokens [B, C*Pt, D]
      → reshape [B, C, Pt, D] → mean over channels → [B, Pt, D]
      → + time positional embedding
      → 1 × TransformerEncoderLayer (self-attn over Pt time patches)
      → LayerNorm → Linear → [B, Pt, 23*6 = 138]   (6D joint rotations per patch)

Metric: mean geodesic rotation error (degrees), per joint and overall.

Usage:
    python scripts/eval_pose_nymeria_ssl.py \
        --runs vit1d_ssl_mae vit1d_ssl_limu_bert vit1d_ssl_simmtm vit1d_ssl_focal \
        --mode probe \
        --out-dir data_vis/nymeria_pose_probe_ssl/

    python scripts/eval_pose_nymeria_ssl.py \
        --runs vit1d_ssl_mae vit1d_ssl_limu_bert vit1d_ssl_simmtm vit1d_ssl_focal \
        --mode finetune \
        --out-dir data_vis/nymeria_pose_finetune_ssl/
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
except ImportError:  # pragma: no cover - optional experiment tracking
    wandb = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import (
    make_nymeria_real_wrist_full_body_loaders_flu,
)
from imu_lm.utils.helpers import cfg_get

NUM_JOINTS = 23
ROT6D_DIM  = 6
OUT_DIM    = NUM_JOINTS * ROT6D_DIM   # 138

SMPL_JOINT_NAMES = [
    "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]


# ── Rotation helpers ──────────────────────────────────────────────────────────

def rot6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def geodesic_error_deg(pred_6d, gt_6d):
    R_err = rot6d_to_matrix(pred_6d) @ rot6d_to_matrix(gt_6d).transpose(-1, -2)
    trace = R_err.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1 + 1e-6, 3 - 1e-6)
    return (torch.acos((trace - 1.0) / 2.0).mean() * (180.0 / torch.pi))


def slice_joints_1_23(y):
    """Nymeria loader yields [B, Pt, 144] (24 joints); drop root → [B, Pt, 138]."""
    return y[..., ROT6D_DIM:]


@torch.no_grad()
def mpjpe_mm(pred_6d, y_full_6d, gt_pos, body_model):
    """MPJPE in mm for predicted joints 1..23, with GT root rotation spliced in."""
    B, Pt, _ = pred_6d.shape
    j_loader = y_full_6d.shape[-1] // ROT6D_DIM

    root_6d = y_full_6d.reshape(B, Pt, j_loader, ROT6D_DIM)[:, :, 0:1, :]
    root_R = rot6d_to_matrix(root_6d)
    pred_R = rot6d_to_matrix(pred_6d.reshape(B, Pt, NUM_JOINTS, ROT6D_DIM))
    full_R = torch.cat([root_R, pred_R], dim=2)

    bm_device = body_model._J_regressor.device
    full_R_flat = full_R.reshape(B * Pt, -1).to(bm_device)
    _, pos_pred = body_model.forward_kinematics(full_R_flat)

    pos_pred = pos_pred.to(pred_6d.device)
    pos_pred_rel = pos_pred - pos_pred[:, 0:1]
    pos_pred_rel = pos_pred_rel.reshape(B, Pt, j_loader, 3)
    gt_pos = gt_pos.reshape(B, Pt, j_loader, 3).to(pred_6d.device)

    return float((pos_pred_rel[:, :, 1:] - gt_pos[:, :, 1:]).norm(dim=-1).mean().item() * 1000.0)


def load_body_model(device):
    try:
        import core.articulate as art
        from core.paths import Paths
        return art.model.ParametricModel(Paths.SMPL_FILE, device=device)
    except Exception as exc:
        print(f"  W&B/console MPJPE unavailable: could not load SMPL body model ({exc})")
        return None


# ── One-transformer-block pose head ───────────────────────────────────────────

class OneBlockPoseHead(nn.Module):
    def __init__(self, embed_dim, in_channels, num_patches, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.num_patches = num_patches
        self.time_pos = nn.Embedding(num_patches, embed_dim)
        self.block = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, OUT_DIM)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens):
        """tokens [B, C*Pt, D] → [B, Pt, 138]."""
        B = tokens.shape[0]
        x = tokens.reshape(B, self.in_channels, self.num_patches, -1).mean(dim=1)  # [B, Pt, D]
        t = torch.arange(self.num_patches, device=tokens.device)
        x = x + self.time_pos(t).unsqueeze(0)
        x = self.norm(self.block(x))
        return self.head(x)                                                        # [B, Pt, 138]


# ── Model loading ─────────────────────────────────────────────────────────────

def load_encoder(run_dir, step, device):
    ckpt_path = (os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt")
                 if step is not None else os.path.join(run_dir, "checkpoints", "best.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    enc = ViT1DEncoder(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    short = os.path.basename(run_dir).replace("vit1d_ssl_", "").replace("vit1d_", "")
    print(f"  {short}  step={ckpt.get('step','?')}  embed_dim={enc.embed_dim}")
    return enc, cfg, short


# ── Train / eval ──────────────────────────────────────────────────────────────

def run_one(enc, cfg, train_loader, val_loader, device, mode,
            lr, epochs, patience, body_model=None, wb_run=None):
    Pt = int(round(float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56)) * 50)) // enc.patch_size
    head = OneBlockPoseHead(enc.embed_dim, enc.in_channels, Pt).to(device)

    if mode == "probe":
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
        params = list(head.parameters())
    else:  # finetune
        for p in enc.parameters():
            p.requires_grad_(True)
        params = list(enc.parameters()) + list(head.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler()

    best_err, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        head.train()
        if mode == "finetune":
            enc.train()
        train_losses = []
        train_mpjpe = None          # FK is costly → compute once/epoch on the 1st batch
        for bi, batch in enumerate(train_loader):
            if batch is None:
                continue
            x = batch[0].to(device)
            y_full = batch[1].to(device)
            y = slice_joints_1_23(y_full)
            gt_pos = batch[2].to(device) if len(batch) > 2 else None
            opt.zero_grad()
            with autocast():
                if mode == "probe":
                    with torch.no_grad():
                        toks, _ = enc.forward_tokens(x.float())
                else:
                    toks, _ = enc.forward_tokens(x.float())
                pred = head(toks)
                loss = F.mse_loss(pred, y)
            train_losses.append(float(loss.detach().item()))
            # Train MPJPE: only on the first batch of each epoch (FK is expensive)
            if train_mpjpe is None and body_model is not None and gt_pos is not None:
                train_mpjpe = mpjpe_mm(pred.detach().float(), y_full, gt_pos, body_model)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

        val_metrics = evaluate(enc, head, val_loader, device, body_model)
        val_err = val_metrics["geo_err_deg"]
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        if val_err < best_err - 1e-4:
            best_err = val_err
            best_state = ({k: v.cpu().clone() for k, v in head.state_dict().items()},
                          {k: v.cpu().clone() for k, v in enc.state_dict().items()} if mode == "finetune" else None)
            no_improve = 0
        else:
            no_improve += 1
        if wb_run is not None:
            wb_payload = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/geo_err_deg": val_err,
                "lr": sched.get_last_lr()[0],
            }
            if train_mpjpe is not None:
                wb_payload["train/mpjpe_mm"] = train_mpjpe
            if val_metrics.get("mpjpe_mm") is not None:
                wb_payload["val/mpjpe_mm"] = val_metrics["mpjpe_mm"]
            wb_run.log(wb_payload, step=epoch + 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            mpjpe_txt = ""
            if train_mpjpe is not None and val_metrics.get("mpjpe_mm") is not None:
                mpjpe_txt = f"  train_mpjpe={train_mpjpe:.1f}mm  val_mpjpe={val_metrics['mpjpe_mm']:.1f}mm"
            print(f"    epoch {epoch+1:3d}  val_geo_err={val_err:.3f}°{mpjpe_txt}  best={best_err:.3f}°")
        if patience > 0 and no_improve >= patience:
            print(f"    early stop @ epoch {epoch+1}")
            break

    if best_state is not None:
        head.load_state_dict(best_state[0])
        if mode == "finetune" and best_state[1] is not None:
            enc.load_state_dict(best_state[1])
    per_j = per_joint_error(enc, head, val_loader, device)
    return best_err, per_j


@torch.no_grad()
def evaluate(enc, head, loader, device, body_model=None):
    head.eval(); enc.eval()
    errs, mpjpes = [], []
    for batch in loader:
        if batch is None:
            continue
        x = batch[0].to(device)
        y_full = batch[1].to(device)
        y = slice_joints_1_23(y_full)
        gt_pos = batch[2].to(device) if len(batch) > 2 else None
        toks, _ = enc.forward_tokens(x.float())
        pred_raw = head(toks)
        pred = pred_raw.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
        gt = y.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
        errs.append(geodesic_error_deg(pred, gt).item())
        if body_model is not None and gt_pos is not None:
            mpjpes.append(mpjpe_mm(pred_raw, y_full, gt_pos, body_model))
    return {
        "geo_err_deg": float(np.mean(errs)),
        "mpjpe_mm": float(np.mean(mpjpes)) if mpjpes else None,
    }


@torch.no_grad()
def per_joint_error(enc, head, loader, device):
    head.eval(); enc.eval()
    acc = [[] for _ in range(NUM_JOINTS)]
    for batch in loader:
        x = batch[0].to(device)
        y = slice_joints_1_23(batch[1].to(device))
        toks, _ = enc.forward_tokens(x.float())
        pred = head(toks).reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
        gt = y.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
        R = rot6d_to_matrix(pred) @ rot6d_to_matrix(gt).transpose(-1, -2)
        tr = R.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1 + 1e-6, 3 - 1e-6)
        ang = torch.acos((tr - 1.0) / 2.0) * (180.0 / torch.pi)   # [B, Pt, J]
        for j in range(NUM_JOINTS):
            acc[j].append(ang[:, :, j].mean().item())
    return np.array([float(np.mean(a)) for a in acc])


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(results, shorts, mode, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(max(6, len(shorts) * 1.8), 4.5))
    vals = [results[s]["geo_err"] for s in shorts]
    bars = ax.bar(shorts, vals, color=[colors[i % len(colors)] for i in range(len(shorts))],
                  alpha=0.85, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}°",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean Geodesic Error (°)")
    ax.set_title(f"Nymeria 23-Joint Pose — {mode}\nSSL pretrained encoders + 1-block head",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pose_{mode}_overall.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    y = np.arange(NUM_JOINTS); w = 0.8 / len(shorts)
    fig, ax = plt.subplots(figsize=(9, max(6, NUM_JOINTS * 0.42)))
    for ri, s in enumerate(shorts):
        ax.barh(y + ri * w, results[s]["per_joint"], height=w * 0.9, label=s,
                color=colors[ri % len(colors)], alpha=0.85)
    ax.set_yticks(y + w * (len(shorts) - 1) / 2)
    ax.set_yticklabels(SMPL_JOINT_NAMES, fontsize=8)
    ax.set_xlabel("Geodesic Error (°)")
    ax.set_title(f"Nymeria Per-Joint Error — {mode}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pose_{mode}_per_joint.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plots → {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+",
                   default=["vit1d_ssl_mae", "vit1d_ssl_limu_bert",
                            "vit1d_ssl_simmtm", "vit1d_ssl_focal"])
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--train-dir", default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir",   default="/mnt/nas2/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=200,
                   help="cap Nymeria train sessions (867 total is slow to load)")
    p.add_argument("--max-val-files",   type=int, default=101)
    p.add_argument("--lr",       type=float, default=None,
                   help="default: 3e-4 probe / 1e-4 finetune")
    p.add_argument("--epochs",   type=int, default=60)
    p.add_argument("--patience", type=int, default=12,
                   help="early-stop patience; 0 (or --no-early-stop) disables it")
    p.add_argument("--no-early-stop", action="store_true",
                   help="run all --epochs without early stopping")
    p.add_argument("--out-dir",  default=None)
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable Weights & Biases logging")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project; default reads cfg wandb.project or TransfHAR_project")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity; default reads cfg wandb.entity")
    p.add_argument("--wandb-name-prefix", default="nymeria_pose_ssl",
                   help="prefix for per-encoder W&B run names")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else (3e-4 if args.mode == "probe" else 1e-4)
    out_dir = args.out_dir or f"data_vis/nymeria_pose_{args.mode}_ssl"
    os.makedirs(out_dir, exist_ok=True)

    # Build Nymeria loaders once from the first run's cfg (same windowing for all)
    first_dir = os.path.join(args.runs_root, args.runs[0])
    _, base_cfg, _ = load_encoder(first_dir, args.step, device)
    print(f"\nLoading Nymeria real-wrist full-body data ...")
    train_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))[:args.max_train_files]
    val_files   = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]
    print(f"  train files: {len(train_files)}  val files: {len(val_files)}")
    loaders = make_nymeria_real_wrist_full_body_loaders_flu(
        cfg=base_cfg, pkl_root=args.train_dir,
        train_files=train_files, val_files=val_files, add_gravity=False)
    train_loader, val_loader = loaders["train_loader"], loaders["val_loader"]
    b = next(iter(val_loader))
    print(f"  batch: x={tuple(b[0].shape)}  y={tuple(b[1].shape)}")

    body_model = load_body_model(device)
    use_wandb = (wandb is not None) and (not args.no_wandb)
    if use_wandb:
        try:
            wandb.login()
        except Exception as exc:
            print(f"  W&B login failed ({exc}); continuing without W&B logging.")
            use_wandb = False
    elif not args.no_wandb:
        print("  W&B is not installed; continuing without W&B logging.")

    wandb_project = args.wandb_project or cfg_get(base_cfg, ["wandb", "project"], "TransfHAR_project")
    wandb_entity = args.wandb_entity or cfg_get(base_cfg, ["wandb", "entity"], None)

    results, shorts = {}, []
    for run in args.runs:
        print(f"\n{'='*55}\n{run}  [{args.mode}]\n{'='*55}")
        enc, cfg, short = load_encoder(os.path.join(args.runs_root, run), args.step, device)
        wb_run = None
        if use_wandb:
            wb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"{args.wandb_name_prefix}_{args.mode}_{short}",
                config={
                    "script": "scripts/eval_pose_nymeria_ssl.py",
                    "source_run": run,
                    "mode": args.mode,
                    "ssl_step": args.step,
                    "lr": lr,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "max_train_files": args.max_train_files,
                    "max_val_files": args.max_val_files,
                    "train_dir": args.train_dir,
                    "val_dir": args.val_dir,
                    "out_dir": out_dir,
                    "metric": "mpjpe_mm",
                },
                reinit=True,
            )
        _patience = 0 if args.no_early_stop else args.patience
        err, per_j = run_one(enc, cfg, train_loader, val_loader, device,
                             args.mode, lr, args.epochs, _patience,
                             body_model=body_model, wb_run=wb_run)
        results[short] = {"geo_err": err, "per_joint": per_j}
        shorts.append(short)
        if wb_run is not None:
            wb_run.log({"best/val_geo_err_deg": err})
            wb_run.finish()
        print(f"  → {short}: {err:.3f}° mean geodesic error")

    # Summary
    print(f"\n{'='*50}\nNymeria 23-Joint Pose — {args.mode}")
    print(f"{'Model':<16} {'Geo Err (°)':>12}")
    print("-" * 50)
    for s in shorts:
        print(f"{s:<16} {results[s]['geo_err']:>12.3f}")
    print("=" * 50)

    rows = [{"joint": SMPL_JOINT_NAMES[j], **{s: results[s]["per_joint"][j] for s in shorts}}
            for j in range(NUM_JOINTS)]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"pose_{args.mode}_per_joint.csv"), index=False)
    pd.DataFrame([{"model": s, "geo_err": results[s]["geo_err"]} for s in shorts]).to_csv(
        os.path.join(out_dir, f"pose_{args.mode}_overall.csv"), index=False)
    plot(results, shorts, args.mode, out_dir)


if __name__ == "__main__":
    main()
