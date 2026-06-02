#!/usr/bin/env python3
"""
Probe / finetune the alt-attention (VGGT-style) MAE encoder on Nymeria
real-wrist 23-joint full-body pose, with a per-section register-token decoder.

Encoder : ViT1DEncoderAltAttn  (loaded from runs/<run>/checkpoints).
Readout : one learnable REGISTER token per 0.32 s section (time-patch). The
          register query cross-attends to that section's 6 channel tokens, then
          a temporal transformer over the Pt section features predicts pose —
          same 2-stage structure / losses as vit1d_full_body_pose_finetune_*:
              loss = rot_MSE + joint_pos_weight·FK_pos_MSE + jitter_weight·jitter
          (FK position loss + weight decay are what keep finetune from overfitting.)

Modes:
  --mode probe     : freeze encoder, train register-decoder only.
  --mode finetune  : train encoder + register-decoder jointly.

Metrics: geodesic rotation error (deg) and MPJPE (mm), train + val, → W&B.

Usage:
    python scripts/eval_pose_nymeria_altattn.py \
        --run vit1d_ssl_mae_varlen_altattn --mode probe \
        --out-dir data_vis/nymeria_pose_altattn_probe/
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

from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import (
    make_nymeria_real_wrist_full_body_loaders_flu,
)
from imu_lm.objectives.pose_1d import _compute_fk_pos_loss, _compute_val_mpjpe
from imu_lm.utils.helpers import cfg_get

NUM_JOINTS = 23
ROT6D_DIM  = 6
OUT_DIM    = NUM_JOINTS * ROT6D_DIM   # 138
FS         = 50.0

SMPL_JOINT_NAMES = [
    "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]


# ── Rotation / metric helpers ─────────────────────────────────────────────────

def rot6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def geodesic_error_deg(pred_6d, gt_6d):
    R = rot6d_to_matrix(pred_6d) @ rot6d_to_matrix(gt_6d).transpose(-1, -2)
    tr = R.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1 + 1e-6, 3 - 1e-6)
    return torch.acos((tr - 1.0) / 2.0).mean() * (180.0 / torch.pi)


def slice_joints_1_23(y):
    return y[..., ROT6D_DIM:]


def load_body_model(device):
    try:
        import core.articulate as art
        from core.paths import Paths
        return art.model.ParametricModel(Paths.SMPL_FILE, device=device)
    except Exception as exc:
        print(f"  SMPL body model unavailable ({exc}) — FK position loss + MPJPE disabled")
        return None


# ── Register-token pose decoder ───────────────────────────────────────────────

class RegisterPoseDecoder(nn.Module):
    """Per-section register token → temporal transformer → per-section pose.

    Stage 1 (register readout): for each 0.32 s section, a learnable register
    query cross-attends to that section's C channel tokens → section feature.
    Stage 2 (temporal): transformer over the Pt section features → 23×6 pose.
    Same 2-stage shape as ViT1DPoseDecoder, but the channel mean-pool is replaced
    by a register-token cross-attention.
    """

    def __init__(self, embed_dim, in_channels, num_patches, num_joints=23,
                 hidden=192, n_temporal=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.num_patches = num_patches
        self.hidden = hidden

        self.proj = nn.Linear(embed_dim, hidden)
        # Stage 1: register token (shared query) + per-section positional id
        self.register   = nn.Parameter(torch.zeros(1, 1, hidden))
        self.reg_secpos = nn.Embedding(num_patches, hidden)
        self.frame_attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.norm1      = nn.LayerNorm(hidden)
        # Stage 2: temporal transformer over Pt section features
        self.time_pos = nn.Embedding(num_patches, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=n_temporal)
        self.norm2    = nn.LayerNorm(hidden)
        self.dropout  = nn.Dropout(dropout)
        self.head     = nn.Linear(hidden, num_joints * ROT6D_DIM)
        nn.init.trunc_normal_(self.register, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, tokens):
        """tokens [B, C*Pt, D] (channel-major) → [B, Pt, 138]."""
        B = tokens.shape[0]
        C, Pt = self.in_channels, self.num_patches
        h = self.proj(tokens)                                       # [B, C*Pt, H]
        # section grid [B, Pt, C, H] → [B*Pt, C, H]
        grid = h.reshape(B, C, Pt, self.hidden).permute(0, 2, 1, 3).reshape(B * Pt, C, self.hidden)
        sec = torch.arange(Pt, device=tokens.device)
        # register query per section: shared token + section positional id
        reg = self.register.expand(B * Pt, 1, -1) + \
              self.reg_secpos(sec).repeat(B, 1).unsqueeze(1)        # [B*Pt, 1, H]
        sec_feat, _ = self.frame_attn(reg, grid, grid)             # [B*Pt, 1, H]
        sec_feat = self.norm1(sec_feat).reshape(B, Pt, self.hidden)  # [B, Pt, H]
        # Stage 2 temporal
        sec_feat = sec_feat + self.time_pos(sec).unsqueeze(0)
        out = self.norm2(self.temporal(sec_feat))
        return self.head(self.dropout(out))                        # [B, Pt, 138]


class TemporalPoseHead(nn.Module):
    """Pose head that consumes the encoder's PRETRAINED per-section register
    features [B, Pt, D] directly (no from-scratch query) → temporal transformer.
    Used when the encoder was pretrained with use_register: true."""

    def __init__(self, embed_dim, num_patches, num_joints=23,
                 hidden=192, n_temporal=2, n_heads=4, dropout=0.2):
        super().__init__()
        self.num_patches = num_patches
        self.proj    = nn.Linear(embed_dim, hidden)
        self.time_pos = nn.Embedding(num_patches, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=n_temporal)
        self.norm    = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(hidden, num_joints * ROT6D_DIM)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, reg):
        """reg [B, Pt, D] (pretrained register features) → [B, Pt, 138]."""
        sec = torch.arange(reg.shape[1], device=reg.device)
        h = self.proj(reg) + self.time_pos(sec).unsqueeze(0)
        h = self.norm(self.temporal(h))
        return self.head(self.dropout(h))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_encoder(run_dir, step, device):
    ckpt_path = (os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt")
                 if step is not None else os.path.join(run_dir, "checkpoints", "best.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    enc = ViT1DEncoderAltAttn(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    print(f"  loaded {os.path.basename(run_dir)}  step={ckpt.get('step','?')}  embed_dim={enc.embed_dim}")
    return enc, cfg


# ── Loss (rotation MSE + FK position + jitter) ────────────────────────────────

def pose_loss(pred, y_full, gt_pos, body_model, joint_pos_w, jitter_w):
    """pred [B,Pt,138], y_full [B,Pt,144], gt_pos [B,Pt,72]."""
    y = slice_joints_1_23(y_full)
    loss = F.mse_loss(pred, y)                                     # rotation MSE
    logs = {"rot": float(loss.detach())}
    if body_model is not None and gt_pos is not None and joint_pos_w > 0:
        pos_loss = _compute_fk_pos_loss(pred, y_full, gt_pos, 1, NUM_JOINTS, body_model)
        loss = loss + joint_pos_w * pos_loss
        logs["pos"] = float(pos_loss.detach())
    if jitter_w > 0 and pred.shape[1] > 1:
        jit = ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()
        loss = loss + jitter_w * jit
        logs["jitter"] = float(jit.detach())
    return loss, logs


# ── Train / eval ──────────────────────────────────────────────────────────────

def _pose_feat(enc, x, use_reg):
    """Per-section pose feature: pretrained registers [B,Pt,D] if available,
    else the channel token grid [B,C*Pt,D]."""
    if use_reg:
        return enc.forward_registers(x.float())[0]
    return enc.forward_tokens(x.float())[0]


def run_one(enc, train_loader, val_loader, device, mode, Pt, lr, wd, epochs,
            patience, body_model, joint_pos_w, jitter_w, dropout, wb_run=None):
    use_reg = bool(getattr(enc, "_use_register", False))
    if use_reg:
        dec = TemporalPoseHead(enc.embed_dim, Pt, dropout=dropout).to(device)
        print("  using PRETRAINED per-section register features → TemporalPoseHead")
    else:
        dec = RegisterPoseDecoder(enc.embed_dim, enc.in_channels, Pt, dropout=dropout).to(device)
        print("  encoder has no registers → from-scratch RegisterPoseDecoder")

    if mode == "probe":
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
        params = list(dec.parameters())
    else:
        for p in enc.parameters():
            p.requires_grad_(True)
        params = list(enc.parameters()) + list(dec.parameters())

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler()

    best_err, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        dec.train()
        if mode == "finetune":
            enc.train()
        train_losses, train_mpjpe = [], None
        for batch in train_loader:
            if batch is None:
                continue
            x = batch[0].to(device); y_full = batch[1].to(device)
            gt_pos = batch[2].to(device) if len(batch) > 2 else None
            opt.zero_grad()
            with autocast():
                if mode == "probe":
                    with torch.no_grad():
                        feat = _pose_feat(enc, x, use_reg)
                else:
                    feat = _pose_feat(enc, x, use_reg)
                pred = dec(feat)
            # FK loss needs fp32 for the body model — compute outside autocast
            loss, _ = pose_loss(pred.float(), y_full, gt_pos, body_model, joint_pos_w, jitter_w)
            train_losses.append(float(loss.detach()))
            if train_mpjpe is None and body_model is not None and gt_pos is not None:
                train_mpjpe = _compute_val_mpjpe(pred.detach().float(), y_full, gt_pos, 1, NUM_JOINTS, body_model)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update()
        sched.step()

        vm = evaluate(enc, dec, val_loader, device, body_model, use_reg)
        val_err = vm["geo_err_deg"]
        if val_err < best_err - 1e-4:
            best_err = val_err
            best_state = ({k: v.cpu().clone() for k, v in dec.state_dict().items()},
                          {k: v.cpu().clone() for k, v in enc.state_dict().items()} if mode == "finetune" else None)
            no_improve = 0
        else:
            no_improve += 1
        if wb_run is not None:
            payload = {"epoch": epoch + 1, "train/loss": float(np.mean(train_losses)),
                       "val/geo_err_deg": val_err, "lr": sched.get_last_lr()[0]}
            if train_mpjpe is not None:
                payload["train/mpjpe_mm"] = train_mpjpe
            if vm.get("mpjpe_mm") is not None:
                payload["val/mpjpe_mm"] = vm["mpjpe_mm"]
            wb_run.log(payload, step=epoch + 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            mtxt = (f"  train_mpjpe={train_mpjpe:.1f}mm  val_mpjpe={vm['mpjpe_mm']:.1f}mm"
                    if train_mpjpe is not None and vm.get("mpjpe_mm") is not None else "")
            print(f"    epoch {epoch+1:3d}  val_geo={val_err:.3f}°{mtxt}  best={best_err:.3f}°")
        if patience > 0 and no_improve >= patience:
            print(f"    early stop @ epoch {epoch+1}")
            break

    if best_state is not None:
        dec.load_state_dict(best_state[0])
        if mode == "finetune" and best_state[1] is not None:
            enc.load_state_dict(best_state[1])
    final = evaluate(enc, dec, val_loader, device, body_model, use_reg)
    per_j = per_joint_error(enc, dec, val_loader, device, use_reg)
    return final, per_j


@torch.no_grad()
def evaluate(enc, dec, loader, device, body_model=None, use_reg=False):
    dec.eval(); enc.eval()
    errs, mpjpes = [], []
    for batch in loader:
        if batch is None:
            continue
        x = batch[0].to(device); y_full = batch[1].to(device)
        gt_pos = batch[2].to(device) if len(batch) > 2 else None
        pred = dec(_pose_feat(enc, x, use_reg))
        y = slice_joints_1_23(y_full)
        errs.append(geodesic_error_deg(pred.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM),
                                       y.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)).item())
        if body_model is not None and gt_pos is not None:
            mpjpes.append(_compute_val_mpjpe(pred.float(), y_full, gt_pos, 1, NUM_JOINTS, body_model))
    return {"geo_err_deg": float(np.mean(errs)),
            "mpjpe_mm": float(np.mean(mpjpes)) if mpjpes else None}


@torch.no_grad()
def per_joint_error(enc, dec, loader, device, use_reg=False):
    dec.eval(); enc.eval()
    acc = [[] for _ in range(NUM_JOINTS)]
    for batch in loader:
        x = batch[0].to(device)
        y = slice_joints_1_23(batch[1].to(device))
        pred = dec(_pose_feat(enc, x, use_reg)).reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
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
    p.add_argument("--run", default="vit1d_ssl_mae_varlen_altattn")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--mode", choices=["probe", "finetune"], default="probe")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--train-dir", default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir",   default="/mnt/nas2/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=200)
    p.add_argument("--max-val-files",   type=int, default=101)
    p.add_argument("--window-seconds",  type=float, default=2.56,
                   help="pose window length (2.56s → 8 sections of 0.32s)")
    p.add_argument("--lr", type=float, default=None, help="default 3e-4 probe / 1e-4 finetune")
    p.add_argument("--weight-decay", type=float, default=0.05,
                   help="strong WD (like the finetune config) curbs overfitting")
    p.add_argument("--joint-pos-weight", type=float, default=10.0,
                   help="FK joint-position loss weight (physically-grounded regulariser)")
    p.add_argument("--jitter-weight", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name-prefix", default="nymeria_pose_altattn")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    lr = args.lr if args.lr is not None else (3e-4 if args.mode == "probe" else 1e-4)
    out_dir = args.out_dir or f"data_vis/nymeria_pose_altattn_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)

    enc, cfg = load_encoder(os.path.join(args.runs_root, args.run), args.step, device)
    # Pose at a fixed window (the model is variable-length; pose uses 2.56s = 8 sections)
    cfg = dict(cfg)
    cfg.setdefault("windowing", {})
    cfg["windowing"] = {**cfg.get("windowing", {}), "window_seconds": args.window_seconds}
    T  = int(round(args.window_seconds * FS))
    Pt = T // enc.patch_size

    print(f"\nLoading Nymeria real-wrist full-body data (window {args.window_seconds}s → {Pt} sections) ...")
    train_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))[:args.max_train_files]
    val_files   = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]
    print(f"  train files: {len(train_files)}  val files: {len(val_files)}")
    loaders = make_nymeria_real_wrist_full_body_loaders_flu(
        cfg=cfg, pkl_root=args.train_dir,
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
            print(f"  W&B login failed ({exc}); continuing without W&B."); use_wandb = False
    project = args.wandb_project or cfg_get(cfg, ["wandb", "project"], "TransfHAR_project")
    entity  = args.wandb_entity or cfg_get(cfg, ["wandb", "entity"], None)
    wb_run = None
    if use_wandb:
        wb_run = wandb.init(project=project, entity=entity,
                            name=f"{args.wandb_name_prefix}_{args.mode}",
                            config=vars(args), reinit=True)

    patience = 0 if args.no_early_stop else args.patience
    final, per_j = run_one(enc, train_loader, val_loader, device, args.mode, Pt,
                           lr, args.weight_decay, args.epochs, patience, body_model,
                           args.joint_pos_weight, args.jitter_weight, args.dropout, wb_run)
    if wb_run is not None:
        wb_run.log({"best/val_geo_err_deg": final["geo_err_deg"],
                    "best/val_mpjpe_mm": final.get("mpjpe_mm")})
        wb_run.finish()

    mpjpe_txt = f"  MPJPE={final['mpjpe_mm']:.1f}mm" if final.get("mpjpe_mm") is not None else ""
    print(f"\n{'='*55}\n{args.run} [{args.mode}]: "
          f"geodesic={final['geo_err_deg']:.3f}°{mpjpe_txt}\n{'='*55}")

    pd.DataFrame([{"joint": SMPL_JOINT_NAMES[j], "geo_err_deg": per_j[j]}
                  for j in range(NUM_JOINTS)]).to_csv(
        os.path.join(out_dir, f"pose_{args.mode}_per_joint.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(np.arange(NUM_JOINTS), per_j, color="#4c72b0", alpha=0.85)
    ax.set_yticks(np.arange(NUM_JOINTS)); ax.set_yticklabels(SMPL_JOINT_NAMES, fontsize=8)
    ax.set_xlabel("Geodesic Error (°)")
    ax.set_title(f"{args.run} [{args.mode}]\nper-joint  (overall {final['geo_err_deg']:.2f}°{mpjpe_txt})",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pose_{args.mode}_per_joint.png"), dpi=150, bbox_inches="tight")
    print(f"Saved → {out_dir}")


if __name__ == "__main__":
    main()
