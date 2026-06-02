#!/usr/bin/env python3
"""Probe SSL baseline checkpoints on Nymeria real-wrist 23-joint pose.

This is the baseline-model counterpart to ``eval_pose_nymeria_altattn.py``.
The SimMTM / LIMU-BERT / FOCAL / MOMENT wrappers expose a window-level
``encode(x)`` representation for [B, T, C] inputs, so the pose head maps one
frozen window embedding to the 8 pose sections in a 2.56 s Nymeria window.

Usage:
    python scripts/eval_pose_nymeria_ssl_baseline.py \
        --model simmtm \
        --run ssl_simmtm_fullocd_realdata_ocdetect_sizematched \
        --mode probe \
        --out-dir data_vis/nymeria_pose_baseline_probe/simmtm \
        --max-train-files 867
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import (
    make_nymeria_real_wrist_full_body_loaders_flu,
)
from imu_lm.models.baselines.ssl_baselines import build_ssl_model
from imu_lm.objectives.pose_1d import _compute_fk_pos_loss, _compute_val_mpjpe
from imu_lm.utils.helpers import cfg_get

NUM_JOINTS = 23
ROT6D_DIM = 6
OUT_DIM = NUM_JOINTS * ROT6D_DIM
FS = 50.0

SMPL_JOINT_NAMES = [
    "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]


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
        print(f"  SMPL body model unavailable ({exc}) - FK loss + MPJPE disabled")
        return None


class WindowEmbeddingPoseDecoder(nn.Module):
    """Window embedding -> per-section pose sequence.

    A learned time query for each pose section attends to the frozen SSL window
    embedding, then a small temporal transformer predicts [B, Pt, 23*6].
    """

    def __init__(self, embed_dim, num_patches, hidden=384, n_heads=4,
                 n_temporal=2, dropout=0.2):
        super().__init__()
        self.num_patches = num_patches
        self.proj = nn.Linear(embed_dim, hidden)
        self.time_query = nn.Embedding(num_patches, hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                                batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, num_layers=n_temporal)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, OUT_DIM)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, emb):
        B = emb.shape[0]
        memory = self.proj(emb).unsqueeze(1)                 # [B, 1, H]
        t = torch.arange(self.num_patches, device=emb.device)
        q = self.time_query(t).unsqueeze(0).expand(B, -1, -1) # [B, Pt, H]
        h, _ = self.cross_attn(q, memory, memory)
        h = self.norm1(h + q)
        h = self.norm2(self.temporal(h))
        return self.head(self.dropout(h))


def load_ssl_model(model_name, run_dir, step, device):
    ckpt_path = (os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt")
                 if step is not None else os.path.join(run_dir, "checkpoints", "best.pt"))
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(run_dir, "checkpoints", "latest.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model_cfg = cfg_get(cfg, ["model"], {}) or {}
    model = build_ssl_model(model_name, model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"  loaded {model_name} from {ckpt_path}  step={ckpt.get('step', '?')}")
    return model, cfg


@torch.no_grad()
def encode_window(model, x_ct):
    """Nymeria x [B,C,T] -> baseline encode input [B,T,C] -> [B,D]."""
    return model.encode(x_ct.float().transpose(1, 2).contiguous())


def pose_loss(pred, y_full, gt_pos, body_model, joint_pos_w, jitter_w):
    y = slice_joints_1_23(y_full)
    loss = F.mse_loss(pred, y)
    logs = {"rot": float(loss.detach())}
    if body_model is not None and gt_pos is not None and joint_pos_w > 0:
        pos = _compute_fk_pos_loss(pred, y_full, gt_pos, 1, NUM_JOINTS, body_model)
        loss = loss + joint_pos_w * pos
        logs["pos"] = float(pos.detach())
    if jitter_w > 0 and pred.shape[1] > 1:
        jit = ((pred[:, 1:] - pred[:, :-1]) ** 2).mean()
        loss = loss + jitter_w * jit
        logs["jitter"] = float(jit.detach())
    return loss, logs


def run_probe(model, train_loader, val_loader, device, Pt, lr, wd, epochs,
              patience, body_model, joint_pos_w, jitter_w, dropout, wb_run=None):
    with torch.no_grad():
        sample = next(iter(val_loader))[0][:1].to(device)
        embed_dim = int(encode_window(model, sample).shape[-1])
    head = WindowEmbeddingPoseDecoder(embed_dim, Pt, dropout=dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler()

    best_err, best_state, no_improve = float("inf"), None, 0
    for epoch in range(epochs):
        head.train()
        losses, train_mpjpe = [], None
        for batch in train_loader:
            if batch is None:
                continue
            x = batch[0].to(device)
            y_full = batch[1].to(device)
            gt_pos = batch[2].to(device) if len(batch) > 2 else None
            opt.zero_grad()
            with autocast():
                with torch.no_grad():
                    emb = encode_window(model, x)
                pred = head(emb)
            loss, _ = pose_loss(pred.float(), y_full, gt_pos, body_model,
                                joint_pos_w, jitter_w)
            losses.append(float(loss.detach()))
            if train_mpjpe is None and body_model is not None and gt_pos is not None:
                train_mpjpe = _compute_val_mpjpe(
                    pred.detach().float(), y_full, gt_pos, 1, NUM_JOINTS, body_model)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

        vm = evaluate(model, head, val_loader, device, body_model)
        val_err = vm["geo_err_deg"]
        if val_err < best_err - 1e-4:
            best_err = val_err
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if wb_run is not None:
            payload = {
                "epoch": epoch + 1,
                "train/loss": float(np.mean(losses)),
                "val/geo_err_deg": val_err,
                "lr": sched.get_last_lr()[0],
            }
            if train_mpjpe is not None:
                payload["train/mpjpe_mm"] = train_mpjpe
            if vm.get("mpjpe_mm") is not None:
                payload["val/mpjpe_mm"] = vm["mpjpe_mm"]
            wb_run.log(payload, step=epoch + 1)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            mtxt = (f"  train_mpjpe={train_mpjpe:.1f}mm  val_mpjpe={vm['mpjpe_mm']:.1f}mm"
                    if train_mpjpe is not None and vm.get("mpjpe_mm") is not None else "")
            print(f"    epoch {epoch+1:3d}  val_geo={val_err:.3f}deg{mtxt}  best={best_err:.3f}deg")
        if patience > 0 and no_improve >= patience:
            print(f"    early stop @ epoch {epoch+1}")
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    return evaluate(model, head, val_loader, device, body_model), per_joint_error(model, head, val_loader, device)


@torch.no_grad()
def evaluate(model, head, loader, device, body_model=None):
    head.eval()
    model.eval()
    errs, mpjpes = [], []
    for batch in loader:
        if batch is None:
            continue
        x = batch[0].to(device)
        y_full = batch[1].to(device)
        gt_pos = batch[2].to(device) if len(batch) > 2 else None
        pred = head(encode_window(model, x))
        y = slice_joints_1_23(y_full)
        errs.append(geodesic_error_deg(
            pred.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM),
            y.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)).item())
        if body_model is not None and gt_pos is not None:
            mpjpes.append(_compute_val_mpjpe(pred.float(), y_full, gt_pos, 1, NUM_JOINTS, body_model))
    return {"geo_err_deg": float(np.mean(errs)),
            "mpjpe_mm": float(np.mean(mpjpes)) if mpjpes else None}


@torch.no_grad()
def per_joint_error(model, head, loader, device):
    head.eval()
    model.eval()
    acc = [[] for _ in range(NUM_JOINTS)]
    for batch in loader:
        x = batch[0].to(device)
        y = slice_joints_1_23(batch[1].to(device))
        pred = head(encode_window(model, x)).reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
        gt = y.reshape(*y.shape[:2], NUM_JOINTS, ROT6D_DIM)
        R = rot6d_to_matrix(pred) @ rot6d_to_matrix(gt).transpose(-1, -2)
        tr = R.diagonal(dim1=-2, dim2=-1).sum(-1).clamp(-1 + 1e-6, 3 - 1e-6)
        ang = torch.acos((tr - 1.0) / 2.0) * (180.0 / torch.pi)
        for j in range(NUM_JOINTS):
            acc[j].append(ang[:, :, j].mean().item())
    return np.array([float(np.mean(a)) for a in acc])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["simmtm", "limu_bert", "focal", "moment"])
    p.add_argument("--run", required=True)
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--mode", choices=["probe"], default="probe")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--train-dir", default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir", default="/mnt/nas2/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=200)
    p.add_argument("--max-val-files", type=int, default=101)
    p.add_argument("--window-seconds", type=float, default=2.56)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
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
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name-prefix", default="nymeria_pose_ssl_baseline")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = args.out_dir or f"data_vis/nymeria_pose_ssl_baseline_{args.model}_{args.mode}"
    os.makedirs(out_dir, exist_ok=True)

    model, cfg = load_ssl_model(args.model, os.path.join(args.runs_root, args.run),
                                args.step, device)
    cfg = dict(cfg)
    cfg.setdefault("windowing", {})
    cfg["windowing"] = {**cfg.get("windowing", {}), "window_seconds": args.window_seconds}
    cfg.setdefault("data", {})
    cfg["data"] = {**cfg.get("data", {}),
                   "batch_size": args.batch_size,
                   "eval_batch_size": args.eval_batch_size}
    T = int(round(args.window_seconds * FS))
    Pt = T // 16

    print(f"\nLoading Nymeria real-wrist full-body data (window {args.window_seconds}s -> {Pt} sections) ...")
    train_files = sorted(glob.glob(os.path.join(args.train_dir, "*.pkl")))[:args.max_train_files]
    val_files = sorted(glob.glob(os.path.join(args.val_dir, "*.pkl")))[:args.max_val_files]
    print(f"  train files: {len(train_files)}  val files: {len(val_files)}")
    loaders = make_nymeria_real_wrist_full_body_loaders_flu(
        cfg=cfg, pkl_root=args.train_dir,
        train_files=train_files, val_files=val_files, add_gravity=False)
    train_loader, val_loader = loaders["train_loader"], loaders["val_loader"]
    b = next(iter(val_loader))
    print(f"  batch: x={tuple(b[0].shape)}  y={tuple(b[1].shape)}")

    body_model = load_body_model(device)
    use_wandb = (wandb is not None) and (not args.no_wandb)
    project = args.wandb_project or cfg_get(cfg, ["wandb", "project"], "TransfHAR_project")
    entity = args.wandb_entity or cfg_get(cfg, ["wandb", "entity"], None)
    wb_run = None
    if use_wandb:
        wb_run = wandb.init(
            project=project, entity=entity,
            name=f"{args.wandb_name_prefix}_{args.model}_{args.mode}",
            config=vars(args), reinit=True)

    patience = 0 if args.no_early_stop else args.patience
    final, per_j = run_probe(
        model, train_loader, val_loader, device, Pt, args.lr,
        args.weight_decay, args.epochs, patience, body_model,
        args.joint_pos_weight, args.jitter_weight, args.dropout, wb_run)
    if wb_run is not None:
        wb_run.log({"best/val_geo_err_deg": final["geo_err_deg"],
                    "best/val_mpjpe_mm": final.get("mpjpe_mm")})
        wb_run.finish()

    mpjpe_txt = f"  MPJPE={final['mpjpe_mm']:.1f}mm" if final.get("mpjpe_mm") is not None else ""
    print(f"\n{'='*55}\n{args.run} [{args.model} probe]: "
          f"geodesic={final['geo_err_deg']:.3f}deg{mpjpe_txt}\n{'='*55}")

    csv_path = os.path.join(out_dir, f"pose_{args.model}_{args.mode}_per_joint.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["joint", "geo_err_deg"])
        writer.writeheader()
        for j in range(NUM_JOINTS):
            writer.writerow({"joint": SMPL_JOINT_NAMES[j], "geo_err_deg": per_j[j]})
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(np.arange(NUM_JOINTS), per_j, color="#4c72b0", alpha=0.85)
        ax.set_yticks(np.arange(NUM_JOINTS))
        ax.set_yticklabels(SMPL_JOINT_NAMES, fontsize=8)
        ax.set_xlabel("Geodesic Error (deg)")
        ax.set_title(f"{args.model} {args.mode}\noverall {final['geo_err_deg']:.2f}deg{mpjpe_txt}",
                     fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pose_{args.model}_{args.mode}_per_joint.png"),
                    dpi=150, bbox_inches="tight")
    else:
        print("matplotlib is not installed; skipped per-joint PNG plot")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
