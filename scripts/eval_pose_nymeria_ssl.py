#!/usr/bin/env python3
"""
Evaluate SSL-pretrained STANDARD ViT1D encoders (ViT1DEncoder) on Nymeria
real-wrist 23-joint full-body pose estimation.

This is a THIN WRAPPER around eval_pose_nymeria_altattn.py: it reuses that
script's pose decoder (RegisterPoseDecoder), training loss (rotation MSE +
forward-kinematics joint-position MSE + temporal jitter), optimizer, training
loop, and metrics VERBATIM by importing `run_one`. The ONLY difference between
this probe and the alt-attn probe is the encoder architecture:

    this script : ViT1DEncoder           (standard 1D ViT, full self-attention)
    altattn     : ViT1DEncoderAltAttn     (alternating frame/global attention)

Everything downstream of the encoder is identical, so a difference in pose error
is attributable to the encoder alone.

Two modes:
  --mode probe     : freeze the encoder, train only the pose decoder.
  --mode finetune  : train the encoder + pose decoder jointly (full parameter).

Metric: mean geodesic rotation error (degrees), per joint and overall.

Usage:
    python scripts/eval_pose_nymeria_ssl.py \
        --runs vit1d_ssl_mae_varlen \
        --mode probe \
        --out-dir data_vis/nymeria_pose_probe_ssl/
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

try:
    import wandb
except ImportError:  # pragma: no cover - optional experiment tracking
    wandb = None

# repo root (for imu_lm.*) and scripts dir (for the sibling alt-attn import)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imu_lm.models.ViT1D.model import ViT1DEncoder
from imu_lm.data.nymeria_real_wrist_full_body_loader_FLU import (
    make_nymeria_real_wrist_full_body_loaders_flu,
)
from imu_lm.utils.helpers import cfg_get

# Reuse the alt-attn probe's training core VERBATIM (RegisterPoseDecoder, the
# rot+FK+jitter pose loss, AdamW config, train loop, evaluate, per-joint metric)
# so the encoder is the ONLY difference between the two probes.
from eval_pose_nymeria_altattn import (
    run_one,
    load_body_model,
    NUM_JOINTS,
    SMPL_JOINT_NAMES,
)

FS = 50  # Hz; window_seconds * FS = samples, // patch_size = sections (Pt)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_encoder(run_dir, step, device):
    """Build the STANDARD ViT1DEncoder from a run checkpoint.

    Mirrors the alt-attn loader except it instantiates ViT1DEncoder instead of
    ViT1DEncoderAltAttn (the one and only intended difference).
    """
    ckpt_path = (os.path.join(run_dir, "checkpoints", f"step_{step:07d}.pt")
                 if step is not None else os.path.join(run_dir, "checkpoints", "best.pt"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    enc = ViT1DEncoder(cfg).to(device)
    enc.load_state_dict(ckpt["model"], strict=True)
    short = os.path.basename(run_dir).replace("vit1d_ssl_", "").replace("vit1d_", "")
    print(f"  {short}  step={ckpt.get('step','?')}  embed_dim={enc.embed_dim}")
    return enc, cfg, short


# ── Plot (multi-run comparison) ───────────────────────────────────────────────

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
    ax.set_title(f"Nymeria 23-Joint Pose — {mode}\nSSL pretrained encoders + register-token decoder",
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
    p.add_argument("--head", choices=["shared", "indep"], default="shared",
                   help="pose decoder variant (matches eval_pose_nymeria_altattn.py)")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--train-dir", default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--val-dir",   default="/mnt/nas2/processed_data_test_body_Ty_rwrist")
    p.add_argument("--max-train-files", type=int, default=200,
                   help="cap Nymeria train sessions (867 total is slow to load)")
    p.add_argument("--max-val-files",   type=int, default=101)
    p.add_argument("--window-seconds", type=float, default=2.56,
                   help="probe window length; overrides cfg windowing.window_seconds "
                        "(default 2.56 s to match eval_pose_nymeria_altattn.py)")
    p.add_argument("--lr",       type=float, default=None,
                   help="default: 3e-4 probe / 1e-4 finetune")
    # The following four match eval_pose_nymeria_altattn.py exactly (loss + optim).
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--joint-pos-weight", type=float, default=10.0,
                   help="weight on the forward-kinematics joint-position MSE term")
    p.add_argument("--jitter-weight", type=float, default=1e-4,
                   help="weight on the temporal-smoothness (jitter) term")
    p.add_argument("--dropout", type=float, default=0.2,
                   help="dropout inside the pose decoder")
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

    # Build Nymeria loaders once from the first run's cfg (same windowing for all).
    first_dir = os.path.join(args.runs_root, args.runs[0])
    _, base_cfg, _ = load_encoder(first_dir, args.step, device)
    # Override the probe window so every run is evaluated at the same window
    # length (default 2.56 s) — matches eval_pose_nymeria_altattn.py.
    base_cfg = dict(base_cfg)
    base_cfg["windowing"] = {**base_cfg.get("windowing", {}),
                             "window_seconds": args.window_seconds}
    print(f"\nLoading Nymeria real-wrist full-body data (window {args.window_seconds}s) ...")
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
        Pt = int(round(args.window_seconds * FS)) // enc.patch_size
        wb_run = None
        if use_wandb:
            wb_run = wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=f"{args.wandb_name_prefix}_{args.mode}_{short}",
                config=vars(args),
                reinit=True,
            )
        patience = 0 if args.no_early_stop else args.patience
        # Identical training to eval_pose_nymeria_altattn.py — encoder is the only
        # difference. run_one builds RegisterPoseDecoder (no registers on the
        # standard encoder) and optimizes rot+FK+jitter loss with AdamW(wd, betas).
        final, per_j = run_one(enc, train_loader, val_loader, device, args.mode, Pt,
                               lr, args.weight_decay, args.epochs, patience, body_model,
                               args.joint_pos_weight, args.jitter_weight, args.dropout, wb_run,
                               head=args.head)
        results[short] = {"geo_err": final["geo_err_deg"], "per_joint": per_j}
        shorts.append(short)
        if wb_run is not None:
            wb_run.log({"best/val_geo_err_deg": final["geo_err_deg"],
                        "best/val_mpjpe_mm": final.get("mpjpe_mm")})
            wb_run.finish()
        mj = f"  MPJPE={final['mpjpe_mm']:.1f}mm" if final.get("mpjpe_mm") is not None else ""
        print(f"  → {short}: {final['geo_err_deg']:.3f}° mean geodesic error{mj}")

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
