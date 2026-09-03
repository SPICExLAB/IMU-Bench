#!/usr/bin/env python3
"""Stage 1 of the RelCon SSL baseline: train the MotifDist distance network
(vendored from baselines/SSL/relcon, Xu et al. ICLR 2025) on the SAME
combined multidata pretraining source used by MAE/FOCAL/SimMTM
(imu_lm/data/combined_mae_multidata_loader.py), so Stage 2
(imu_lm/objectives/relcon_ssl.py::relcon_forward_loss, wired into
run.py's mae_multidata_varlen_altattn_flu objective with ssl_method="relcon")
has a frozen candidate-ranking network trained on the same data distribution.

Standalone (not routed through run.py's ViT1DEncoder-centric dispatch) --
MotifDist is a genuinely different, much smaller network (a cross-attention
reconstruction model, not a ViT1DEncoder), so it doesn't fit that dispatch's
"one shared encoder + objective-specific head" assumption. This mirrors how
the TLIO/AirIO odometry baselines (baselines/odometry/*/train_nymeria_*.py)
are also standalone scripts reusing vendored network code directly.

Usage:
    python train_relcon_motifdist.py --config ../configs/relcon_motifdist_multidata.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _REPO_ROOT)

from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders
from imu_lm.objectives.relcon_ssl import MotifDistNet, motifdist_forward_loss
from imu_lm.utils.helpers import deep_update


def build_cfg(config_path, model_config_path):
    base = yaml.safe_load(open(config_path))
    model = yaml.safe_load(open(model_config_path))
    return deep_update(base, model)


@torch.no_grad()
def evaluate(net, val_loader, device, max_batches=50):
    net.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if batch is None or i >= max_batches:
            if i >= max_batches:
                break
            continue
        x = batch[0].to(device)
        loss, _ = motifdist_forward_loss((x,), net, {})
        losses.append(loss.item())
    net.train()
    return sum(losses) / max(1, len(losses))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=os.path.join(_REPO_ROOT, "configs", "base.yaml"))
    p.add_argument("--model-config", default=os.path.join(_REPO_ROOT, "configs", "relcon_motifdist_multidata.yaml"))
    p.add_argument("--run-name", default="relcon_motifdist_multidata")
    p.add_argument("--runs-root", default=os.path.join(_REPO_ROOT, "runs"))
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--val-every-steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=15)
    p.add_argument("--in-channels", type=int, default=6)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = build_cfg(args.config, args.model_config)
    device = torch.device(args.device)

    run_dir = os.path.join(args.runs_root, args.run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    print("building combined multidata loaders ...")
    loaders = make_combined_mae_multidata_loaders(cfg)
    train_loader, val_loader = loaders["train_loader"], loaders["val_loader"]
    print(f"train batches/epoch={len(train_loader)}  val batches/epoch={len(val_loader)}")

    net = MotifDistNet(in_channels=args.in_channels, embed_dim=args.embed_dim,
                        kernel_size=args.kernel_size, double_receptivefield=2).to(device)
    print(f"MotifDist params: {sum(p.numel() for p in net.parameters()):,}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    step = 0
    best_val, best_state, no_imp = float("inf"), None, 0
    done = False
    while not done:
        for batch in train_loader:
            if batch is None:
                continue
            x = batch[0].to(device)
            opt.zero_grad()
            loss, _ = motifdist_forward_loss((x,), net, {})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            step += 1

            if step % 20 == 0:
                print(f"step={step} train_loss={loss.item():.5f}")

            if step % args.val_every_steps == 0:
                val_loss = evaluate(net, val_loader, device)
                improved = val_loss < best_val - 1e-6
                if improved:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    no_imp = 0
                    torch.save({"model": best_state, "step": step, "val_loss": best_val,
                                "in_channels": args.in_channels, "embed_dim": args.embed_dim,
                                "kernel_size": args.kernel_size},
                               os.path.join(ckpt_dir, "best.pt"))
                else:
                    no_imp += 1
                print(f"[eval] step={step} val_loss={val_loss:.5f} best={best_val:.5f} "
                      f"{'*' if improved else ''}")
                torch.save({"model": net.state_dict(), "step": step, "val_loss": val_loss},
                           os.path.join(ckpt_dir, "latest.pt"))
                if no_imp >= args.patience:
                    print(f"early stopping at step {step}")
                    done = True
                    break

            if step >= args.max_steps:
                done = True
                break

    print(f"\nDONE. best_val_loss={best_val:.5f}  checkpoint={os.path.join(ckpt_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
