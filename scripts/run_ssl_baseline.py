#!/usr/bin/env python3
"""
SSL pretraining launcher for baseline models (SimMTM / LIMU-BERT / FOCAL)
on the OCDetect + realdata mixture (FLU + z-norm).

Reuses the exact OCDetect + parquet data pipeline as the ViT1D
`mae_realdata_ocdetect_flu` objective, but yields [B, T, C] windows that the
sequence-based baselines expect.

Usage:
    python scripts/run_ssl_baseline.py \
        --model   simmtm \
        --config  configs/ssl_simmtm_realdata_ocdetect.yaml \
        --run-name ssl_simmtm_realdata_ocdetect
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import (ConcatDataset, DataLoader, WeightedRandomSampler)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.models.baselines.ssl_baselines import build_ssl_model
from imu_lm.utils.helpers import cfg_get

try:
    import wandb
except ImportError:
    wandb = None


# ── Data: OCDetect + realdata, yielding [B, T, C] ─────────────────────────────

def build_loaders(cfg: dict):
    """Replicates mae_realdata_ocdetect_flu data pipeline but transposes to [T, C]."""
    from imu_lm.data.OCDetect_dataloader import (
        _collect_windows, _list_files, select_ocd_files,
        DATA_DIR_DEFAULT as OCD_DIR, OCDetectDataset, SAMPLE_HZ,
    )
    from imu_lm.data.loaders import (
        WindowDataset, build_session_index, make_splits,
    )

    rdo = cfg_get(cfg, ["objective", "mae_realdata_ocdetect_flu"], {}) or {}
    ocd_dir   = rdo.get("ocdetect_data_dir", OCD_DIR)
    ocd_split = tuple(rdo.get("ocdetect_splits", [0.8, 0.1, 0.1]))
    max_ocd_h = float(rdo.get("max_ocdetect_hours", 500.0))
    max_ocd_files = rdo.get("max_ocdetect_files", None)
    win_sec   = float(cfg_get(cfg, ["windowing", "window_seconds"], 2.56))
    vl_cfg = cfg_get(cfg, ["data", "variable_length"], {}) or {}
    var_len = bool(vl_cfg.get("enabled", False))
    vl_min = int(vl_cfg.get("min_samples", 64))
    vl_max = int(vl_cfg.get("max_samples", round(win_sec * SAMPLE_HZ)))
    window_samples = vl_max if var_len else int(round(win_sec * SAMPLE_HZ))
    hop_samples = max(1, int(round(window_samples * float(
        cfg_get(cfg, ["windowing", "window_hop_ratio"], 0.5)))))

    bs        = int(cfg_get(cfg, ["data", "batch_size"], 256))
    eval_bs   = int(cfg_get(cfg, ["data", "eval_batch_size"], bs))
    nworkers  = int(cfg_get(cfg, ["data", "num_workers"], 4))
    pin       = bool(cfg_get(cfg, ["data", "pin_memory"], True))

    # ── OCDetect ──────────────────────────────────────────────────────────────
    print("Loading OCDetect windows (FLU) ...")
    # Pre-cap files (deterministic) BEFORE windowing to bound RAM, then
    # window-level cap to exactly max_ocdetect_hours — identical to the v0 path.
    _ocd_file_list = select_ocd_files(
        ocd_dir, max_files=int(max_ocd_files) if max_ocd_files else None)
    print(f"  using {len(_ocd_file_list)} OCDetect files")
    X_ocd, y_ocd = _collect_windows(
        _ocd_file_list,
        window_samples=window_samples,
        hop_samples=hop_samples,
        apply_flu_transform=True,
    )
    max_wins = int(max_ocd_h * 3600.0 / win_sec)
    if len(X_ocd) > max_wins:
        idx = np.random.default_rng(42).choice(len(X_ocd), max_wins, replace=False)
        X_ocd, y_ocd = X_ocd[idx], y_ocd[idx]
    elif max_ocd_files:
        print(f"  WARNING: only {len(X_ocd)} windows from {len(_ocd_file_list)} files "
              f"(< {max_wins}-window cap) — raise max_ocdetect_files")
    n_tr = int(ocd_split[0] * len(X_ocd))
    n_va = int(ocd_split[1] * len(X_ocd))
    ocd_train = OCDetectDataset(X_ocd[:n_tr],            y_ocd[:n_tr],            cfg=cfg)
    ocd_val   = OCDetectDataset(X_ocd[n_tr:n_tr+n_va],   y_ocd[n_tr:n_tr+n_va],   cfg=cfg)
    print(f"  OCDetect: train={len(ocd_train)} val={len(ocd_val)}")

    # ── Realdata parquet ──────────────────────────────────────────────────────
    print("Loading realdata parquet windows ...")
    parquet = cfg_get(cfg, ["paths", "dataset_path"])
    train_on = cfg_get(cfg, ["data", "train_on_datasets"], None)
    si = build_session_index(parquet, cfg,
                             dataset_filter=list(train_on) if train_on else None)
    splits = make_splits(si, cfg)
    parq_train = WindowDataset(parquet, si, splits["train_keys"], cfg, split_name="train")
    parq_val   = WindowDataset(parquet, si, splits["val_keys"],   cfg, split_name="val")
    print(f"  Realdata: train={len(parq_train)} val={len(parq_val)}")

    # ── Wrap so every item is x transposed to [T, C] ──────────────────────────
    class _TC(torch.utils.data.Dataset):
        def __init__(self, ds): self.ds = ds
        def __len__(self):      return len(self.ds)
        def __getitem__(self, i):
            item = self.ds[i]
            if item is None:
                return None
            x = item[0] if isinstance(item, (tuple, list)) else item   # [C, T]
            return x.transpose(0, 1).contiguous()                      # [T, C]

    train_ds = ConcatDataset([_TC(parq_train), _TC(ocd_train)])
    val_ds   = ConcatDataset([_TC(parq_val),   _TC(ocd_val)])

    # ── Sampler ───────────────────────────────────────────────────────────────
    n_p, n_o = len(parq_train), len(ocd_train)
    balanced = bool(rdo.get("balanced_sampling", True))
    if balanced:
        weights = [1.0 / max(n_p, 1)] * n_p + [1.0 / max(n_o, 1)] * n_o
        train_sampler = WeightedRandomSampler(
            weights, num_samples=2 * min(n_p, n_o), replacement=True)
        train_shuffle = False
        print(f"  sampler: balanced 50/50 (epoch={2 * min(n_p, n_o)} windows)")
    else:
        train_sampler = None
        train_shuffle = True
        ratio = n_o / max(n_p, 1)
        print(f"  sampler: natural ratio (realdata={n_p} ocdetect={n_o} "
              f"≈ 1:{ratio:.1f}, epoch={n_p + n_o} windows)")

    def collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return torch.stack(batch, 0)              # [B, T, C]

    if var_len:
        step = int(vl_cfg.get("sample_multiple", 8))
        lengths = list(range(vl_min, vl_max + 1, step))

        def collate_train(batch):
            x = collate(batch)
            if x is None:
                return None
            L = int(np.random.choice(lengths))
            if L < x.shape[1]:
                off = int(np.random.randint(0, x.shape[1] - L + 1))
                x = x[:, off:off + L, :]
            return x.contiguous()

        def collate_val(batch):
            x = collate(batch)
            if x is None:
                return None
            return x[:, :vl_max, :].contiguous()
    else:
        collate_train = collate_val = collate

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler,
                              shuffle=train_shuffle,
                              num_workers=nworkers, pin_memory=pin,
                              drop_last=True, collate_fn=collate_train,
                              persistent_workers=nworkers > 0)
    val_loader   = DataLoader(val_ds, batch_size=eval_bs, shuffle=False,
                              num_workers=nworkers, pin_memory=pin,
                              drop_last=False, collate_fn=collate_val,
                              persistent_workers=nworkers > 0)
    return train_loader, val_loader


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, cfg, run_dir, device, use_wandb):
    tr  = cfg_get(cfg, ["trainer"], {}) or {}
    lr  = float(cfg_get(cfg, ["trainer", "optim", "lr"], 1e-4))
    wd  = float(cfg_get(cfg, ["trainer", "optim", "weight_decay"], 1e-4))
    max_steps    = int(tr.get("max_steps", 100000))
    val_every    = int(tr.get("val_every_steps", 1000))
    ckpt_every   = int(cfg_get(cfg, ["logging", "ckpt_every_steps"], 5000))
    log_every    = int(cfg_get(cfg, ["logging", "log_every_steps"], 20))
    es_cfg       = cfg_get(cfg, ["trainer", "early_stopping"], {}) or {}
    patience     = int(es_cfg.get("patience", 10))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                            betas=(0.9, 0.95))
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val   = float("inf")
    no_improve = 0
    step       = 0
    stop       = False
    t0 = time.time()

    while step < max_steps and not stop:
        model.train()
        for batch in train_loader:
            if batch is None:
                continue
            opt.zero_grad()
            loss, logs = model.forward_loss(batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % log_every == 0:
                rate = step / (time.time() - t0)
                print(f"step {step:>7} | loss {logs['loss']:.4f} | {rate:.1f} it/s")
                if use_wandb:
                    wandb.log({f"train/{k}": v for k, v in logs.items()}, step=step)

            if step % val_every == 0:
                vloss, vlogs = evaluate(model, val_loader, device)
                print(f"  [val] step {step} | loss {vloss:.4f}")
                if use_wandb:
                    wandb.log({f"val/{k}": v for k, v in vlogs.items()}, step=step)
                if vloss < best_val - 1e-4:
                    best_val   = vloss
                    no_improve = 0
                    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                               os.path.join(ckpt_dir, "best.pt"))
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at step {step}")
                    stop = True
                    break

            if step % ckpt_every == 0:
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                           os.path.join(ckpt_dir, f"step_{step:07d}.pt"))
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                           os.path.join(ckpt_dir, "latest.pt"))

            if step >= max_steps:
                stop = True
                break

    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
               os.path.join(ckpt_dir, "latest.pt"))
    print(f"Done. best_val={best_val:.4f} at step≤{step}")


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    agg, n = {}, 0
    for batch in val_loader:
        if batch is None:
            continue
        _, logs = model.forward_loss(batch, device)
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    avg = {k: v / max(n, 1) for k, v in agg.items()}
    return avg.get("loss", 0.0), avg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",    required=True,
                    choices=["simmtm", "limu_bert", "focal", "moment"])
    ap.add_argument("--config",   required=True, help="Model+data config YAML")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data", {})["sensor_columns"] = \
        ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"ssl_{args.model}"
    run_dir  = os.path.join(cfg_get(cfg, ["paths", "runs_root"], args.runs_root), run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    # Model
    model_cfg = cfg_get(cfg, ["model"], {}) or {}
    model = build_ssl_model(args.model, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  ({n_params/1e6:.1f}M params)  → {run_dir}")

    # wandb
    use_wandb = (wandb is not None) and (not args.no_wandb)
    if use_wandb:
        wandb.init(
            project=cfg_get(cfg, ["wandb", "project"], "TransfHAR_project"),
            entity=cfg_get(cfg, ["wandb", "entity"], None),
            name=run_name, config=cfg,
        )

    train_loader, val_loader = build_loaders(cfg)
    train(model, train_loader, val_loader, cfg, run_dir, device, use_wandb)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
