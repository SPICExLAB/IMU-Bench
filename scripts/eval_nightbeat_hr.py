#!/usr/bin/env python3
"""Nightbeat-DB evaluation: heart rate from an overnight wrist accelerometer.

Task
----
Regress instantaneous heart rate (bpm) from a window of wrist accelerometer,
using the ballistocardiogram — the faint body-recoil each heartbeat imparts to
the wrist. Reference HR comes from a chest ECG. Dataset and task from
Moebus et al., IEEE BHI 2024 (https://github.com/eth-siplab/Nightbeat).

The gyro mask token (``--gyro-mode``)
-------------------------------------
Nightbeat-DB is accelerometer-only, but the multidata MAE encoder was
pretrained on 6 channels. Rather than invent a gyroscope, we reuse the exact
convention pretraining already established for its own accelerometer-only
sources (capture24, NHANES — see ``masked_1d_freq_channel_mask.py``): the 3
gyro patch positions are forced into the encoder's mask, so their embeddings
are replaced by the **learned mask token**. That is the state the encoder saw
for capture24/NHANES gyro on every pretraining step, so it is in-distribution.

  ``--gyro-mode mask``  gyro patches -> pretraining mask token   [default]
  ``--gyro-mode zero``  gyro channels fed as literal zeros (the naive
                        alternative; the encoder reads them as a gyroscope
                        lying perfectly still). Kept so the cost of getting
                        this wrong is measurable rather than assumed.
  ``--gyro-mode acc3``  3-channel input, for an accelerometer-only checkpoint
                        (e.g. vit1d_ssl_mae_varlen_acconly_multidata_flu).

Protocol
--------
Participant-grouped k-fold: every participant appears in exactly one test
fold, so pooling folds yields one held-out prediction per window. Grouping is
mandatory — HR is strongly subject-specific, and the paper's own
per-participant-mean baseline already reaches 3.16 bpm MAE, so any window-level
split leaks the answer outright.

Modes: ``probe`` (frozen encoder + head), ``finetune`` (encoder + head trained
end to end), ``scratch`` (same, random init) — the same three-row structure the
PADS benchmark in this repo uses.

Baselines reported alongside, because they are what make the numbers readable:
  * ``global_mean``  — predict the training folds' mean HR. Beating this is the
                       minimum bar for "learned anything".
  * ``subject_mean`` — predict each test participant's OWN mean HR. This is the
                       paper's "Mean-Pred." row (3.16 bpm MAE). It is an
                       ORACLE: it uses test-participant labels, so a model is
                       not required to beat it, but a model that does not is
                       tracking sleep-time HR level rather than the BCG.

Published reference numbers on Nightbeat-DB (Moebus et al., Table 1), for
context only — they come from dedicated signal-processing methods on the full
100 Hz signal, not from a learned probe on 50 Hz windows:
  Nightbeat 0.88 MAE | Mean-Pred. 3.16 | PWR 8.83 | Jerks 13.23 | BioInsights 17.47

Usage
-----
    python scripts/eval_nightbeat_hr.py \
        --config configs/base_nightbeat.yaml \
        --model-config configs/vit1d_nightbeat_hr_probe_mae_multidata_flu.yaml \
        --gyro-mode mask --mode probe --folds 5 \
        --out runs/nightbeat_hr_probe_mae/results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.nightbeat_loader import (
    concat_datasets,
    make_nightbeat_datasets,
    participant_folds,
)
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logger = logging.getLogger("eval_nightbeat")

# Published Nightbeat-DB results (Moebus et al. 2024, Table 1). Context only.
PAPER_REFERENCE = {
    "Nightbeat (paper)":   {"mae": 0.88,  "rmse": 2.24,  "cor": 0.81},
    "Mean-Pred. (paper)":  {"mae": 3.16,  "rmse": 4.07,  "cor": 0.00},
    "PWR (paper)":         {"mae": 8.83,  "rmse": 10.91, "cor": 0.22},
    "Jerks (paper)":       {"mae": 13.23, "rmse": 20.63, "cor": 0.04},
    "BioInsights (paper)": {"mae": 17.47, "rmse": 21.38, "cor": 0.00},
}


# ─────────────────────────── encoder + masked encoding ───────────────────────

def build_encoder(cfg: Any, pretrained: bool = True) -> nn.Module:
    """Build the ViT1D encoder named by ``cfg`` and optionally load its ckpt."""
    from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn
    from imu_lm.models.ViT1D.run import _load_encoder_state_dict

    if bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False)):
        encoder = ViT1DEncoderAltAttn(cfg)
    else:
        encoder = ViT1DEncoder(cfg)

    if not pretrained:
        logger.info("[encoder] random init (scratch)")
        return encoder

    probe_cfg = cfg_get(cfg, ["objective", "nightbeat_hr"], {}) or {}
    run_name = probe_cfg.get("pretrained_run")
    if not run_name:
        raise ValueError("objective.nightbeat_hr.pretrained_run must be set for "
                         "probe/finetune modes (or use --mode scratch)")
    ckpt_name = probe_cfg.get("pretrained_ckpt", "best")
    runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
    ckpt_path = os.path.join(runs_root, run_name, "checkpoints", f"{ckpt_name}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Pretrain checkpoints written by this repo put the encoder weights at
    # top-level ``state["model"]`` (the MAE decoder lives under "decoder").
    # Some downstream checkpoints nest them under ``state["encoder"]["model"]``.
    # _load_encoder_state_dict expects the dict that holds "model", so hand it
    # whichever of the two applies.
    enc_state = state["encoder"] if "encoder" in state else state
    if "model" not in enc_state:
        raise KeyError(f"{ckpt_path}: no 'model' weights found (keys: {list(state)[:8]})")
    _load_encoder_state_dict(encoder, enc_state)
    logger.info("[encoder] loaded pretrained weights from %s", ckpt_path)
    return encoder


def encode_batch(encoder: nn.Module, x: torch.Tensor, gyro_mode: str) -> torch.Tensor:
    """[B, C, T] -> [B, D] pooled embedding, honouring the gyro convention.

    Token layout is channel-major (``PatchEmbed1D`` concatenates per channel),
    so channel ``c``'s patches occupy indices ``c*Pt : (c+1)*Pt`` and the gyro
    block is ``3*Pt : 6*Pt``.
    """
    if gyro_mode in ("zero", "acc3"):
        # No mask: encoder's own pooled/CLS output, gyro (if present) read as zeros.
        return encoder(x)

    if gyro_mode != "mask":
        raise ValueError(f"unknown gyro_mode: {gyro_mode}")

    B, C, T = x.shape
    if C != 6:
        raise ValueError(f"gyro_mode='mask' expects 6 channels, got {C}")
    Pt = T // encoder.patch_size
    mask = torch.zeros(B, C * Pt, dtype=torch.bool, device=x.device)
    mask[:, 3 * Pt :] = True   # every gyro patch position

    if getattr(encoder, "_use_bert_mask", False):
        if hasattr(encoder, "_backbone"):
            # ViT1DEncoderAltAttn: masked positions get self.mask_token, and the
            # global CLS token is the pooled representation used for probing.
            _tokens, _reg, cls, _Pt = encoder._backbone(x, mask=mask)
            return cls
        # ViT1DEncoder with bert_masking: masked positions get global_mask_token,
        # all N positions returned. Pool over the real (acc) tokens only.
        tokens = encoder(x, mask=mask)                 # [B, N, D]
        return tokens[:, : 3 * Pt].mean(dim=1)

    # Standard MAE encoder: masked tokens are dropped, so what comes back is
    # already the acc-only visible set.
    tokens = encoder(x, mask=mask)                     # [B, N_keep, D]
    if isinstance(tokens, tuple):
        tokens = tokens[0]
    return tokens.mean(dim=1)


@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    gyro_mode: str,
    device: torch.device,
    amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    feats, ys = [], []
    for x, y, _cvm in loader:
        x = x.to(device, non_blocking=True)
        if gyro_mode == "acc3":
            x = x[:, :3]
        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            z = encode_batch(encoder, x, gyro_mode)
        feats.append(z.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


# ───────────────────────────────── metrics ───────────────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(y: np.ndarray, yhat: np.ndarray, pids: np.ndarray) -> Dict[str, Any]:
    """Pooled and per-participant MAE / RMSE / Pearson r.

    The paper's per-participant correlations are the headline claim, and a
    pooled correlation can look strong purely from between-subject HR spread,
    so both are reported.
    """
    err = yhat - y
    out: Dict[str, Any] = {
        "n": int(y.size),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "cor": _pearson(y, yhat),
    }
    per: Dict[str, Dict[str, float]] = {}
    for p in np.unique(pids):
        m = pids == p
        e = yhat[m] - y[m]
        per[str(int(p))] = {
            "n": int(m.sum()),
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "cor": _pearson(y[m], yhat[m]),
            "hr_mean": float(y[m].mean()),
            "hr_std": float(y[m].std()),
        }
    out["per_participant"] = per
    vals = list(per.values())
    out["mae_participant_mean"] = float(np.mean([v["mae"] for v in vals]))
    out["rmse_participant_mean"] = float(np.mean([v["rmse"] for v in vals]))
    cors = [v["cor"] for v in vals if np.isfinite(v["cor"])]
    out["cor_participant_mean"] = float(np.mean(cors)) if cors else float("nan")
    return out


# ────────────────────────────────── heads ────────────────────────────────────

def ridge_fit_predict(
    Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray,
    alphas: Tuple[float, ...] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4),
    val_frac: float = 0.15,
    seed: int = 0,
) -> np.ndarray:
    """Closed-form ridge with alpha chosen on a held-out slice of train.

    Features are standardized and the target is centered, so the intercept is
    exact and alpha is comparable across folds.
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(Ztr.shape[0])
    n_val = max(1, int(val_frac * idx.size))
    va, tr = idx[:n_val], idx[n_val:]

    mu, sd = Ztr[tr].mean(0), Ztr[tr].std(0) + 1e-6
    A = (Ztr[tr] - mu) / sd
    b = ytr[tr] - ytr[tr].mean()
    G = A.T @ A
    rhs = A.T @ b
    eye = np.eye(G.shape[0], dtype=G.dtype)

    Av = (Ztr[va] - mu) / sd
    best_alpha, best_mae = None, np.inf
    for alpha in alphas:
        w = np.linalg.solve(G + alpha * eye, rhs)
        pred = Av @ w + ytr[tr].mean()
        mae = float(np.abs(pred - ytr[va]).mean())
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha

    # Refit on all of train with the selected alpha.
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    A = (Ztr - mu) / sd
    ymean = ytr.mean()
    w = np.linalg.solve(A.T @ A + best_alpha * np.eye(A.shape[1]), A.T @ (ytr - ymean))
    logger.info("[ridge] alpha=%g (val MAE %.3f)", best_alpha, best_mae)
    return ((Zte - mu) / sd) @ w + ymean


class MLPHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def train_mlp_on_features(
    Ztr, ytr, Zte, device, epochs=100, lr=1e-3, wd=1e-4,
    batch_size=1024, val_frac=0.15, seed=0, patience=15,
) -> np.ndarray:
    """Train an MLP head on frozen features; early-stop on val MAE."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(Ztr.shape[0])
    n_val = max(1, int(val_frac * idx.size))
    va, tr = idx[:n_val], idx[n_val:]

    mu, sd = Ztr[tr].mean(0), Ztr[tr].std(0) + 1e-6
    ymu, ysd = ytr[tr].mean(), ytr[tr].std() + 1e-6

    def prep(Z):
        return torch.from_numpy(((Z - mu) / sd).astype(np.float32))

    Xtr, Xva, Xte = prep(Ztr[tr]), prep(Ztr[va]), prep(Zte)
    Ytr = torch.from_numpy(((ytr[tr] - ymu) / ysd).astype(np.float32))
    yva = ytr[va]

    head = MLPHead(Ztr.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_state, best_mae, bad = None, np.inf, 0
    n = Xtr.shape[0]
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            j = perm[i : i + batch_size]
            xb, yb = Xtr[j].to(device), Ytr[j].to(device)
            loss = nn.functional.smooth_l1_loss(head(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            pv = head(Xva.to(device)).cpu().numpy() * ysd + ymu
        mae = float(np.abs(pv - yva).mean())
        if mae < best_mae - 1e-4:
            best_mae, bad = mae, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred = head(Xte.to(device)).cpu().numpy() * ysd + ymu
    logger.info("[mlp] best val MAE %.3f bpm", best_mae)
    return pred


# ─────────────────────────── end-to-end train (finetune/scratch) ─────────────

def train_end_to_end(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: DataLoader,
    gyro_mode: str,
    device: torch.device,
    cfg: Any,
) -> np.ndarray:
    """Train encoder + head jointly, early-stopping on val MAE. Returns test preds."""
    tr_cfg = cfg_get(cfg, ["trainer"], {}) or {}
    epochs = int(tr_cfg.get("epochs", 20))
    lr = float(cfg_get(cfg, ["trainer", "optim", "lr"], 1e-4))
    wd = float(cfg_get(cfg, ["trainer", "optim", "weight_decay"], 0.05))
    patience = int(cfg_get(cfg, ["trainer", "early_stopping", "patience"], 5))
    amp = bool(tr_cfg.get("amp", True)) and device.type == "cuda"
    grad_clip = float(tr_cfg.get("grad_clip_norm", 1.0))

    head_hidden = int(tr_cfg.get("head_hidden", 256))
    loss_name = str(tr_cfg.get("loss", "smooth_l1"))
    warmup_frac = float(tr_cfg.get("warmup_frac", 0.0))
    llrd = float(tr_cfg.get("layer_lr_decay", 1.0))   # 1.0 = off

    head = MLPHead(encoder.embed_dim, hidden=head_hidden).to(device)
    encoder = encoder.to(device)
    params = list(encoder.parameters()) + list(head.parameters())

    if llrd < 1.0:
        # Layer-wise LR decay: deeper blocks move faster than early ones, the
        # standard recipe for finetuning a pretrained ViT without washing out
        # the low-level features. AltAttn alternates frame/global blocks, so
        # depth is indexed by the pair number.
        groups, seen = [], set()
        n_pairs = int(getattr(encoder, "n_pairs", 6))
        def depth_of(name):
            m = re.search(r"(?:frame|global)_blocks\.(\d+)\.", name)
            if m:
                return int(m.group(1)) + 1
            return 0 if ("patch_embed" in name or "pos_embed" in name
                         or "token" in name) else n_pairs + 1
        buckets = {}
        for n, p in encoder.named_parameters():
            buckets.setdefault(depth_of(n), []).append(p)
            seen.add(id(p))
        max_d = n_pairs + 1
        for d, ps in sorted(buckets.items()):
            groups.append({"params": ps, "lr": lr * (llrd ** (max_d - d))})
        groups.append({"params": list(head.parameters()), "lr": lr})
        opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
        logger.info("[e2e] layer-wise lr decay %.2f over %d depths (top lr %.2g, "
                    "bottom lr %.2g)", llrd, len(buckets), lr, lr * (llrd ** max_d))
    else:
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    if warmup_frac > 0.0:
        n_warm = max(1, int(round(warmup_frac * epochs)))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt,
            [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=n_warm),
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - n_warm))],
            milestones=[n_warm])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    loss_fn = {
        "smooth_l1": nn.functional.smooth_l1_loss,
        "l1": nn.functional.l1_loss,       # matches the MAE metric directly
        "mse": nn.functional.mse_loss,
    }[loss_name]

    # Standardize the target; raw bpm (~40-100) makes the loss scale awkward.
    ytr_all = train_loader.dataset.targets()
    ymu, ysd = float(ytr_all.mean()), float(ytr_all.std() + 1e-6)

    def run_eval(loader):
        encoder.eval(); head.eval()
        preds, ys = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device, non_blocking=True)
                if gyro_mode == "acc3":
                    x = x[:, :3]
                with torch.autocast("cuda", enabled=amp):
                    p = head(encode_batch(encoder, x, gyro_mode))
                preds.append(p.float().cpu().numpy() * ysd + ymu)
                ys.append(y.numpy())
        return np.concatenate(preds), np.concatenate(ys)

    best_state, best_mae, bad = None, np.inf, 0
    for ep in range(epochs):
        encoder.train(); head.train()
        t0, nb, run_loss = time.time(), 0, 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = ((y.to(device) - ymu) / ysd)
            if gyro_mode == "acc3":
                x = x[:, :3]
            with torch.autocast("cuda", enabled=amp):
                loss = loss_fn(head(encode_batch(encoder, x, gyro_mode)), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.detach()); nb += 1
        sched.step()

        msg = f"[e2e] epoch {ep+1}/{epochs} loss={run_loss/max(nb,1):.4f} ({time.time()-t0:.0f}s)"
        if val_loader is not None:
            pv, yv = run_eval(val_loader)
            mae = float(np.abs(pv - yv).mean())
            msg += f" val_mae={mae:.3f}"
            if mae < best_mae - 1e-3:
                best_mae, bad = mae, 0
                best_state = {
                    "enc": {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()},
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                }
            else:
                bad += 1
        logger.info(msg)
        if val_loader is not None and bad >= patience:
            logger.info("[e2e] early stop at epoch %d (best val MAE %.3f)", ep + 1, best_mae)
            break

    if best_state is not None:
        encoder.load_state_dict(best_state["enc"])
        head.load_state_dict(best_state["head"])
    preds, _ = run_eval(test_loader)
    return preds, best_mae


# ────────────────────────────────── main ─────────────────────────────────────

def make_loader(ds, batch_size, num_workers, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base_nightbeat.yaml")
    ap.add_argument("--model-config", default=None,
                    help="model/objective YAML merged over --config")
    ap.add_argument("--gyro-mode", choices=["mask", "zero", "acc3"], default="mask")
    ap.add_argument("--mode", choices=["probe", "finetune", "scratch"], default="probe")
    ap.add_argument("--head", choices=["ridge", "mlp"], default="ridge",
                    help="probe head only; finetune/scratch always use the MLP head")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default=None, help="prepared Nightbeat root")
    ap.add_argument("--pids", nargs="*", type=int, default=None)
    ap.add_argument("--max-windows-per-pid", type=int, default=None,
                    help="uniformly subsample windows per participant (speed)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--window-seconds", type=float, default=None,
                    help="override windowing.window_seconds. NOTE: the MAE "
                         "pretrain capped windows at 512 samples (variable_length"
                         ".max_samples), i.e. 32 time patches, so a frozen probe "
                         "on a longer window indexes time-positional-embedding "
                         "rows pretraining never exercised. Longer windows are "
                         "fine for finetune/scratch, which train the pos-embed.")
    ap.add_argument("--out", default=None, help="write results JSON here")
    # Finetune/scratch tuning knobs. Select configs by the reported val MAE
    # (held-out TRAIN participants) -- picking by test MAE would overfit the
    # very folds the headline number is computed on.
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--loss", choices=["smooth_l1", "l1", "mse"], default=None)
    ap.add_argument("--warmup-frac", type=float, default=None,
                    help="fraction of epochs spent in linear LR warmup")
    ap.add_argument("--layer-lr-decay", type=float, default=None,
                    help="layer-wise LR decay (<1 slows early blocks); 1.0 = off")
    ap.add_argument("--head-hidden", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.model_config:
        cfg = deep_update(cfg, load_yaml(args.model_config))
    tr = cfg.setdefault("trainer", {})
    for key, val in (("lr", args.lr), ("weight_decay", args.weight_decay)):
        if val is not None:
            tr.setdefault("optim", {})[key] = val
    for key, val in (("epochs", args.epochs), ("loss", args.loss),
                     ("warmup_frac", args.warmup_frac),
                     ("layer_lr_decay", args.layer_lr_decay),
                     ("head_hidden", args.head_hidden)):
        if val is not None:
            tr[key] = val
    if args.patience is not None:
        tr.setdefault("early_stopping", {})["patience"] = args.patience
    if args.batch_size is not None:
        cfg.setdefault("data", {})["batch_size"] = args.batch_size

    if args.window_seconds is not None:
        cfg.setdefault("windowing", {})["window_seconds"] = args.window_seconds
        max_pretrain_patches = 32   # variable_length max 512 samples / patch 16
        n_patches = int(round(args.window_seconds * 50.0)) // int(
            cfg_get(cfg, ["vit1d", "patch_size"], 16))
        if args.mode == "probe" and n_patches > max_pretrain_patches:
            logger.warning(
                "[eval] window_seconds=%.2f -> %d time patches, but the MAE "
                "pretrain never fed more than %d (variable_length.max_samples=512"
                ", patch 16). Time-positional-embedding rows %d+ were therefore "
                "never exercised, so treat this frozen-probe row as confounded "
                "by an untrained pos-embed rather than as a clean read of the "
                "representation. Prefer --mode finetune for long windows.",
                args.window_seconds, n_patches, max_pretrain_patches,
                max_pretrain_patches)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("[eval] device=%s mode=%s gyro_mode=%s head=%s folds=%d",
                device, args.mode, args.gyro_mode, args.head, args.folds)

    if args.gyro_mode == "acc3":
        # A 6-channel patch embed with shared_kernel: acc_gyro splits at C//2,
        # so feeding it 3 channels would route ch1/ch2 through proj_gyro. acc3
        # is only valid against a genuinely 3-channel (acc-only) checkpoint.
        n_ch = len(cfg_get(cfg, ["data", "sensor_columns"], []) or [])
        if n_ch != 3:
            raise SystemExit(
                f"--gyro-mode acc3 needs a 3-channel config (data.sensor_columns "
                f"has {n_ch}); point --model-config at an accelerometer-only "
                f"checkpoint such as vit1d_ssl_mae_varlen_acconly_multidata_flu"
            )

    # Always window as 6 channels; acc3 slices back to 3 at encode time.
    datasets, _traces, T = make_nightbeat_datasets(
        cfg, root=args.root, pids=args.pids, pad_to_6ch=True,
    )
    if not datasets:
        raise SystemExit("no participants loaded")

    if args.max_windows_per_pid:
        for pid, ds in datasets.items():
            if len(ds.specs) > args.max_windows_per_pid:
                sel = np.linspace(0, len(ds.specs) - 1, args.max_windows_per_pid).astype(int)
                ds.specs = [ds.specs[i] for i in sel]

    all_pids = sorted(datasets.keys())
    n_win = sum(len(datasets[p].specs) for p in all_pids)
    logger.info("[eval] %d participants, %d windows, T=%d (%.2fs @ 50Hz)",
                len(all_pids), n_win, T, T / 50.0)

    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    bs = int(data_cfg.get("batch_size", 256))
    nw = int(data_cfg.get("num_workers", 4))

    folds = participant_folds(all_pids, args.folds, seed=args.seed)
    logger.info("[eval] participant folds: %s", folds)

    # For a frozen encoder the features do not depend on the fold, so extract
    # them once for every participant and let the folds re-slice the cache.
    # Re-extracting per fold would repeat ~(folds-1)/folds of the forward
    # passes for nothing -- the dominant cost at 100k+ windows.
    feat_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if args.mode == "probe":
        encoder = build_encoder(cfg, pretrained=True).to(device)
        encoder.eval()
        for prm in encoder.parameters():
            prm.requires_grad_(False)
        t0 = time.time()
        for i, pid in enumerate(all_pids):
            Z, y = extract_embeddings(
                encoder, make_loader(datasets[pid], bs, nw), args.gyro_mode, device)
            feat_cache[pid] = (Z, y)
            logger.info("[embed] pid %02d (%d/%d): %s  [%.0fs elapsed]",
                        pid, i + 1, len(all_pids), Z.shape, time.time() - t0)
        del encoder
        logger.info("[embed] done in %.0fs", time.time() - t0)

    def cached(pids_subset):
        """Assemble (features, targets, participant-ids) from the frozen cache.

        Order is the shuffle=False DataLoader order, i.e. spec order, so the
        three arrays stay row-aligned.
        """
        Z = np.concatenate([feat_cache[q][0] for q in pids_subset])
        y = np.concatenate([feat_cache[q][1] for q in pids_subset])
        pid_col = np.concatenate(
            [np.full(feat_cache[q][1].size, q, dtype=np.int64) for q in pids_subset])
        return Z, y, pid_col

    y_all: List[np.ndarray] = []
    p_all: List[np.ndarray] = []
    pred_model: List[np.ndarray] = []
    pred_gmean: List[np.ndarray] = []
    val_maes: List[float] = []
    pred_smean: List[np.ndarray] = []

    for k, test_pids in enumerate(folds):
        train_pids = [p for p in all_pids if p not in test_pids]
        if not train_pids:
            continue
        logger.info("[fold %d/%d] train=%d participants, test=%s",
                    k + 1, args.folds, len(train_pids), test_pids)

        if args.mode == "probe":
            Ztr, ytr, _ = cached(train_pids)
            Zte, y_te, p_te = cached(test_pids)
            y_tr = ytr
            logger.info("[fold %d] features train=%s test=%s", k + 1, Ztr.shape, Zte.shape)
            if args.head == "ridge":
                pred = ridge_fit_predict(Ztr, ytr, Zte, seed=args.seed)
            else:
                pred = train_mlp_on_features(Ztr, ytr, Zte, device, seed=args.seed)
        else:
            te_ds = concat_datasets(datasets, test_pids)
            y_te = te_ds.targets()
            p_te = te_ds.pids()
            # finetune / scratch: hold out a few train participants for early stop.
            rng = np.random.RandomState(args.seed + k)
            tp = list(train_pids)
            rng.shuffle(tp)
            # Hold out a slice of TRAIN participants for early stopping. Cap it
            # so it can never consume the whole training set -- with very few
            # participants (small --pids runs) an uncapped n_val leaves
            # fit_pids empty, which otherwise surfaces as an opaque
            # "NoneType has no len()" from inside DataLoader.
            n_val = max(1, min(len(tp) - 1, len(tp) // 8))
            val_pids, fit_pids = sorted(tp[:n_val]), sorted(tp[n_val:])
            if not fit_pids:
                raise SystemExit(
                    f"fold {k+1}: only {len(tp)} train participant(s) -- too few to "
                    f"split into fit/val for {args.mode}. Use more participants or "
                    f"fewer folds.")
            fit_ds = concat_datasets(datasets, fit_pids)
            val_ds = concat_datasets(datasets, val_pids)
            encoder = build_encoder(cfg, pretrained=(args.mode == "finetune"))
            pred, fold_val_mae = train_end_to_end(
                encoder,
                make_loader(fit_ds, bs, nw, shuffle=True),
                make_loader(val_ds, bs, nw),
                make_loader(te_ds, bs, nw),
                args.gyro_mode, device, cfg,
            )
            val_maes.append(fold_val_mae)
            y_tr = fit_ds.targets()
            del encoder

        # Baselines on the same test windows.
        gmean = np.full_like(y_te, float(y_tr.mean()))
        smean = np.empty_like(y_te)
        for p in np.unique(p_te):
            m = p_te == p
            smean[m] = y_te[m].mean()   # ORACLE: uses test-participant labels

        y_all.append(y_te); p_all.append(p_te)
        pred_model.append(pred.astype(np.float32))
        pred_gmean.append(gmean); pred_smean.append(smean)

        logger.info("[fold %d] model MAE=%.3f | global_mean=%.3f | subject_mean(oracle)=%.3f",
                    k + 1,
                    float(np.abs(pred - y_te).mean()),
                    float(np.abs(gmean - y_te).mean()),
                    float(np.abs(smean - y_te).mean()))

    y = np.concatenate(y_all); pids = np.concatenate(p_all)
    results = {
        "config": {
            "config": args.config, "model_config": args.model_config,
            "mode": args.mode, "gyro_mode": args.gyro_mode, "head": args.head,
            "folds": args.folds, "seed": args.seed,
            "window_T": int(T), "window_seconds": T / 50.0,
            "n_participants": len(all_pids), "n_windows": int(y.size),
            "participants": all_pids,
            "hparams": {
                "lr": cfg_get(cfg, ["trainer", "optim", "lr"], None),
                "weight_decay": cfg_get(cfg, ["trainer", "optim", "weight_decay"], None),
                "epochs": cfg_get(cfg, ["trainer", "epochs"], None),
                "patience": cfg_get(cfg, ["trainer", "early_stopping", "patience"], None),
                "loss": cfg_get(cfg, ["trainer", "loss"], "smooth_l1"),
                "warmup_frac": cfg_get(cfg, ["trainer", "warmup_frac"], 0.0),
                "layer_lr_decay": cfg_get(cfg, ["trainer", "layer_lr_decay"], 1.0),
                "head_hidden": cfg_get(cfg, ["trainer", "head_hidden"], 256),
                "batch_size": cfg_get(cfg, ["data", "batch_size"], None),
            },
        },
        "val_mae_per_fold": val_maes,
        "val_mae_mean": (float(np.mean(val_maes)) if val_maes else None),
        "hr_target_stats": {
            "mean": float(y.mean()), "std": float(y.std()),
            "min": float(y.min()), "max": float(y.max()),
        },
        "model": compute_metrics(y, np.concatenate(pred_model), pids),
        "baseline_global_mean": compute_metrics(y, np.concatenate(pred_gmean), pids),
        "baseline_subject_mean_ORACLE": compute_metrics(y, np.concatenate(pred_smean), pids),
        "paper_reference_Moebus2024": PAPER_REFERENCE,
    }

    name = f"{args.mode}/{args.gyro_mode}" + (f"/{args.head}" if args.mode == "probe" else "")
    print("\n" + "=" * 72)
    print(f"Nightbeat-DB HR regression — {name}")
    print(f"{len(all_pids)} participants, {y.size} windows of {T/50.0:.2f}s, "
          f"HR {y.mean():.1f} +- {y.std():.1f} bpm")
    print("=" * 72)
    print(f"{'row':<34}{'MAE':>8}{'RMSE':>8}{'Cor':>8}{'MAE/subj':>10}{'Cor/subj':>10}")
    print("-" * 72)
    for label, key in (
        (f"Ours ({name})", "model"),
        ("Base: global mean", "baseline_global_mean"),
        ("Base: subject mean [ORACLE]", "baseline_subject_mean_ORACLE"),
    ):
        m = results[key]
        print(f"{label:<34}{m['mae']:>8.2f}{m['rmse']:>8.2f}{m['cor']:>8.2f}"
              f"{m['mae_participant_mean']:>10.2f}{m['cor_participant_mean']:>10.2f}")
    print("-" * 72)
    if val_maes:
        print(f"val MAE (held-out TRAIN participants, model-selection metric): "
              f"{np.mean(val_maes):.3f}  per-fold {[round(v,3) for v in val_maes]}")
        print("-" * 72)
    print("Published on Nightbeat-DB (Moebus et al. 2024), for context:")
    for label, m in PAPER_REFERENCE.items():
        print(f"{label:<34}{m['mae']:>8.2f}{m['rmse']:>8.2f}{m['cor']:>8.2f}")
    print("=" * 72)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
