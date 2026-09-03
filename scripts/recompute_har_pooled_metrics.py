#!/usr/bin/env python3
"""Recompute correct (pooled) macro_f1/bal_acc/macro_precision/macro_recall
for HAR classification runs whose val_loss-selected checkpoint numbers in
EXPERIMENTS.md were sourced from Trainer._eval() before it was fixed to pool
predictions across the whole validation pass instead of averaging per-batch
metric estimates (see imu_lm/runtime_consistency/trainer.py's _eval
docstring / the fix commit). DeepConvLSTM rows are unaffected -- they use
imu_lm/probe/eval.py's eval_head(), which already pools correctly.

Usage
-----
    python scripts/recompute_har_pooled_metrics.py --out-json corrected_har_metrics.json
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.loaders import make_loaders
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn
from imu_lm.probe.head import LinearHead
from imu_lm.utils.training import remap_labels
from imu_lm.utils.metrics import compute_metrics

# (display name, run dir, alt_attention)
RUNS = [
    # WISDM
    ("WISDM / Ours (MAE)",              "vit1d_har_probe_wisdm_mae_multidata",           True),
    ("WISDM / Ours (MAE) finetune",     "vit1d_har_finetune_wisdm_mae_multidata",        True),
    ("WISDM / Ours (MAE, no alt-attn)", "vit1d_har_probe_wisdm_mae_noaltattn_multidata", False),
    ("WISDM / FOCAL",                   "vit1d_har_probe_wisdm_focal_multidata",         False),
    ("WISDM / SimMTM",                  "vit1d_har_probe_wisdm_simmtm_multidata",        False),
    ("WISDM / Scratch",                 "vit1d_har_supervised_wisdm_altattn_scratch",    True),
    # UTD-MHAD (expected ~unchanged -- val set is tiny, ~1 batch, no batching bias)
    ("UTD-MHAD / Ours (MAE)",              "vit1d_har_probe_utd_mhad_mae_multidata",           True),
    ("UTD-MHAD / Ours (MAE) finetune",     "vit1d_har_finetune_utd_mhad_mae_multidata",        True),
    ("UTD-MHAD / Ours (MAE, no alt-attn)", "vit1d_har_probe_utd_mhad_mae_noaltattn_multidata", False),
    ("UTD-MHAD / FOCAL",                   "vit1d_har_probe_utd_mhad_focal_multidata",         False),
    ("UTD-MHAD / SimMTM",                  "vit1d_har_probe_utd_mhad_simmtm_multidata",        False),
    ("UTD-MHAD / Scratch",                 "vit1d_har_supervised_utd_mhad_altattn_scratch",    True),
]


@torch.no_grad()
def eval_run(run_name: str, alt_attention: bool, device):
    run_dir = os.path.join("runs", run_name)
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]

    label_map = json.load(open(os.path.join(run_dir, "artifacts", "label_map.json")))
    raw_to_idx = {int(k): v for k, v in label_map["raw_to_idx"].items()}

    EncCls = ViT1DEncoderAltAttn if alt_attention else ViT1DEncoder
    encoder = EncCls(cfg).to(device)
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval()

    head = LinearHead(encoder.embed_dim, len(raw_to_idx)).to(device)
    head.load_state_dict(ckpt["head"], strict=True)
    head.eval()

    loaders = make_loaders(cfg)
    val_loader = loaders["val_loader"]

    all_true, all_pred = [], []
    for batch in val_loader:
        if batch is None:
            continue
        x, y = batch
        x = x.to(device)
        y = remap_labels(y, raw_to_idx)
        valid = y >= 0
        if not valid.any():
            continue
        x, y = x[valid], y[valid]
        z = encoder(x.float())
        preds = head(z).argmax(dim=1)
        all_true.extend(y.tolist())
        all_pred.extend(preds.tolist())

    metrics = compute_metrics(all_true, all_pred)
    acc = sum(int(t == p) for t, p in zip(all_true, all_pred)) / len(all_true)
    return {
        "n_val_samples": len(all_true),
        "n_classes_seen": len(set(all_true)),
        "acc": acc,
        "bal_acc": metrics["bal_acc"],
        "macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default="corrected_har_metrics.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name, run_name, alt_attention in RUNS:
        print(f"=== {name} ({run_name}) ===")
        try:
            m = eval_run(run_name, alt_attention, device)
            results[name] = {**m, "run": run_name}
            print(f"  acc={m['acc']:.4f} bal_acc={m['bal_acc']:.4f} macro_f1={m['macro_f1']:.4f} "
                  f"(n={m['n_val_samples']}, classes_seen={m['n_classes_seen']})")
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e), "run": run_name}

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
