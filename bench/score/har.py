"""HAR scorer: window-level macro-F1 / balanced accuracy / accuracy on the val
split, RE-EVALUATED from the selected checkpoint with predictions pooled over
the whole val pass (sklearn macro over the classes present).

Why re-evaluate instead of reading logs/metrics.txt: runs trained before
2026-07-18 logged per-batch-averaged F1/bal-acc (EXPERIMENTS.md §3 bug note;
WISDM MAE probe logs 0.206 vs the true 0.639), so the log is not a safe source
for every legacy run. This mirrors scripts/recompute_har_pooled_metrics.py but
builds the encoder from the checkpoint's own cfg (ViT1D / alt-attn / acc-only /
Inertia-1 adapter). Reference DeepConvLSTM runs (no ViT1D encoder) are read from
their summary.json, which was produced by the already-pooled probe/eval.py path.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from bench.score.common import acc_only_slice, load_ckpt_meta, parse_metrics_txt, val_row_at_step


def _score_deepconvlstm(run_dir: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    v = s["val"]
    return ({"macro_f1": float(v["macro_f1"]), "bal_acc": float(v["bal_acc"]), "acc": float(v["acc"]),
             "val_loss": float(v.get("loss", float("nan")))},
            {"checkpoint_step": None, "via": "summary.json (imu_lm/probe/eval.py pooled eval)",
             "num_classes": s.get("num_classes")})


def score(run_dir: str, ckpt_path: str, task: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if os.path.basename(run_dir).startswith("deepconvlstm") and os.path.exists(os.path.join(run_dir, "summary.json")):
        return _score_deepconvlstm(run_dir)

    import torch
    from imu_lm.data.loaders import make_loaders
    from imu_lm.models.baselines.inertia1_encoder import build_encoder
    from imu_lm.probe.head import LinearHead
    from imu_lm.utils.metrics import compute_metrics
    from imu_lm.utils.training import remap_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = load_ckpt_meta(ckpt_path)
    ckpt, cfg, step = meta["_obj"], meta["cfg"], int(meta["step"])

    label_map = json.load(open(os.path.join(run_dir, "artifacts", "label_map.json")))
    raw_to_idx = {int(k): v for k, v in label_map["raw_to_idx"].items()}

    encoder = build_encoder(cfg).to(device)
    encoder.load_state_dict(ckpt["model"], strict=True)
    encoder.eval()
    head = LinearHead(encoder.embed_dim, len(raw_to_idx)).to(device)
    head.load_state_dict(ckpt["head"], strict=True)
    head.eval()

    val_loader = make_loaders(cfg)["val_loader"]
    all_true, all_pred, n_batches = [], [], 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            x, y = batch
            x = acc_only_slice(x.to(device).float(), cfg)
            y = remap_labels(y, raw_to_idx)
            valid = y >= 0
            if not valid.any():
                continue
            x, y = x[valid], y[valid]
            preds = head(encoder(x)).argmax(dim=1)
            all_true.extend(y.tolist())
            all_pred.extend(preds.tolist())
            n_batches += 1
    m = compute_metrics(all_true, all_pred)
    acc = sum(int(t == p) for t, p in zip(all_true, all_pred)) / max(1, len(all_true))
    metrics = {"macro_f1": float(m["macro_f1"]), "bal_acc": float(m["bal_acc"]), "acc": float(acc)}
    rows = parse_metrics_txt(run_dir, "val")
    row = val_row_at_step(rows, step) if rows else None
    if row is not None:
        metrics["val_loss"] = float(row["loss"])
    info = {"checkpoint_step": step, "n_val_samples": len(all_true), "n_classes_seen": len(set(all_true)),
            "n_classes_head": len(raw_to_idx), "n_val_batches": n_batches,
            "logged_macro_f1_at_ckpt": (float(row["macro_f1"]) if row and "macro_f1" in row else None)}
    return metrics, info
