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

from bench.score.common import (acc_only_slice, classification_kind, load_ckpt_meta, parse_metrics_txt, save_units,
                                 units_safely, val_row_at_step)


class _WithIndex:
    """Wrap a map-style dataset so every item also carries its dataset index. Iterated with the
    val loader's own batch size / order / None-skipping, the batches hold exactly the same
    windows as the original val loader, so predictions (and metrics) are unchanged."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return None if item is None else (item[0], item[1], i)


def _collate_with_index(batch):
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys, ids = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), torch.as_tensor(ids, dtype=torch.long)


def _indexed_val_loader(val_loader):
    """Same batches as `val_loader`, plus the dataset index of every window (None if the
    dataset does not expose the WindowDataset session index)."""
    from torch.utils.data import DataLoader
    ds = val_loader.dataset
    if not (hasattr(ds, "_keys") and hasattr(ds, "_sess_idx") and hasattr(ds, "_starts")):
        return None
    return DataLoader(_WithIndex(ds), batch_size=val_loader.batch_size, shuffle=False,
                      num_workers=val_loader.num_workers, pin_memory=val_loader.pin_memory,
                      drop_last=False, collate_fn=_collate_with_index)


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
    idx_loader = _indexed_val_loader(val_loader)     # identical batches + window index (subject ids)
    all_true, all_pred, all_idx, n_batches = [], [], [], 0
    with torch.no_grad():
        for batch in (idx_loader if idx_loader is not None else val_loader):
            if batch is None:
                continue
            if idx_loader is not None:
                x, y, idx = batch
            else:
                (x, y), idx = batch, None
            x = acc_only_slice(x.to(device).float(), cfg)
            y = remap_labels(y, raw_to_idx)
            valid = y >= 0
            if not valid.any():
                continue
            x, y = x[valid], y[valid]
            preds = head(encoder(x)).argmax(dim=1)
            all_true.extend(y.tolist())
            all_pred.extend(preds.tolist())
            if idx is not None:
                all_idx.extend(idx[valid.cpu()].tolist())
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
    units_safely(_save_har_units, run_dir, ckpt_path, task, val_loader.dataset, all_true, all_pred,
                 all_idx if idx_loader is not None else None, metrics, step)
    return metrics, info


def _save_har_units(run_dir, ckpt_path, task, ds, y_true, y_pred, idx, metrics, step):
    """Per-window sidecar: y_true / y_pred + subject and session of every window (subject_id /
    session_id parquet columns via the WindowDataset session index)."""
    if idx is None or len(idx) != len(y_true):
        subj = sess = None
        keys = [str(i) for i in range(len(y_true))]
        note = "window subject ids unavailable (val dataset exposes no session index)"
    else:
        ks = [ds._keys[int(ds._sess_idx[i])] for i in idx]
        subj = [k.subject_id for k in ks]
        sess = [f"{k.subject_id}/{k.session_id}" for k in ks]
        keys = [f"{k.dataset}|{k.subject_id}|{k.session_id}|{int(ds._starts[i])}" for k, i in zip(ks, idx)]
        note = "unit = 2.56 s val window; subject = parquet subject_id; session = subject_id/session_id"
    save_units(run_dir, ckpt_path, task, kind=classification_kind(task["metrics"]["primary"]),
               arrays={"unit_key": keys, "y_true": y_true, "y_pred": y_pred},
               groups={"subject": subj, "session": sess},
               metric_reported=metrics.get(task["metrics"]["primary"]), checkpoint_step=step, notes=note)
