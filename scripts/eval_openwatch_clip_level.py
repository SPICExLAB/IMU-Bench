#!/usr/bin/env python3
"""Window-level vs. clip-level (k=3) eval for the OpenWatch 6-class gesture task.

Every existing OpenWatch gesture number in EXPERIMENTS.md/Experiments_table.md
Section 5 is a *single-window* accuracy: `OpenWatchDataset` resamples one
whole per-gesture CSV recording straight to the model's fixed 512-sample
window (see `imu_lm/data/openwatch_loader.py`), so one forward pass = one
prediction = one recording. The OpenWatch paper's own Table 2 (Bonazzi et
al.) instead reports two aggregation levels, "Window-Level" and "Clip-Level
(k=3)" — the latter noticeably higher for every model in that table (e.g.
their NormWear-Base: window acc 0.682 -> clip acc 0.788), so comparing our
single-window number directly against their clip-level headline number
mixes up two different metrics, not just two different models.

This script adds that missing clip-level number for our own checkpoints so
the comparison is apples-to-apples: for each val recording, split the RAW
(pre-resample) signal into k=3 equal consecutive thirds, resample each third
independently to the model's window length (same as the existing loader does
for the whole recording) and run it through the frozen encoder+head to get a
per-window prediction. "Window-level" pools all k*N of those individual
predictions; "clip-level" soft-votes (mean softmax) the k predictions for
each recording back into one prediction per recording, compared against that
recording's single ground-truth label.

Caveat: the paper's own exact windowing (window length, stride, whether k=3
windows overlap) isn't published in the README, so this is our own
best-effort reconstruction of "3 windows per clip" — soft-voted mean-softmax
aggregation over 3 equal temporal thirds of each recording — not a verified
replication of their exact protocol.

Usage
-----
    python scripts/eval_openwatch_clip_level.py --out-json results_openwatch_clip_level.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.data.openwatch_loader import (
    _read_csv_array,
    apply_unit_scale,
    _resample_to_len,
    _split_subjects,
    _stratified_split_subjects_by_position,
    canonical_subject,
    _to_flu,
    discover_openwatch_records,
)
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.models.ViT1D.model import ViT1DEncoder, ViT1DEncoderAltAttn
from imu_lm.probe.head import LinearHead
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.metrics import compute_metrics

RUNS_ROOT = "/gpfs/home/vbc1591/IMU-SSL/runs"
K = 3

# (display name, run dir under runs/, objective key that doubles as the data config block)
RUNS = [
    ("Ours (MAE)",              "vit1d_gesture_probe_openwatch_6class_mae_multidata",           "openwatch_gesture_probe"),
    ("Ours (MAE) finetune",     "vit1d_gesture_finetune_openwatch_6class_mae_multidata",        "openwatch_gesture_finetune"),
    ("Ours (MAE, no alt-attn)", "vit1d_gesture_probe_openwatch_6class_mae_noaltattn_multidata", "openwatch_gesture_probe"),
    ("Ours (MAE, acc-only)",    "vit1d_gesture_probe_openwatch_6class_mae_acconly_multidata",   "openwatch_gesture_probe"),
    ("FOCAL",                   "vit1d_gesture_probe_openwatch_6class_focal_multidata",         "openwatch_gesture_probe"),
    ("SimMTM",                  "vit1d_gesture_probe_openwatch_6class_simmtm_multidata",        "openwatch_gesture_probe"),
    ("RelCon",                  "vit1d_gesture_probe_openwatch_6class_relcon_multidata",        "openwatch_gesture_probe"),
    ("Scratch",                 "vit1d_gesture_supervised_openwatch_6class_altattn_scratch",    "openwatch_gesture_supervised"),
]


def build_model(ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["cfg"]
    from imu_lm.models.baselines.inertia1_encoder import build_encoder as _build_encoder
    encoder = _build_encoder(cfg)   # ViT1D / alt-attn / Inertia-1 adapter, per the ckpt's own cfg
    encoder.load_state_dict(state["model"], strict=True)
    encoder.eval()

    head_state = state["head"]
    num_classes = head_state["fc.weight"].shape[0]
    head = LinearHead(encoder.embed_dim, num_classes)
    head.load_state_dict(head_state)
    head.eval()
    return encoder, head, cfg


def load_val_records(cfg, objective_key: str, split: str = "val"):
    """Reconstruct a checkpoint's held-out records from its own stored cfg.

    Mirrors `make_openwatch_loaders`' split logic exactly: honors
    `splits.strategy: posture_stratified` (else the plain shuffle) and
    `eval_root` (windowed cross-root runs train on the augmented release but
    hold out *real* no-aug recordings — and their subject split is computed
    from the eval_root discovery). `split` selects "val" or "test".
    """
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    ow_cfg = cfg_get(cfg, ["objective", objective_key], {}) or {}
    cfg_block = {**data_cfg, **ow_cfg}

    root = cfg_block.get("eval_root") or cfg_block.get("openwatch_root", cfg_block.get("root"))
    seed = int(cfg_get(cfg, ["splits", "seed"], cfg_get(cfg, ["trainer", "seed"], 0)))
    contamination_safe = bool(cfg_block.get("contamination_safe", False))

    all_records, _label_map = discover_openwatch_records(
        root=root,
        subjects=None,
        positions=cfg_block.get("positions"),
        gestures=cfg_block.get("gestures"),
        label_ids=cfg_block.get("label_ids"),
        collapse_background=bool(cfg_block.get("collapse_background", False)),
        background_label=cfg_block.get("background_label", "background"),
        background_fraction=float(cfg_block.get("background_fraction", 1.0)),
        seed=seed,
        dedup_exact=contamination_safe,
    )
    _canon = canonical_subject if contamination_safe else (lambda s: int(s))
    if contamination_safe:
        from dataclasses import replace as _dc_replace
        split_source_records = [_dc_replace(r, subject_id=_canon(r.subject_id)) for r in all_records]
    else:
        split_source_records = all_records
    all_subjects = sorted({r.subject_id for r in split_source_records})
    val_fraction = float(cfg_get(cfg, ["splits", "val_fraction"], 0.1))
    test_fraction = float(cfg_get(cfg, ["splits", "test_fraction"], 0.1))
    strategy = str(cfg_get(cfg, ["splits", "strategy"], "random")).lower()
    if strategy == "posture_stratified":
        _train, val_subjects, test_subjects = _stratified_split_subjects_by_position(
            split_source_records, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
        )
    else:
        _train, val_subjects, test_subjects = _split_subjects(
            all_subjects, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
        )
    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    want = set(val_subjects if split == "val" else test_subjects)
    return [r for r in all_records if _canon(r.subject_id) in want], cfg_block


def _acc_only_truncate(x_t: torch.Tensor, cfg) -> torch.Tensor:
    """Mirror Trainer._acc_only_slice: truncate [1, C, T] to the first N
    (acc-only) channels when data.acc_only is set, since the OpenWatch loader
    itself always emits the full acc+gyro concat regardless of config."""
    if not bool(cfg_get(cfg, ["data", "acc_only"], False)):
        return x_t
    n = len(cfg_get(cfg, ["data", "sensor_columns"], []) or [])
    return x_t[:, :n, :] if n and x_t.shape[1] > n else x_t


def predict_windows(encoder, head, rec, cfg, cfg_block, T: int, k: int = K):
    include_ppg = bool(cfg_block.get("include_ppg", False))
    frame = str(cfg_block.get("coordinate_frame", "raw")).lower()

    x = _read_csv_array(rec.path, include_ppg=include_ppg)
    x = apply_unit_scale(x, include_ppg, cfg_block.get("acc_scale", 1.0), cfg_block.get("gyro_scale", 1.0))   # same units as training
    if frame == "flu":
        x = _to_flu(x, include_ppg=include_ppg)

    n = len(x)
    if n < k:
        segments = [x] * k
    else:
        bounds = np.linspace(0, n, k + 1).astype(int)
        segments = [x[bounds[i]:bounds[i + 1]] for i in range(k)]
        segments = [seg if len(seg) > 0 else x for seg in segments]

    probs, preds = [], []
    for seg in segments:
        xw = _resample_to_len(seg, T)
        xproc = preprocess_window(xw.copy(), cfg, None)
        if xproc is None:
            continue
        x_t = torch.from_numpy(xproc).float().unsqueeze(0)
        x_t = _acc_only_truncate(x_t, cfg)
        with torch.no_grad():
            z = encoder(x_t)
            logits = head(z)
            p = torch.softmax(logits, dim=-1)[0].numpy()
        probs.append(p)
        preds.append(int(p.argmax()))

    return preds, probs


def eval_run(name: str, run_dir: str, objective_key: str, k: int = K):
    ckpt_path = os.path.join(RUNS_ROOT, run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[skip] {name}: no checkpoint at {ckpt_path}")
        return None

    encoder, head, cfg = build_model(ckpt_path)
    val_records, cfg_block = load_val_records(cfg, objective_key)
    T, _ = compute_T_and_hop(cfg)

    window_true, window_pred = [], []
    clip_true, clip_pred = [], []

    for rec in val_records:
        preds, probs = predict_windows(encoder, head, rec, cfg, cfg_block, T, k=k)
        if not probs:
            continue
        window_true.extend([rec.label] * len(preds))
        window_pred.extend(preds)
        clip_true.append(rec.label)
        clip_pred.append(int(np.mean(probs, axis=0).argmax()))

    window_metrics = compute_metrics(window_true, window_pred)
    clip_metrics = compute_metrics(clip_true, clip_pred)

    result = {
        "name": name,
        "run": run_dir,
        "n_clips": len(clip_true),
        "window": {
            "acc": float(np.mean(np.array(window_true) == np.array(window_pred))),
            "macro_f1": window_metrics["macro_f1"],
            "macro_recall": window_metrics["macro_recall"],
        },
        "clip": {
            "acc": float(np.mean(np.array(clip_true) == np.array(clip_pred))),
            "macro_f1": clip_metrics["macro_f1"],
            "macro_recall": clip_metrics["macro_recall"],
        },
    }
    print(
        f"{name:28s} n_clips={result['n_clips']:4d}  "
        f"window[F1={result['window']['macro_f1']:.3f} Acc={result['window']['acc']:.3f} "
        f"Rec={result['window']['macro_recall']:.3f}]  "
        f"clip[F1={result['clip']['macro_f1']:.3f} Acc={result['clip']['acc']:.3f} "
        f"Rec={result['clip']['macro_recall']:.3f}]"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="results_openwatch_clip_level.json")
    parser.add_argument("--k", type=int, default=K)
    args = parser.parse_args()

    results = []
    for name, run_dir, objective_key in RUNS:
        res = eval_run(name, run_dir, objective_key, k=args.k)
        if res is not None:
            results.append(res)

    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {len(results)} rows to {args.out_json}")


if __name__ == "__main__":
    main()
