#!/usr/bin/env python3
"""Window-level / clip-level(k=3) eval for our own 8 OpenWatch checkpoints,
using the OpenWatch paper's *actual* protocol (arXiv:2605.04791), correcting
`eval_openwatch_clip_level.py` (EXPERIMENTS.md §5b), which used a guessed
"3 disjoint thirds + soft-vote" scheme before the paper's real methodology
was found.

Real protocol: 1s windows (100 samples @ 100Hz) at a 20-sample stride
(~6 overlapping windows/clip); clip label = first run of 3 consecutive
identical window predictions, else majority vote (`imu_lm.utils.clip_voting`).

Important asymmetry vs. `eval_openwatch_normwear_windowed.py`: our 8
checkpoints were already trained on *whole-recording* inputs (each CSV
resampled straight to the model's fixed 512-sample window) — retraining all
8 on window-level data would require relaunching every SLURM training job.
So this script reuses the existing trained encoder+head as-is and only
corrects the *evaluation-time* windowing/aggregation, unlike the NormWear
windowed script (which retrains its linear head on window-level embeddings,
since NormWear has no such fixed-window architectural constraint forcing an
existing whole-recording-trained checkpoint). This is a real, documented
train/eval mismatch for our own rows that NormWear's windowed reproduction
doesn't share.

Usage
-----
    python scripts/eval_openwatch_windowed_protocol.py --out-json results_openwatch_windowed_protocol.json
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
from imu_lm.data.openwatch_loader import _read_csv_array, _resample_to_len, _to_flu, apply_unit_scale
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.clip_voting import STRIDE_SAMPLES, WINDOW_SAMPLES, aggregate_clip_label, restricted_window_starts, sliding_window_starts
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.metrics import compute_metrics
from scripts.eval_openwatch_clip_level import RUNS, RUNS_ROOT, _acc_only_truncate, build_model, load_val_records

NATIVE_HZ = 100.0


def predict_windows(encoder, head, rec, cfg, cfg_block, T: int, stride: int = STRIDE_SAMPLES):
    include_ppg = bool(cfg_block.get("include_ppg", False))
    frame = str(cfg_block.get("coordinate_frame", "raw")).lower()
    # model-input handling must match how the checkpoint was TRAINED:
    #   window_zero_pad -> raw 1s slice, preprocess, zero-pad to next patch
    #                      multiple (100 -> 112); no resampling, no stretch
    #   window_target_len -> resample slice to that length (varlen encoder)
    #   neither -> resample to the full fixed T (5.12x stretch; matches
    #              whole-recording-trained checkpoints, §5d convention)
    zero_pad = bool(cfg_block.get("window_zero_pad", False))
    patch = int(cfg_get(cfg, ["vit1d", "patch_size"], 16))
    pad_T = int(cfg_block.get("window_pad_len") or ((WINDOW_SAMPLES + patch - 1) // patch) * patch)
    restrict = cfg_block.get("window_restrict_len")

    x = _read_csv_array(rec.path, include_ppg=include_ppg)
    x = apply_unit_scale(x, include_ppg, cfg_block.get("acc_scale", 1.0), cfg_block.get("gyro_scale", 1.0))   # same units as training
    if frame == "flu":
        x = _to_flu(x, include_ppg=include_ppg)

    if restrict:
        starts = restricted_window_starts(len(x), stride=stride, restrict_len=int(restrict))
    else:
        starts = sliding_window_starts(len(x), stride=stride)
    preds = []
    for s in starts:
        seg = x[s : s + WINDOW_SAMPLES]
        if zero_pad:
            xproc = preprocess_window(seg.copy(), cfg, None)
            if xproc is None:
                continue
            if xproc.shape[1] < pad_T:
                xproc = np.pad(xproc, ((0, 0), (0, pad_T - xproc.shape[1])))
        else:
            xw = _resample_to_len(seg, T)
            xproc = preprocess_window(xw.copy(), cfg, None)
            if xproc is None:
                continue
        x_t = torch.from_numpy(xproc).float().unsqueeze(0)
        x_t = _acc_only_truncate(x_t, cfg)
        with torch.no_grad():
            z = encoder(x_t)
            logits = head(z)
        preds.append(int(logits.argmax(dim=-1).item()))
    return preds


def eval_run(name: str, run_dir: str, objective_key: str, stride: int = STRIDE_SAMPLES, split: str = "val"):
    ckpt_path = os.path.join(RUNS_ROOT, run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[skip] {name}: no checkpoint at {ckpt_path}")
        return None

    encoder, head, cfg = build_model(ckpt_path)
    val_records, cfg_block = load_val_records(cfg, objective_key, split=split)
    T, _ = compute_T_and_hop(cfg)
    wtl = cfg_block.get("window_target_len")
    if wtl:
        T = int(wtl)
        print(f"[{name}] windowed checkpoint: resampling eval windows to window_target_len={T}")

    window_true, window_pred = [], []
    clip_true, clip_pred = [], []

    for rec in val_records:
        preds = predict_windows(encoder, head, rec, cfg, cfg_block, T, stride=stride)
        if not preds:
            continue
        window_true.extend([rec.label] * len(preds))
        window_pred.extend(preds)
        clip_true.append(rec.label)
        clip_pred.append(aggregate_clip_label(preds))

    window_metrics = compute_metrics(window_true, window_pred)
    clip_metrics = compute_metrics(clip_true, clip_pred)

    result = {
        "name": name,
        "run": run_dir,
        "split": split,
        "n_clips": len(clip_true),
        "n_windows": len(window_true),
        "window": {
            "acc": float(np.mean(np.array(window_true) == np.array(window_pred))),
            "bal_acc": window_metrics["bal_acc"],
            "macro_f1": window_metrics["macro_f1"],
            "macro_recall": window_metrics["macro_recall"],
        },
        "clip": {
            "acc": float(np.mean(np.array(clip_true) == np.array(clip_pred))),
            "bal_acc": clip_metrics["bal_acc"],
            "macro_f1": clip_metrics["macro_f1"],
            "macro_recall": clip_metrics["macro_recall"],
        },
    }
    print(
        f"{name:28s} [{split}] n_clips={result['n_clips']:4d} n_windows={result['n_windows']:5d}  "
        f"window[F1={result['window']['macro_f1']:.3f} Acc={result['window']['acc']:.3f} "
        f"Rec={result['window']['macro_recall']:.3f}]  "
        f"clip[F1={result['clip']['macro_f1']:.3f} Acc={result['clip']['acc']:.3f} "
        f"Rec={result['clip']['macro_recall']:.3f}]"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="results_openwatch_windowed_protocol.json")
    parser.add_argument("--stride", type=int, default=STRIDE_SAMPLES)
    parser.add_argument(
        "--splits", nargs="+", default=["val"], choices=["val", "test"],
        help="which held-out split(s) to score each run on",
    )
    parser.add_argument(
        "--runs", nargs="+", default=None, metavar="NAME:RUN_DIR:OBJECTIVE_KEY",
        help="override the built-in RUNS list, e.g. "
             "'Ours (MAE) windowed:vit1d_gesture_probe_openwatch_6class_mae_windowed_strat:openwatch_gesture_probe'",
    )
    args = parser.parse_args()

    if args.runs is not None:
        runs = []
        for spec in args.runs:
            name, run_dir, objective_key = spec.rsplit(":", 2)
            runs.append((name, run_dir, objective_key))
    else:
        runs = RUNS

    print(f"window={WINDOW_SAMPLES} stride={args.stride} (default {STRIDE_SAMPLES}) @ {NATIVE_HZ}Hz splits={args.splits}")

    results = []
    for name, run_dir, objective_key in runs:
        for split in args.splits:
            res = eval_run(name, run_dir, objective_key, stride=args.stride, split=split)
            if res is not None:
                results.append(res)

    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {len(results)} rows to {args.out_json}")


if __name__ == "__main__":
    main()
