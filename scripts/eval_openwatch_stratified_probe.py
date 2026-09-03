#!/usr/bin/env python3
"""Whole-recording OpenWatch probe reproduction under a posture-stratified
subject split, testing whether the plain-random split's train/val posture
mismatch (§5e finding: default seed=42 val split is 40% walking despite only
3/50 subjects total having any walking data) explains part of the gap to the
paper's cited NormWear-Base* number.

Retrains a fresh linear head (reusing `eval_openwatch_normwear.py`'s
class-weighted-CE/AdamW/early-stop-on-macro-F1 recipe for consistency across
all 7 rows) on top of each already-frozen encoder -- no encoder retraining
needed for these, since they're all frozen-probe architectures. The two
full-finetune rows (Ours(MAE) finetune, Scratch) can't be redone this way
(their encoders are trainable, not frozen) and are retrained separately via
real SLURM jobs using `configs/*_stratified.yaml`.

Embeddings are whole-recording (matching the original §5/§5c convention, not
the windowed §5d protocol), so results here are directly comparable to
§5/§5c's numbers under the new split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "baselines", "SSL"))

from imu_lm.data.augmentations.preprocess import preprocess_window
from imu_lm.data.openwatch_loader import (
    OPENWATCH_ROOT,
    _read_csv_array,
    _resample_to_len,
    _stratified_split_subjects_by_position,
    _to_flu,
    discover_openwatch_records,
)
from imu_lm.data.windowing import compute_T_and_hop
from imu_lm.utils.helpers import cfg_get
from imu_lm.utils.metrics import compute_metrics
from scripts.eval_openwatch_clip_level import RUNS, RUNS_ROOT, _acc_only_truncate, build_model
from scripts.eval_openwatch_normwear import CKPT_PATH as NORMWEAR_CKPT_PATH
from scripts.eval_openwatch_normwear import train_and_eval_head
from NormWear.main_model import NormWearModel

# Same OpenWatch task constants used everywhere else in this benchmark.
GESTURES = ["double_clench", "double_pinch", "pinch_down", "pinch_up", "slide"]
SEED = 42
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
BACKGROUND_FRACTION = 0.2
NATIVE_HZ = 100.0


def stratified_records_for_cfg_block(cfg_block):
    """Rebuild train/val records for a given per-model cfg_block, but using
    the posture-stratified split instead of whatever the original checkpoint
    was trained with."""
    root = cfg_block.get("openwatch_root", cfg_block.get("root", OPENWATCH_ROOT))
    all_records, label_map = discover_openwatch_records(
        root=root,
        subjects=None,
        positions=cfg_block.get("positions"),
        gestures=cfg_block.get("gestures", GESTURES),
        label_ids=cfg_block.get("label_ids"),
        collapse_background=bool(cfg_block.get("collapse_background", True)),
        background_label=cfg_block.get("background_label", "background"),
        background_fraction=float(cfg_block.get("background_fraction", BACKGROUND_FRACTION)),
        seed=SEED,
    )
    train_subj, val_subj, _test_subj = _stratified_split_subjects_by_position(
        all_records, seed=SEED, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION,
    )
    train_set, val_set = set(train_subj), set(val_subj)
    train_records = [r for r in all_records if r.subject_id in train_set]
    val_records = [r for r in all_records if r.subject_id in val_set]
    return train_records, val_records, label_map


def extract_vit1d_whole_recording_embeddings(records, encoder, cfg, cfg_block, T):
    include_ppg = bool(cfg_block.get("include_ppg", False))
    frame = str(cfg_block.get("coordinate_frame", "raw")).lower()

    embeds, labels = [], []
    for rec in records:
        x = _read_csv_array(rec.path, include_ppg=include_ppg)
        if frame == "flu":
            x = _to_flu(x, include_ppg=include_ppg)
        xw = _resample_to_len(x, T)
        xproc = preprocess_window(xw.copy(), cfg, None)
        if xproc is None:
            continue
        x_t = torch.from_numpy(xproc).float().unsqueeze(0)
        x_t = _acc_only_truncate(x_t, cfg)
        with torch.no_grad():
            z = encoder(x_t)
        embeds.append(z[0].numpy())
        labels.append(rec.label)
    return np.stack(embeds).astype(np.float32), np.array(labels, dtype=np.int64)


def extract_normwear_whole_recording_embeddings(records, model, device):
    embeds, labels = [], []
    for rec in records:
        x = _read_csv_array(rec.path, include_ppg=False)
        x_t = torch.from_numpy(x.T[None]).float()  # CPU -- see eval_openwatch_normwear.py note
        with torch.no_grad():
            out = model.get_embedding(x_t, sampling_rate=NATIVE_HZ, device=device)
        emb = out.mean(dim=2).flatten(start_dim=1)
        embeds.append(emb[0].cpu().numpy())
        labels.append(rec.label)
    return np.stack(embeds).astype(np.float32), np.array(labels, dtype=np.int64)


def eval_vit1d_row(name, run_dir, objective_key, device):
    ckpt_path = os.path.join(RUNS_ROOT, run_dir, "checkpoints", "best.pt")
    if not os.path.exists(ckpt_path):
        print(f"[skip] {name}: no checkpoint at {ckpt_path}")
        return None
    encoder, _old_head, cfg = build_model(ckpt_path)
    T, _ = compute_T_and_hop(cfg)
    data_cfg = cfg_get(cfg, ["data"], {}) or {}
    ow_cfg = cfg_get(cfg, ["objective", objective_key], {}) or {}
    cfg_block = {**data_cfg, **ow_cfg}

    train_records, val_records, label_map = stratified_records_for_cfg_block(cfg_block)
    Xtr, ytr = extract_vit1d_whole_recording_embeddings(train_records, encoder, cfg, cfg_block, T)
    Xva, yva = extract_vit1d_whole_recording_embeddings(val_records, encoder, cfg, cfg_block, T)

    result = train_and_eval_head(Xtr, ytr, Xva, yva, num_classes=len(label_map), device=device)
    result.update({"name": name, "run": run_dir})
    print(f"{name:28s} n_train={len(ytr):4d} n_val={result['n_val']:3d}  "
          f"acc={result['acc']:.4f} macro_f1={result['macro_f1']:.4f}")
    return result


def eval_normwear_row(device):
    all_records, label_map = discover_openwatch_records(
        root=OPENWATCH_ROOT, subjects=None, positions=None, gestures=GESTURES, label_ids=None,
        collapse_background=True, background_label="background",
        background_fraction=BACKGROUND_FRACTION, seed=SEED,
    )
    train_subj, val_subj, _test_subj = _stratified_split_subjects_by_position(
        all_records, seed=SEED, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION,
    )
    train_set, val_set = set(train_subj), set(val_subj)
    train_records = [r for r in all_records if r.subject_id in train_set]
    val_records = [r for r in all_records if r.subject_id in val_set]

    model = NormWearModel(weight_path=NORMWEAR_CKPT_PATH, optimized_cwt=True).to(device)
    model.eval()
    Xtr, ytr = extract_normwear_whole_recording_embeddings(train_records, model, device)
    Xva, yva = extract_normwear_whole_recording_embeddings(val_records, model, device)

    result = train_and_eval_head(Xtr, ytr, Xva, yva, num_classes=len(label_map), device=device)
    result.update({"name": "NormWear-Base (first-party, stratified)", "run": "scripts/eval_openwatch_normwear.py"})
    print(f"{'NormWear-Base (stratified)':28s} n_train={len(ytr):4d} n_val={result['n_val']:3d}  "
          f"acc={result['acc']:.4f} macro_f1={result['macro_f1']:.4f}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", default="results_openwatch_stratified_probe.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    results = []
    for name, run_dir, objective_key in RUNS:
        if objective_key in ("openwatch_gesture_supervised", "openwatch_gesture_finetune"):
            # Scratch's and Ours(MAE) finetune's encoders are trainable, not
            # frozen, and were already trained under the OLD split -- reusing
            # them with a fresh head under the NEW split would mix an
            # old-split-adapted encoder with a new-split-trained head, not a
            # clean test. These two need the real SLURM retraining instead
            # (configs/*_stratified.yaml).
            print(f"[skip] {name}: full-finetune row, see SLURM retraining instead")
            continue
        res = eval_vit1d_row(name, run_dir, objective_key, device)
        if res is not None:
            results.append(res)

    results.append(eval_normwear_row(device))

    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {len(results)} rows to {args.out_json}")


if __name__ == "__main__":
    main()
