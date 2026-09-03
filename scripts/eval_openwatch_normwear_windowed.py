#!/usr/bin/env python3
"""NormWear-Base reproduction using the OpenWatch paper's *actual* window/
clip-level protocol (arXiv:2605.04791), correcting `eval_openwatch_normwear.py`
(EXPERIMENTS.md §5c), which only evaluated one embedding per whole recording.

The paper: "windows of length W=100 samples (1s at 100Hz) were extracted
with stride 20 samples" (~6 overlapping windows per clip), and "an early-stop
rule assigns the clip label once any class appears in k=3 consecutive
windows; if no class meets the threshold, majority voting is used as a
fallback." Both window-level and clip-level accuracy are reported on the
*same* trained model, which implies training happens on window-level data
too (not whole-recording embeddings) — this script trains the linear head on
per-window embeddings (each window inherits its recording's label) and
reports both aggregation levels.

Still NOT a certified exact replication: the paper's subject-split algorithm
is "randomized selection balanced per posture and activity condition" (a
stratified split); ours (`_split_subjects`) is a plain unstratified random
subject shuffle at an assumed 80/10/10 ratio. See EXPERIMENTS.md §5c/§5d.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "baselines", "SSL"))

from imu_lm.data.openwatch_loader import _read_csv_array, _split_subjects, discover_openwatch_records
from imu_lm.utils.clip_voting import STRIDE_SAMPLES, WINDOW_SAMPLES, aggregate_clip_label, sliding_window_starts
from imu_lm.utils.metrics import compute_metrics
from NormWear.main_model import NormWearModel

OPENWATCH_ROOT = "/projects/b53017/vbc1591/openwatch_raw/open_watch_without_augmentations"
GESTURES = ["double_clench", "double_pinch", "pinch_down", "pinch_up", "slide"]
SEED = 42
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
BACKGROUND_FRACTION = 0.2
NATIVE_HZ = 100.0
CKPT_PATH = os.path.join(REPO_ROOT, "baselines/SSL/NormWear/checkpoints/normwear_pretrain_ckpt.pth")


def build_split():
    all_records, label_map = discover_openwatch_records(
        root=OPENWATCH_ROOT,
        subjects=None,
        positions=None,
        gestures=GESTURES,
        label_ids=None,
        collapse_background=True,
        background_label="background",
        background_fraction=BACKGROUND_FRACTION,
        seed=SEED,
    )
    all_subjects = sorted({r.subject_id for r in all_records})
    train_subj, val_subj, _test_subj = _split_subjects(
        all_subjects, seed=SEED, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION,
    )
    train_set, val_set = set(train_subj), set(val_subj)
    train_records = [r for r in all_records if r.subject_id in train_set]
    val_records = [r for r in all_records if r.subject_id in val_set]
    return train_records, val_records, label_map


def extract_window_embeddings(records, model, device, stride=STRIDE_SAMPLES):
    """Returns per-window embeddings/labels/record-group-index (temporal
    order preserved within each record, needed for clip aggregation)."""
    embeds, labels, groups = [], [], []
    for ridx, rec in enumerate(records):
        x = _read_csv_array(rec.path, include_ppg=False)  # [L, 6]
        starts = sliding_window_starts(len(x), stride=stride)
        for s in starts:
            seg = x[s : s + WINDOW_SAMPLES]
            x_t = torch.from_numpy(seg.T[None]).float()  # [1, 6, l], CPU (see eval_openwatch_normwear.py note)
            with torch.no_grad():
                out = model.get_embedding(x_t, sampling_rate=NATIVE_HZ, device=device)  # [1, nvar, P, E]
            emb = out.mean(dim=2).flatten(start_dim=1)  # mean over patches -> concat over channels
            embeds.append(emb[0].cpu().numpy())
            labels.append(rec.label)
            groups.append(ridx)
    return (
        np.stack(embeds).astype(np.float32),
        np.array(labels, dtype=np.int64),
        np.array(groups, dtype=np.int64),
    )


def train_head(Xtr, ytr, Xva, yva, num_classes, device, max_steps=500, patience=25, lr=1e-3):
    class_counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
    class_weights = torch.tensor(
        class_counts.sum() / (num_classes * np.maximum(class_counts, 1.0)), dtype=torch.float32
    ).to(device)

    head = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    best_f1, best_state, bad_steps = -1.0, None, 0
    for step in range(max_steps):
        head.train()
        opt.zero_grad()
        loss = F.cross_entropy(head(Xtr_t), ytr_t, weight=class_weights)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            val_preds = head(Xva_t).argmax(dim=1).cpu().numpy()
        f1 = compute_metrics(yva, val_preds)["macro_f1"]
        if f1 > best_f1:
            best_f1, bad_steps = f1, 0
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            bad_steps += 1
            if bad_steps >= patience:
                print(f"[early stop] step={step} best_val_window_macro_f1={best_f1:.4f}")
                break

    head.load_state_dict(best_state)
    head.eval()
    return head


def clip_level_eval(head, Xva, groups_va, val_records, device):
    with torch.no_grad():
        window_preds = head(torch.from_numpy(Xva).to(device)).argmax(dim=1).cpu().numpy()

    clip_preds, clip_true = [], []
    for ridx, rec in enumerate(val_records):
        preds_this_clip = window_preds[groups_va == ridx].tolist()
        clip_preds.append(aggregate_clip_label(preds_this_clip))
        clip_true.append(rec.label)
    return np.array(clip_true), np.array(clip_preds), window_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="results_openwatch_normwear_windowed_embeddings.npz")
    parser.add_argument("--out-json", default="results_openwatch_normwear_windowed.json")
    parser.add_argument("--stride", type=int, default=STRIDE_SAMPLES)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_records, val_records, label_map = build_split()
    print(f"train={len(train_records)} val={len(val_records)} classes={label_map}")
    print(f"stride override: {args.stride} (default {STRIDE_SAMPLES})")
    print(f"window={WINDOW_SAMPLES} stride={STRIDE_SAMPLES} @ {NATIVE_HZ}Hz")

    if os.path.exists(args.cache):
        d = np.load(args.cache)
        Xtr, ytr, Xva, yva, groups_va = d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["groups_va"]
        print(f"loaded cached embeddings from {args.cache}")
    else:
        model = NormWearModel(weight_path=CKPT_PATH, optimized_cwt=True).to(device)
        model.eval()
        Xtr, ytr, _groups_tr = extract_window_embeddings(train_records, model, device, stride=args.stride)
        Xva, yva, groups_va = extract_window_embeddings(val_records, model, device, stride=args.stride)
        np.savez(args.cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, groups_va=groups_va)
        print(f"cached embeddings to {args.cache} (Xtr={Xtr.shape}, Xva={Xva.shape})")

    num_classes = len(label_map)
    head = train_head(Xtr, ytr, Xva, yva, num_classes=num_classes, device=device)

    clip_true, clip_preds, window_preds = clip_level_eval(head, Xva, groups_va, val_records, device)

    window_metrics = compute_metrics(yva, window_preds)
    clip_metrics = compute_metrics(clip_true, clip_preds)

    result = {
        "n_val_clips": len(val_records),
        "n_val_windows": int(len(yva)),
        "window": {
            "acc": float((yva == window_preds).mean()),
            "macro_f1": window_metrics["macro_f1"],
            "macro_recall": window_metrics["macro_recall"],
        },
        "clip": {
            "acc": float((clip_true == clip_preds).mean()),
            "macro_f1": clip_metrics["macro_f1"],
            "macro_recall": clip_metrics["macro_recall"],
        },
    }
    print(result)
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
