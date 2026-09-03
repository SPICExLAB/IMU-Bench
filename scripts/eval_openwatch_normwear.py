#!/usr/bin/env python3
"""First-party NormWear-Base reproduction on our own OpenWatch 6-class split.

EXPERIMENTS.md §5/§5a previously only *cited* the OpenWatch paper's own
NormWear-Base numbers (Bonazzi et al. Table 2) — this script actually loads
their official pretrained checkpoint (Apache-2.0, vendored under
`baselines/SSL/NormWear/`, ~136M params, matching the paper's stated size)
and evaluates it under our own protocol: identical val split (same seed/
subject split as `configs/vit1d_gesture_probe_openwatch_6class_mae_multidata_flu.yaml`),
identical `compute_metrics`, so the result is directly comparable to
"Ours (MAE)" in §5, not just to the paper's self-reported number under an
unknown/different protocol.

Recipe (matches the paper's own NormWear-Base description, arXiv:2605.04791):
frozen NormWear backbone + a trainable linear classification head, trained
with class-weighted cross-entropy, AdamW, early stopping on val macro-F1
(the paper uses patience=5/100 epochs on full dataset passes; since this is
a single linear layer over cached embeddings we run more, smaller full-batch
gradient steps instead of literal "epochs" — same spirit, finer granularity).

Embedding convention follows the *paper's own* downstream pipeline
(`downstream_pipeline/model_apis.py::NormWear_API.get_embedding`): mean-pool
over patches per channel, then concatenate (not average) across channels.

Input is the RAW (unnormalized) per-gesture recording at its native 100Hz
(per the paper's stated IMU sampling rate) — no resampling/stretching to a
fixed window the way our own ViT1D loader does; NormWear's own
`get_embedding` handles resampling to its internal 65Hz internally.
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


def build_split(root: str = OPENWATCH_ROOT):
    all_records, label_map = discover_openwatch_records(
        root=root,
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


def extract_embeddings(records, model, device):
    embeds, labels = [], []
    for rec in records:
        x = _read_csv_array(rec.path, include_ppg=False)  # [L, 6]
        # NOTE: get_embedding expects sample_data as CPU tensor/ndarray (it
        # calls .numpy() on it internally) and takes `device` separately to
        # place the *computed CWT spectrogram* on GPU before the backbone
        # forward -- passing an already-GPU tensor here throws inside their
        # code (can't .numpy() a cuda tensor).
        x_t = torch.from_numpy(x.T[None]).float()  # [1, 6, L], stays on CPU
        with torch.no_grad():
            out = model.get_embedding(x_t, sampling_rate=NATIVE_HZ, device=device)  # [1, nvar, P, E]
        emb = out.mean(dim=2).flatten(start_dim=1)  # mean over patches -> concat over channels
        embeds.append(emb[0].cpu().numpy())
        labels.append(rec.label)
    return np.stack(embeds).astype(np.float32), np.array(labels, dtype=np.int64)


def train_and_eval_head(Xtr, ytr, Xva, yva, num_classes, device, max_steps=500, patience=25, lr=1e-3):
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
                print(f"[early stop] step={step} best_val_macro_f1={best_f1:.4f}")
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_preds = head(Xva_t).argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(yva, val_preds)
    return {
        "acc": float((yva == val_preds).mean()),
        "bal_acc": metrics["bal_acc"],
        "macro_f1": metrics["macro_f1"],
        "macro_recall": metrics["macro_recall"],
        "n_val": int(len(yva)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="results_openwatch_normwear_embeddings.npz")
    parser.add_argument("--out-json", default="results_openwatch_normwear.json")
    # `open_watch_augmentations` reruns this same recipe against OpenWatch's own
    # augmented release, so the first-party row is on the same data as §5a's
    # table instead of §5's no-aug one. Default stays no-aug (§5c's number).
    parser.add_argument("--root", default=OPENWATCH_ROOT)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_records, val_records, label_map = build_split(root=args.root)
    print(f"root={args.root}")
    print(f"train={len(train_records)} val={len(val_records)} classes={label_map}")

    if os.path.exists(args.cache):
        d = np.load(args.cache)
        Xtr, ytr, Xva, yva = d["Xtr"], d["ytr"], d["Xva"], d["yva"]
        print(f"loaded cached embeddings from {args.cache}")
    else:
        model = NormWearModel(weight_path=CKPT_PATH, optimized_cwt=True).to(device)
        model.eval()
        Xtr, ytr = extract_embeddings(train_records, model, device)
        Xva, yva = extract_embeddings(val_records, model, device)
        np.savez(args.cache, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva)
        print(f"cached embeddings to {args.cache} (Xtr={Xtr.shape}, Xva={Xva.shape})")

    result = train_and_eval_head(Xtr, ytr, Xva, yva, num_classes=len(label_map), device=device)
    result["root"] = args.root
    print(result)
    with open(args.out_json, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
