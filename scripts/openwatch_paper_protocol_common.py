#!/usr/bin/env python3
"""Shared data plumbing for the OpenWatch paper-protocol fair comparison
(EXPERIMENTS.md §5f): every model family (our ViT1D MAE, Mix-Token,
NormWear-Base) MUST evaluate on the identical contamination-safe
posture-stratified split and identical restricted-sub-interval windows, so
this lives in exactly one place.

Protocol constants (Bonazzi et al., arXiv:2605.04791 §3.2):
- windows: W=100 samples (1s @ 100Hz), stride 20, from a restricted
  sub-interval yielding 6 windows/clip (we center the 2.0s sub-interval —
  the paper does not say where it sits);
- clip label: k=3 consecutive-run else majority (imu_lm.utils.clip_voting);
- split: subject-independent, posture/activity balanced -> our
  posture-stratified split over CANONICAL identities (duplicate subjects
  collapsed, mirrored aug ids 51-85 routed to their source subject), with
  byte-identical duplicate files deduped. Seed 42, 80/10/10.
- aug condition: train on `open_watch_augmentations`, hold out REAL
  recordings — identical val/test sets in both conditions.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace as dc_replace
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.openwatch_loader import (  # noqa: E402
    OpenWatchRecord,
    _read_csv_array,
    _stratified_split_subjects_by_position,
    _to_flu,
    canonical_subject,
    discover_openwatch_records,
)
from imu_lm.utils.clip_voting import (  # noqa: E402
    CONSECUTIVE_K,
    RESTRICT_SAMPLES,
    STRIDE_SAMPLES,
    WINDOW_SAMPLES,
    aggregate_clip_label,
    restricted_window_starts,
)
from imu_lm.utils.metrics import compute_metrics  # noqa: E402

NOAUG_ROOT = "/projects/b53017/vbc1591/openwatch_raw/open_watch_without_augmentations"
AUG_ROOT = "/projects/b53017/vbc1591/openwatch_raw/open_watch_augmentations"
GESTURES = ["double_clench", "double_pinch", "pinch_down", "pinch_up", "slide"]
NUM_CLASSES = 6
SEED = 42
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1
BACKGROUND_FRACTION = 0.2
NATIVE_HZ = 100.0


def _discover(root: str) -> List[OpenWatchRecord]:
    records, _label_map = discover_openwatch_records(
        root=root,
        subjects=None,
        positions=None,
        gestures=GESTURES,
        label_ids=None,
        collapse_background=True,
        background_label="background",
        background_fraction=BACKGROUND_FRACTION,
        seed=SEED,
        dedup_exact=True,
    )
    return records


def build_protocol_split(condition: str) -> Tuple[List[OpenWatchRecord], List[OpenWatchRecord], List[OpenWatchRecord]]:
    """(train, val, test) records for `condition` in {'noaug','aug'}.

    Mirrors make_openwatch_loaders' contamination_safe + eval_root path
    exactly: split computed over canonical identities of the REAL release;
    val/test always drawn from the real release; aug-condition train drawn
    from the augmented release with membership decided by canonical source.
    """
    if condition not in {"noaug", "aug"}:
        raise ValueError(condition)
    eval_records = _discover(NOAUG_ROOT)
    train_pool = _discover(AUG_ROOT) if condition == "aug" else eval_records

    split_source = [dc_replace(r, subject_id=canonical_subject(r.subject_id)) for r in eval_records]
    train_s, val_s, test_s = _stratified_split_subjects_by_position(
        split_source, seed=SEED, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION,
    )
    train_set, val_set, test_set = set(train_s), set(val_s), set(test_s)

    train = [r for r in train_pool if canonical_subject(r.subject_id) in train_set]
    val = [r for r in eval_records if canonical_subject(r.subject_id) in val_set]
    test = [r for r in eval_records if canonical_subject(r.subject_id) in test_set]
    return train, val, test


def load_record_windows(rec: OpenWatchRecord, include_ppg: bool = False, flu: bool = False) -> np.ndarray:
    """All protocol windows of one recording -> [n_win, W, C] float32 (raw
    device axes unless flu=True)."""
    x = _read_csv_array(rec.path, include_ppg=include_ppg)
    if flu:
        x = _to_flu(x, include_ppg=include_ppg)
    starts = restricted_window_starts(len(x), window=WINDOW_SAMPLES, stride=STRIDE_SAMPLES,
                                      restrict_len=RESTRICT_SAMPLES)
    return np.stack([x[s : s + WINDOW_SAMPLES] for s in starts]).astype(np.float32)


def build_window_arrays(records: List[OpenWatchRecord], include_ppg: bool = False, flu: bool = False):
    """Windows + labels + clip index for a record list.

    Returns (X [N, W, C], y [N], clip_id [N]) where clip_id maps each window
    back to its recording index for clip-level voting.
    """
    xs, ys, cids = [], [], []
    for ci, rec in enumerate(records):
        w = load_record_windows(rec, include_ppg=include_ppg, flu=flu)
        xs.append(w)
        ys.extend([rec.label] * len(w))
        cids.extend([ci] * len(w))
    return np.concatenate(xs, 0), np.asarray(ys, np.int64), np.asarray(cids, np.int64)


def window_and_clip_metrics(y_true: np.ndarray, y_pred: np.ndarray, clip_id: np.ndarray,
                            clip_labels: Dict[int, int]) -> Dict:
    """Standard §5f result block from per-window predictions."""
    wm = compute_metrics(y_true, y_pred)
    clip_true, clip_pred = [], []
    for ci in np.unique(clip_id):
        m = clip_id == ci
        # windows of one clip are stored in temporal order
        clip_pred.append(aggregate_clip_label(list(y_pred[m]), k=CONSECUTIVE_K))
        clip_true.append(clip_labels[int(ci)])
    cm = compute_metrics(clip_true, clip_pred)
    return {
        "n_windows": int(len(y_true)),
        "n_clips": int(len(clip_true)),
        "window": {"acc": float((y_true == y_pred).mean()), "bal_acc": wm["bal_acc"],
                   "macro_f1": wm["macro_f1"], "macro_recall": wm["macro_recall"]},
        "clip": {"acc": float(np.mean(np.array(clip_true) == np.array(clip_pred))), "bal_acc": cm["bal_acc"],
                 "macro_f1": cm["macro_f1"], "macro_recall": cm["macro_recall"]},
    }


def inverse_frequency_weights(y: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = counts.sum() / (num_classes * np.maximum(counts, 1.0))
    return w.astype(np.float32)
