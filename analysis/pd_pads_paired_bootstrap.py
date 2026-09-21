#!/usr/bin/env python3
"""Paired bootstrap between the PADS authors' classifiers and the benchmark's Parkinson's cells.

    python analysis/pd_pads_paired_bootstrap.py

Both sides are scored on the same 94 evaluation participants (per-participant sidecars in
bench/results/_units/pd_pads/; the PADS ones come from `baselines/pd/pads_pipeline.py --units-only`, which
reproduces the reported metric exactly). Resampling is over participants within class strata, 5,000 paired
resamples, percentile interval. One training seed per cell, so this covers evaluation-sample variability only.
Writes paper_results/appendix_pd_pads/paired_bootstrap.csv.
"""
import os
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
U = os.path.join(REPO, "bench", "results", "_units", "pd_pads")
PAIRS = [("pads_features__external__s0", "mae_5src__finetune__s42"), ("pads_features__external__s0", "limubert_5src__finetune__s42"),
         ("pads_boss__external__s0", "mae_5src__finetune__s42"), ("pads_boss__external__s0", "limubert_5src__finetune__s42"),
         ("pads_boss__external__s0", "scratch__scratch__s42")]
METRICS = {"subject_macro_f1": lambda a, b: 100 * f1_score(a, b, average="macro", zero_division=0),
           "subject_bal_acc": lambda a, b: 100 * balanced_accuracy_score(a, b)}


def load(cell):
    z = np.load(os.path.join(U, cell + ".npz"), allow_pickle=True)
    k = z["unit_key"].astype(str); o = np.argsort(k)
    return k[o], z["y_true"][o], z["y_pred"][o]


def paired(a, b, fn, n_boot=5000, seed=0):
    ka, yt, pa = load(a); kb, yt2, pb = load(b)
    assert (ka == kb).all() and (yt == yt2).all(), "cells are not on the same participants"
    rng = np.random.default_rng(seed); strata = [np.where(yt == c)[0] for c in np.unique(yt)]
    d = []
    for _ in range(n_boot):
        s = np.concatenate([rng.choice(i, len(i)) for i in strata]); d.append(fn(yt[s], pa[s]) - fn(yt[s], pb[s]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return fn(yt, pa) - fn(yt, pb), lo, hi


rows = []
for a, b in PAIRS:
    for name, fn in METRICS.items():
        d, lo, hi = paired(a, b, fn)
        rows.append([a, b, name, round(d, 2), round(lo, 2), round(hi, 2), bool(lo > 0 or hi < 0)])
        print(f"{a.split('__')[0]:14s} - {b.split('__s')[0]:26s} {name:17s} {d:+6.1f} [{lo:+6.1f}, {hi:+6.1f}] {'excludes 0' if rows[-1][-1] else ''}")
sys.path.insert(0, os.path.join(REPO, "paper_results"))
from paper_io import write_csv  # noqa: E402
write_csv("appendix_pd_pads", "paired_bootstrap.csv", ["cell_a", "cell_b", "metric", "diff_a_minus_b", "lo", "hi", "excludes_zero"], rows)
