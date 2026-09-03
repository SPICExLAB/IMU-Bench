"""Re-implementation of the PADS paper's smartwatch-only ML pipeline.

Paper: Varghese, Brenner, Fujarski, van Alen, Plagwitz, Warnecke,
"Machine Learning in the Parkinson's disease smartwatch (PADS) dataset",
npj Parkinson's Disease 10, 9 (2024).  Official code (BSD-3):
https://imigitlab.uni-muenster.de/published/pads-project

What the paper actually does for the smartwatch modality (Table 1 / Table 2):

  * ONE sample per SUBJECT: the release's own preprocessing concatenates all
    assessment steps into a 132 x 976 matrix per subject
    (11 task slots [Relaxed1/2, RelaxedTask1/2, StretchHold, HoldWeight,
    DrinkGlas, CrossArms, TouchNose, Entrainment1/2] x 2 wrists x
    (acc XYZ + gyro XYZ); LiftHold / PointFinger / TouchIndex are dropped,
    the first 0.5 s (notification vibration) is cut, gravity is removed with
    an L1 trend filter).  This script reads those files straight from the
    PhysioNet download (`preprocessed/movement/<id>_ml.bin`).
  * Option A: hand-crafted features per channel -- log10 Welch PSD in 1 Hz
    bins for 1..19 Hz (19) + std / abs-energy / abs-max over 4 equal
    segments (12) = 31 features per channel, all channels concatenated.
  * Option B: Bag-of-SFA-Symbols (BOSS) histograms per channel, three window
    sizes 20/40/80, window_step 2, word_size 2, n_bins 3, concatenated.
  * (Option C, XceptionTime deep learning, is not listed in the paper's tables
    "due to underperformance" and is not re-implemented here.)
  * Input choice: acceleration only / rotation (gyro) only / both.
  * Classifiers: RBF-SVM (C in {0.1,1,10,100}), CatBoost (500 trees, depth
    4/5/6), MLP (LeakyReLU, hidden {[50],[200],[200,20],[500,100,20]}, AdamW
    lr 1e-3, batch 16, <=100 epochs, early stopping patience 10 on a 10 %
    stratified validation split, softmax output fed to a sample-weighted
    cross-entropy -- reproduced as in the official code).
  * Sample weights inversely proportional to the (class x gender) group
    frequency in the training fold, passed to every classifier's fit.
  * Nested CV: outer RepeatedStratifiedKFold(5, random_state=42) over
    subjects; inside each outer training fold every (option, input,
    classifier, hyper-parameter) is scored by an inner
    StratifiedKFold(5, shuffle, random_state=42) grid search on balanced
    accuracy; the single best setup per outer fold is applied to that fold's
    test set.  Reported number = mean over the 5 outer test folds.

Published smartwatch-only results (Table 2, "complete" rows):
    PD vs HC : 78.99 % (7.66) balanced accuracy
    PD vs DD : 69.18 % (7.45) balanced accuracy
The 72.42 % (PD vs DD) / 91.16 % (PD vs HC) figures are the classifier
STACKING of smartwatch + NMS questionnaire (CatBoost) -- they are NOT
sensor-only numbers.

Two protocols are provided:

  --protocol paper   exact nested 5-fold CV over the 390 (PD+DD) or 355
                     (PD+HC) subjects, to check that this re-implementation
                     reproduces the published numbers.
  --protocol ours    the same pipeline trained on the 312 training subjects
                     and scored on the 78 validation subjects of the repo's
                     `pads_pddd_dataset.parquet` split (subject-grouped
                     80/20, seed 0), i.e. the exact split of the
                     `vit1d_pddd_*` probe / finetune / scratch runs, so the
                     paper's method can be placed in the same table.
                     Model selection is still done only on the training
                     subjects (inner 5-fold CV).

Two deliberate deviations from the official code, both documented here:
  * Option B's BOSS transformers are fitted once per OUTER training fold and
    the transformed features are then grid-searched, instead of re-fitting
    BOSS inside every inner fold.  BOSS is label-free (anova=False) and only
    ever sees the outer training subjects, so the test fold is untouched; it
    cuts the cost ~20x (one 132-channel MultiBOSS fit is ~6 CPU-min).
  * For Option B + NN the official CustomNN creates its input layer lazily in
    forward() (c_in=None), i.e. AFTER the optimizer was built, so that layer
    never trains.  Here the input width is known up front and the whole MLP
    trains -- this can only make the baseline stronger.

The full grid is 2 options x 3 inputs x 3 classifiers = 18 configurations.
One invocation runs ONE configuration (so it can be spread over a SLURM
array) and writes a JSON with, per outer fold, the inner-CV score, the chosen
hyper-parameters, test predictions and decision scores.  Use
`scripts/summarize_pads_paper_baseline.py` to pick the best configuration
per fold (paper rule: argmax of the inner-CV balanced accuracy) and to build
the tables.

Usage:
    python scripts/pads_paper_baseline.py --mode pd_vs_dd --protocol paper \
        --option a --channels acc --classifier svm --out out/paper_a_acc_svm.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PADS_ROOT_DEFAULT = "/projects/b53017/vbc1591/physionet.org/files/parkinsons-disease-smartwatch/1.0.0"
PDDD_PARQUET_DEFAULT = "/projects/b53017/vbc1591/pads_pddd_dataset.parquet"
N_SAMPLES_PER_CHANNEL = 976  # 1024 @ ~100 Hz minus the first 48 samples

TASKS = ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold", "HoldWeight",
         "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1", "Entrainment2"]

HYPERPARAMS = {
    "nn": {"clf__h_layer_sizes": [[50], [200], [200, 20], [500, 100, 20]]},  # official name: clf__module__h_layer_sizes
    "svm": {"clf__C": [0.1, 1.0, 10.0, 100.0], "clf__gamma": ["scale"]},
    "cat": {"clf__num_trees": [500], "clf__depth": [4, 5, 6]},
}


# --------------------------------------------------------------------------- data
def movement_channels() -> list[str]:
    ch = []
    for task in TASKS:
        for wrist in ["LeftWrist", "RightWrist"]:
            for sensor in ["Acceleration", "Rotation"]:
                for axis in ["X", "Y", "Z"]:
                    ch.append(f"{task}_{sensor}_{wrist}_{axis}")
    return ch


def load_file_list(pads_root: str, mode: str) -> pd.DataFrame:
    """Same subject ordering / label recoding as the official `get_file_list`."""
    fl = pd.read_csv(os.path.join(pads_root, "preprocessed", "file_list.csv"))
    if mode == "pd_vs_hc":
        fl = fl.loc[(fl["label"] == 0) | (fl["label"] == 1)].copy()
        # 0 = HC, 1 = PD (unchanged)
    elif mode == "pd_vs_dd":
        fl = fl.loc[(fl["label"] == 1) | (fl["label"] == 2)].copy()
        # official code: PD -> 0, DD -> 1
        fl["label"] = fl["label"].replace({1: 0, 2: 1})
    else:
        raise ValueError(mode)
    fl["id"] = fl["id"].astype(int)
    return fl.reset_index(drop=True)


def load_movement(pads_root: str, ids, channel_filter: str | None):
    channels = np.array(movement_channels())
    if channel_filter is None:
        mask = np.ones(len(channels), dtype=bool)
    else:
        mask = np.array([channel_filter in c for c in channels])
    xs = []
    for sid in ids:
        arr = np.fromfile(os.path.join(pads_root, "preprocessed", "movement", f"{int(sid):03d}_ml.bin"),
                          dtype=np.float32).reshape(-1, N_SAMPLES_PER_CHANNEL)
        assert arr.shape[0] == len(channels), arr.shape
        xs.append(arr[mask])
    return np.stack(xs), channels[mask]


# ------------------------------------------------------------ option A features
def bandpower(x, srate=100, steps=100, cut=(1, 20), scale=True):
    _, pxx = welch(x, fs=srate, nperseg=steps, scaling="density")
    bp = pxx[cut[0]:cut[1]] if cut is not None else pxx
    return np.log10(bp) if scale else bp


def _segments(x, n=4):
    ln = int(len(x) / n)
    return [x[i * ln:(i + 1) * ln] for i in range(n)]


def std_windowed(x):
    return [np.std(s) for s in _segments(x)]


def abs_energy_windowed(x):
    return [np.sum(np.abs(s)) for s in _segments(x)]


def abs_max_windowed(x):
    return [np.max(np.abs(s)) for s in _segments(x)]


def feature_extraction_a(x_ch):  # x_ch: [C, T] -> flat vector [C*31] (per-channel blocks concatenated by feature)
    feats = [np.apply_along_axis(f, 1, x_ch) for f in (bandpower, std_windowed, abs_energy_windowed, abs_max_windowed)]
    return np.concatenate(feats, axis=1).flatten()


# ------------------------------------------------------------ option B (BOSS)
class MultiBOSS(BaseEstimator, TransformerMixin):
    """One pyts BOSS transformer per (channel, window size); histograms concatenated.

    Mirrors ml/multi_boss.py of the official code, incl. an on-disk cache keyed
    by the md5 of the training array so the (expensive, label-free) fit is not
    repeated for every classifier / hyper-parameter combination.
    """

    def __init__(self, n_channels=None, window_sizes=(20, 40, 80), window_step=2, word_size=2, n_bins=3,
                 cache_dir=None):
        self.n_channels = n_channels
        self.window_sizes = window_sizes
        self.window_step = window_step
        self.word_size = word_size
        self.n_bins = n_bins
        self.cache_dir = cache_dir

    def _new_boss(self, ws):
        from pyts.transformation import BOSS
        # sparse=True + .toarray(): pyts 0.13's sparse=False path calls the scipy `.A` attribute that
        # scipy>=1.14 removed; the values are identical.
        return BOSS(sparse=True, window_size=ws, window_step=self.window_step, word_size=self.word_size,
                    n_bins=self.n_bins, norm_std=False, norm_mean=False, anova=False, drop_sum=False)

    def _cache_path(self, X):
        if not self.cache_dir:
            return None
        h = hashlib.md5(np.ascontiguousarray(X).tobytes()).hexdigest()
        h += "_" + "_".join(str(w) for w in self.window_sizes) + f"_{self.word_size}_{self.n_bins}"
        return os.path.join(self.cache_dir, h + ".pkl")

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        p = self._cache_path(X)
        if p and os.path.isfile(p):
            with open(p, "rb") as f:
                self.boss_list_ = pickle.load(f)
            return self
        self.boss_list_ = []
        for c in range(X.shape[1]):
            for ws in self.window_sizes:
                b = self._new_boss(ws)
                b.fit(X[:, c, :], y)
                self.boss_list_.append(b)
        if p:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self.boss_list_, f)
            os.replace(tmp, p)  # atomic: array-job tasks may race on the same key
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        out, k = [], 0
        for c in range(X.shape[1]):
            for _ in self.window_sizes:
                z = self.boss_list_[k].transform(X[:, c, :])
                out.append(z.toarray() if hasattr(z, "toarray") else np.asarray(z))
                k += 1
        return np.concatenate(out, axis=1).astype(np.int32)


# ------------------------------------------------------------ classifiers
def make_nn():
    import torch
    from skorch import NeuralNetClassifier
    from skorch.callbacks import EarlyStopping
    from skorch.dataset import ValidSplit
    from torch import nn, optim

    class CustomNN(nn.Module):
        # verbatim structure of ml/nn.py (note: softmax output into CrossEntropyLoss, as in the original)
        def __init__(self, c_in=30, h_layer_sizes=None, nonlin=None):
            super().__init__()
            nonlin = nonlin or nn.LeakyReLU()
            self.h_layer_sizes = list(h_layer_sizes)
            self.nonlin = nonlin
            self.input = nn.Linear(c_in, self.h_layer_sizes[0])
            self.h_layers = None
            if len(self.h_layer_sizes) > 1:
                layers, prev = [], self.h_layer_sizes[0]
                for h in self.h_layer_sizes:
                    layers += [nn.Linear(prev, h), nonlin]
                    prev = h
                self.h_layers = nn.Sequential(*layers)
            self.output = nn.Linear(self.h_layer_sizes[-1], 2)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x, sample_weight=None, **kwargs):
            x = self.nonlin(self.input(x.float()))
            if self.h_layers is not None:
                x = self.h_layers(x)
            return self.softmax(self.output(x))

    class WeightedNN(NeuralNetClassifier):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("criterion__reduction", "none")
            super().__init__(*args, **kwargs)

        def get_loss(self, y_pred, y_true, X, *args, **kwargs):
            loss = super().get_loss(y_pred, y_true, X, *args, **kwargs)
            if isinstance(X, dict) and "sample_weight" in X:
                return (X["sample_weight"] * loss).mean()
            return loss.mean()

        def fit(self, X, y, sample_weight=None, **kw):
            X = torch.as_tensor(np.asarray(X, dtype=np.float32))
            y = np.asarray(y, dtype=np.int64)
            if sample_weight is None:
                return super().fit(X, y)
            sw = torch.as_tensor(np.asarray(sample_weight, dtype=np.float32))
            return super().fit({"x": X, "sample_weight": sw}, y)

    def build(c_in):
        return WeightedNN(module=CustomNN, module__c_in=c_in, module__h_layer_sizes=[50], max_epochs=100,
                          verbose=0, iterator_train__shuffle=True, criterion=nn.CrossEntropyLoss,
                          optimizer=optim.AdamW, train_split=ValidSplit(cv=0.1, stratified=True, random_state=42),
                          batch_size=16, lr=0.001, callbacks=[EarlyStopping(patience=10)],
                          device="cuda" if torch.cuda.is_available() else "cpu")

    return build


def make_estimator(classifier: str):
    steps = []
    if classifier == "svm":
        steps += [("scaler", StandardScaler()), ("clf", SVC(probability=False, verbose=False))]
    elif classifier == "cat":
        from catboost import CatBoostClassifier
        steps += [("scaler", "passthrough"),
                  ("clf", CatBoostClassifier(random_state=42, verbose=False, thread_count=int(os.environ.get("CATBOOST_THREADS", "8"))))]
    elif classifier == "nn":
        build = make_nn()
        steps += [("scaler", StandardScaler()), ("clf", _LazyNN(build))]
    else:
        raise ValueError(classifier)
    return Pipeline(steps)


class _LazyNN(BaseEstimator):
    """Instantiate the skorch net once the (post-BOSS) feature width is known."""

    def __init__(self, build=None, h_layer_sizes=None):
        self.build = build
        self.h_layer_sizes = h_layer_sizes
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, sample_weight=None):
        import torch
        torch.manual_seed(42)
        self.net_ = self.build(np.asarray(X).shape[1])
        if self.h_layer_sizes is not None:
            self.net_.set_params(module__h_layer_sizes=list(self.h_layer_sizes))
        self.net_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(X)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


def balanced_weights(y, groups=None):
    """Official get_balanced_weights: weight = N / (n_groups * count(group)), group = label[_gender]."""
    y = np.asarray(y).astype(str)
    if groups is not None:
        y = np.array([f"{a}_{b}" for a, b in zip(y, np.asarray(groups).astype(str))])
    uniq, counts = np.unique(y, return_counts=True)
    w = {u: len(y) / (len(uniq) * c) for u, c in zip(uniq, counts)}
    return np.array([w[v] for v in y], dtype=np.float32)


def scores_of(est, X):
    """Continuous score for class 1 (for AUROC / threshold sweeps)."""
    clf = est.named_steps["clf"]
    if hasattr(clf, "predict_proba") and not isinstance(clf, SVC):
        return est.predict_proba(X)[:, 1]
    return est.decision_function(X)


def metrics_of(y_true, y_pred, score=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_pos": f1_score(y_true, y_pred, zero_division=0),
        "precision_pos": precision_score(y_true, y_pred, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion": pd.crosstab(pd.Series(y_true, name="t"), pd.Series(y_pred, name="p")).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0).values.tolist(),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        m["roc_auc"] = roc_auc_score(y_true, score)
        # oracle threshold on the test scores -- optimistic by construction, reported only because the
        # table this feeds carries the same column for the neural rows
        best = 0.0
        for thr in np.unique(score):
            best = max(best, balanced_accuracy_score(y_true, (score >= thr).astype(int)))
        m["balanced_accuracy_best_threshold"] = best
    return m


# --------------------------------------------------------------------------- splits
def our_split_subjects(parquet: str, val_ratio=0.2, seed=0):
    """Reproduce imu_lm.data.splits.make_splits for the pads_pddd parquet: group by subject_id
    (string, groupby order), shuffle with RandomState(seed), first round(N*val_ratio) groups = val."""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet, columns=["subject_id"]).to_pandas()
    subs = sorted(t["subject_id"].astype(str).unique().tolist())
    groups = list(subs)  # groupby(sorted keys) yields one group per subject in sorted order
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    n_val = int(round(len(groups) * val_ratio))
    return sorted(groups[:n_val]), sorted(groups[n_val:])


# --------------------------------------------------------------------------- main
def boss_features(X_train, X_test, cache_dir):
    """Fit MultiBOSS on the training subjects only, transform both; cached on disk by content hash."""
    key = None
    if cache_dir:
        h = hashlib.md5(np.ascontiguousarray(X_train).tobytes()).hexdigest()[:16]
        h += "_" + hashlib.md5(np.ascontiguousarray(X_test).tobytes()).hexdigest()[:16]
        key = os.path.join(cache_dir, f"bossfeat_{h}.npz")
        if os.path.isfile(key):
            z = np.load(key)
            return z["ztr"], z["zte"]
    t0 = time.time()
    mb = MultiBOSS(n_channels=X_train.shape[1], cache_dir=cache_dir).fit(X_train)
    ztr, zte = mb.transform(X_train), mb.transform(X_test)
    print(f"  BOSS: fit on {X_train.shape} -> {ztr.shape} features ({time.time()-t0:.0f}s)", flush=True)
    if key:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp.npz")
        os.close(fd)
        np.savez_compressed(tmp, ztr=ztr, zte=zte)  # path already ends in .npz -> no suffix appended
        os.replace(tmp, key)
    return ztr, zte


def run_fold(args, X, y, gender, train_idx, test_idx, n_channels, fold_tag):
    t0 = time.time()
    if args.option == "b":
        Xtr, Xte = boss_features(X[train_idx], X[test_idx], args.boss_cache)
    else:
        Xtr, Xte = X[train_idx], X[test_idx]
    est = make_estimator(args.classifier)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(est, HYPERPARAMS[args.classifier], cv=inner, scoring="balanced_accuracy",
                      n_jobs=args.n_jobs, verbose=0, refit=True)
    w = balanced_weights(y[train_idx], gender[train_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(Xtr, y[train_idx], clf__sample_weight=w)
        pred = gs.predict(Xte)
        sc = scores_of(gs.best_estimator_, Xte)
    m = metrics_of(y[test_idx], pred, sc)
    rec = {
        "fold": fold_tag,
        "inner_cv_balanced_accuracy": float(gs.best_score_),
        "best_params": {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in gs.best_params_.items()},
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "test": {k: (float(v) if not isinstance(v, list) else v) for k, v in m.items()},
        "test_pred": [int(p) for p in pred],
        "test_score": [float(s) for s in sc],
        "seconds": time.time() - t0,
    }
    print(f"[{fold_tag}] inner-cv bal_acc={gs.best_score_*100:.2f}  params={gs.best_params_}  "
          f"TEST bal_acc={m['balanced_accuracy']*100:.2f} acc={m['accuracy']*100:.2f} "
          f"macro_f1={m['macro_f1']:.3f} auc={m.get('roc_auc', float('nan')):.3f}  ({rec['seconds']:.0f}s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pads-root", default=PADS_ROOT_DEFAULT)
    ap.add_argument("--mode", default="pd_vs_dd", choices=["pd_vs_dd", "pd_vs_hc"])
    ap.add_argument("--protocol", default="paper", choices=["paper", "ours"])
    ap.add_argument("--pddd-parquet", default=PDDD_PARQUET_DEFAULT,
                    help="(protocol=ours) parquet whose subject split defines train/val")
    ap.add_argument("--option", default="a", choices=["a", "b"])
    ap.add_argument("--channels", default="both", choices=["acc", "rot", "both"])
    ap.add_argument("--classifier", default="svm", choices=["svm", "cat", "nn"])
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--boss-cache", default=None, help="dir for cached BOSS fits (option b)")
    ap.add_argument("--only-fold", type=int, default=None, help="(paper) run a single outer fold 0..4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fl = load_file_list(args.pads_root, args.mode)
    chan_filter = {"acc": "Acceleration", "rot": "Rotation", "both": None}[args.channels]
    X_raw, channels = load_movement(args.pads_root, fl["id"].values, chan_filter)
    y = fl["label"].values.astype(int)
    gender = fl["gender"].values
    ids = fl["id"].values
    print(f"mode={args.mode} protocol={args.protocol} option={args.option} channels={args.channels} "
          f"clf={args.classifier}  X={X_raw.shape}  y counts={np.bincount(y).tolist()}", flush=True)

    if args.option == "a":
        X = np.stack([feature_extraction_a(x) for x in X_raw])
        print(f"option-a features: {X.shape}", flush=True)
    else:
        X = X_raw  # [N, C, T]; BOSS lives inside the pipeline
    n_channels = X_raw.shape[1]

    folds = []
    if args.protocol == "paper":
        cv = RepeatedStratifiedKFold(n_repeats=1, n_splits=5, random_state=42)
        for k, (tr, te) in enumerate(cv.split(X_raw.reshape(len(X_raw), -1)[:, :1], y)):
            if args.only_fold is not None and k != args.only_fold:
                continue
            folds.append(run_fold(args, X, y, gender, tr, te, n_channels, f"outer{k+1}"))
    else:
        val_subs, train_subs = our_split_subjects(args.pddd_parquet)
        id_str = np.array([f"{int(i):03d}" for i in ids])
        tr = np.where(np.isin(id_str, train_subs))[0]
        te = np.where(np.isin(id_str, val_subs))[0]
        assert len(tr) + len(te) == len(ids), (len(tr), len(te), len(ids))
        print(f"our split: train={len(tr)} subjects (y={np.bincount(y[tr]).tolist()})  "
              f"val={len(te)} subjects (y={np.bincount(y[te]).tolist()})", flush=True)
        rec = run_fold(args, X, y, gender, tr, te, n_channels, "ours_val")
        rec["val_subjects"] = [str(s) for s in id_str[te]]
        folds.append(rec)

    out = {
        "mode": args.mode, "protocol": args.protocol, "option": args.option, "channels": args.channels,
        "classifier": args.classifier, "label_convention": ("0=PD,1=DD" if args.mode == "pd_vs_dd" else "0=HC,1=PD"),
        "n_subjects": int(len(y)), "folds": folds,
        "mean_test_balanced_accuracy": float(np.mean([f["test"]["balanced_accuracy"] for f in folds])),
        "mean_inner_cv_balanced_accuracy": float(np.mean([f["inner_cv_balanced_accuracy"] for f in folds])),
    }
    Path(os.path.dirname(os.path.abspath(args.out))).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"mean TEST bal_acc over {len(folds)} fold(s): {out['mean_test_balanced_accuracy']*100:.2f}  -> {args.out}")


if __name__ == "__main__":
    main()
