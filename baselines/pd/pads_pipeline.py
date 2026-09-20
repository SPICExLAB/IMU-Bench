#!/usr/bin/env python3
"""The PADS authors' movement pipeline (Varghese et al., npj Parkinson's Disease 2024), run under the
IMU-Bench Parkinson's protocol so it can sit in the task table as a task-specific system.

Their smartwatch classifiers are (a) hand-crafted spectral/statistical features and (b) per-channel BOSS
histograms (pyts), each followed by a scaler and an SVM / MLP / CatBoost chosen by nested CV, with
class-balanced sample weights. Ported here as faithfully as the protocol allows:

  same windows        the benchmark's WindowDataset for pd_pads: 2.56 s / 128 samples at 50 Hz, six
                      channels, per-window z-score, both wrists as separate single-IMU sessions
  same split          375 training / 94 evaluation participants (splits.seed 42, grouped by subject)
  same aggregation    mean softmax over a participant's windows -> argmax (bench/score/pd.py)
  their features      (a) Welch band power 1-19 Hz in 1 Hz steps (srate 50 here, 100 in the original),
                          windowed std / abs-energy / abs-max over 4 segments;
                      (b) MultiBOSS: one BOSS per channel and window size (20, 40, 80), step 2, word 2, 3 bins
  their classifiers   CatBoost (500 trees, depth in {4,5,6}) and RBF SVM (C in {0.1,1,10,100}); the grid is
                      resolved by 5-fold GroupKFold over TRAINING participants on window balanced accuracy,
                      never on the 94 evaluation participants

Deviations, all recorded in the result JSON: 50 Hz instead of 100 Hz (the benchmark's cache), 2.56 s
windows instead of whole 10 s assessment steps, SVM fitted on a participant-stratified subsample of
<= --svm-cap windows (RBF SVM is quadratic in N), probability=True on the SVM for mean-prob aggregation.

    python baselines/pd/pads_pipeline.py            # writes bench/results/pd_pads/pads_{boss,features}__external__s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from imu_lm.data.loaders import WindowDataset                       # noqa: E402
from imu_lm.data.splits import build_session_index, make_splits     # noqa: E402
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml    # noqa: E402
from bench.schema import Result, save_result, stamp_provenance      # noqa: E402

FS = 50
CLASSES = 3


# ---- the authors' feature extractors (ml/feature_extraction.py), srate made explicit ------------------
def bandpower(x, srate=FS, steps=FS, cut=(1, 20), scale=True):
    from scipy.signal import welch
    _, pxx = welch(x, fs=srate, nperseg=steps, scaling="density")
    bp = pxx[cut[0]:cut[1]] if cut is not None else pxx
    return np.log10(bp + 1e-12) if scale else bp


def _segmented(x, fn):
    n = len(x) // 4
    return [fn(x[k * n:(k + 1) * n]) for k in range(4)]


def features_a(X):
    """X [N, C, T] -> [N, C * 31] (19 band powers + 4 std + 4 abs-energy + 4 abs-max per channel)."""
    out = []
    for w in X:
        f = []
        for ch in w:
            f.extend(bandpower(ch)); f.extend(_segmented(ch, np.std))
            f.extend(_segmented(ch, lambda s: np.sum(np.abs(s)))); f.extend(_segmented(ch, lambda s: np.max(np.abs(s))))
        out.append(f)
    return np.asarray(out, dtype=np.float32)


class MultiBOSS:
    """ml/multi_boss.py without the on-disk cache: one BOSS per (channel, window size)."""
    def __init__(self, n_channels, window_sizes=(20, 40, 80), window_step=2, word_size=2, n_bins=3):
        from pyts.transformation import BOSS
        self.boss = [[BOSS(sparse=False, window_size=ws, window_step=window_step, word_size=word_size,
                           n_bins=n_bins, norm_std=False) for ws in window_sizes] for _ in range(n_channels)]

    def fit(self, X, y):
        for ch, row in enumerate(self.boss):
            for b in row:
                b.fit(X[:, ch, :], y)
        return self

    def transform(self, X):
        return np.concatenate([b.transform(X[:, ch, :]) for ch, row in enumerate(self.boss) for b in row],
                              axis=1).astype(np.float32)


# ---- data through the benchmark's own dataset class ---------------------------------------------------
def load_windows(cfg, si, keys, split_name):
    ds = WindowDataset(cfg_get(cfg, ["paths", "dataset_path"]), si, keys, cfg, split_name=split_name)
    X, y, subj = [], [], []
    t0 = time.time()
    for i in range(len(ds)):
        item = ds[i]
        if item is None:
            continue
        x, lab = item
        X.append(x.numpy()); y.append(int(lab)); subj.append(ds._keys[ds._sess_idx[i]].subject_id)
        if i and i % 20000 == 0:
            print(f"    {split_name}: {i}/{len(ds)} windows ({time.time() - t0:.0f}s)", flush=True)
    return np.stack(X), np.asarray(y), np.asarray(subj)


def balanced_weights(y):
    counts = np.bincount(y, minlength=CLASSES).astype(np.float64)
    w = (len(y) / (CLASSES * counts))[y]
    return w / w.mean()


def metrics_of(yt, yp):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    return {"acc": float(accuracy_score(yt, yp)), "bal_acc": float(balanced_accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0))}


def subject_aggregate(P, subj):
    """Mean posterior per participant -> argmax; also majority vote. Returns (ids, y_meanprob, y_vote, P_subj)."""
    ids = sorted(set(subj))
    Pm = np.stack([P[subj == s].mean(0) for s in ids]); vote = [np.bincount(P[subj == s].argmax(1), minlength=CLASSES).argmax() for s in ids]
    return ids, Pm.argmax(1), np.asarray(vote), Pm


def select_and_fit(name, make_clf, grid, Xtr, ytr, gtr, wtr, n_jobs):
    """Grid resolved by 5-fold GroupKFold over training participants (window balanced accuracy)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score
    best, best_score = None, -1
    for params in grid:
        scores = []
        for tr, te in GroupKFold(n_splits=5).split(Xtr, ytr, gtr):
            clf = make_clf(params, n_jobs); clf.fit(Xtr[tr], ytr[tr], sample_weight=wtr[tr])
            scores.append(balanced_accuracy_score(ytr[te], clf.predict(Xtr[te])))
        s = float(np.mean(scores)); print(f"    [{name}] {params}: inner bal-acc {s:.4f}", flush=True)
        if s > best_score:
            best, best_score = params, s
    clf = make_clf(best, n_jobs); clf.fit(Xtr, ytr, sample_weight=wtr)
    return clf, best, best_score


def make_cat(params, n_jobs):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=500, depth=params["depth"], random_state=42, verbose=False,
                              thread_count=n_jobs, loss_function="MultiClass")


def make_svm(params, n_jobs):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(), SVC(C=params["C"], gamma="scale", probability=True, random_state=42))


class _SvmWrap:
    """sample_weight has to reach the SVC step of the pipeline."""
    def __init__(self, pipe): self.pipe = pipe
    def fit(self, X, y, sample_weight=None): self.pipe.fit(X, y, svc__sample_weight=sample_weight); return self
    def predict(self, X): return self.pipe.predict(X)
    def predict_proba(self, X): return self.pipe.predict_proba(X)


def write(encoder, metrics, info, notes):
    r = Result(task="pd_pads", encoder=encoder, protocol="external", seed=0, metrics=metrics,
               primary_metric="subject_macro_f1", eval_split="val", run=None, checkpoint=None,
               selection={"split": "train (5-fold GroupKFold by participant)", "monitor": "window bal_acc",
                          "mode": "max"},
               job_ids={"score": os.environ.get("SLURM_JOB_ID")},
               provenance=stamp_provenance("scorer:baselines/pd/pads_pipeline.py", info=info),
               status="ok", notes=notes)
    print("  ->", save_result(r), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", default="configs/vit1d_pd_probe_pads_mae_multidata_flu.yaml")
    ap.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--svm-cap", type=int, default=12000, help="max training windows for the RBF SVM")
    ap.add_argument("--skip-svm", action="store_true")
    a = ap.parse_args()

    cfg = deep_update(load_yaml(os.path.join(REPO, "configs/base.yaml")), load_yaml(os.path.join(REPO, a.model_config)))
    for k in ("augment", "augmentation", "random_rotation", "gain_jitter", "gaussian_noise", "time_warp", "axis_bias"):
        if isinstance(cfg.get(k), dict):
            cfg[k]["enabled"] = False            # the benchmark evaluates un-augmented windows
    si = build_session_index(cfg_get(cfg, ["paths", "dataset_path"]), cfg, dataset_filter=["pads"])
    sp = make_splits(si, cfg)
    print("[pads] loading windows through the benchmark's WindowDataset ...", flush=True)
    Xtr, ytr, gtr = load_windows(cfg, si, sp["train_keys"], "val")   # split_name 'val' => no train-time augmentation
    Xva, yva, gva = load_windows(cfg, si, sp["val_keys"], "val")
    print(f"[pads] train {Xtr.shape} from {len(set(gtr))} participants | val {Xva.shape} from {len(set(gva))} participants", flush=True)
    assert len(set(gtr)) == 375 and len(set(gva)) == 94 and not (set(gtr) & set(gva))
    wtr = balanced_weights(ytr)
    base_info = {"n_train_windows": int(len(ytr)), "n_val_windows": int(len(yva)), "n_train_subjects": 375,
                 "n_val_subjects": 94, "window_seconds": 2.56, "hz": FS, "deviations":
                 "50 Hz cache (original 100 Hz); 2.56 s windows (original 10 s assessment steps); grid by "
                 "GroupKFold on training participants; SVM on a <= svm-cap participant-stratified subsample"}

    def score_and_write(encoder, clf, Xv, feat_name, chosen, inner, extra_note=""):
        P = clf.predict_proba(Xv)
        win = metrics_of(yva, P.argmax(1))
        ids, ymp, yvote, _ = subject_aggregate(P, gva)
        ys = np.asarray([int(yva[gva == s][0]) for s in ids])
        sm, sv = metrics_of(ys, ymp), metrics_of(ys, yvote)
        conf = np.bincount(ymp, minlength=CLASSES).tolist()
        metrics = {"subject_macro_f1": sm["macro_f1"], "subject_bal_acc": sm["bal_acc"], "subject_acc": sm["acc"],
                   "subject_vote_bal_acc": sv["bal_acc"], "window_bal_acc": win["bal_acc"],
                   "window_macro_f1": win["macro_f1"], "window_acc": win["acc"]}
        info = dict(base_info, features=feat_name, classifier=type(clf).__name__, chosen_params=chosen,
                    inner_bal_acc=inner, subject_pred_class_counts=conf)
        print(f"  {encoder}: window {win} | subject mean-prob {sm} | vote bal-acc {sv['bal_acc']:.3f} | pred counts {conf}", flush=True)
        write(encoder, metrics, info, "PADS authors' pipeline (Varghese et al. 2024) under the IMU-Bench protocol. " + extra_note)

    # ---- (a) hand-crafted features -> CatBoost ------------------------------------------------------------
    print("[pads] (a) features ...", flush=True)
    Fa_tr, Fa_va = features_a(Xtr), features_a(Xva)
    clf, chosen, inner = select_and_fit("features+cat", make_cat, [{"depth": d} for d in (4, 5, 6)], Fa_tr, ytr, gtr, wtr, a.n_jobs)
    score_and_write("pads_features", clf, Fa_va, "spectral+statistical (experiment a)", chosen, inner,
                    "Band power 1-19 Hz and windowed std / abs-energy / abs-max per channel, CatBoost.")

    # ---- (b) MultiBOSS -> CatBoost (the paper's best smartwatch-only option) ------------------------------
    print("[pads] (b) MultiBOSS ...", flush=True)
    mb = MultiBOSS(Xtr.shape[1]).fit(Xtr, ytr)
    Fb_tr, Fb_va = mb.transform(Xtr), mb.transform(Xva)
    print(f"    BOSS features: {Fb_tr.shape[1]} per window", flush=True)
    clf, chosen, inner = select_and_fit("boss+cat", make_cat, [{"depth": d} for d in (4, 5, 6)], Fb_tr, ytr, gtr, wtr, a.n_jobs)
    score_and_write("pads_boss", clf, Fb_va, "MultiBOSS (experiment b)", chosen, inner,
                    "Per-channel BOSS histograms (window sizes 20/40/80, word 2, 3 bins), CatBoost.")

    # ---- (b') MultiBOSS -> RBF SVM on a participant-stratified subsample ---------------------------------
    if not a.skip_svm:
        rng = np.random.default_rng(0)
        per = max(1, a.svm_cap // 375)
        keep = np.concatenate([rng.choice(np.where(gtr == s)[0], size=min(per, (gtr == s).sum()), replace=False) for s in sorted(set(gtr))])
        print(f"[pads] (b') BOSS + SVM on {len(keep)} windows ...", flush=True)
        clf, chosen, inner = select_and_fit("boss+svm", lambda p, n: _SvmWrap(make_svm(p, n)),
                                            [{"C": c} for c in (0.1, 1.0, 10.0, 100.0)], Fb_tr[keep], ytr[keep], gtr[keep], wtr[keep], a.n_jobs)
        score_and_write("pads_boss_svm", clf, Fb_va, "MultiBOSS (experiment b)", chosen, inner,
                        f"Per-channel BOSS histograms, RBF SVM with probability outputs fitted on {len(keep)} participant-stratified training windows.")
    print("[pads] done", flush=True)


if __name__ == "__main__":
    main()
