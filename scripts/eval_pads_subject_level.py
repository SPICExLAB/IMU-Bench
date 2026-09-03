"""Subject-level evaluation for the PADS clinical-classification runs.

The training pipeline scores one 2.56 s window at a time, but PADS' label is a
property of the *patient*, and the original paper's benchmark classifies whole
subjects (its own preprocessing writes one `{id}_ml.bin` per subject). A
window-level number is therefore not comparable to the published baselines.

This script reloads a finished run's best checkpoint, re-derives the exact same
validation split, runs every val window through the frozen encoder + head, and
then aggregates the per-window logits back up to one prediction per subject:

    mean_prob   average softmax over all of a subject's windows, then argmax
                (the default; uses classifier confidence, not just counts)
    vote        argmax per window, then majority vote across the subject

Each val subject contributes ~22 recordings x ~8 windows ~= 180 windows, so the
aggregation has plenty to work with.

Reports window-level and both subject-level metrics side by side, so the cost
of aggregation is visible rather than assumed.

For binary tasks (pdhc / pddd) it additionally reports, at window level and at
subject level (mean-prob score): AUROC, and the balanced accuracy at the
best-possible decision threshold on P(class 1) -- an ORACLE number (threshold
picked on the evaluation split itself), reported only to show how much of the
bal_acc deficit is a calibration / class-prior problem versus a ranking one.
`--dump-npz` writes the per-window and per-subject scores so downstream tables
can be rebuilt without re-running the encoder.

Usage:
    python scripts/eval_pads_subject_level.py \
        --config configs/base.yaml \
        --model-config configs/vit1d_pd_probe_pads_mae_multidata_flu.yaml \
        --run vit1d_pd_probe_pads_mae
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from imu_lm.data.loaders import WindowDataset
from imu_lm.data.splits import build_session_index, make_splits
from imu_lm.probe.head import LinearHead
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml
from imu_lm.utils.metrics import compute_metrics


def build_encoder(cfg):
    from imu_lm.models.baselines.inertia1_encoder import build_encoder as _build_encoder
    alt = bool(cfg_get(cfg, ["vit1d", "encoder", "alt_attention"], False))
    return _build_encoder(cfg), alt   # ViT1D / alt-attn / Inertia-1 adapter per cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--run", required=True, help="run name under paths.runs_root")
    ap.add_argument("--ckpt", default="best", help="best | latest | explicit path")
    ap.add_argument("--split", default="val_keys", choices=["val_keys", "train_keys", "test_keys"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--dump-npz", default=None, help="optional .npz with per-window / per-subject scores")
    args = ap.parse_args()

    cfg = deep_update(load_yaml(args.config), load_yaml(args.model_config))
    runs_root = cfg_get(cfg, ["paths", "runs_root"], "runs")
    run_dir = os.path.join(runs_root, args.run)
    ckpt_path = (
        args.ckpt if os.path.sep in args.ckpt
        else os.path.join(run_dir, "checkpoints", f"{args.ckpt}.pt")
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    parquet = cfg_get(cfg, ["paths", "dataset_path"])
    train_on = cfg_get(cfg, ["data", "train_on_datasets"], None)
    session_index = build_session_index(parquet, cfg, dataset_filter=list(train_on) if train_on else None)
    splits = make_splits(session_index, cfg)
    keys = splits[args.split]
    if not keys:
        raise RuntimeError(f"split {args.split} is empty")

    ds = WindowDataset(parquet, session_index, keys, cfg, split_name=args.split.replace("_keys", ""))

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    encoder, alt = build_encoder(cfg)
    encoder.load_state_dict(state["model"], strict=False)
    num_classes = state["head"]["fc.weight"].shape[0]
    head = LinearHead(encoder.embed_dim, num_classes)
    head.load_state_dict(state["head"])

    # Raw dataset label -> contiguous head index, as fitted during training.
    lm_path = os.path.join(run_dir, "artifacts", "label_map.json")
    raw_to_idx = None
    if os.path.exists(lm_path):
        raw_to_idx = {int(k): int(v) for k, v in json.load(open(lm_path))["raw_to_idx"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device).eval()
    head.to(device).eval()

    # Windows carry no subject id, so recover it from the dataset's own index.
    subj_of_window = [ds._keys[i].subject_id for i in ds._sess_idx]

    win_true, win_pred, win_subj, win_p1 = [], [], [], []
    prob_sum = defaultdict(lambda: np.zeros(num_classes, dtype=np.float64))
    prob_n = defaultdict(int)
    vote = defaultdict(lambda: np.zeros(num_classes, dtype=np.int64))
    subj_true = {}

    buf_x, buf_y, buf_s = [], [], []

    def flush():
        if not buf_x:
            return
        x = torch.stack(buf_x).to(device)
        with torch.no_grad():
            p = torch.softmax(head(encoder(x)), dim=1).cpu().numpy()
        for pi, yi, si in zip(p, buf_y, buf_s):
            win_true.append(yi); win_pred.append(int(pi.argmax())); win_subj.append(si)
            win_p1.append(float(pi[1]) if num_classes == 2 else float("nan"))
            prob_sum[si] += pi; prob_n[si] += 1
            vote[si][int(pi.argmax())] += 1
            subj_true[si] = yi
        buf_x.clear(); buf_y.clear(); buf_s.clear()

    n_skipped = 0
    for idx in range(len(ds)):
        item = ds[idx]
        if item is None:
            n_skipped += 1
            continue
        x, y = item
        y = int(y)
        if raw_to_idx is not None:
            if y not in raw_to_idx:
                n_skipped += 1
                continue
            y = raw_to_idx[y]
        buf_x.append(x); buf_y.append(y); buf_s.append(subj_of_window[idx])
        if len(buf_x) >= args.batch_size:
            flush()
    flush()

    subjects = sorted(subj_true)
    y_true_s = [subj_true[s] for s in subjects]
    y_mean = [int((prob_sum[s] / max(prob_n[s], 1)).argmax()) for s in subjects]
    y_vote = [int(vote[s].argmax()) for s in subjects]

    def metrics_of(yt, yp):
        # compute_metrics returns bal_acc/macro_f1/... but no plain accuracy,
        # so add it here rather than assuming a key that isn't there.
        m = dict(compute_metrics(yt, yp))
        m["acc"] = float(np.mean(np.asarray(yt) == np.asarray(yp))) if yt else 0.0
        return m

    m_win = metrics_of(win_true, win_pred)
    m_mean = metrics_of(y_true_s, y_mean)
    m_vote = metrics_of(y_true_s, y_vote)

    def line(tag, m, n):
        print(f"{tag:26s} n={n:5d}  acc={m['acc']*100:6.2f}%  "
              f"bal_acc={m['bal_acc']*100:6.2f}%  macro_f1={m['macro_f1']:.4f}")

    # Binary-only extras: AUROC and oracle-threshold balanced accuracy on P(class 1),
    # at window level and on the per-subject mean probability.
    binary = {}
    subj_p1 = [float((prob_sum[s] / max(prob_n[s], 1))[1]) for s in subjects] if num_classes == 2 else None
    if num_classes == 2:
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score

        def best_thr(yt, sc):
            yt = np.asarray(yt); sc = np.asarray(sc)
            best, best_t = -1.0, 0.5
            for t in np.unique(sc):
                b = balanced_accuracy_score(yt, (sc >= t).astype(int))
                if b > best:
                    best, best_t = b, float(t)
            return float(best), best_t

        for tag, yt, sc in (("window", win_true, win_p1), ("subject_mean_prob", y_true_s, subj_p1)):
            b, t = best_thr(yt, sc)
            binary[tag] = {"auroc": float(roc_auc_score(yt, sc)) if len(set(yt)) == 2 else float("nan"),
                           "bal_acc_best_threshold": b, "best_threshold": t}

    print(f"\nrun={args.run}  ckpt={os.path.basename(ckpt_path)}  classes={num_classes}  "
          f"alt_attn={alt}  device={device.type}")
    print(f"windows scored={len(win_true)} skipped={n_skipped}  subjects={len(subjects)}")
    line("window-level", m_win, len(win_true))
    line("subject-level (mean_prob)", m_mean, len(subjects))
    line("subject-level (vote)", m_vote, len(subjects))
    for tag, b in binary.items():
        print(f"{tag:26s} AUROC={b['auroc']:.4f}  bal_acc@best-thr={b['bal_acc_best_threshold']*100:6.2f}% "
              f"(thr={b['best_threshold']:.3f})")

    result = {
        "run": args.run, "ckpt": ckpt_path, "num_classes": num_classes,
        "n_windows": len(win_true), "n_subjects": len(subjects),
        "window": {k: m_win[k] for k in ("acc", "bal_acc", "macro_f1")},
        "subject_mean_prob": {k: m_mean[k] for k in ("acc", "bal_acc", "macro_f1")},
        "subject_vote": {k: m_vote[k] for k in ("acc", "bal_acc", "macro_f1")},
        "subject_confusion_mean_prob": m_mean.get("confusion"),
        "binary_extras": binary or None,
    }
    if args.dump_npz:
        np.savez_compressed(
            args.dump_npz,
            win_true=np.asarray(win_true), win_pred=np.asarray(win_pred), win_p1=np.asarray(win_p1),
            win_subject=np.asarray(win_subj), subjects=np.asarray(subjects), subj_true=np.asarray(y_true_s),
            subj_pred_mean=np.asarray(y_mean), subj_pred_vote=np.asarray(y_vote),
            subj_p1_mean=np.asarray(subj_p1 if subj_p1 is not None else []),
        )
        print(f"wrote {args.dump_npz}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
