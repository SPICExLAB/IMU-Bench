#!/usr/bin/env python3
"""HHAR / MHEALTH with the static-posture classes set aside.

    python analysis/har_static_vs_dynamic.py [cell ...]

Static classes (HHAR: sit, stand; MHEALTH: standing_still, sitting_relaxing, lying_down) differ mainly in which way
gravity points, the cue a per-window z-score removes. For every cell with a per-window sidecar this reports
  all        macro-F1 / accuracy over all classes (the benchmark's number)
  dynamic    the same over the windows whose TRUE class is dynamic, macro-F1 taken over the dynamic classes only
             (a dynamic window predicted as a static class counts as an error)
  static     the same over the static windows and classes
  merged     all classes, with the static classes merged into ONE class ("still") in truth and prediction:
             what the score would be if the task did not ask which way gravity points
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = {"har_hhar": ("/projects/b53017/vbc1591/hhar_dataset.parquet", {"sit", "stand"}),
      "har_mhealth": ("/projects/b53017/vbc1591/mhealth_dataset.parquet", {"standing_still", "sitting_relaxing", "lying_down"})}
CELLS = sys.argv[1:] or ["mae_5src__probe", "mae_5src__finetune", "scratch__scratch", "harnet5__probe", "harnet5__finetune"]

for task, (pq, static_names) in DS.items():
    d = pd.read_parquet(pq, columns=["dataset_activity_id", "dataset_activity_label"]).drop_duplicates()
    raw2name = dict(zip(d.dataset_activity_id, d.dataset_activity_label))
    print(f"\n=== {task}: static = {sorted(static_names)} ===")
    print(f"{'cell':22s} {'all F1':>7s} {'acc':>6s} | {'dyn F1':>7s} {'acc':>6s} | {'stat F1':>7s} {'acc':>6s} | {'merged F1':>9s} {'acc':>6s}")
    for cell in CELLS:
        f = os.path.join(REPO, "bench", "results", "_units", task, f"{cell}__s42.npz")
        run = json.load(open(os.path.join(REPO, "bench", "results", task, f"{cell}__s42.json")))["run"] if os.path.exists(os.path.join(REPO, "bench", "results", task, f"{cell}__s42.json")) else None
        if not os.path.exists(f) or run is None:
            print(f"{cell:22s} (no per-window predictions yet)"); continue
        lm = json.load(open(os.path.join(REPO, "runs", run, "artifacts", "label_map.json")))
        idx2name = {int(k): raw2name.get(v, str(v)) for k, v in lm["idx_to_raw"].items()}
        z = np.load(f, allow_pickle=True); yt, yp = z["y_true"].astype(int), z["y_pred"].astype(int)
        static = {i for i, n in idx2name.items() if n in static_names}; dyn = sorted(set(idx2name) - static)
        def part(mask, labels):
            return 100 * f1_score(yt[mask], yp[mask], labels=sorted(labels), average="macro", zero_division=0), 100 * float((yt[mask] == yp[mask]).mean())
        m_dyn, m_st = ~np.isin(yt, list(static)), np.isin(yt, list(static))
        S = min(static); merge = np.vectorize(lambda v: S if v in static else v)
        ytm, ypm = merge(yt), merge(yp)
        print(f"{cell:22s} {100*f1_score(yt,yp,average='macro'):7.1f} {100*(yt==yp).mean():6.1f} | {part(m_dyn,dyn)[0]:7.1f} {part(m_dyn,dyn)[1]:6.1f} | "
              f"{part(m_st,static)[0]:7.1f} {part(m_st,static)[1]:6.1f} | {100*f1_score(ytm,ypm,average='macro'):9.1f} {100*(ytm==ypm).mean():6.1f}")
