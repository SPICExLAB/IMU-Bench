#!/usr/bin/env python3
"""t-SNE of the reference 5-source MAE encoder -- v6: fine-grained labels.

v5 coloured seven semantic FAMILIES because v4's per-activity legend could not be
resolved at column width. v6 keeps the detail and drops the legend instead: hue is
the family, and the individual activity names are written on the plot at each
activity's medoid. The reader gets ~200 distinct activities in the point cloud and
~30 of them named in place, with no key to cross-reference.

What v6 adds to the window set (user request 2026-09-23):

  TransfHAR   60 fine-grained everyday hand activities, a downstream set that was
              missing entirely. This is the region the paper's claim is about --
              "locomotion travels between corpora, fine manipulation does not" --
              and v5 had almost no fine manipulation in it.
  PRISM       its 96 procedure steps instead of the single "cooking" bucket. PRISM
              is held out, so this also strengthens the exposure control.
  UTD-MHAD    its 18 hand gestures, dropped wholesale in 2026-08, restored.
  OpenWatch   5 microgestures + background, the finest motions in the benchmark.
  OCDetect    two of the FIVE sources of the reference corpus, which carry no
  EatingSpeed activity label and so were absent from every earlier version: the
              figure claimed to show this encoder's space while omitting 40 % of
              what it was trained on. Plotted unlabelled, in grey.

Labels: the printed NAME is the raw (synonym-normalised) label; the COLOUR is the
family that v4's validated bucket_label maps it to. So un-bucketing changes what is
written, never how a point is coloured, and the families stay the validated ones
plus "gesture" and "object handling" for the two vocabularies v4 never carried.

Windows shorter than 10.24 s are wrap-padded, as in v4. OpenWatch (3.0 s) and
UTD-MHAD are the most heavily padded sources, so the report prints the wrap factor
per source: an island at a high wrap factor is a padding artefact, not a finding.

Usage
-----
    python scripts/plot_ssl_embeddings_v6.py                 # SLURM, needs a GPU
    python scripts/plot_ssl_embeddings_v6.py --cache <npz>   # redraw only
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_ssl_embeddings_v4 import (  # noqa: E402
    ACT_NORM,
    DATASETS as V4_DATASETS,
    DS_COLOR_MAP as V4_DS_COLOR,
    FAMILY_COLORS,
    PER_DATASET_CAP,
    SEED,
    SENSORS,
    SKIP_LABELS,
    T_MIN_SHORT,
    T_WIN,
    UTD_GESTURE_LABELS,
    bucket_label,
)
from plot_ssl_embeddings_v5 import (  # noqa: E402
    CELL_COLOR,
    CELL_ORDER,
    MARK_IN,
    MARK_OUT,
    REF_CKPT,
    SYNTHETIC,
    embed,
)
from plot_ssl_embeddings_v5 import IN_POOL as V5_IN_POOL  # noqa: E402

# ------------------------------------------------------------------ sources ----
USERSTUDY = "/projects/b53017/vbc1591/userstudy_watch_dataset.parquet"
DATASETS = dict(V4_DATASETS, userstudy_watch=USERSTUDY)
# Sources whose own vocabulary is the point: keep the raw label as the printed name
# instead of collapsing it. bucket_label still runs, but only to pick the colour.
FINE_SOURCES = {"userstudy_watch", "prism", "utd_mhad"}
# RecoFit's 72 gym variants stay bucketed: un-bucketing them would put 72 near-
# identical dumbbell names into a region the paper makes no claim about.

# Preprocessed [N, 6, 512] validation windows, no activity label.
CACHE = "/projects/b53017/vbc1591/mae_multidata_cache"
UNLABELLED = {"ocdetect": f"{CACHE}/ocdetect_val_x.npy",
              "eatingspeed": f"{CACHE}/eatingspeed_val_x.npy"}
SYNTH_SOURCES = {"nymeria": f"{CACHE}/nymeria_val_x.npy", "motionx": f"{CACHE}/motionx_val_x.npy",
                 "sonic": f"{CACHE}/sonic_val_x.npy", "humanml3d": f"{CACHE}/humanml3d_val_x.npy"}
OPENWATCH_ROOT = "/projects/b53017/vbc1591/openwatch_raw/open_watch_without_augmentations"
OPENWATCH_FS = 100.0            # raw rate; resampled to the 50 Hz of every other source

IN_POOL = V5_IN_POOL | {"ocdetect", "eatingspeed"}      # both are reference-corpus sources
NO_LABEL = "(no label)"
DS_COLOR_MAP = dict(V4_DS_COLOR, userstudy_watch="#c2427a", openwatch="#8a2c58",
                    ocdetect="#7a9a3f", eatingspeed="#b58a2b")

# ----------------------------------------------------------------- families ----
FAMILY_MAIN = {"locomotion": "#2a78d6", "cycling": "#eb6834", "household": "#1baf7a",
               "posture": "#eda100", "hand/desk": "#e87ba4", "sport": "#008300",
               "stairs": "#4a3aa7",
               # the two vocabularies v4 never carried
               "gesture": "#00838f", "object handling": "#795548"}
FAMILY_ORDER = ["locomotion", "cycling", "stairs", "sport", "household",
                "object handling", "hand/desk", "gesture", "posture"]

# TransfHAR's 60 names, split by what the hand is doing. Everything not matched is
# object handling, which is what most of that vocabulary is ("unscrewing bottle",
# "tying shoelace", "organizing fridge").
TRANSFHAR_DESK = ("typing", "using mouse", "using stylus", "clicking pen", "phone scrolling",
                  "reading manga", "drawing", "writing")
TRANSFHAR_GESTURE = ("air heart", "in air", "finger snapping", "fist bump", "fist pump",
                     "hand gesture sequence", "pointing", "rock paper scissors", "waving")
TRANSFHAR_SPORT = ("basketball shooting", "shadow boxing", "karate chop", "badminton")


def family_of(lab: str, ds: str) -> str:
    """Colour family. The validated v4 map decides wherever it can; the two new
    vocabularies (watch gestures, TransfHAR manipulation) get explicit rules."""
    if lab == NO_LABEL:
        return "other"
    if ds == "openwatch" or (ds == "utd_mhad" and lab in UTD_GESTURE_LABELS):
        return "gesture"
    if ds == "userstudy_watch":
        if any(k in lab for k in TRANSFHAR_GESTURE):
            return "gesture"
        if any(k in lab for k in TRANSFHAR_SPORT):
            return "sport"
        if any(k in lab for k in TRANSFHAR_DESK):
            return "hand/desk"
        return "object handling"
    b = bucket_label(ACT_NORM.get(lab, lab))
    return FAMILY_COLORS[b][1] if b in FAMILY_COLORS else "other"


def cell_of(ds: str) -> str:
    return (f"{'mocap-synth' if ds in SYNTHETIC else 'real'}, "
            f"{'in pool' if ds in IN_POOL else 'held out'}")


# ------------------------------------------------------------------ windows ----
def _prep_cfg():
    return {"preprocess": {
        "impute": {"enabled": True, "method": "linear", "max_missing_frac": 0.2},
        "filter": {"enabled": True, "type": "lowpass", "high_hz": 24, "low_hz": 0.25, "order": 4},
        "normalize": {"enabled": True, "method": "zscore", "eps": 1e-6}}}


def _finish(w, cfg, out_X, out_ds, out_act, ds, lab, wraps):
    from imu_lm.data.augmentations.preprocess import preprocess_window
    xp = preprocess_window(w, cfg, None)
    if xp is None:
        return False
    n = xp.shape[1]
    if n < T_WIN:
        wraps.setdefault(ds, []).append(T_WIN / n)
        xp = np.pad(xp, ((0, 0), (0, T_WIN - n)), mode="wrap")
    else:
        wraps.setdefault(ds, []).append(1.0)
    out_X.append(xp.astype(np.float32)); out_ds.append(ds); out_act.append(lab)
    return True


def _openwatch_records(rng):
    """Raw OpenWatch recordings with their gesture name. 3.0 s at 100 Hz -> 150
    samples at 50 Hz, so every one of them is wrap-padded (factor ~3.4).

    dedup_exact=True: the release ships mirrored and duplicated recordings
    (Appendix Data Audit), and a duplicate pair is its own nearest neighbour, which
    would inflate this source's isolation for a reason that has nothing to do with
    the representation."""
    from imu_lm.data.openwatch_loader import (_read_csv_array, apply_unit_scale,
                                              discover_openwatch_records)
    recs, _label_map = discover_openwatch_records(root=OPENWATCH_ROOT, dedup_exact=True)
    out = [(r.path, str(r.gesture).replace("_", " ").strip().lower()) for r in recs]
    rng.shuffle(out)
    return out, _read_csv_array, apply_unit_scale


def extract_windows():
    import pyarrow.dataset as pa_ds

    cfg, rng = _prep_cfg(), np.random.default_rng(SEED)
    X, ds_l, act_l, wraps = [], [], [], {}

    for ds_name, path in DATASETS.items():
        d = pa_ds.dataset(path, format="parquet")
        cols = SENSORS + ["dataset", "subject_id", "session_id", "dataset_activity_label"]
        df = d.to_table(columns=cols,
                        filter=(pa_ds.field("dataset") == ds_name)).to_pandas()
        fine = ds_name in FINE_SOURCES
        wins = []
        for _, g in df.groupby(["subject_id", "session_id"], sort=False):
            arr = g[SENSORS].to_numpy(dtype=np.float32)
            labs = (g["dataset_activity_label"].astype(str).str.replace("_", " ", regex=False)
                    .str.strip().str.lower().to_numpy())
            spans = ([(s, s + T_WIN) for s in range(0, len(arr) - T_WIN + 1, T_WIN)]
                     if len(arr) >= T_WIN else
                     ([(0, len(arr))] if len(arr) >= T_MIN_SHORT else []))
            for a0, a1 in spans:
                wl = labs[a0:a1]
                vals, cnts = np.unique(wl, return_counts=True)
                top = int(np.argmax(cnts))
                if cnts[top] / len(wl) < 0.8:
                    continue
                lab = vals[top]
                if lab in ("nan", "none", "", "unknown", "-1"):
                    continue
                # v4 dropped UTD-MHAD's gestures wholesale; v6 keeps them (user request)
                lab = ACT_NORM.get(lab, lab)
                if not fine:
                    lab = bucket_label(lab)
                if lab in SKIP_LABELS:
                    continue
                wins.append((arr[a0:a1], lab))
        rng.shuffle(wins)
        kept = sum(_finish(w, cfg, X, ds_l, act_l, ds_name, lab, wraps)
                   for w, lab in wins[:PER_DATASET_CAP])
        print(f"[{ds_name}] {kept} windows, {len({l for _, l in wins[:PER_DATASET_CAP]})} labels",
              flush=True)

    # OpenWatch: labelled, but every clip is short and therefore wrap-padded
    try:
        recs, read_csv, unit_scale = _openwatch_records(rng)
        from scipy.signal import resample_poly
        kept, labs_seen = 0, set()
        for path, name in recs:
            if kept >= PER_DATASET_CAP:
                break
            try:
                a = read_csv(path, include_ppg=False)
                a = unit_scale(a, False)
            except Exception:
                continue
            a = np.asarray(a, dtype=np.float32)
            if a.ndim != 2 or a.shape[0] < 6:
                continue
            if a.shape[0] < a.shape[1]:            # [C, T] -> [T, C]
                a = a.T
            a = resample_poly(a[:, :6], 1, 2, axis=0).astype(np.float32)   # 100 -> 50 Hz
            if len(a) < T_MIN_SHORT:
                continue
            if _finish(a[:T_WIN], cfg, X, ds_l, act_l, "openwatch", name, wraps):
                kept += 1; labs_seen.add(name)
        print(f"[openwatch] {kept} windows, {len(labs_seen)} labels: "
              f"{sorted(labs_seen)}", flush=True)
    except Exception as e:                                   # never lose the figure over one source
        print(f"[openwatch] SKIPPED: {type(e).__name__}: {e}", flush=True)

    # preprocessed caches: the mocap-synth sources, plus the two unlabelled
    # reference-corpus sources that no earlier version plotted
    for src, path in {**SYNTH_SOURCES, **UNLABELLED}.items():
        arr = np.load(path, mmap_mode="r")
        idx = rng.choice(len(arr), size=min(PER_DATASET_CAP, len(arr)), replace=False)
        for i in idx:
            X.append(np.asarray(arr[i], dtype=np.float32))
            ds_l.append(src); act_l.append(NO_LABEL)
            wraps.setdefault(src, []).append(1.0)
        print(f"[{src}] {len(idx)} windows (unlabelled)" if src in UNLABELLED
              else f"[{src}] {len(idx)} windows", flush=True)

    wrap_mean = {k: float(np.mean(v)) for k, v in wraps.items()}
    return (np.stack(X), np.array(ds_l), np.array(act_l), wrap_mean)


# --------------------------------------------------------------- statistics ----
def analysis(Z, ds_lab, act_lab, wrap, out, ckpt_path):
    """v5's metrics on the v6 window set, plus the per-source wrap factor, so an
    island produced by padding a 3 s clip out to 10.24 s cannot pass as a finding."""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    Zn = normalize(Z)
    N, K = len(Zn), 10
    S = Zn @ Zn.T
    np.fill_diagonal(S, -np.inf)
    nn_idx = np.argpartition(-S, K, axis=1)[:, :K]

    same_ds_row = (ds_lab[nn_idx] == ds_lab[:, None]).mean(axis=1)
    same_ds = float(same_ds_row.mean())
    lab_m = act_lab != NO_LABEL
    same_act_row = (act_lab[nn_idx] == act_lab[:, None]).mean(axis=1)
    same_act = float(same_act_row[lab_m].mean())

    _, ds_cnt = np.unique(ds_lab, return_counts=True)
    ds_chance = float(((ds_cnt / N) ** 2).sum())
    _, ac_cnt = np.unique(act_lab[lab_m], return_counts=True)
    act_chance = float(((ac_cnt / lab_m.sum()) ** 2).sum())

    fam = np.array([family_of(a, d) for a, d in zip(act_lab, ds_lab)])
    fam_m = fam != "other"
    same_fam_row = (fam[nn_idx] == fam[:, None]).mean(axis=1)
    same_fam = float(same_fam_row[fam_m].mean())
    _, f_cnt = np.unique(fam[fam_m], return_counts=True)
    fam_chance = float(((f_cnt / fam_m.sum()) ** 2).sum())

    sil_ds = float(silhouette_score(Zn, ds_lab, metric="cosine"))
    big = [a for a, c in zip(*np.unique(act_lab, return_counts=True)) if c >= 20 and a != NO_LABEL]
    keep = np.isin(act_lab, big)
    sil_act = float(silhouette_score(Zn[keep], act_lab[keep], metric="cosine"))

    per_ds = sorted(((ds, int((ds_lab == ds).sum()), float(same_ds_row[ds_lab == ds].mean()),
                      float((ds_lab == ds).mean()), cell_of(ds), wrap.get(ds, 1.0))
                     for ds in sorted(set(ds_lab))), key=lambda r: -r[2])
    cells = {}
    for c in CELL_ORDER:
        m = np.array([cell_of(d) == c for d in ds_lab])
        if m.any():
            cells[c] = (int(m.sum()), float(same_ds_row[m].mean()))
    held = np.array([d not in IN_POOL for d in ds_lab]) & lab_m
    inp = np.array([d in IN_POOL for d in ds_lab]) & lab_m
    act_held = float(same_act_row[held].mean()) if held.any() else float("nan")
    act_in = float(same_act_row[inp].mean()) if inp.any() else float("nan")

    # cross-dataset activity retrieval: neighbours restricted to OTHER datasets
    S_cross = S.copy()
    for ds in np.unique(ds_lab):
        m = ds_lab == ds
        S_cross[np.ix_(m, m)] = -np.inf
    nn_x = np.argpartition(-S_cross, K, axis=1)[:, :K]
    cross_act = (act_lab[nn_x] == act_lab[:, None]).mean(axis=1)
    cross_fam = (fam[nn_x] == fam[:, None]).mean(axis=1)

    shared, lines = [], []
    for a in np.unique(act_lab[lab_m]):
        m = act_lab == a
        nd = len(np.unique(ds_lab[m]))
        if nd >= 2 and m.sum() >= 20:
            shared.append(a)
            lines.append((a, int(m.sum()), nd, float(cross_act[m].mean()), float(m.mean())))
    lines.sort(key=lambda r: -r[3])

    fam_lines, fam_single = [], []
    for f_ in FAMILY_ORDER:
        m = fam == f_
        if m.sum() < 20:
            continue
        nd = len(np.unique(ds_lab[m]))
        row = (f_, int(m.sum()), nd, float(cross_fam[m].mean()), float(m.mean()))
        # a family only one dataset carries scores 0 because there is nothing to
        # retrieve from, which says nothing about the representation
        (fam_lines if nd >= 2 else fam_single).append(row)
    fam_lines.sort(key=lambda r: -r[3])

    lodo, sh = [], np.isin(act_lab, shared)
    for ds in np.unique(ds_lab):
        te, tr = sh & (ds_lab == ds), sh & (ds_lab != ds)
        if te.sum() < 10 or tr.sum() < 50:
            continue
        te = te & np.isin(act_lab, list(set(act_lab[tr])))
        if te.sum() < 10:
            continue
        sims = Zn[te] @ Zn[tr].T
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        lab_tr = act_lab[tr]
        preds = [np.unique(lab_tr[r], return_counts=True)[0][
                     np.argmax(np.unique(lab_tr[r], return_counts=True)[1])] for r in top]
        lodo.append((ds, int(te.sum()), float((np.array(preds) == act_lab[te]).mean()),
                     float(np.max(np.unique(act_lab[te], return_counts=True)[1]) / te.sum()),
                     ds in IN_POOL))

    p = f"{out}/analysis_report_v6.md"
    with open(p, "w") as f:
        f.write("# SSL embedding space, reference 5-source encoder -- v6 (fine labels)\n\n")
        f.write(f"Encoder: `{ckpt_path}`\n\n")
        f.write(f"N={N} windows from {len(set(ds_lab))} sources, {int(lab_m.sum())} of them "
                f"activity-labelled across **{len(set(act_lab[lab_m]))} distinct activities**; "
                f"k={K} cosine neighbours on {Z.shape[1]}-d frozen embeddings.\n\n")
        f.write("## 1. kNN neighbourhood purity\n\n")
        f.write(f"- same-DATASET: **{same_ds:.3f}** (chance {ds_chance:.3f}, {same_ds/ds_chance:.1f}x)\n")
        f.write(f"- same-ACTIVITY: **{same_act:.3f}** (chance {act_chance:.3f}, {same_act/act_chance:.1f}x)\n")
        f.write(f"- same-FAMILY: **{same_fam:.3f}** (chance {fam_chance:.3f}, {same_fam/fam_chance:.1f}x)\n\n")
        f.write("## 2. Silhouette (cosine)\n\n")
        f.write(f"- by dataset: **{sil_ds:.3f}**\n- by activity (n>=20): **{sil_act:.3f}**\n\n")
        f.write("## 3. Isolation by exposure and signal origin (the control)\n\n")
        f.write("| cell | n | same-dataset@10 |\n|---|---|---|\n")
        for c in CELL_ORDER:
            if c in cells:
                f.write(f"| {c} | {cells[c][0]} | {cells[c][1]:.3f} |\n")
        f.write(f"\nActivity purity among labelled windows: in-pool {act_in:.3f}, "
                f"held-out {act_held:.3f}.\n\n")
        pad_ok = np.array([wrap.get(d, 1.0) <= 2.0 for d in ds_lab])
        f.write("The same control with the wrap-padded sources dropped, because a clip "
                "padded out to 10.24 s can be isolated by its own periodicity:\n\n")
        f.write("| cell | n | same-dataset@10 |\n|---|---|---|\n")
        for c in CELL_ORDER:
            m = np.array([cell_of(d) == c for d in ds_lab]) & pad_ok
            if m.any():
                f.write(f"| {c} | {int(m.sum())} | {float(same_ds_row[m].mean()):.3f} |\n")
        f.write(f"\nDropped: {', '.join(sorted(d for d in set(ds_lab) if wrap.get(d, 1.0) > 2.0))}.\n\n")
        # How much of the exposure contrast rides on one source? Recompute it with
        # each real source left out in turn. A contrast that changes sign under
        # leave-one-out is not a group-level property.
        real = np.array([d not in SYNTHETIC for d in ds_lab]) & pad_ok
        def _contrast(keep):
            a = np.array([d in IN_POOL for d in ds_lab]) & real & keep
            b = np.array([d not in IN_POOL for d in ds_lab]) & real & keep
            if not a.any() or not b.any():
                return None
            return float(same_ds_row[b].mean() - same_ds_row[a].mean())
        base = _contrast(np.ones(N, bool))
        loo = sorted(((d, _contrast(ds_lab != d)) for d in sorted(set(ds_lab[real]))),
                     key=lambda r: (r[1] is None, r[1]))
        f.write("**Robustness of the exposure contrast.** held-out minus in-pool, real "
                f"sources only, padded sources already dropped: **{base:+.3f}**. "
                "Leaving out one source at a time:\n\n")
        f.write("| source left out | held-out minus in-pool |\n|---|---|\n")
        for d, v in loo:
            if v is not None:
                f.write(f"| {d} | {v:+.3f} |\n")
        vals = [v for _, v in loo if v is not None]
        if vals:
            flips = "yes" if min(vals) < 0 < max(vals) else "no"
            f.write(f"\nRange {min(vals):+.3f} to {max(vals):+.3f}; changes sign under "
                    f"leave-one-out: **{flips}**. For comparison, the mocap-synth minus "
                    f"real contrast is "
                    f"{float(same_ds_row[np.array([d in SYNTHETIC for d in ds_lab])].mean()) - float(same_ds_row[np.array([d not in SYNTHETIC for d in ds_lab])].mean()):+.3f}.\n\n")
        f.write("## 4. Per-dataset isolation, with the padding control\n\n")
        f.write("`wrap` is the mean factor a source's windows were wrap-padded by to reach "
                "10.24 s. A source that is BOTH isolated and heavily padded is not evidence "
                "of anything about the representation.\n\n")
        f.write("| dataset | n | same-dataset@10 | chance | cell | wrap |\n|---|---|---|---|---|---|\n")
        for ds, n, v, ch, c, w in per_ds:
            f.write(f"| {ds} | {n} | {v:.3f} | {ch:.3f} | {c} | {w:.2f}x |\n")
        f.write("\n## 5. Cross-dataset retrieval, by family\n\n")
        f.write("| family | n | #datasets | cross-ds same-family@10 | chance |\n|---|---|---|---|---|\n")
        for a, n, nd, sc, ch in fam_lines:
            f.write(f"| {a} | {n} | {nd} | {sc:.3f} | {ch:.3f} |\n")
        for a, n, nd, sc, ch in fam_single:
            f.write(f"| {a} | {n} | 1 | n/a | {ch:.3f} |\n")
        if fam_single:
            f.write("\nn/a: only one dataset carries this family, so there is nothing to "
                    "retrieve from and the score is 0 by construction.\n")
        f.write("\n## 6. Cross-dataset retrieval, by activity (>=2 datasets, n>=20)\n\n")
        f.write("| activity | n | #datasets | cross-ds same-act@10 | chance |\n|---|---|---|---|---|\n")
        for a, n, nd, sc, ch in lines:
            f.write(f"| {a} | {n} | {nd} | {sc:.3f} | {ch:.3f} |\n")
        f.write("\n## 7. Leave-one-dataset-out kNN activity classification\n\n")
        f.write("| held-out dataset | in pool | n test | kNN acc | majority |\n|---|---|---|---|---|\n")
        for ds, n, acc, maj, isin in lodo:
            f.write(f"| {ds} | {'yes' if isin else 'no'} | {n} | {acc:.3f} | {maj:.3f} |\n")
    print("wrote", p, flush=True)
    return dict(N=N, n_act=len(set(act_lab[lab_m])), same_ds=same_ds, ds_chance=ds_chance,
                same_act=same_act, act_chance=act_chance, same_fam=same_fam,
                fam_chance=fam_chance, sil_ds=sil_ds, sil_act=sil_act, cells=cells,
                act_in=act_in, act_held=act_held)


# ------------------------------------------------------------------ figures ----
def _frame(ax):
    ax.set_facecolor("#ffffff")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d7d6cc"); sp.set_linewidth(0.8)


def _anchors(xy, labels, min_n, keep):
    """Per label: the 2-D median (robust to the tail t-SNE throws out) and a
    compactness score. A label is worth naming when its points sit together; a
    label smeared over the whole panel has no place to put its name."""
    out = []
    for a in np.unique(labels):
        if a not in keep:
            continue
        m = labels == a
        if m.sum() < min_n:
            continue
        p = xy[m]
        c = np.median(p, axis=0)
        spread = float(np.median(np.linalg.norm(p - c, axis=1)))
        out.append((a, c, int(m.sum()), spread))
    if not out:
        return []
    scale = float(np.median([np.linalg.norm(xy - np.median(xy, axis=0), axis=1).max()]))
    # compact first, then large: a tight cluster of 40 is more nameable than a
    # diffuse cloud of 300
    out.sort(key=lambda r: (r[3] / max(scale, 1e-9), -r[2]))
    return out


def _density_anchor(p, bins=30):
    """The densest spot of a point set: a bimodal family's median falls between its
    lobes, which is usually somewhere another family occupies."""
    from scipy.ndimage import uniform_filter
    if len(p) < 12:
        return np.median(p, axis=0)
    H, xe, ye = np.histogram2d(p[:, 0], p[:, 1], bins=bins)
    i, j = np.unravel_index(np.argmax(uniform_filter(H, size=3, mode="constant")), H.shape)
    m = ((p[:, 0] >= xe[i]) & (p[:, 0] <= xe[i + 1])
         & (p[:, 1] >= ye[j]) & (p[:, 1] <= ye[j + 1]))
    return np.median(p[m], axis=0) if m.sum() >= 3 else np.median(p, axis=0)


def _place(ax, anchors, xy, fontsize, max_labels, color, weight="normal", pad=1.2,
           placed=None, force=False):
    """Greedy placement that rejects a name whose box would touch one already
    placed. The box is sized from the name's own length, so "drawing circle in
    air" reserves the room it actually needs and a short name does not."""
    import matplotlib.patheffects as pe
    fig = ax.figure
    fig.canvas.draw()                       # the axes box is only known once drawn
    bb = ax.get_window_extent()
    ax_w_pt = max(bb.width * 72.0 / fig.dpi, 1.0)
    ax_h_pt = max(bb.height * 72.0 / fig.dpi, 1.0)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    h = fontsize * 1.45 / ax_h_pt * pad     # one line of text, in axis fraction
    placed = [] if placed is None else placed
    n = 0
    short = lambda t: (t if len(t) <= 24 else t[:23].rstrip() + "\u2026")
    for name, c, cnt, _ in anchors:
        if n >= max_labels:
            break
        f = (c - lo) / span
        name = short(str(name))
        w = len(name) * fontsize * 0.50 / ax_w_pt * pad
        free = lambda g: not any(abs(g[0] - q[0]) < (w + q[2]) / 2
                                 and abs(g[1] - q[1]) < h for q in placed)
        if not free(f):
            if not force:
                continue
            # a family name is the only key this figure has, so nudge it rather
            # than drop it: rings of candidate offsets, nearest first
            for r in (1.0, 2.0, 3.0):
                cand = [(f[0] + dx * h * r * 1.6, f[1] + dy * h * r)
                        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0),
                                       (1, 1), (-1, 1), (1, -1), (-1, -1))]
                cand = [g for g in cand if 0.02 < g[0] < 0.98 and 0.02 < g[1] < 0.98]
                hit = next((g for g in cand if free(g)), None)
                if hit is not None:
                    f = np.array(hit)
                    break
            c = lo + f * span
        col = color(name) if callable(color) else color
        ax.text(c[0], c[1], name, fontsize=fontsize, color=col, weight=weight,
                ha="center", va="center", zorder=6, clip_on=True,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="#ffffff")])
        placed.append((f[0], f[1], w)); n += 1
    return n, placed


def panel_activity(ax, xy, ds_lab, act_lab, s=2.4, name_fs=4.6, fam_fs=7.2, max_names=34):
    """Hue = family, names written in place, no legend (user request 2026-09-23)."""
    fam = np.array([family_of(a, d) for a, d in zip(act_lab, ds_lab)])
    other = fam == "other"
    if other.any():                       # unlabelled / unmapped windows as the backdrop
        ax.scatter(xy[other, 0], xy[other, 1], s=s * 0.8, c="#dbdad2", alpha=0.65,
                   linewidths=0, zorder=1)
    for f_ in FAMILY_ORDER:
        m = fam == f_
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=s, c=FAMILY_MAIN[f_], alpha=0.85,
                       linewidths=0, zorder=2)
    # family names go down first and keep their room: they are the figure's key,
    # since there is no legend. Activity names then fill what is left.
    fam_anchor = [(f_, _density_anchor(xy[fam == f_]), int((fam == f_).sum()), 0.0)
                  for f_ in FAMILY_ORDER if (fam == f_).sum() >= 20]
    fam_anchor.sort(key=lambda r: -r[2])           # biggest family gets first pick
    _, occupied = _place(ax, fam_anchor, xy, fam_fs, len(fam_anchor),
                         lambda f_: FAMILY_MAIN[f_], weight="bold", pad=1.5, force=True)
    named = set(np.unique(act_lab[~other]))
    n_act, _ = _place(ax, _anchors(xy[~other], act_lab[~other], 12, named), xy,
                      name_fs, max_names, "#1a1a1a", pad=1.5, placed=occupied)
    _frame(ax)
    return n_act


def panel_dataset(ax, xy, ds_lab, fontsize=6.0, s=7):
    for ds in sorted(set(ds_lab)):
        m = ds_lab == ds
        isin = ds in IN_POOL
        ax.scatter(xy[m, 0], xy[m, 1], s=s if isin else s * 1.5,
                   c=DS_COLOR_MAP.get(ds, "#9b9a92"), marker=MARK_IN if isin else MARK_OUT,
                   alpha=0.8, linewidths=0,
                   label=f"{ds} ({int(m.sum())})" + ("" if isin else " — held out"))
    _frame(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=fontsize,
              framealpha=0.95, markerscale=1.8, handletextpad=0.3, borderpad=0.4)


def panel_cells(ax, xy, ds_lab, fontsize=8.0, s=7):
    cl = np.array([cell_of(d) for d in ds_lab])
    for c in CELL_ORDER:
        m = cl == c
        if not m.any():
            continue
        isin = "in pool" in c
        ax.scatter(xy[m, 0], xy[m, 1], s=s if isin else s * 1.5, c=CELL_COLOR[c],
                   marker=MARK_IN if isin else MARK_OUT, alpha=0.75, linewidths=0,
                   label=f"{c} ({int(m.sum())})")
    _frame(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=fontsize,
              framealpha=0.95, markerscale=1.8, handletextpad=0.3, borderpad=0.4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=REF_CKPT)
    ap.add_argument("--out", default="analysis/embedding_viz")
    ap.add_argument("--cache", default=None, help="reuse an existing _v6.npz")
    ap.add_argument("--no-analysis", action="store_true")
    ap.add_argument("--max-names", type=int, default=34)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    os.makedirs(args.out, exist_ok=True)
    npz = args.cache or f"{args.out}/ssl_embeddings_v6.npz"
    if args.cache and os.path.exists(npz):
        d = np.load(npz, allow_pickle=True)
        Z, ts, ds_lab, act_lab = d["Z"], d["tsne"], d["dataset"], d["activity"]
        wrap = dict(d["wrap"].item()) if "wrap" in d else {}
        print(f"[cache] {npz}: {Z.shape}", flush=True)
    else:
        X, ds_lab, act_lab, wrap = extract_windows()
        print(f"total windows: {len(X)}; distinct activities: "
              f"{len(set(act_lab)) - 1}", flush=True)
        Z = embed(X, args.ckpt)
        print("t-SNE ...", flush=True)
        ts = TSNE(n_components=2, perplexity=30, metric="cosine", init="pca",
                  random_state=SEED).fit_transform(Z)
        np.savez_compressed(npz, Z=Z, tsne=ts, dataset=ds_lab, activity=act_lab,
                            wrap=np.array(wrap, dtype=object))
        print("wrote", npz, flush=True)

    stats = (None if args.no_analysis
             else analysis(Z, ds_lab, act_lab, wrap, args.out, args.ckpt))

    # the paper figure: one wide panel, names in place, no legend
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=400)
    n_named = panel_activity(ax, ts, ds_lab, act_lab, s=2.6, max_names=args.max_names)
    fig.tight_layout()
    fig.savefig(f"{args.out}/ssl_tsne_by_activity_v6.png", facecolor="#ffffff",
                bbox_inches="tight")
    fig.savefig("/gpfs/home/vbc1591/paper_iclr/Pictures/ssl_tsne_by_activity_v6.png",
                facecolor="#ffffff", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote the activity figure, {n_named} activities named in place", flush=True)

    for fname, fn in [("ssl_tsne_by_dataset_v6.png", panel_dataset),
                      ("ssl_tsne_by_exposure_v6.png", panel_cells)]:
        fig, ax = plt.subplots(figsize=(5.5, 3.1), dpi=400)
        fn(ax, ts, ds_lab, s=2.4)
        fig.tight_layout()
        fig.savefig(f"{args.out}/{fname}", facecolor="#ffffff", bbox_inches="tight")
        plt.close(fig)
        print("wrote", f"{args.out}/{fname}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(28, 7.0), dpi=130)
    panel_cells(axes[0], ts, ds_lab, fontsize=9, s=7)
    axes[0].set_title("signal origin x pretraining exposure", fontsize=12)
    panel_dataset(axes[1], ts, ds_lab, fontsize=7.5, s=7)
    axes[1].set_title("dataset", fontsize=12)
    panel_activity(axes[2], ts, ds_lab, act_lab, s=7, name_fs=7.0, fam_fs=11,
                   max_names=args.max_names)
    axes[2].set_title("activity (family hue, names in place)", fontsize=12)
    fig.suptitle(f"Reference 5-source MAE — t-SNE — {len(set(ds_lab))} sources, "
                 f"{len(ds_lab)} windows, "
                 + (f"{stats['n_act']} activities — same-dataset@10 {stats['same_ds']:.3f} "
                    f"(chance {stats['ds_chance']:.3f}), same-activity {stats['same_act']:.3f} "
                    f"(chance {stats['act_chance']:.3f})" if stats else ""), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{args.out}/ssl_embedding_projections_v6.png", facecolor="#ffffff",
                bbox_inches="tight")
    print("wrote", f"{args.out}/ssl_embedding_projections_v6.png", flush=True)


if __name__ == "__main__":
    main()
