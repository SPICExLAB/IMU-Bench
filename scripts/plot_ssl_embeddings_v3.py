#!/usr/bin/env python3
"""UMAP + t-SNE of the MAE SSL encoder across 11 real HAR datasets -- v3
(2026-08, user request): MANY activities colored via SEMANTIC FAMILIES
(hue = family, shade = member) + a quantitative cross-dataset /
cross-activity analysis of the embedding space.

Families -> validated categorical slots (reference palette order, which
passes all adjacency checks; within-family shades are deliberate composite
encoding -- the hue carries the family, the legend disambiguates members):
  blue=locomotion  orange=cycling  aqua=household  yellow=static posture
  magenta=hand/desk  green=sport/exercise  violet=stairs  red=UTD gestures

Analysis metrics (written to analysis/embedding_viz/analysis_report.md):
  1. kNN neighborhood purity (k=10, cosine): same-dataset vs same-activity
     fraction, each vs its chance level.
  2. Silhouette score (cosine) under dataset labels vs activity labels.
  3. Cross-dataset activity retrieval: same-activity fraction among the 10
     nearest neighbors drawn ONLY from OTHER datasets (per shared activity).
  4. Leave-one-dataset-out kNN activity classification on shared activities.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SENSORS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
T_WIN = 512
T_MIN_SHORT = 128
PER_DATASET_CAP = 800
SEED = 42

# -- activity normalization: merge cross-dataset synonyms + variant groups --
ACT_NORM = {
    "walk": "walking",
    "sit": "sitting", "sitting relaxing": "sitting",
    "stand": "standing", "standing still": "standing",
    "bike": "cycling",
    "jog": "jogging",
    "run": "running",
    "lying down": "lying", "laying": "lying",
    "stairsup": "stairs up", "ascending stairs": "stairs up", "climbing stairs": "stairs up",
    "stairsdown": "stairs down", "descending stairs": "stairs down",
    "eating soup": "eating", "eating chips": "eating", "eating pasta": "eating",
    "eating sandwich": "eating",
    "drinking from cup": "drinking",
    "playing catch tennis ball": "catch",
    "kicking soccer ball": "soccer", "playing soccer": "soccer",
    "dribbling basketball": "basketball", "basketball shoot": "basketball",
    "folding clothes": "folding", "folding laundry": "folding",
    "rope jumping": "jumping", "jump front and back": "jumping",
    "clap": "clapping",
    "computer work": "computer work",
    "waist bends forward": "waist bends",
    # shoaib
    "upstairs": "stairs up", "downstairs": "stairs down",
    "type": "typing", "write": "writing", "eat": "eating",
    "coffee": "drinking", "talk": "talking", "smoke": "smoking",
    # ut_watch / samosa
    "keyboard": "typing", "browsing": "phone use", "mobile": "phone use",
    "teeth": "brushing teeth", "toothbrushing": "brushing teeth",
    "hair": "grooming", "brushing hair": "grooming",
    "hair dryer in use": "grooming", "shaver in use": "grooming",
    "comb hair": "grooming",
    "vacuuming": "vacuum cleaning", "vacuum in use": "vacuum cleaning",
    "sweeping": "sweeping",
    "dishes": "dishes", "washing utensils": "dishes",
    "wiping": "wiping", "wiping with rag": "wiping",
    "washing": "washing hands", "washing hands": "washing hands",
    "drawing": "drawing",
    "chopping": "cooking", "grating": "cooking", "cutting": "cooking",
    "frying": "cooking", "microwave": "cooking", "filling": "cooking",
    "blender in use": "cooking", "pouring pitcher": "cooking",
    "pour water": "cooking",
    "screwing": "tools", "sanding": "tools", "hammering": "tools",
    "drill in use": "tools", "twisting jar": "tools", "knocking": "tools",
    # wharf
    "brush teeth": "brushing teeth", "climb stairs": "stairs up",
    "descend stairs": "stairs down", "drink glass": "drinking",
    "eat meat": "eating", "eat soup": "eating",
    "getup bed": "posture transition", "liedown bed": "posture transition",
    "sitdown chair": "posture transition", "standup chair": "posture transition",
    "use telephone": "phone use",
    # recofit direct
    "running (treadmill)": "running", "jump rope": "jumping",
    "jumping jacks": "jumping", "elliptical machine": "elliptical",
    "rowing machine": "rowing",
}


def bucket_label(lab: str) -> str:
    """Prefix/keyword buckets for PRISM's 96 kitchen sub-steps and RecoFit's
    72 gym exercise variants (applied AFTER ACT_NORM exact matches miss)."""
    for p in ("cooking", "maketea", "makecoffee", "makesandwich", "makecereal",
              "latte making"):
        if lab.startswith(p):
            return "cooking"
    if lab.startswith("makestencil"):
        return "tools"
    if "stretch" in lab:
        return "stretching"
    if any(k in lab for k in ("squat", "lunge", "burpee", "box jump", "wall ball",
                              "kettlebell swing", "medicine ball", "deadlift")):
        return "leg exercise"
    if any(k in lab for k in ("curl", "press", "raise", "fly", "extension",
                              "kickback", "dumbbell row", "band pull", "pushup",
                              "push-up", "dip", "lawnmower", "punches",
                              "chest press", "back fly")):
        return "arm exercise"
    if any(k in lab for k in ("sit-up", "sit-ups", "crunch", "v-up", "plank",
                              "russian twist", "power boat")):
        return "core exercise"
    return lab

# UTD-MHAD-only fine hand gestures -> one family bucket
UTD_GESTURES = {
    "swipe left", "swipe right", "wave", "throw", "arm cross", "draw x",
    "draw circle cw", "draw circle ccw", "draw triangle", "bowling", "boxing",
    "baseball swing", "tennis swing", "tennis serve", "arm curl", "push",
    "knock", "pickup and throw",
}

# -- families: (validated anchor slot) -> ordered members with shades --------
FAMILY_COLORS = {  # activity -> (hex, family)
    # blue: locomotion
    "walking":       ("#2a78d6", "locomotion"),
    "jogging":       ("#155a9e", "locomotion"),
    "running":       ("#0c3f75", "locomotion"),
    "nordic walking":("#7fabe6", "locomotion"),
    # orange: cycling
    "cycling":       ("#eb6834", "cycling"),
    # aqua: household
    "vacuum cleaning": ("#1baf7a", "household"),
    "house cleaning":  ("#128a5f", "household"),
    "ironing":         ("#0c6a49", "household"),
    "folding":         ("#67cda4", "household"),
    # yellow/amber: static posture
    "sitting":       ("#eda100", "posture"),
    "standing":      ("#b97d00", "posture"),
    "lying":         ("#845800", "posture"),
    "watching tv":   ("#f4c04d", "posture"),
    # magenta: hand/desk
    "typing":        ("#e87ba4", "hand/desk"),
    "writing":       ("#d1548b", "hand/desk"),
    "computer work": ("#aa3a6e", "hand/desk"),
    "brushing teeth":("#f2aec7", "hand/desk"),
    "eating":        ("#8a2c58", "hand/desk"),
    "drinking":      ("#672043", "hand/desk"),
    # green: sport / exercise / calisthenics
    "soccer":        ("#008300", "sport"),
    "basketball":    ("#2ea52e", "sport"),
    "catch":         ("#63c463", "sport"),
    "jumping":       ("#005c00", "sport"),
    "clapping":      ("#98d998", "sport"),
    "waist bends":   ("#00a651", "sport"),
    "knees bending": ("#7cb342", "sport"),
    "frontal elevation of arms": ("#c5e1a5", "sport"),
    # magenta additions (hand/desk)
    "drawing":       ("#f7c9da", "hand/desk"),
    "phone use":     ("#c2427a", "hand/desk"),
    "talking":       ("#a03063", "hand/desk"),
    "smoking":       ("#7c2450", "hand/desk"),
    "grooming":      ("#5a1a3c", "hand/desk"),
    # aqua additions (household/kitchen)
    "sweeping":      ("#37bd8c", "household"),
    "dishes":        ("#0a5a3e", "household"),
    "wiping":        ("#84d7b4", "household"),
    "washing hands": ("#a5e3c9", "household"),
    "cooking":       ("#063f2c", "household"),
    "tools":         ("#4cc39a", "household"),
    # amber addition (posture)
    "posture transition": ("#6b4700", "posture"),
    # green additions (gym / sport)
    "stretching":    ("#4cb04c", "sport"),
    "leg exercise":  ("#1d701d", "sport"),
    "arm exercise":  ("#77c877", "sport"),
    "core exercise": ("#0f4d0f", "sport"),
    "elliptical":    ("#b2e3b2", "sport"),
    "rowing":        ("#d2eed2", "sport"),
    # violet: stairs
    "stairs up":     ("#4a3aa7", "stairs"),
    "stairs down":   ("#8d81d6", "stairs"),
    "stairs":        ("#685bc0", "stairs"),
    # red: UTD fine hand gestures (single bucket)
    "hand gesture (utd)": ("#e34948", "gestures"),
}
FAMILY_ORDER = ["locomotion", "cycling", "household", "posture",
                "hand/desk", "sport", "stairs", "gestures"]
OTHER_GRAY = "#9b9a92"

_MASTER = "/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet"
DATASETS = {
    "wisdm":    _MASTER, "utd_mhad": _MASTER, "pamap2": _MASTER,
    "prism":    _MASTER, "recofit":  _MASTER, "samosa": _MASTER,
    "shoaib":   _MASTER, "ut_watch": _MASTER,
    "hhar":     "/projects/b53017/vbc1591/hhar_dataset.parquet",
    "mhealth":  "/projects/b53017/vbc1591/mhealth_dataset.parquet",
    "wharf":    "/projects/b53017/vbc1591/wharf_dataset.parquet",
}
ZERO_GYRO = {"wharf"}   # acc-only dataset: gyro columns are NaN -> zero-fill
# 11 datasets: 8 validated anchors + 3 same-hue shades grouped semantically
# (blues = phone/locomotion sets, violets = kitchen/ADL watch sets, reds = ADL)
DS_COLOR_MAP = {
    "hhar": "#2a78d6", "shoaib": "#155a9e",
    "mhealth": "#eb6834", "pamap2": "#1baf7a", "utd_mhad": "#eda100",
    "wisdm": "#e87ba4", "recofit": "#008300",
    "prism": "#4a3aa7", "ut_watch": "#8d81d6",
    "samosa": "#e34948", "wharf": "#a33230",
}
# windows with these labels are dropped entirely (device-idle / null classes)
SKIP_LABELS = {"device on table", "rest", "non-exercise", "note",
               "arm band adjustment", "tap left device", "tap right device",
               "other", "alarm clock", "shuffling", "scratching", "laughing",
               "coughing", "toilet flushing", "car driving"}


def extract_windows():
    import pyarrow.dataset as pa_ds
    from imu_lm.data.augmentations.preprocess import preprocess_window

    prep_cfg = {"preprocess": {
        "impute":   {"enabled": True, "method": "linear", "max_missing_frac": 0.2},
        "filter":   {"enabled": True, "type": "lowpass", "high_hz": 24, "low_hz": 0.25, "order": 4},
        "normalize": {"enabled": True, "method": "zscore", "eps": 1e-6}}}

    rng = np.random.default_rng(SEED)
    X_all, ds_labels, act_labels = [], [], []
    for ds_name, path in DATASETS.items():
        d = pa_ds.dataset(path, format="parquet")
        cols = SENSORS + ["dataset", "subject_id", "session_id", "dataset_activity_label"]
        df = d.to_table(columns=cols, filter=(pa_ds.field("dataset") == ds_name)).to_pandas()
        wins = []
        for (_, _), g in df.groupby(["subject_id", "session_id"], sort=False):
            arr = g[SENSORS].to_numpy(dtype=np.float32)
            if ds_name in ZERO_GYRO:
                arr[:, 3:6] = 0.0
            labs = (g["dataset_activity_label"].astype(str).str.replace("_", " ", regex=False)
                     .str.strip().str.lower().to_numpy())
            spans = ([(s0, s0 + T_WIN) for s0 in range(0, len(arr) - T_WIN + 1, T_WIN)]
                     if len(arr) >= T_WIN else
                     ([(0, len(arr))] if len(arr) >= T_MIN_SHORT else []))
            for a0, a1 in spans:
                w, wl = arr[a0:a1], labs[a0:a1]
                vals, cnts = np.unique(wl, return_counts=True)
                top = int(np.argmax(cnts))
                if cnts[top] / len(wl) < 0.8:
                    continue
                lab = vals[top]
                if lab in ("nan", "none", "", "unknown", "-1"):
                    continue
                lab = ACT_NORM.get(lab, lab)
                lab = bucket_label(lab)
                if lab in SKIP_LABELS:
                    continue
                if lab in UTD_GESTURES:
                    lab = "hand gesture (utd)"
                wins.append((w, lab))
        rng.shuffle(wins)
        kept = 0
        for w, lab in wins[:PER_DATASET_CAP]:
            xp = preprocess_window(w, prep_cfg, None)
            if xp is None:
                continue
            if xp.shape[1] < T_WIN:
                xp = np.pad(xp, ((0, 0), (0, T_WIN - xp.shape[1])), mode="wrap")
            X_all.append(xp.astype(np.float32)); ds_labels.append(ds_name); act_labels.append(lab)
            kept += 1
        print(f"[{ds_name}] windows kept: {kept}", flush=True)
    return np.stack(X_all), np.array(ds_labels), np.array(act_labels)


def embed(X):
    import torch
    from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
    ckpt = torch.load("runs/vit1d_ssl_mae_varlen_altattn_multidata/checkpoints/best.pt",
                      map_location="cpu", weights_only=False)
    enc = ViT1DEncoderAltAttn(ckpt["cfg"])
    missing, _ = enc.load_state_dict(ckpt["model"], strict=False)
    assert not missing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc.eval().to(device)
    import numpy as _np
    Z = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            Z.append(enc(torch.from_numpy(X[i:i + 256]).to(device)).cpu().numpy())
    return _np.concatenate(Z, 0)


def analysis(Z, ds_lab, act_lab, out):
    """Quantitative cross-dataset / cross-activity structure metrics."""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    Zn = normalize(Z)                      # cosine == euclidean on normalized
    N = len(Zn)
    S = Zn @ Zn.T                          # cosine similarity [N,N]
    np.fill_diagonal(S, -np.inf)
    K = 10
    nn_idx = np.argpartition(-S, K, axis=1)[:, :K]          # top-K neighbors

    same_ds = (ds_lab[nn_idx] == ds_lab[:, None]).mean()
    same_act = (act_lab[nn_idx] == act_lab[:, None]).mean()
    # chance levels
    _, ds_cnt = np.unique(ds_lab, return_counts=True)
    ds_chance = float(((ds_cnt / N) ** 2).sum())
    _, ac_cnt = np.unique(act_lab, return_counts=True)
    act_chance = float(((ac_cnt / N) ** 2).sum())

    sil_ds = silhouette_score(Zn, ds_lab, metric="cosine")
    # silhouette over activities with >=20 samples
    keep = np.isin(act_lab, [a for a, c in zip(*np.unique(act_lab, return_counts=True)) if c >= 20])
    sil_act = silhouette_score(Zn[keep], act_lab[keep], metric="cosine")

    # cross-dataset activity retrieval: neighbors restricted to OTHER datasets
    S_cross = S.copy()
    for ds in np.unique(ds_lab):
        m = ds_lab == ds
        S_cross[np.ix_(m, m)] = -np.inf
    nn_x = np.argpartition(-S_cross, K, axis=1)[:, :K]
    same_act_cross = (act_lab[nn_x] == act_lab[:, None]).mean(axis=1)

    # per-activity cross-dataset score, only for activities in >=2 datasets
    lines = []
    shared = []
    for a in np.unique(act_lab):
        m = act_lab == a
        n_ds = len(np.unique(ds_lab[m]))
        if n_ds >= 2 and m.sum() >= 20:
            shared.append(a)
            chance = float(m.mean())
            lines.append((a, int(m.sum()), n_ds, float(same_act_cross[m].mean()), chance))
    lines.sort(key=lambda r: -r[3])

    # leave-one-dataset-out kNN classification on shared activities
    lodo = []
    sh_mask = np.isin(act_lab, shared)
    for ds in np.unique(ds_lab):
        te = sh_mask & (ds_lab == ds)
        tr = sh_mask & (ds_lab != ds)
        if te.sum() < 10 or tr.sum() < 50:
            continue
        # only test activities that exist in the train pool
        tr_acts = set(act_lab[tr])
        te = te & np.isin(act_lab, list(tr_acts))
        if te.sum() < 10:
            continue
        sims = Zn[te] @ Zn[tr].T
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        lab_tr = act_lab[tr]
        preds = []
        for row in top:
            v, c = np.unique(lab_tr[row], return_counts=True)
            preds.append(v[np.argmax(c)])
        acc = float((np.array(preds) == act_lab[te]).mean())
        maj = float(np.max(np.unique(act_lab[te], return_counts=True)[1]) / te.sum())
        lodo.append((ds, int(te.sum()), acc, maj))

    with open(f"{out}/analysis_report.md", "w") as f:
        f.write("# SSL embedding space: cross-dataset / cross-activity analysis\n\n")
        f.write(f"N={N} windows, k={K} cosine neighbors, 384-d frozen MAE embeddings.\n\n")
        f.write("## 1. kNN neighborhood purity\n\n")
        f.write(f"- same-DATASET fraction: **{same_ds:.3f}** (chance {ds_chance:.3f})\n")
        f.write(f"- same-ACTIVITY fraction: **{same_act:.3f}** (chance {act_chance:.3f})\n\n")
        f.write(f"Dataset purity is {same_ds/ds_chance:.1f}x chance; activity purity is "
                f"{same_act/act_chance:.1f}x chance.\n\n")
        f.write("## 2. Silhouette (cosine)\n\n")
        f.write(f"- by dataset: **{sil_ds:.3f}**\n- by activity (n>=20): **{sil_act:.3f}**\n\n")
        f.write("## 3. Cross-dataset activity retrieval\n\n")
        f.write("Same-activity fraction among 10 NN restricted to OTHER datasets\n"
                "(activities present in >=2 datasets, n>=20):\n\n")
        f.write("| activity | n | #datasets | cross-ds same-act@10 | chance |\n|---|---|---|---|---|\n")
        for a, n, nd, sc, ch in lines:
            f.write(f"| {a} | {n} | {nd} | {sc:.3f} | {ch:.3f} |\n")
        f.write("\n## 4. Leave-one-dataset-out kNN activity classification\n\n")
        f.write("(shared activities only; majority-class baseline in parens)\n\n")
        f.write("| held-out dataset | n test | kNN acc | majority baseline |\n|---|---|---|---|\n")
        for ds, n, acc, maj in lodo:
            f.write(f"| {ds} | {n} | {acc:.3f} | {maj:.3f} |\n")
    print("wrote", f"{out}/analysis_report.md", flush=True)


def scatter(ax, coords, labels, entries, title, ncol=1):
    ax.set_facecolor("#ffffff")
    for lab, col in entries:
        m = labels == lab
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=7, c=col, alpha=0.8,
                   linewidths=0, label=f"{lab} ({int(m.sum())})")
    ax.set_title(title, fontsize=12, color="#1a1a19")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d7d6cc"); sp.set_linewidth(0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5,
              framealpha=0.95, markerscale=1.8, handletextpad=0.3,
              borderpad=0.4, ncol=ncol)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import umap

    out = "analysis/embedding_viz"
    os.makedirs(out, exist_ok=True)

    X, ds_lab, act_lab = extract_windows()
    print(f"total windows: {len(X)}", flush=True)
    Z = embed(X)

    print("t-SNE ...", flush=True)
    ts = TSNE(n_components=2, perplexity=30, metric="cosine", init="pca",
              random_state=SEED).fit_transform(Z)
    print("UMAP ...", flush=True)
    um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="cosine", random_state=SEED).fit_transform(Z)

    np.savez_compressed(f"{out}/ssl_embeddings_v2.npz", Z=Z, umap=um, tsne=ts,
                        dataset=ds_lab, activity=act_lab)

    analysis(Z, ds_lab, act_lab, out)

    # activity display labels: colored families; everything else -> other
    act_plot = np.array([a if a in FAMILY_COLORS else "other" for a in act_lab])
    fam_of = {a: fam for a, (c, fam) in FAMILY_COLORS.items()}
    act_entries = []
    for fam in FAMILY_ORDER:
        for a, (c, f_) in FAMILY_COLORS.items():
            if f_ == fam and (act_plot == a).any():
                act_entries.append((a, c))
    if (act_plot == "other").any():
        act_entries.append(("other", OTHER_GRAY))

    ds_order = sorted(set(ds_lab))
    ds_entries = [(d_, DS_COLOR_MAP[d_]) for d_ in ds_order]

    panels = [
        (um, ds_lab,   ds_entries,  "UMAP — colored by dataset",   "ssl_umap_by_dataset_v2.png", 1),
        (ts, ds_lab,   ds_entries,  "t-SNE — colored by dataset",  "ssl_tsne_by_dataset_v2.png", 1),
        (um, act_plot, act_entries, "UMAP — activities in semantic families", "ssl_umap_by_activity_v2.png", 2),
        (ts, act_plot, act_entries, "t-SNE — activities in semantic families", "ssl_tsne_by_activity_v2.png", 2),
    ]
    for coords, labels, entries, title, fname, ncol in panels:
        fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=150)
        scatter(ax, coords, labels, entries, f"MAE SSL encoder — {title}", ncol)
        fig.tight_layout()
        fig.savefig(f"{out}/{fname}", facecolor="#ffffff", bbox_inches="tight")
        plt.close(fig)
        print("wrote", f"{out}/{fname}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(19, 13), dpi=140)
    for ax, (coords, labels, entries, title, _, ncol) in zip(axes.ravel(), panels):
        scatter(ax, coords, labels, entries, title, ncol)
    fig.suptitle(f"MAE SSL encoder embeddings — 5 real HAR datasets, {len(X)} windows",
                 fontsize=14, color="#1a1a19")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{out}/ssl_embedding_projections_v2.png", facecolor="#ffffff",
                bbox_inches="tight")
    print("wrote", f"{out}/ssl_embedding_projections_v2.png", flush=True)


if __name__ == "__main__":
    main()
