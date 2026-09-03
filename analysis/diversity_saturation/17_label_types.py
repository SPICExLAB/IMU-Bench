#!/usr/bin/env python
"""17_label_types.py -- put activity NAMES on the motion types.

Everything so far describes the types geometrically or by signal statistics.
The realdata block of the pool (93,510 windows from pamap2 / recofit / samosa /
shoaib / ut_watch / utd_mhad / wisdm) carries real HAR activity labels, so if
each of those windows can be located in the pool, the types can be read as
activity mixtures instead of geometry.

Two rebuild assumptions were tested and BOTH failed, which is why this script
looks the way it does:

  1. positional join   a fresh rebuild yields the same COUNT (93,510) but in a
                       different order -- 5/200 index-probes matched.
  2. train-split join  content-hashing the rebuilt TRAIN split against the
                       cache matched only 4.16% of windows, i.e. the rebuilt
                       train split is not even the same SET of windows. The
                       session-level train/val/test assignment does not
                       reproduce, so most cached windows sit in a split the
                       rebuild calls val or test.

So the rebuild covers EVERY session in the index (train + val + test), a strict
superset of the cached block, and the join is on window CONTENT. Labels are
taken only for windows that hash-match a cache row, and the join must cover
every cache row or the script writes nothing (see fingerprint_join.py).

Outputs analysis/diversity_saturation/out/type_activity_labels.json
        /projects/b53017/vbc1591/diversity_saturation/realdata_train_labels.npy
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")
CACHE_X = "/projects/b53017/vbc1591/mae_multidata_cache_v3/realdata_train_x.npy"
LABELS_NPY = os.path.join(WORK_DIR, "realdata_train_labels.npy")   # cache-order
_AUG = ("random_rotation", "gaussian_noise", "jitter", "bias_drift", "sensor_dropout",
        "time_crop", "frequency_mask", "gain_jitter", "axis_bias", "time_warp")


def load_cfg():
    from imu_lm.utils.helpers import deep_update, load_yaml
    cfg = deep_update(load_yaml(os.path.join(REPO_ROOT, "configs", "base.yaml")),
                      load_yaml(os.path.join(
                          REPO_ROOT, "configs",
                          "vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe100pct.yaml")))
    for k in _AUG:
        b = cfg.get(k)
        if isinstance(b, dict):
            b["enabled"] = False
    return cfg


def activity_name_map(parquet_path, datasets):
    """dataset_activity_id -> (dataset, dataset_activity_label, global_activity_label)."""
    import pyarrow.dataset as pds
    import pyarrow.compute as pc
    d = pds.dataset(parquet_path, format="parquet")
    cols = ["dataset", "dataset_activity_id", "dataset_activity_label", "global_activity_label"]
    tbl = d.to_table(columns=cols, filter=pc.field("dataset").isin(datasets))
    df = tbl.to_pandas().drop_duplicates()
    m = {}
    for _, r in df.iterrows():
        m[int(r["dataset_activity_id"])] = {
            "dataset": str(r["dataset"]),
            "activity": str(r["dataset_activity_label"]),
            "global": str(r["global_activity_label"]),
        }
    return m


def rebuild_labels(cfg):
    from imu_lm.data.loaders import WindowDataset
    from imu_lm.data.splits import SessionKey, build_session_index, make_splits

    rd = cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["realdata"]
    pq = cfg["paths"]["dataset_path"]
    dsets = list(rd["train_on_datasets"])

    t0 = time.time()
    print("[1/4] scanning session index ...", flush=True)
    si = build_session_index(pq, cfg, dataset_filter=dsets)
    sp = make_splits(si, cfg)
    # EVERY session, not sp["train_keys"] -- the split does not reproduce, so the
    # cached train block is only recoverable from the union of all splits.
    all_keys = [SessionKey(r["dataset"], str(r["subject_id"]), str(r["session_id"]))
                for _, r in si.iterrows()]
    print(f"      {len(si)} sessions ({len(sp['train_keys'])} would be train), "
          f"rebuilding all {len(all_keys)} ({time.time() - t0:.0f}s)", flush=True)

    print("[2/4] constructing WindowDataset over all sessions ...", flush=True)
    ds = WindowDataset(pq, si, all_keys, cfg, split_name="train")
    print(f"      {len(ds)} raw window slots ({time.time() - t0:.0f}s)", flush=True)

    print("[3/4] iterating + fingerprinting ...", flush=True)
    from fingerprint_join import fingerprint, build_index, join

    fps, lab = [], []
    for i in range(len(ds)):
        item = ds[i]
        if item is None:
            continue
        x = item[0]
        x = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        fps.append(fingerprint(x))
        lab.append(int(item[1]))
        if i % 20000 == 0:
            print(f"      {i}/{len(ds)} slots, {len(lab)} kept "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"      kept {len(lab)} windows", flush=True)

    print("[4/4] content-joining onto the cache ...", flush=True)
    cache = np.load(CACHE_X, mmap_mode="r")
    assert len(lab) >= len(cache), (
        f"rebuild produced {len(lab)} windows, fewer than the cache's {len(cache)} "
        "-- cannot be a superset")
    cidx = build_index(cache)
    labels = np.full(len(cache), -1, dtype=np.int64)
    hit = 0
    for fp, y in zip(fps, lab):
        row = cidx.get(fp)
        if row is not None and labels[row] < 0:
            labels[row] = y
            hit += 1
    cov = hit / len(cache)
    print(f"      [realdata] {hit}/{len(cache)} cache rows labelled by content "
          f"({100 * cov:.2f}%) from {len(lab)} rebuilt windows", flush=True)
    assert cov > 0.999, (
        f"only {100 * cov:.2f}% of cached windows were found in the rebuild -- the "
        "rebuild is not reproducing cached content, refusing to join")
    np.save(LABELS_NPY, labels)
    return labels


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_cfg()
    if os.path.exists(LABELS_NPY):
        labels = np.load(LABELS_NPY)
        print(f"[cache] reusing {LABELS_NPY} ({len(labels)} labels)", flush=True)
    else:
        labels = rebuild_labels(cfg)

    man = json.load(open(os.path.join(WORK_DIR, "pool_manifest.json")))
    names, cum = man["sources"], np.array(man["cumulative_sizes"])
    i_rd = names.index("realdata")
    lo = 0 if i_rd == 0 else int(cum[i_rd - 1])
    hi = int(cum[i_rd])
    assert hi - lo == len(labels), (hi - lo, len(labels))

    types = np.load(os.path.join(WORK_DIR, "labels11src_k1000.npy")).astype(np.int32)[lo:hi]
    nmap = activity_name_map(cfg["paths"]["dataset_path"],
                             list(cfg["objective"]["mae_multidata_varlen_altattn_flu"]
                                  ["sources"]["realdata"]["train_on_datasets"]))

    # collapse to the harmonised cross-dataset vocabulary
    gl = np.array([nmap.get(int(v), {}).get("global", f"id{int(v)}") for v in labels])
    vocab = sorted(set(gl.tolist()))
    gid = {g: i for i, g in enumerate(vocab)}
    g_idx = np.array([gid[g] for g in gl])

    tab = np.zeros((len(vocab), 1000), dtype=np.int64)
    np.add.at(tab, (g_idx, types), 1)
    per_type_total = tab.sum(0)

    # per-activity: which types does it use, how concentrated
    per_activity = {}
    for a, name in enumerate(vocab):
        c = tab[a]
        tot = int(c.sum())
        if tot == 0:
            continue
        srt = np.sort(c)[::-1]
        cs = np.cumsum(srt) / tot
        per_activity[name] = {
            "n_windows": tot,
            "types_touched": int((c > 0).sum()),
            "types_for_90pct": int(np.searchsorted(cs, 0.90) + 1),
            "top_types": [{"type": int(t), "share_of_activity_pct": round(100 * c[t] / tot, 1),
                           "purity_within_realdata_pct": round(100 * c[t] / per_type_total[t], 1)}
                          for t in np.argsort(c)[::-1][:3] if c[t] > 0],
        }

    # per-type: what is it, in activity terms (only types with enough realdata support)
    named = []
    for t in range(1000):
        n_t = int(per_type_total[t])
        if n_t < 30:
            continue
        col = tab[:, t]
        a = int(np.argmax(col))
        purity = float(col[a] / n_t)
        ent = col / n_t
        ent = float(-(ent[ent > 0] * np.log(ent[ent > 0])).sum())
        named.append({"type": t, "realdata_windows": n_t, "top_activity": vocab[a],
                      "purity_pct": round(100 * purity, 1),
                      "entropy_nats": round(ent, 3),
                      "runner_up": vocab[int(np.argsort(col)[-2])] if (col > 0).sum() > 1 else None,
                      "runner_up_pct": round(100 * float(np.sort(col)[-2] / n_t), 1)
                      if (col > 0).sum() > 1 else None})
    named.sort(key=lambda r: -r["purity_pct"])

    purities = np.array([r["purity_pct"] for r in named])
    # chance baseline: purity you would get assigning windows to types at random
    p_glob = tab.sum(1) / tab.sum()
    chance = float(p_glob.max() * 100)

    out = {
        "note": ("activity names come from the realdata block only (93,510 windows, "
                 "7 HAR datasets); the other 10 sources have no usable activity labels "
                 "in the cache, so these names describe the types' realdata content"),
        "n_labelled_windows": int(len(labels)),
        "vocabulary": vocab,
        "types_with_ge30_realdata_windows": len(named),
        "purity_pcts": {p: round(float(np.percentile(purities, p)), 1)
                        for p in (10, 25, 50, 75, 90)},
        "chance_purity_pct": round(chance, 1),
        "median_activity_entropy_nats": round(float(np.median(
            [r["entropy_nats"] for r in named])), 3),
        "per_activity": per_activity,
        "named_types": named,
    }
    p = os.path.join(OUT_DIR, "type_activity_labels.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nvocabulary ({len(vocab)}): {vocab}")
    print(f"\ntypes with >=30 realdata windows: {len(named)}")
    print(f"purity percentiles: {out['purity_pcts']}   (chance = {chance:.1f}%)")
    print(f"\n{'activity':22s}{'windows':>9s}{'types':>7s}{'n@90%':>7s}  top types")
    for a, v in sorted(per_activity.items(), key=lambda kv: -kv[1]["n_windows"]):
        tt = ", ".join(f"#{t['type']}({t['share_of_activity_pct']}%)" for t in v["top_types"])
        print(f"{a:22s}{v['n_windows']:>9,}{v['types_touched']:>7d}{v['types_for_90pct']:>7d}  {tt}")
    print(f"\nmost activity-pure types:")
    for r in named[:15]:
        print(f"  #{r['type']:<4d} {r['purity_pct']:>5.1f}% {r['top_activity']:<20s} "
              f"(n={r['realdata_windows']}, runner-up {r['runner_up']} {r['runner_up_pct']}%)")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
