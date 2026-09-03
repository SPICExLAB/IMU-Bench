#!/usr/bin/env python
"""19_label_types_capture24.py -- activity names for the accelerometer-only half of the type space.

17_label_types.py names types using the realdata block, but realdata has zero
windows in the A-families (the acc-only island), which is 38 % of the pool.
Capture-24 is the only labelled source that lives there, and its windows carry
free-living activity annotations mapped to a published taxonomy.

`Capture24Dataset.__getitem__` never returns None, so its label vector covers
the cached block exactly -- but NOT necessarily in the same ORDER. A realdata
rebuild was measured to permute its windows (5/200 index-probes matched), so a
positional join is not safe here either. The join is done on window CONTENT via
fingerprint_join.py and must be a clean bijection or the script writes nothing.

Both taxonomies are reported: Walmsley2020 (4 coarse classes) and
WillettsSpecific2018 (10 specific classes).

Outputs analysis/diversity_saturation/out/type_activity_labels_capture24.json
        /projects/b53017/vbc1591/diversity_saturation/capture24_train_labels.npy
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
CACHE_X = "/projects/b53017/vbc1591/mae_multidata_cache_v3/capture24_train_x.npy"
LABELS_NPY = os.path.join(WORK_DIR, "capture24_train_labels.npy")   # cache order
TAXONOMIES = ["Walmsley2020", "WillettsSpecific2018"]
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


def build(taxonomy):
    from imu_lm.data.capture24_loader import make_capture24_loaders_random
    cfg = load_cfg()
    src = cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["capture24"]
    t0 = time.time()
    # taxonomy is a KWARG, not a cfg key -- passing it via cfg silently kept the
    # default and mislabelled the second pass with the first pass's class ids.
    loaders = make_capture24_loaders_random(
        data_dir=src["root"], taxonomy=taxonomy,
        splits=tuple(src.get("splits", [0.8, 0.1, 0.1])),
        target_hz=src.get("target_hz"), max_files=None, cfg=cfg)
    ds = loaders["train_loader"].dataset
    print(f"[{taxonomy}] rebuilt {len(ds)} train windows ({time.time() - t0:.0f}s)", flush=True)
    return cfg, ds


def fingerprints_of(ds, every=None):
    """Fingerprint the 3 real acc channels of each rebuilt window."""
    from fingerprint_join import fingerprint
    rows = range(len(ds)) if every is None else every
    out = {}
    for n, j in enumerate(rows):
        item = ds[int(j)]
        x = item[0] if isinstance(item, (tuple, list)) else item
        x = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        out[int(j)] = fingerprint(np.asarray(x)[:3])
        if every is None and n % 200_000 == 0:
            print(f"      [fp] rebuilt {n}/{len(ds)}", flush=True)
    return out


def analyse(y, types, vocab, n_min=30):
    keep = y >= 0
    tab = np.zeros((len(vocab), 1000), dtype=np.int64)
    np.add.at(tab, (y[keep], types[keep]), 1)
    per_type_total = tab.sum(0)

    per_activity = {}
    for a, nm in enumerate(vocab):
        c = tab[a]
        tot = int(c.sum())
        if tot == 0:
            continue
        cs = np.cumsum(np.sort(c)[::-1]) / tot
        per_activity[nm] = {
            "n_windows": tot,
            "share_of_labelled_pct": round(100 * tot / int(keep.sum()), 2),
            "types_touched": int((c > 0).sum()),
            "types_for_90pct": int(np.searchsorted(cs, 0.90) + 1),
            "top_types": [{"type": int(t),
                           "share_of_activity_pct": round(100 * c[t] / tot, 1),
                           "purity_pct": round(100 * c[t] / per_type_total[t], 1)}
                          for t in np.argsort(c)[::-1][:3] if c[t] > 0],
        }

    named = []
    for t in range(1000):
        n_t = int(per_type_total[t])
        if n_t < n_min:
            continue
        col = tab[:, t]
        a = int(np.argmax(col))
        named.append({"type": t, "capture24_windows": n_t, "top_activity": vocab[a],
                      "purity_pct": round(100 * float(col[a] / n_t), 1)})
    named.sort(key=lambda r: -r["purity_pct"])
    pur = np.array([r["purity_pct"] for r in named]) if named else np.array([np.nan])
    chance = float((tab.sum(1) / max(tab.sum(), 1)).max() * 100)
    return {
        "vocabulary": vocab,
        "n_labelled": int(keep.sum()),
        "n_unlabelled": int((~keep).sum()),
        "types_with_ge30_windows": len(named),
        "purity_pcts": {p: round(float(np.nanpercentile(pur, p)), 1)
                        for p in (10, 25, 50, 75, 90)},
        "chance_purity_pct": round(chance, 1),
        "per_activity": per_activity,
        "named_types": named,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from imu_lm.data.capture24_loader import label_names
    from fingerprint_join import build_index, join

    man = json.load(open(os.path.join(WORK_DIR, "pool_manifest.json")))
    names, cum = man["sources"], np.array(man["cumulative_sizes"])
    i_c = names.index("capture24")
    lo, hi = int(cum[i_c - 1]), int(cum[i_c])
    types = np.load(os.path.join(WORK_DIR, "labels11src_k1000.npy")).astype(np.int32)[lo:hi]
    cache = np.load(CACHE_X, mmap_mode="r")
    assert hi - lo == len(cache), (hi - lo, len(cache))
    assert np.abs(np.asarray(cache[:2000, 3:])).max() == 0.0, \
        "expected zero-filled gyro channels in the capture24 cache"
    print(f"capture24 block: global [{lo}, {hi}) = {hi - lo:,} windows", flush=True)

    cidx = build_index(cache, channels=slice(0, 3))
    results, pos, matched, prev_y = {}, None, None, None

    for k, tax in enumerate(TAXONOMIES):
        cfg, ds = build(tax)
        assert len(ds) == len(cache), (
            f"{tax}: rebuild has {len(ds)} windows, cache has {len(cache)}")

        if k == 0:
            fps_map = fingerprints_of(ds)
            fps = [fps_map[j] for j in range(len(ds))]
            pos, matched = join(fps, cidx, name="capture24")
        else:
            # the split is seeded, so window order should repeat -- spot-check
            # rather than re-fingerprint 1.4M windows
            probe = np.random.default_rng(1).choice(len(ds), 5000, replace=False)
            fm = fingerprints_of(ds, every=probe)
            same = sum(1 for j in probe if cidx.get(fm[int(j)], -1) == pos[int(j)])
            print(f"      [{tax}] order re-check: {same}/{len(probe)} probes agree "
                  f"with the first pass", flush=True)
            assert same == len(probe), \
                f"{tax} rebuild ordering differs from the first pass -- refusing to reuse the join"

        y_raw = ds._y.numpy() if hasattr(ds._y, "numpy") else np.asarray(ds._y)
        y = np.full(len(cache), -2, dtype=np.int64)          # -2 = unassigned
        y[pos[matched]] = y_raw[matched]
        assert (y > -2).all(), "some cache rows received no label"

        vocab = label_names(cfg["objective"]["mae_multidata_varlen_altattn_flu"]
                            ["sources"]["capture24"]["root"], tax)
        if k == 0:
            np.save(LABELS_NPY, y)
        if k > 0:
            assert not np.array_equal(y, prev_y), (
                f"{tax} produced the same label vector as {TAXONOMIES[0]} -- the "
                "taxonomy argument is not taking effect")
        prev_y = y.copy()
        assert len(vocab) >= int(y.max()) + 1, (tax, len(vocab), int(y.max()))
        results[tax] = analyse(y, types, vocab)

        r = results[tax]
        print(f"[{tax}] {len(vocab)} classes, {r['n_labelled']:,} labelled "
              f"({r['n_unlabelled']:,} unlabelled), {r['types_with_ge30_windows']} types "
              f">=30 win, median purity {r['purity_pcts'][50]}% "
              f"(chance {r['chance_purity_pct']}%)", flush=True)
        for nm, v in sorted(r["per_activity"].items(), key=lambda kv: -kv[1]["n_windows"]):
            print(f"    {nm:28s}{v['n_windows']:>10,}  types={v['types_touched']:>4d}  "
                  f"n@90%={v['types_for_90pct']:>4d}", flush=True)

    p = os.path.join(OUT_DIR, "type_activity_labels_capture24.json")
    with open(p, "w") as f:
        json.dump({"block": [lo, hi], "taxonomies": results}, f, indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
