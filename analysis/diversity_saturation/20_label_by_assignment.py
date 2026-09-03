#!/usr/bin/env python
"""20_label_by_assignment.py -- name the motion types by ASSIGNING labelled windows to them.

Three attempts to join realdata's HAR labels onto the cached pool by identity
all failed, and the failures were informative:

  positional        rebuild has the same count (93,510) but a different order
                    -- 5/200 index probes matched.
  content, train    hashing the rebuilt train split against the cache matched
                    3,887 windows, 4.16 %.
  content, all      hashing EVERY session (97,699 windows, a superset of the
                    cached block) matched the SAME 3,887 windows.

So the v3 cache's realdata block cannot be regenerated from the current code
and config -- it is not a reordering and not a re-split, the window content
itself differs. (capture24, by contrast, content-matched 100 %, so this is a
realdata-specific provenance problem, not a cache-wide one.)

This script does not need an identity join. Rather than asking "where is this
cached window in my rebuild", it asks the question we actually care about:
**which motion type does this labelled window fall into**. Every rebuilt window
goes through the same frozen encoder, the same PCA-64 projection and the same
1000 centroids, and is assigned to its nearest type.

The PCA basis was not persisted by 01_embed_select_analyze.py, so it is
recovered by least squares from the cached (embedding, pca64) pairs -- exact,
because the projection is linear. The whole encode -> project -> assign chain
is then validated end-to-end on pool windows whose type is already known, and
the script refuses to name anything if that round-trip is poor.

Outputs analysis/diversity_saturation/out/type_activity_labels.json
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

WORK_DIR = "/projects/b53017/vbc1591/diversity_saturation"
OUT_DIR = os.path.join(REPO_ROOT, "analysis", "diversity_saturation", "out")
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "base.yaml")
MODEL_CONFIG = os.path.join(
    REPO_ROOT, "configs", "vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe100pct.yaml")
CKPT = os.path.join(REPO_ROOT, "runs", "vit1d_ssl_mae_recipe_100pct", "checkpoints", "best.pt")
_AUG = ("random_rotation", "gaussian_noise", "jitter", "bias_drift", "sensor_dropout",
        "time_crop", "frequency_mask", "gain_jitter", "axis_bias", "time_warp")
BATCH = 512


def load_cfg():
    from imu_lm.utils.helpers import deep_update, load_yaml
    cfg = deep_update(load_yaml(BASE_CONFIG), load_yaml(MODEL_CONFIG))
    for k in _AUG:
        b = cfg.get(k)
        if isinstance(b, dict):
            b["enabled"] = False
    return cfg


def get_encoder(cfg, device):
    from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn
    enc = ViT1DEncoderAltAttn(cfg).to(device)
    state = torch.load(CKPT, map_location=device, weights_only=False)
    enc.load_state_dict(state["model"])
    enc.eval()
    print(f"[enc] {CKPT} step={state.get('step')} on {device}", flush=True)
    return enc


@torch.no_grad()
def embed(enc, X, device, tag=""):
    out = np.empty((len(X), 384), dtype=np.float32)
    for s in range(0, len(X), BATCH):
        xb = torch.from_numpy(np.ascontiguousarray(X[s:s + BATCH])).float().to(device)
        out[s:s + BATCH] = enc(xb).float().cpu().numpy()
        if (s // BATCH) % 40 == 0:
            print(f"      [enc{tag}] {s}/{len(X)}", flush=True)
    return out


def assign(emb, mu, W, cent, chunk=20_000):
    cn = (cent * cent).sum(1)[None, :]
    out = np.empty(len(emb), dtype=np.int32)
    for s in range(0, len(emb), chunk):
        z = (emb[s:s + chunk].astype(np.float64) - mu) @ W
        d = (z * z).sum(1)[:, None] - 2 * z @ cent.T + cn
        out[s:s + chunk] = d.argmin(1)
    return out


def validate(enc, mu, W, cent, device, n=4000):
    """End-to-end: re-embed known pool windows and re-derive their stored type."""
    from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders
    ds = make_combined_mae_multidata_loaders(load_cfg())["train_loader"].dataset
    labels = np.load(os.path.join(WORK_DIR, "labels11src_k1000.npy")).astype(np.int32)
    idx = np.sort(np.random.default_rng(0).choice(len(ds), n, replace=False))
    X = np.stack([np.asarray(ds[int(i)][0]) for i in idx]).astype(np.float32)
    a = assign(embed(enc, X, device, tag=":val"), mu, W, cent)
    agree = float((a == labels[idx]).mean())
    print(f"[validate] re-embedding {n} pool windows reproduces their stored type "
          f"{100 * agree:.1f}% of the time", flush=True)
    assert agree > 0.85, (
        f"only {100 * agree:.1f}% round-trip agreement -- the encode/project/assign "
        "chain does not reproduce the pool's own labels, refusing to name types")
    return agree


def rebuild_realdata(cfg):
    from imu_lm.data.loaders import WindowDataset
    from imu_lm.data.splits import SessionKey, build_session_index
    rd = cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"]["realdata"]
    pq = cfg["paths"]["dataset_path"]
    t0 = time.time()
    si = build_session_index(pq, cfg, dataset_filter=list(rd["train_on_datasets"]))
    keys = [SessionKey(r["dataset"], str(r["subject_id"]), str(r["session_id"]))
            for _, r in si.iterrows()]
    print(f"[rebuild] {len(si)} sessions ({time.time() - t0:.0f}s)", flush=True)
    ds = WindowDataset(pq, si, keys, cfg, split_name="train")
    print(f"[rebuild] {len(ds)} window slots ({time.time() - t0:.0f}s)", flush=True)
    X, y = [], []
    for i in range(len(ds)):
        item = ds[i]
        if item is None:
            continue
        X.append(np.asarray(item[0], dtype=np.float32))
        y.append(int(item[1]))
        if i % 20000 == 0:
            print(f"[rebuild] {i}/{len(ds)} ({time.time() - t0:.0f}s)", flush=True)
    return np.stack(X), np.array(y, dtype=np.int64)


def activity_name_map(parquet_path, datasets):
    import pyarrow.compute as pc
    import pyarrow.dataset as pds
    d = pds.dataset(parquet_path, format="parquet")
    tbl = d.to_table(columns=["dataset", "dataset_activity_id", "global_activity_label"],
                     filter=pc.field("dataset").isin(datasets))
    df = tbl.to_pandas().drop_duplicates()
    return {int(r["dataset_activity_id"]): str(r["global_activity_label"])
            for _, r in df.iterrows()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_cfg()
    m = np.load(os.path.join(WORK_DIR, "pca_map.npz"))
    mu, W, cent = m["mean"], m["W"], m["centroids"]
    enc = get_encoder(cfg, device)
    agree = validate(enc, mu, W, cent, device)

    X, yid = rebuild_realdata(cfg)
    print(f"[rebuild] {len(X)} labelled windows", flush=True)
    types = assign(embed(enc, X, device, tag=":rd"), mu, W, cent)

    nmap = activity_name_map(cfg["paths"]["dataset_path"],
                             list(cfg["objective"]["mae_multidata_varlen_altattn_flu"]
                                  ["sources"]["realdata"]["train_on_datasets"]))
    gl = np.array([nmap.get(int(v), f"id{int(v)}") for v in yid])
    vocab = sorted(set(gl.tolist()))
    vidx = {g: i for i, g in enumerate(vocab)}
    gidx = np.array([vidx[g] for g in gl])

    tab = np.zeros((len(vocab), 1000), dtype=np.int64)
    np.add.at(tab, (gidx, types), 1)
    tot_t = tab.sum(0)

    per_activity = {}
    for a, nm in enumerate(vocab):
        c = tab[a]
        t = int(c.sum())
        if t == 0:
            continue
        cs = np.cumsum(np.sort(c)[::-1]) / t
        per_activity[nm] = {
            "n_windows": t,
            "share_pct": round(100 * t / len(X), 2),
            "types_touched": int((c > 0).sum()),
            "types_for_50pct": int(np.searchsorted(cs, 0.50) + 1),
            "types_for_90pct": int(np.searchsorted(cs, 0.90) + 1),
            "top_types": [{"type": int(k), "share_of_activity_pct": round(100 * c[k] / t, 1),
                           "purity_pct": round(100 * c[k] / tot_t[k], 1)}
                          for k in np.argsort(c)[::-1][:3] if c[k] > 0],
        }

    named = []
    for k in range(1000):
        n_k = int(tot_t[k])
        if n_k < 30:
            continue
        col = tab[:, k]
        a = int(np.argmax(col))
        p = col / n_k
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        order = np.argsort(col)[::-1]
        named.append({"type": int(k), "n_windows": n_k, "top_activity": vocab[a],
                      "purity_pct": round(100 * float(col[a] / n_k), 1),
                      "entropy_nats": round(ent, 3),
                      "runner_up": vocab[int(order[1])] if col[order[1]] > 0 else None,
                      "runner_up_pct": round(100 * float(col[order[1]] / n_k), 1)})
    named.sort(key=lambda r: -r["purity_pct"])
    pur = np.array([r["purity_pct"] for r in named])
    chance = float((tab.sum(1) / tab.sum()).max() * 100)

    out = {
        "method": ("labelled realdata windows re-embedded with the frozen encoder, "
                   "projected with the recovered PCA-64 map, assigned to the nearest of "
                   "the 1000 centroids. NOT an identity join with the cached pool, which "
                   "is not reproducible from the current code (4.16% content match)."),
        "roundtrip_agreement": round(agree, 4),
        "n_windows": int(len(X)), "vocabulary": vocab,
        "types_touched_by_realdata": int((tot_t > 0).sum()),
        "types_with_ge30_windows": len(named),
        "purity_pcts": {p: round(float(np.percentile(pur, p)), 1) for p in (10, 25, 50, 75, 90)},
        "chance_purity_pct": round(chance, 1),
        "median_entropy_nats": round(float(np.median([r["entropy_nats"] for r in named])), 3),
        "per_activity": per_activity, "named_types": named,
    }
    p = os.path.join(OUT_DIR, "type_activity_labels.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nvocabulary ({len(vocab)}): {vocab}")
    print(f"types touched {out['types_touched_by_realdata']}, >=30 windows {len(named)}, "
          f"purity {out['purity_pcts']} (chance {chance:.1f}%)")
    print(f"\n{'activity':24s}{'windows':>9s}{'types':>7s}{'n@50%':>7s}{'n@90%':>7s}  top types")
    for a, v in sorted(per_activity.items(), key=lambda kv: -kv[1]["n_windows"]):
        tt = ", ".join(f"#{t['type']}({t['share_of_activity_pct']}%)" for t in v["top_types"])
        print(f"{a:24s}{v['n_windows']:>9,}{v['types_touched']:>7d}"
              f"{v['types_for_50pct']:>7d}{v['types_for_90pct']:>7d}  {tt}")
    print("\nmost activity-pure types:")
    for r in named[:20]:
        print(f"  #{r['type']:<4d} {r['purity_pct']:>5.1f}%  {r['top_activity']:<22s}"
              f"(n={r['n_windows']}, then {r['runner_up']} {r['runner_up_pct']}%)")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
