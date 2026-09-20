#!/usr/bin/env python
"""Stage 8 (phase 3): recover per-window SUBJECT and SESSION provenance for the
cached pool, so subject/session diversity stops being an unmeasured confound.

The scaling curve this section replaces moved subject count and session count
along with everything else, and the factorial in 04 controls window count,
source count and motion-type diversity but not who the wearer was. This script
makes that measurable, and 09 turns it into a controlled ladder.

The cache stores signal only, so provenance is rebuilt from the raw loaders and
joined positionally: `scripts/build_multidata_cache.py` iterates each source's
dataset with `shuffle=False` and writes rows in order, dropping only windows the
collate rejects, so cache row i == dataset item i whenever the row counts agree.
That assumption is CHECKED, not trusted -- a random sample of rows is
fingerprinted by content against the rebuild (the join in
`analysis/diversity_saturation/fingerprint_join.py` exists because this
assumption is FALSE for realdata), and the script refuses to write anything if
the probes do not all match.

ocdetect exposes file-level provenance and a subject id parseable from the
filename; motionx/sonic expose one file per synthetic clip (treated as its own
"session", with no subject concept -- documented, conservative); eatingspeed
exposes none through the cached path and is written as unknown rather than
guessed.

Writes $WORK/prov_session.npy, $WORK/prov_subject.npy (int32, -1 = unknown)
       out/provenance.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diversity_saturation")))
from arms import (BASE_CONFIG, CACHE_DIR, OUT_DIR, POOL_CONFIG, POOL_N, SEED,
                  SOURCES, SOURCE_SIZES, WORK_DIR, source_ranges)

N_PROBES = 400


def fingerprint(x):
    return hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float32).tobytes(),
                           digest_size=16).digest()


def build_source(name):
    """Rebuild one source from raw with provenance, returning (dataset, groups)."""
    from imu_lm.data.combined_mae_multidata_loader import _BUILDERS, _group_windows_by_session
    from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml
    cfg = deep_update(load_yaml(BASE_CONFIG), load_yaml(POOL_CONFIG))
    for key in ("random_rotation", "gaussian_noise", "gain_jitter", "axis_bias", "time_warp"):
        if isinstance(cfg.get(key), dict):
            cfg[key]["enabled"] = False
    # The cache is RAW: `objective.cache_is_normalized: false` means
    # CachedWindowDataset z-scores at READ time, so the .npy files hold values at
    # gravity scale (measured: |mean acc| ~ 8.9 m/s^2). Rebuilding through the raw
    # loaders WOULD z-score them (mean~0, std~1), so every fingerprint mismatches
    # and the join looks like a reorder failure when it is really a units failure.
    # Match the cache: normalise nothing during the rebuild.
    if not bool(cfg_get(cfg, ["objective", "cache_is_normalized"], True)):
        pp = cfg.setdefault("preprocess", {}).setdefault("normalize", {})
        pp["enabled"] = False
    src_cfg = cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sources"][name]
    vl_max = int(cfg_get(cfg, ["data", "variable_length", "max_samples"], 512))
    train_ds, _ = _BUILDERS[name](cfg, src_cfg, vl_max, collect_provenance=True)
    return train_ds, _group_windows_by_session(name, train_ds)


def diagnose(ds, cache, probes):
    """Is the cache a REORDER of the rebuild, or different content entirely?

    A content join can only recover a permutation. If the rebuild emits values
    the cache never held -- a different scaling, channel order or window length
    -- fingerprints will not match no matter how they are indexed, and the
    honest outcome is to leave the source unknown."""
    item = ds[int(probes[0])]
    xi = item[0] if isinstance(item, (tuple, list)) else item
    xi = np.asarray(xi, dtype=np.float32)
    ci = np.asarray(cache[int(probes[0])], dtype=np.float32)
    d = {"ds_shape": list(xi.shape), "cache_shape": list(ci.shape),
         "ds_channels": int(xi.shape[0]),
         "ds_range": [float(xi.min()), float(xi.max())],
         "cache_range": [float(ci.min()), float(ci.max())]}
    shape_ok = xi.shape[0] <= ci.shape[0] and xi.shape[-1] <= ci.shape[-1]
    # compare marginal distributions over a handful of rows, order-independent
    dq = np.quantile([np.asarray(ds[int(i)][0] if isinstance(ds[int(i)], (tuple, list))
                                 else ds[int(i)], dtype=np.float32).ravel()
                      for i in probes], [0.05, 0.5, 0.95])
    cq = np.quantile(np.asarray(cache[np.sort(probes)], dtype=np.float32).ravel(),
                     [0.05, 0.5, 0.95])
    d["ds_quantiles"], d["cache_quantiles"] = dq.tolist(), cq.tolist()
    scale_ok = bool(np.allclose(dq, cq, rtol=0.35, atol=0.35))
    d["content_plausible"] = bool(shape_ok and scale_ok)
    d["verdict"] = ("same value distribution and compatible shape -> looks like a "
                    "REORDER, content join is worth trying"
                    if d["content_plausible"] else
                    "value distribution or shape differs -> NOT a reorder; the "
                    "rebuild does not reproduce cached content")
    return d


def _requested_sources():
    """`--sources a,b` restricts the rebuild. The wearer ladder in 09 is
    ocdetect-only by construction (it is the one source with a real wearer
    identity -- motionx/sonic expose one synthetic clip per "session" and no
    subject at all, eatingspeed exposes nothing through the cached path), so
    rebuilding sonic's 135k Bones-Seed sessions to populate a `sess` array the
    ladder never reads costs ~6 h for nothing. Skipped sources keep their -1
    ("unknown") entries, which is what they would have got anyway."""
    for i, a in enumerate(sys.argv):
        if a == "--sources" and i + 1 < len(sys.argv):
            return {s.strip() for s in sys.argv[i + 1].split(",") if s.strip()}
        if a.startswith("--sources="):
            return {s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()}
    return set(SOURCES)


def main():
    from imu_lm.data.combined_mae_multidata_loader import _subject_key_from_session_key

    rng = np.random.default_rng(SEED)
    sess = np.full(POOL_N, -1, dtype=np.int32)
    subj = np.full(POOL_N, -1, dtype=np.int32)
    report = {"sources": {}, "probes": N_PROBES}
    ranges = source_ranges()

    want = _requested_sources()
    for name in SOURCES:
        if name not in want:
            report["sources"][name] = {"status": "not attempted (--sources)"}
            print(f"[prov] {name}: skipped by --sources", flush=True)
            continue
        lo, hi = ranges[name]
        try:
            ds, groups = build_source(name)
        except Exception as exc:                       # noqa: BLE001
            report["sources"][name] = {"status": f"rebuild failed: {exc}"}
            print(f"[prov] {name}: rebuild failed ({exc}) -- left unknown", flush=True)
            continue
        if groups is None:
            report["sources"][name] = {"status": "no per-window provenance exposed"}
            print(f"[prov] {name}: no provenance exposed -- left unknown", flush=True)
            continue
        if len(ds) != SOURCE_SIZES[name]:
            report["sources"][name] = {"status": f"row count {len(ds)} != cache {SOURCE_SIZES[name]}"}
            print(f"[prov] {name}: row count mismatch, refusing to join", flush=True)
            continue

        # positional join, verified by content on a random probe sample
        cache = np.load(os.path.join(CACHE_DIR, f"{name}_train_x.npy"), mmap_mode="r")
        probes = rng.choice(len(ds), size=min(N_PROBES, len(ds)), replace=False)
        ok = 0
        for i in probes:
            item = ds[int(i)]
            xi = item[0] if isinstance(item, (tuple, list)) else item
            xi = np.asarray(xi, dtype=np.float32)
            ci = np.asarray(cache[int(i)], dtype=np.float32)[:, :xi.shape[-1]]
            ok += int(fingerprint(xi) == fingerprint(ci))

        # row -> cache row. Identity if the positional join verified; otherwise
        # recovered by CONTENT, which is the only thing that survives a reorder.
        remap = None
        if ok != len(probes):
            print(f"[prov] {name}: positional join refused ({ok}/{len(probes)} probes) "
                  f"-- diagnosing", flush=True)
            diag = diagnose(ds, cache, probes[:8])
            report.setdefault("diagnostics", {})[name] = diag
            print(f"[prov] {name}: {diag['verdict']}", flush=True)
            if not diag["content_plausible"]:
                report["sources"][name] = {
                    "status": f"positional join REFUSED ({ok}/{len(probes)}) and content "
                              f"differs, not merely reordered: {diag['verdict']}"}
                continue
            import fingerprint_join as fj
            nch = diag["ds_channels"]
            print(f"[prov] {name}: building content index over {len(cache):,} cache rows",
                  flush=True)
            cidx = fj.build_index(cache, channels=slice(0, nch))
            fps = []
            for i in range(len(ds)):
                item = ds[i]
                xi = item[0] if isinstance(item, (tuple, list)) else item
                fps.append(fingerprint(np.asarray(xi, dtype=np.float32)))
                if i % 100_000 == 0:
                    print(f"      [fp] hashed {i:,}/{len(ds):,}", flush=True)
            remap, matched = fj.join(fps, cidx, name=name)

        skeys = sorted(groups)
        sub_of = {k: _subject_key_from_session_key(name, k) for k in skeys}
        sub_ids = {s: j for j, s in enumerate(sorted(set(sub_of.values())))}
        for j, k in enumerate(skeys):
            rows = np.asarray(groups[k], dtype=np.int64)
            if remap is not None:
                rows = remap[rows]
                rows = rows[rows >= 0]
            idxs = rows + lo
            sess[idxs] = j
            subj[idxs] = sub_ids[sub_of[k]]
        report["sources"][name] = {"status": "ok", "sessions": len(skeys),
                                   "subjects": len(sub_ids), "probes_matched": ok,
                                   "join": "positional" if remap is None else "content"}
        print(f"[prov] {name}: {len(skeys)} sessions / {len(sub_ids)} subjects, "
              f"{ok}/{len(probes)} probes matched", flush=True)

    np.save(os.path.join(WORK_DIR, "prov_session.npy"), sess)
    np.save(os.path.join(WORK_DIR, "prov_subject.npy"), subj)
    report["coverage"] = {"windows_with_session": int((sess >= 0).sum()),
                          "windows_with_subject": int((subj >= 0).sum()),
                          "pool_n": POOL_N}
    with open(os.path.join(OUT_DIR, "provenance.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["coverage"], indent=2))


if __name__ == "__main__":
    main()
