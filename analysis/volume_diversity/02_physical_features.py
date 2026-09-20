#!/usr/bin/env python
"""Stage 2: an ENCODER-FREE description of every window in the 4-source pool.

The reviewer objection this exists to close: in 11j/11l a "motion type" was a
k-means cluster in *our own pretrained encoder's* embedding space, so the
diversity axis was defined by the very model whose behaviour it was used to
explain. Here each window is described by 40 hand-specified physical features
computed straight from the cached signal -- no learned model anywhere in the
loop -- and the SUBSET SELECTION in 04 runs on these. The encoder embedding is
kept only as an independent read-out, and 03 reports how much the two
definitions agree.

Every feature is invariant to a global rotation of the sensor frame (only
magnitudes, spectra of magnitudes, and gravity-direction *changes* are used).
That matters twice over: pretraining applies octahedral random rotations, so a
frame-dependent "type" would be a type the model never sees; and the four
sources do not share a device mounting convention, so frame-dependent features
would cluster by dataset rather than by motion.

Variable-length windows (motionx 25 % and sonic 5 % of windows are short clips,
64-512 samples) are wrap-tiled to 512 samples before the spectral features, so
the frequency grid is identical for every window and clusters cannot form on
clip length. Autocorrelation lags are capped at L/2 of the ORIGINAL clip so the
tiling period can never masquerade as a cadence.

Writes $WORK/phys_feats_f32.npy [N, 40] and $WORK/phys_feature_names.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import CACHE_DIR, POOL_N, SOURCES, SOURCE_SIZES, WORK_DIR

FS = 50.0
T_FIX = 512
BANDS = [(0.25, 1.0), (1.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 24.0)]
CHUNK = 8192

NAMES = (
    ["acc_mag_mean", "acc_mag_std", "acc_mag_p05", "acc_mag_p95", "acc_mag_iqr"]
    + ["gyr_mag_mean", "gyr_mag_std", "gyr_mag_p05", "gyr_mag_p95", "gyr_mag_iqr"]
    + ["acc_jerk_rms", "gyr_angacc_rms", "corr_acc_gyr_mag", "acc_zcr", "gyr_zcr",
       "stillness_frac", "static_acc_norm", "gravity_dir_change_deg",
       "gravity_instability_deg", "block_std_nonstationarity",
       "autocorr_peak", "autocorr_peak_hz"]
    + [f"acc_bandfrac_{lo}_{hi}" for lo, hi in BANDS]
    + [f"gyr_bandfrac_{lo}_{hi}" for lo, hi in BANDS]
    + ["acc_spec_entropy", "gyr_spec_entropy", "acc_dom_hz", "gyr_dom_hz",
       "acc_spec_centroid_hz", "gyr_spec_centroid_hz", "acc_log_power", "gyr_log_power"]
)
assert len(NAMES) == 40, len(NAMES)


def _tile_to(x: np.ndarray, lens: np.ndarray, t: int = T_FIX) -> np.ndarray:
    """Wrap-tile each [C, L_i] valid prefix out to [C, t]."""
    b, c, _ = x.shape
    out = np.empty((b, c, t), dtype=np.float32)
    for L in np.unique(lens):
        m = lens == L
        L = int(L)
        if L >= t:
            out[m] = x[m, :, :t]
        else:
            reps = int(np.ceil(t / L))
            out[m] = np.tile(x[m, :, :L], (1, 1, reps))[:, :, :t]
    return out


def _zcr(sig: np.ndarray) -> np.ndarray:
    c = sig - sig.mean(axis=1, keepdims=True)
    return (np.diff(np.signbit(c), axis=1).sum(axis=1) / (c.shape[1] - 1)).astype(np.float32)


def _spec_feats(mag: np.ndarray, freqs: np.ndarray, band_idx, prefix_out):
    """Band fractions, spectral entropy, dominant frequency, centroid, log power."""
    c = mag - mag.mean(axis=1, keepdims=True)
    win = np.hanning(c.shape[1]).astype(np.float32)
    p = np.abs(np.fft.rfft(c * win, axis=1)) ** 2
    keep = (freqs >= 0.25) & (freqs <= 24.0)
    pk = p[:, keep]
    fk = freqs[keep]
    tot = pk.sum(axis=1) + 1e-12
    fr = [pk[:, bi].sum(axis=1) / tot for bi in band_idx]
    q = pk / tot[:, None]
    ent = -(np.where(q > 0, q * np.log(q + 1e-12), 0.0)).sum(axis=1) / np.log(q.shape[1])
    dom = fk[np.argmax(pk, axis=1)]
    cen = (pk * fk[None, :]).sum(axis=1) / tot
    logp = np.log10(tot)
    prefix_out.extend(fr)
    return ent, dom, cen, logp


def features(x: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """x [B, 6, T] cache rows (physical units, impute+filter only), lens [B]."""
    xt = _tile_to(x, lens)
    acc, gyr = xt[:, :3], xt[:, 3:]
    amag = np.linalg.norm(acc, axis=1)
    gmag = np.linalg.norm(gyr, axis=1)

    cols = []
    for m in (amag, gmag):
        qs = np.percentile(m, [5, 25, 75, 95], axis=1)
        cols += [m.mean(1), m.std(1), qs[0], qs[3], qs[2] - qs[1]]

    cols.append(np.sqrt((np.linalg.norm(np.diff(acc, axis=2) * FS, axis=1) ** 2).mean(1)))
    cols.append(np.sqrt((np.linalg.norm(np.diff(gyr, axis=2) * FS, axis=1) ** 2).mean(1)))
    ac = amag - amag.mean(1, keepdims=True)
    gc = gmag - gmag.mean(1, keepdims=True)
    cols.append((ac * gc).mean(1) / (ac.std(1) * gc.std(1) + 1e-9))
    cols.append(_zcr(amag))
    cols.append(_zcr(gmag))
    cols.append((np.abs(amag - 9.81) < 0.5).mean(1))

    g = acc.mean(axis=2)                                   # static/gravity direction
    cols.append(np.linalg.norm(g, axis=1))
    half = xt.shape[2] // 2
    g1, g2 = acc[:, :, :half].mean(2), acc[:, :, half:].mean(2)
    cosang = (g1 * g2).sum(1) / (np.linalg.norm(g1, axis=1) * np.linalg.norm(g2, axis=1) + 1e-9)
    cols.append(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    nb = xt.shape[2] // 25                                  # 0.5 s blocks
    ab = acc[:, :, :nb * 25].reshape(len(xt), 3, nb, 25).mean(3)       # [B,3,nb]
    gu = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
    cb = (ab * gu[:, :, None]).sum(1) / (np.linalg.norm(ab, axis=1) + 1e-9)
    cols.append(np.degrees(np.arccos(np.clip(cb, -1, 1))).mean(1))
    mb = amag[:, :nb * 25].reshape(len(xt), nb, 25)
    cols.append(mb.std(2).std(1))

    # autocorrelation of the centred acc magnitude, lags capped at L/2 of the
    # ORIGINAL clip so wrap-tiling cannot fake a cadence
    n2 = 1 << int(np.ceil(np.log2(2 * xt.shape[2])))
    F = np.fft.rfft(ac, n=n2, axis=1)
    r = np.fft.irfft(F * np.conj(F), n=n2, axis=1)[:, :xt.shape[2]]
    r = r / (r[:, :1] + 1e-9)
    lag_lo, lag_hi = int(0.25 * FS), int(2.5 * FS)
    lags = np.arange(lag_lo, lag_hi + 1)
    valid = lags[None, :] <= np.minimum(lens, xt.shape[2])[:, None] // 2
    rr = np.where(valid, r[:, lag_lo:lag_hi + 1], -np.inf)
    has = valid.any(1)
    j = np.where(has, np.argmax(rr, axis=1), 0)
    peak = np.where(has, rr[np.arange(len(rr)), j], 0.0)
    cols.append(np.where(np.isfinite(peak), peak, 0.0))
    cols.append(np.where(has, FS / np.maximum(lags[j], 1), 0.0))

    freqs = np.fft.rfftfreq(xt.shape[2], d=1.0 / FS)
    keep = (freqs >= 0.25) & (freqs <= 24.0)
    fk = freqs[keep]
    band_idx = [np.where((fk >= lo) & (fk < hi))[0] for lo, hi in BANDS]
    band_cols_a, band_cols_g = [], []
    ent_a, dom_a, cen_a, pw_a = _spec_feats(amag, freqs, band_idx, band_cols_a)
    ent_g, dom_g, cen_g, pw_g = _spec_feats(gmag, freqs, band_idx, band_cols_g)
    cols += band_cols_a + band_cols_g
    cols += [ent_a, ent_g, dom_a, dom_g, cen_a, cen_g, pw_a, pw_g]

    out = np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)
    assert out.shape[1] == len(NAMES), (out.shape, len(NAMES))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    out_path = os.path.join(WORK_DIR, "phys_feats_f32.npy")
    if os.path.exists(out_path) and np.load(out_path, mmap_mode="r").shape == (POOL_N, len(NAMES)):
        print("[phys] features already on disk", flush=True)
        return
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(POOL_N, len(NAMES)))
    row = 0
    for s in SOURCES:
        x = np.load(os.path.join(CACHE_DIR, f"{s}_train_x.npy"), mmap_mode="r")
        tpath = os.path.join(CACHE_DIR, f"{s}_train_tmask.npy")
        lens = (np.load(tpath).astype(np.int32) if os.path.exists(tpath)
                else np.full(len(x), x.shape[-1], dtype=np.int32))
        assert len(x) == SOURCE_SIZES[s], f"{s}: cache has {len(x)}, arms.py says {SOURCE_SIZES[s]}"
        for a in range(0, len(x), CHUNK):
            b = min(len(x), a + CHUNK)
            out[row + a:row + b] = features(np.asarray(x[a:b], dtype=np.float32), lens[a:b])
            if (a // CHUNK) % 20 == 0:
                print(f"[phys] {s} {b:,}/{len(x):,}", flush=True)
        row += len(x)
    out.flush()
    assert row == POOL_N
    with open(os.path.join(WORK_DIR, "phys_feature_names.json"), "w") as f:
        json.dump({"names": NAMES, "fs": FS, "t_fixed": T_FIX, "bands": BANDS,
                   "rotation_invariant": True}, f, indent=2)
    print(f"[phys] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
