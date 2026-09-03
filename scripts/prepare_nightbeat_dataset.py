#!/usr/bin/env python3
"""Convert raw NightbeatDB (.npz per participant) into a compact per-participant
cache for the HR-from-wrist-accelerometer evaluation.

Source
------
Nightbeat: Heart Rate Estimation From a Wrist-Worn Accelerometer During Sleep
(Moebus et al., IEEE BHI 2024).  https://github.com/eth-siplab/Nightbeat
40 participants, ~310 h of overnight recordings, downloaded from ETH polybox
into ``<root>/raw/nightbeatdb`` by ``download_nightbeatdb.sh``.

Raw layout (verified against participant 00)
--------------------------------------------
``{pid:02d}_acc_a.npz``      [5, N100] float64 @100 Hz -- row0 = timestamp (us),
                             rows 1/2/3 = x/y/z in **g** (device range +-2 g,
                             saturating), row4 = magnitude.  This is the WRIST
                             device: all four algorithms in the reference repo
                             (nightbeat/pwr/jerks/bioinsights) read ``acc_a_*``,
                             and ``helpers/ojwalch_prep.py`` maps the Apple
                             Watch wrist dataset onto the same column names.
``{pid:02d}_acc_m.npz``      [4, N100] float32 @100 Hz -- the Movesense *chest*
                             unit's accelerometer (co-located with the ECG).
                             NOT used here: it would leak the reference.
``{pid:02d}_ecg_m.npz``      [N1024] float64 @1024 Hz -- chest ECG reference.
``{pid:02d}_hr_bxb.npz``     [K] int32 -- beat sample indices in 1024 Hz space.
``{pid:02d}_hr_bxb_rate.npz``[K] int32 -- per-beat HR in bpm, **truncated to
                             integer**.  We recompute HR in float from the beat
                             indices instead: the paper's headline MAE is
                             0.88 bpm, so ~0.5 bpm quantization noise in the
                             target is not acceptable.
``{pid:02d}_hrv_isvalid.npz``[N1024] float64 @1024 Hz -- 0/1 ECG validity.

The 100 Hz accelerometer and the 1024 Hz ECG streams share a t=0 origin and an
identical duration (verified: 24034.19 s vs 24034.199 s for pid 00), which is
the same assumption the reference repo's ``chunk_trace`` makes.

Output (one ``{pid:02d}.npz`` per participant, ~18 MB each)
-----------------------------------------------------------
``acc``          [3, N50] float32 -- wrist acc in m/s^2, resampled to the
                 pipeline's canonical 50 Hz (imu_lm.data.windowing) with an
                 anti-aliasing FIR + zero-phase filtering.
``isvalid``      [N50] uint8 -- ECG validity decimated to 50 Hz (min-pooled, so
                 a window is only "valid" if every contributing sample was).
``beat_t``       [K] float64 -- beat times in seconds.
``beat_hr``      [K] float32 -- 60 / IBI for the beat ending each interval;
                 ``beat_hr[0]`` is NaN (no preceding beat).
``clip_frac``    scalar -- fraction of raw acc samples at the +-2 g rail.

Windowing is deliberately NOT baked in: the loader
(``imu_lm.data.nightbeat_loader``) cuts windows from these traces, so window
length / hop can be changed without re-running this script.

Usage
-----
    python scripts/prepare_nightbeat_dataset.py \
        --raw-root /projects/b53017/vbc1591/Nightbeat/raw/nightbeatdb \
        --out      /projects/b53017/vbc1591/Nightbeat/prepared \
        --workers  8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy import signal

RAW_ACC_HZ = 100.0
ECG_HZ = 1024.0
CANONICAL_HZ = 50.0  # imu_lm.data.windowing.CANONICAL_SAMPLE_RATE_HZ
G_TO_MS2 = 9.80665
ACC_RAIL_G = 2.0  # the wrist unit saturates at +-2 g

DEFAULT_RAW_ROOT = "/projects/b53017/vbc1591/Nightbeat/raw/nightbeatdb"
DEFAULT_OUT = "/projects/b53017/vbc1591/Nightbeat/prepared"

PID_RE = re.compile(r"^(\d+)_acc_a\.npz$")


def discover_pids(raw_root: Path) -> list[int]:
    pids = []
    for f in os.listdir(raw_root):
        m = PID_RE.match(f)
        if m:
            pids.append(int(m.group(1)))
    return sorted(pids)


def _decimate_2x(x: np.ndarray) -> np.ndarray:
    """100 Hz -> 50 Hz with an anti-aliasing FIR, zero phase.

    Plain slicing/interp would alias the 25-50 Hz band down onto the 0-25 Hz
    band where the ballistocardiogram lives, so use a real decimator.
    """
    return signal.decimate(x, q=2, ftype="fir", zero_phase=True)


def _min_pool_to_50hz(valid_1024: np.ndarray, n_out: int) -> np.ndarray:
    """1024 Hz 0/1 validity -> 50 Hz, using a min over each output sample's
    span.  Min (not mean) so any invalid ECG inside the span marks the 50 Hz
    sample invalid -- we would rather drop a window than train on a bad target.
    """
    edges = np.linspace(0, valid_1024.shape[0], n_out + 1).astype(np.int64)
    out = np.ones(n_out, dtype=np.uint8)
    # np.minimum.reduceat over ragged spans; guard against zero-length spans.
    starts = edges[:-1]
    ends = edges[1:]
    nonempty = ends > starts
    if nonempty.any():
        red = np.minimum.reduceat(valid_1024, starts[nonempty])
        out[nonempty] = (red > 0.5).astype(np.uint8)
    return out


def process_pid(pid: int, raw_root: str, out_dir: str, overwrite: bool) -> dict:
    raw_root = Path(raw_root)
    out_path = Path(out_dir) / f"{pid:02d}.npz"
    if out_path.exists() and not overwrite:
        return {"pid": pid, "status": "skipped (exists)"}

    acc_a = np.load(raw_root / f"{pid:02d}_acc_a.npz")["arr_0"]
    if acc_a.shape[0] < 4:
        raise ValueError(f"pid {pid}: acc_a has {acc_a.shape[0]} rows, expected >=4")

    ts_us = acc_a[0]
    acc_g = acc_a[1:4]  # [3, N100] in g

    # The reference pipeline assumes a uniform 100 Hz grid; verify rather than
    # trust it, since a dropped block would silently shift every HR target.
    dt = np.diff(ts_us)
    dt_med = float(np.median(dt))
    max_dev_s = float(np.abs(dt - dt_med).max()) / 1e6 if dt.size else 0.0

    clip_frac = float(np.mean(np.abs(acc_g) >= ACC_RAIL_G - 1e-9))

    # NaNs would propagate through the decimation FIR across the whole trace.
    n_nan = int(np.isnan(acc_g).sum())
    if n_nan:
        for c in range(3):
            row = acc_g[c]
            bad = np.isnan(row)
            if bad.all():
                raise ValueError(f"pid {pid}: acc channel {c} is all-NaN")
            idx = np.arange(row.shape[0])
            row[bad] = np.interp(idx[bad], idx[~bad], row[~bad])

    acc_ms2 = (acc_g * G_TO_MS2).astype(np.float64)
    acc_50 = np.stack([_decimate_2x(acc_ms2[c]) for c in range(3)]).astype(np.float32)
    n50 = acc_50.shape[1]

    isvalid_1024 = np.load(raw_root / f"{pid:02d}_hrv_isvalid.npz")["arr_0"]
    isvalid_50 = _min_pool_to_50hz(isvalid_1024, n50)

    beat_idx = np.load(raw_root / f"{pid:02d}_hr_bxb.npz")["arr_0"].astype(np.int64)
    beat_rate_int = np.load(raw_root / f"{pid:02d}_hr_bxb_rate.npz")["arr_0"].astype(np.int32)
    order = np.argsort(beat_idx)
    beat_idx = beat_idx[order]
    beat_rate_int = beat_rate_int[order]

    beat_t = beat_idx.astype(np.float64) / ECG_HZ
    ibi = np.diff(beat_t)
    beat_hr = np.full(beat_t.shape[0], np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        beat_hr[1:] = 60.0 / ibi
    beat_hr = beat_hr.astype(np.float32)

    # Sanity: our float HR must match the shipped integer rate to within the
    # truncation itself. A larger gap means the index/rate pairing is off.
    both = np.isfinite(beat_hr[1:])
    if both.any():
        resid = beat_hr[1:][both] - beat_rate_int[1:][both]
        pair_err = float(np.median(np.abs(resid)))
    else:
        pair_err = float("nan")

    acc_dur = acc_g.shape[1] / RAW_ACC_HZ
    ecg_dur = isvalid_1024.shape[0] / ECG_HZ

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        out_path,
        acc=acc_50,
        isvalid=isvalid_50,
        beat_t=beat_t,
        beat_hr=beat_hr,
        fs=np.float32(CANONICAL_HZ),
        pid=np.int32(pid),
        clip_frac=np.float32(clip_frac),
        duration_s=np.float32(acc_dur),
    )

    return {
        "pid": pid,
        "status": "ok",
        "n_samples_50hz": int(n50),
        "duration_h": round(acc_dur / 3600.0, 3),
        "acc_ecg_dur_gap_s": round(abs(acc_dur - ecg_dur), 3),
        "max_ts_gap_s": round(max_dev_s, 6),
        "n_beats": int(beat_idx.shape[0]),
        "hr_median": round(float(np.nanmedian(beat_hr)), 2),
        "valid_frac": round(float(isvalid_50.mean()), 4),
        "clip_frac": round(clip_frac, 6),
        "n_nan_acc": n_nan,
        "hr_pairing_resid_bpm": round(pair_err, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pids", nargs="*", type=int, default=None,
                    help="subset of participant ids (default: all found)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    if not raw_root.is_dir():
        raise SystemExit(f"raw root not found: {raw_root}")

    pids = args.pids if args.pids else discover_pids(raw_root)
    if not pids:
        raise SystemExit(f"no *_acc_a.npz files under {raw_root}")
    print(f"[prep] {len(pids)} participants -> {args.out}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    results = []
    if args.workers <= 1:
        for pid in pids:
            r = process_pid(pid, str(raw_root), args.out, args.overwrite)
            print(f"[prep] {r}", flush=True)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_pid, pid, str(raw_root), args.out,
                              args.overwrite): pid for pid in pids}
            for fut in as_completed(futs):
                pid = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # keep going; report at the end
                    r = {"pid": pid, "status": f"FAILED: {type(e).__name__}: {e}"}
                print(f"[prep] {r}", flush=True)
                results.append(r)

    results.sort(key=lambda r: r["pid"])
    summary_path = Path(args.out) / "prep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if str(r.get("status", "")).startswith("FAILED")]
    total_h = sum(r["duration_h"] for r in ok)
    print(f"\n[prep] {len(ok)}/{len(results)} ok, {total_h:.1f} h total")
    if ok:
        gaps = [r["acc_ecg_dur_gap_s"] for r in ok]
        print(f"[prep] max acc/ECG duration gap: {max(gaps):.3f} s")
        print(f"[prep] max timestamp irregularity: {max(r['max_ts_gap_s'] for r in ok):.6f} s")
        print(f"[prep] HR pairing residual (median |float-int| bpm): "
              f"{max(r['hr_pairing_resid_bpm'] for r in ok):.3f}")
    if failed:
        print(f"[prep] {len(failed)} FAILED: {[r['pid'] for r in failed]}")
        sys.exit(1)
    print(f"[prep] summary -> {summary_path}")


if __name__ == "__main__":
    main()
