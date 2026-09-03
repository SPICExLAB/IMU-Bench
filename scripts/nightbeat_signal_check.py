#!/usr/bin/env python3
"""Model-free check that the prepared Nightbeat cache still carries the BCG.

Purpose: separate "the representation does not encode heart rate" from "our
preprocessing destroyed the heart rate before the encoder ever saw it". This
runs a plain spectral-peak heart-rate estimator directly on the prepared 50 Hz
windows — no encoder, no training — and reports how well it recovers the ECG
reference.

Two estimators are reported, because the difference between them is the point:

``peak``     band-pass the acceleration magnitude to the plausible HR
             fundamental range and take the periodogram peak. This is the naive
             reading and it does NOT work: the BCG fundamental is far too weak
             to dominate that band, so the peak locks onto noise.
``envelope`` the standard ballistocardiogram reading, and the one the PWR
             baseline is built on: the heartbeat appears as *amplitude
             modulation* of a higher-frequency band (the sharp J-wave complex,
             ~4-11 Hz), not as energy at the heart rate itself. So band-pass to
             that band, take the Hilbert envelope, and find the periodogram
             peak of the *envelope* in the HR range.

If ``envelope`` recovers HR meaningfully better than ``peak`` and better than
chance, the cardiac information survived download -> decimation -> caching, and
a weak probe result is a statement about the representation rather than about
the data.

It also sweeps window length, which is the parameter that matters most here:
raw FFT resolution is 1/T_seconds Hz = 60/T_seconds bpm, so a 10.24 s window
resolves only ~5.9 bpm and a 20.48 s window ~2.9 bpm. Any spectral method is
floored by that, and it bounds how much of a learned probe's error is
attributable to window length rather than to the representation.

Usage:
    python scripts/nightbeat_signal_check.py --pids 0 1 2 3 4 --windows 10.24 20.48 40.96
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.nightbeat_loader import (
    NIGHTBEAT_PREPARED_ROOT,
    build_window_specs,
    load_participant,
)

FS = 50.0
HR_LO_HZ, HR_HI_HZ = 0.6, 3.0    # 36-180 bpm
BCG_LO_HZ, BCG_HI_HZ = 4.0, 11.0  # J-wave complex band carrying the modulation


def _peak_bpm(x: np.ndarray, nfft: int = 8192) -> float:
    """Periodogram peak of ``x`` restricted to the HR band, in bpm."""
    x = x - x.mean()
    if x.std() < 1e-12:
        return float("nan")
    x = x * sps.windows.hann(x.size)
    spec = np.abs(np.fft.rfft(x, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / FS)
    band = (freqs >= HR_LO_HZ) & (freqs <= HR_HI_HZ)
    if not band.any():
        return float("nan")
    return float(freqs[band][np.argmax(spec[band])] * 60.0)


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sos = sps.butter(4, [lo / (FS / 2), hi / (FS / 2)], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x)


def hr_peak(acc: np.ndarray) -> float:
    """Naive: periodogram peak of the acc magnitude in the HR band."""
    mag = np.linalg.norm(acc, axis=0)
    if mag.std() < 1e-12:
        return float("nan")
    return _peak_bpm(_bandpass(mag, HR_LO_HZ, HR_HI_HZ))


def hr_envelope(acc: np.ndarray) -> float:
    """BCG-style: HR from the Hilbert envelope of the 4-11 Hz J-wave band.

    Estimated per axis and combined by median — a single wrist axis often has
    the BCG buried, and PWR's axis-selection step exists for the same reason;
    the median is the label-free stand-in for it.
    """
    ests = []
    for c in range(acc.shape[0]):
        x = acc[c]
        if x.std() < 1e-12:
            continue
        env = np.abs(sps.hilbert(_bandpass(x, BCG_LO_HZ, BCG_HI_HZ)))
        est = _peak_bpm(_bandpass(env, HR_LO_HZ, HR_HI_HZ))
        if np.isfinite(est):
            ests.append(est)
    return float(np.median(ests)) if ests else float("nan")


METHODS = {"peak": hr_peak, "envelope": hr_envelope}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=NIGHTBEAT_PREPARED_ROOT)
    ap.add_argument("--pids", nargs="*", type=int, default=None)
    ap.add_argument("--windows", nargs="*", type=float, default=[10.24, 20.48, 40.96])
    ap.add_argument("--max-windows-per-pid", type=int, default=300)
    ap.add_argument("--methods", nargs="*", default=["peak", "envelope"],
                    choices=list(METHODS))
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob("[0-9]*.npz"))
    if args.pids is not None:
        want = set(args.pids)
        files = [f for f in files if int(f.stem) in want]
    if not files:
        raise SystemExit(f"no participant files under {root}")

    print(f"{'method':>9}{'win_s':>7}{'n':>8}{'MAE':>9}{'RMSE':>9}{'Cor':>7}"
          f"{'FFTres':>9}{'subjMean':>10}")
    print("-" * 68)

    for win_s in args.windows:
        T = int(round(win_s * FS))

        # Cut the window set once per window length, then score every method on
        # exactly the same windows so the rows are directly comparable.
        windows, ys, gs = [], [], []
        for f in files:
            data = load_participant(f)
            specs = build_window_specs(
                data, T=T, hop=T, min_valid_frac=1.0, min_beats=5, max_hr_std=8.0)
            if not specs:
                continue
            if len(specs) > args.max_windows_per_pid:
                sel = np.linspace(0, len(specs) - 1, args.max_windows_per_pid).astype(int)
                specs = [specs[i] for i in sel]
            acc = data["acc"]
            for s in specs:
                windows.append(acc[:, s.start : s.start + T])
                ys.append(s.hr); gs.append(s.pid)

        if not windows:
            continue
        y = np.array(ys); g = np.array(gs)
        sm = np.empty_like(y)
        for q in np.unique(g):
            m = g == q
            sm[m] = y[m].mean()

        for mname in args.methods:
            fn = METHODS[mname]
            p = np.array([fn(w) for w in windows])
            ok = np.isfinite(p)
            if not ok.any():
                continue
            yv, pv = y[ok], p[ok]
            err = pv - yv
            cor = (float(np.corrcoef(yv, pv)[0, 1])
                   if yv.std() > 1e-9 and pv.std() > 1e-9 else 0.0)
            print(f"{mname:>9}{win_s:>7.2f}{yv.size:>8}{np.abs(err).mean():>9.2f}"
                  f"{np.sqrt((err ** 2).mean()):>9.2f}{cor:>7.2f}"
                  f"{60.0 / win_s:>9.2f}{np.abs(sm[ok] - yv).mean():>10.2f}")

    print("\nFFTres = raw FFT bin spacing in bpm (60 / window_seconds) -- the "
          "floor for any\nspectral method at that window length. subjMean = "
          "per-participant-mean oracle\non the same windows (paper's Mean-Pred. "
          "row, 3.16 bpm on their windows).")


if __name__ == "__main__":
    main()
