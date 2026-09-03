"""Window-level preprocessing: impute, filter, normalize, transpose to [C, T]."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from imu_lm.utils.helpers import cfg_get
from imu_lm.data.augmentations.transform import apply_frame_perturb

logger = logging.getLogger(__name__)


try:  # optional heavy dependency; used for filtering/spline if available
    from scipy import signal, interpolate  # type: ignore
except Exception:  # pragma: no cover - optional
    signal = None
    interpolate = None


@dataclass
class PreprocessStats:
    windows_seen: int = 0
    windows_dropped_impute: int = 0
    windows_filtered: int = 0
    windows_normalized: int = 0


def impute_window(Xw: np.ndarray, cfg: Any, stats: Optional[PreprocessStats] = None) -> Optional[np.ndarray]:
    if not bool(cfg_get(cfg, ["preprocess", "impute", "enabled"], False)):
        return Xw

    method = cfg_get(cfg, ["preprocess", "impute", "method"], "linear")
    max_missing = float(cfg_get(cfg, ["preprocess", "impute", "max_missing_frac"], 0.0))

    missing_mask = np.isnan(Xw)
    missing_frac = missing_mask.mean() if Xw.size else 0.0
    if missing_frac > max_missing:
        if stats:
            stats.windows_dropped_impute += 1
        return None

    T = Xw.shape[0]
    t_idx = np.arange(T)
    for c in range(Xw.shape[1]):
        xc = Xw[:, c]
        missing = np.isnan(xc)
        if not missing.any():
            continue
        valid_idx = t_idx[~missing]
        valid_vals = xc[~missing]
        if valid_idx.size == 0:
            if stats:
                stats.windows_dropped_impute += 1
            return None
        if method == "spline" and interpolate is not None and valid_idx.size >= 4:
            spline = interpolate.CubicSpline(valid_idx, valid_vals, extrapolate=True)
            xc[missing] = spline(t_idx[missing])
        else:  # linear fallback
            # np.interp requires sorted valid_idx and at least one value; handle single value
            if valid_idx.size == 1:
                xc[missing] = valid_vals[0]
            else:
                xc[missing] = np.interp(t_idx[missing], valid_idx, valid_vals)
        Xw[:, c] = xc

    return Xw


def filter_window(Xw: np.ndarray, cfg: Any, stats: Optional[PreprocessStats] = None) -> np.ndarray:
    """Low/high/band-pass filter a ``[T, C]`` window.

    Returns a NEW array — never mutates ``Xw``. This is load-bearing, not
    defensive style (bug fixed 2026-07-28): the old version wrote back into
    the caller's array via ``Xw[:, c] = filtfilt(...)`` and returned that same
    object. Every caller reached here through ``preprocess_window``, which
    copies first (``Xw.astype(np.float32)``), so the aliasing was invisible —
    until ``humanml3d_dataset_FLU.py`` began calling ``filter_window``
    directly on (a) a worker-cached clip array and (b) a *view* into one.
    Because those clips are cached per worker and re-drawn many times over a
    run, each draw re-filtered the already-filtered cache cumulatively:
    measured ~0.985 correlation vs. a single pass after 50 draws, with the
    passband decaying without bound as training length grows, and
    non-deterministically (it depends on how often a given worker happened to
    draw a given clip). In the random-crop path it was worse in kind — only
    the cropped sub-range got re-filtered, leaving step discontinuities at
    crop boundaries inside a single clip.
    """
    filt_cfg = cfg_get(cfg, ["preprocess", "filter"], {}) or {}
    if not filt_cfg.get("enabled", False):
        return Xw

    if signal is None:
        raise ImportError("scipy is required for filtering; install scipy to enable filter_window")

    ftype = filt_cfg.get("type", "bandpass")
    order = int(filt_cfg.get("order", 4))
    low = filt_cfg.get("low_hz")
    high = filt_cfg.get("high_hz")
    if (ftype in {"highpass", "bandpass"} and low is None) or (ftype in {"lowpass", "bandpass"} and high is None):
        raise ValueError("filter.low_hz/high_hz must be set for the chosen filter type")
    fs = 50.0  # canonical sample rate

    nyq = 0.5 * fs
    if ftype == "lowpass":
        Wn = high / nyq
        b, a = signal.butter(order, Wn, btype="low")
    elif ftype == "highpass":
        Wn = low / nyq
        b, a = signal.butter(order, Wn, btype="high")
    else:  # bandpass
        Wn = [low / nyq, high / nyq]
        b, a = signal.butter(order, Wn, btype="band")

    # Write into a fresh array, never back into Xw -- see docstring.
    out = np.array(Xw, copy=True)
    for c in range(out.shape[1]):
        out[:, c] = signal.filtfilt(b, a, Xw[:, c], method="gust")

    if stats:
        stats.windows_filtered += 1
    return out


def normalize_window(Xw: np.ndarray, cfg: Any, stats: Optional[PreprocessStats] = None) -> np.ndarray:
    norm_cfg = cfg_get(cfg, ["preprocess", "normalize"], {}) or {}
    if not norm_cfg.get("enabled", False):
        return Xw

    method = norm_cfg.get("method", "zscore")
    eps = float(norm_cfg.get("eps", 1e-6))

    Xn = np.empty_like(Xw, dtype=np.float32)
    for c in range(Xw.shape[1]):
        x = Xw[:, c].astype(np.float32)
        if method == "robust":
            q25, q75 = np.percentile(x, [25, 75])
            scale = (q75 - q25) + eps
            Xn[:, c] = (x - np.median(x)) / scale
        elif method == "minmax":
            xmin, xmax = x.min(), x.max()
            Xn[:, c] = (x - xmin) / ((xmax - xmin) + eps)
        elif method == "physical":
            # DC-PRESERVING fixed scaling (2026-08-18): divide by a constant per modality
            # instead of per-window statistics, so gravity direction (window mean) and motion
            # amplitude (window std) survive into the model. acc: m/s^2 -> g; gyro: rad/s / w_ref.
            # Optional symmetric clipping keeps outliers from dominating the numeric range.
            n_acc = int(norm_cfg.get("n_acc", 3))
            # acc_units: "auto" (default) | "g" | "m/s2". Mixed units across datasets (Capture24 and
            # NHANES are g, the rest m/s^2) would otherwise be scaled 9.81x apart -- z-scoring hid
            # this, fixed physical scaling does not.
            if c < n_acc:
                u = str(norm_cfg.get("acc_units", "auto")).lower()
                if u == "auto":
                    mag = float(np.linalg.norm(Xw[:, :n_acc].mean(axis=0)))
                    spread = float(Xw[:, :n_acc].std())
                    # both conditions: gravity-free m/s^2 data (PADS) has a small mean but large spread
                    u = "g" if (mag < 3.0 and spread < 1.5) else "m/s2"
                sc = float(norm_cfg.get("acc_scale", 9.81)) / (9.80665 if u == "g" else 1.0)
            else:
                sc = float(norm_cfg.get("gyro_scale", 5.0))
            v = x / sc
            clip = norm_cfg.get("clip_acc", 8.0) if c < n_acc else norm_cfg.get("clip_gyro", 8.0)
            Xn[:, c] = np.clip(v, -float(clip), float(clip)) if clip else v
        else:  # zscore
            Xn[:, c] = (x - x.mean()) / (x.std() + eps)

    if stats:
        stats.windows_normalized += 1
    return Xn


def preprocess_window(Xw: np.ndarray, cfg: Any, stats: Optional[PreprocessStats] = None) -> Optional[np.ndarray]:
    """Apply impute -> filter -> normalize, returning [C, T] float32 or None if dropped."""

    if stats:
        stats.windows_seen += 1

    Xw = Xw.astype(np.float32)

    Xw = impute_window(Xw, cfg, stats)
    if Xw is None:
        return None

    # Coordinate-frame perturbation: models a sensor MOUNTED in a different
    # frame, so it applies to every split and runs on the raw signal -- before
    # normalize_window, which is per-channel and does not commute with a
    # rotation. Placed after impute so a rotation can't smear NaNs across axes.
    # No-op unless the config carries `frame_perturb.enabled: true`.
    Xw = apply_frame_perturb(Xw, cfg)

    Xw = filter_window(Xw, cfg, stats)
    Xw = normalize_window(Xw, cfg, stats)

    return np.transpose(Xw, (1, 0))
