"""Label-agnostic augmentations for raw sensor windows.

All functions operate on numpy arrays shaped [T, C] (time first, channels second).
They are intended to be called *before* tensor conversion so they remain
representation-agnostic.

Augmentation catalogue
----------------------
random_rotation   : uniform SO(3) applied to acc (ch0-2) + gyro (ch3-5) together
gaussian_noise    : additive i.i.d. Gaussian noise
jitter            : second independent Gaussian pass at a smaller sigma
bias_drift        : per-channel random-walk bias that drifts over the window
sensor_dropout    : zero out entire channels with probability p
time_crop         : crop a random contiguous sub-window, resample to original length
frequency_mask    : zero-out random frequency bands (SpecAugment-style, per channel)
gain_jitter       : per-channel amplitude scale
axis_bias         : per-channel constant DC offset
time_warp         : smooth monotonic time warp
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imu_lm.utils.helpers import cfg_get


def add_gaussian_noise(x: np.ndarray, sigma: float) -> np.ndarray:
    """Additive sensor jitter: x <- x + N(0, sigma^2)."""

    noise = np.random.normal(0.0, sigma, size=x.shape).astype(x.dtype)
    return x + noise


def apply_gain_jitter(x: np.ndarray, min_scale: float, max_scale: float) -> np.ndarray:
    """Channel-wise gain jitter with a single scale per axis."""

    scale = np.random.uniform(min_scale, max_scale, size=(1, x.shape[1])).astype(x.dtype)
    return x * scale


def apply_axis_bias(x: np.ndarray, max_bias: float) -> np.ndarray:
    """Channel-wise bias offset jitter with a constant offset per axis."""

    bias = np.random.uniform(-max_bias, max_bias, size=(1, x.shape[1])).astype(x.dtype)
    return x + bias


def smooth_time_warp(x: np.ndarray, max_warp: float, sigma: int) -> np.ndarray:
    """Smooth, monotonic time warp (bounded by +/- max_warp of timeline).

    - Preserves ordering (monotonic warp).
    - Uses a Gaussian-smoothed noise curve to create the warp field.
    - Resamples each channel with 1D interpolation.
    """

    T = x.shape[0]
    if T < 2 or max_warp <= 0:
        return x

    t_orig = np.linspace(0.0, 1.0, T)
    noise = np.random.normal(0.0, 1.0, size=T)

    if sigma > 0:
        radius = int(3 * sigma)
        t = np.arange(-radius, radius + 1)
        kernel = np.exp(-(t ** 2) / (2 * (sigma ** 2)))
        kernel = kernel / kernel.sum()
        smooth = np.convolve(noise, kernel, mode="same")
    else:
        smooth = noise

    # Bound the warp displacement to +/- max_warp of normalized time.
    smooth = smooth / (np.max(np.abs(smooth)) + 1e-8) * max_warp

    warp = t_orig + smooth
    warp = np.clip(warp, 0.0, 1.0)
    warp[0] = 0.0
    warp[-1] = 1.0
    warp = np.maximum.accumulate(warp)

    x_warped = np.empty_like(x)
    for c in range(x.shape[1]):
        x_warped[:, c] = np.interp(t_orig, warp, x[:, c]).astype(x.dtype)
    return x_warped


def _random_rotation_matrix() -> np.ndarray:
    """Sample a uniformly random rotation from SO(3) via unit quaternion."""
    q = np.random.randn(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def apply_random_rotation(x: np.ndarray) -> np.ndarray:
    """Apply a single random SO(3) rotation to acc (ch 0-2) and gyro (ch 3-5).

    The same rotation R is applied to both triaxial sensors so the relative
    geometry between acc and gyro is preserved.  x is [T, C] with C >= 6.
    """
    R = _random_rotation_matrix()
    out = x.copy()
    out[:, :3] = (R @ x[:, :3].T).T
    out[:, 3:6] = (R @ x[:, 3:6].T).T
    return out.astype(x.dtype)


def apply_bias_drift(x: np.ndarray, max_drift: float) -> np.ndarray:
    """Per-channel slowly-varying bias drift (random walk).

    Simulates IMU bias instability: the offset starts near zero at the
    beginning of the window and accumulates a random walk, reaching at most
    ``max_drift`` (in sensor units) by the end.  The same drift magnitude is
    applied to all channels but with independent random directions.

    Args:
        x         : [T, C]
        max_drift : controls step size — std of each step is max_drift/sqrt(T)
    """
    T, C = x.shape
    steps = np.random.normal(0.0, max_drift / max(np.sqrt(T), 1.0), size=(T, C))
    drift = np.cumsum(steps, axis=0)
    drift -= drift[0]   # anchor start at zero
    return (x + drift).astype(x.dtype)


def apply_sensor_dropout(x: np.ndarray, p: float) -> np.ndarray:
    """Zero out entire sensor channels independently with probability p.

    Simulates sensor failure or missing data.  Each channel is dropped
    independently; on average p*C channels are zeroed.

    Args:
        x : [T, C]
        p : per-channel drop probability (0 = never, 1 = always)
    """
    mask = (np.random.random(x.shape[1]) > p).astype(x.dtype)   # [C]
    return x * mask[np.newaxis, :]


def apply_time_crop(x: np.ndarray, min_crop_ratio: float) -> np.ndarray:
    """Randomly crop a contiguous sub-window, then resample to original length.

    Simulates temporal misalignment and mild motion speed variation.
    A crop of length in [min_crop_ratio·T, T] is taken from a random start
    position and resampled back to T samples via linear interpolation.

    Args:
        x              : [T, C]
        min_crop_ratio : minimum fraction of the window kept (e.g. 0.8 → 80%)
    """
    T, C = x.shape
    crop_len = int(np.random.uniform(min_crop_ratio, 1.0) * T)
    if crop_len >= T:
        return x
    start = np.random.randint(0, T - crop_len + 1)
    crop  = x[start : start + crop_len]   # [crop_len, C]
    t_src = np.linspace(0.0, 1.0, crop_len)
    t_dst = np.linspace(0.0, 1.0, T)
    out   = np.empty((T, C), dtype=x.dtype)
    for c in range(C):
        out[:, c] = np.interp(t_dst, t_src, crop[:, c])
    return out


def apply_frequency_mask(
    x: np.ndarray,
    max_mask_fraction: float,
    n_masks: int = 1,
) -> np.ndarray:
    """Zero out random frequency bands per channel (SpecAugment-style).

    For each channel, applies ``n_masks`` independent contiguous frequency masks.
    Each mask width is drawn uniformly in [0, max_mask_fraction * F] where F is
    the number of rfft bins.  The signal is reconstructed via irfft.

    Args:
        x                  : [T, C]
        max_mask_fraction  : maximum fraction of frequency bins to zero (e.g. 0.25)
        n_masks            : number of independent masks per channel (default 1)
    """
    T, C = x.shape
    out  = np.empty_like(x)
    for c in range(C):
        spec = np.fft.rfft(x[:, c])          # complex [F]
        F    = len(spec)
        for _ in range(n_masks):
            w = int(np.random.uniform(0.0, max_mask_fraction) * F)
            if w > 0:
                s = np.random.randint(0, max(F - w + 1, 1))
                spec[s : s + w] = 0.0
        out[:, c] = np.fft.irfft(spec, n=T).astype(x.dtype)
    return out


def apply_augment(x: np.ndarray, cfg: Any) -> np.ndarray:
    """Apply configured augmentations to a raw window [T, C].

    Augmentations are declared at the top level of the config dict.
    Each has an ``enabled`` flag; disabled augmentations are skipped entirely.
    Order matches the catalogue in the module docstring.
    """
    out = x

    # 1. SO(3) random rotation (acc ch0-2 + gyro ch3-5, same R)
    rr_cfg = cfg_get(cfg, ["random_rotation"], {}) or {}
    if rr_cfg.get("enabled", False):
        out = apply_random_rotation(out)

    # 2. Gaussian noise
    gn_cfg = cfg_get(cfg, ["gaussian_noise"], {}) or {}
    if gn_cfg.get("enabled", False):
        out = add_gaussian_noise(out, float(gn_cfg.get("sigma", 0.0)))

    # 3. Jitter (second independent Gaussian pass at smaller sigma)
    jt_cfg = cfg_get(cfg, ["jitter"], {}) or {}
    if jt_cfg.get("enabled", False):
        out = add_gaussian_noise(out, float(jt_cfg.get("sigma", 0.0)))

    # 4. Bias drift (per-channel random-walk offset over the window)
    bd_cfg = cfg_get(cfg, ["bias_drift"], {}) or {}
    if bd_cfg.get("enabled", False):
        out = apply_bias_drift(out, float(bd_cfg.get("max_drift", 0.05)))

    # 5. Sensor dropout (zero out whole channels with probability p)
    sd_cfg = cfg_get(cfg, ["sensor_dropout"], {}) or {}
    if sd_cfg.get("enabled", False):
        out = apply_sensor_dropout(out, float(sd_cfg.get("p", 0.1)))

    # 6. Time crop (random sub-window resample)
    tc_cfg = cfg_get(cfg, ["time_crop"], {}) or {}
    if tc_cfg.get("enabled", False):
        out = apply_time_crop(out, float(tc_cfg.get("min_crop_ratio", 0.8)))

    # 7. Frequency masking (SpecAugment-style per-channel FFT mask)
    fm_cfg = cfg_get(cfg, ["frequency_mask"], {}) or {}
    if fm_cfg.get("enabled", False):
        out = apply_frequency_mask(
            out,
            max_mask_fraction=float(fm_cfg.get("max_mask_fraction", 0.25)),
            n_masks=int(fm_cfg.get("n_masks", 1)),
        )

    # 8. Gain jitter (per-channel amplitude scale)
    gj_cfg = cfg_get(cfg, ["gain_jitter"], {}) or {}
    if gj_cfg.get("enabled", False):
        out = apply_gain_jitter(out, float(gj_cfg.get("min_scale", 1.0)),
                                     float(gj_cfg.get("max_scale", 1.0)))

    # 9. Axis bias (per-channel constant DC offset)
    ab_cfg = cfg_get(cfg, ["axis_bias"], {}) or {}
    if ab_cfg.get("enabled", False):
        out = apply_axis_bias(out, float(ab_cfg.get("max_bias", 0.0)))

    # 10. Time warp (smooth monotonic temporal distortion)
    tw_cfg = cfg_get(cfg, ["time_warp"], {}) or {}
    if tw_cfg.get("enabled", False):
        out = smooth_time_warp(out,
                               max_warp=float(tw_cfg.get("max_warp", 0.0)),
                               sigma=int(tw_cfg.get("smooth_sigma", 0)))

    return out

