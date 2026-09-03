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

Separately, `frame_perturb` (below) re-expresses a window in a different
sensor-mounting frame. It is NOT an augmentation: it is applied to every
split and is deterministic, and it lives in preprocess_window(), not here.
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
        # BUG FIX (2026-08-15): np.convolve(mode="same") returns
        # max(len(noise), len(kernel)) samples, so any window shorter than the
        # kernel (2*3*sigma+1; e.g. a 64-sample HumanML3D clip with sigma=20)
        # crashed on `t_orig + smooth`. Reflect-pad by the kernel radius and use
        # mode="valid" -> always exactly T samples, identical result when T is
        # long enough.
        padded = np.pad(noise, radius, mode="reflect") if T > 1 else noise
        smooth = np.convolve(padded, kernel, mode="valid")
        if smooth.shape[0] != T:                       # T <= radius edge case
            smooth = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, smooth.shape[0]), smooth)
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


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Unit-quaternion -> 3x3 rotation matrix (q is normalised here)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _random_rotation_matrix() -> np.ndarray:
    """Sample a uniformly random rotation from SO(3) via unit quaternion."""
    return _quat_to_matrix(np.random.randn(4))


def _apply_rotation(x: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply one 3x3 rotation to acc (ch 0-2) and, when present, gyro (ch 3-5)."""
    out = x.copy()
    out[:, :3] = (R @ x[:, :3].T).T
    if x.shape[1] >= 6:
        out[:, 3:6] = (R @ x[:, 3:6].T).T
    return out.astype(x.dtype)


def _build_octahedral_rotations() -> np.ndarray:
    """The 24 proper rotations of the cube (octahedral rotation group).

    Every element is a signed permutation matrix (exactly one nonzero ±1 per
    row and column — i.e. axis permutation + sign flips) with determinant
    +1, so each one is a genuine 3D rotation restricted to multiples of 90°
    about the coordinate axes. Of the 48 signed permutation matrices, exactly
    half (24) have determinant +1 (the other 24 are improper — reflections).
    Computed once at import time.
    """
    from itertools import permutations, product

    mats = []
    for perm in permutations(range(3)):
        P = np.eye(3)[list(perm)]
        for signs in product((1.0, -1.0), repeat=3):
            M = P * np.array(signs)[:, None]
            if np.isclose(np.linalg.det(M), 1.0):
                mats.append(M)
    assert len(mats) == 24, f"expected 24 proper rotations, got {len(mats)}"
    return np.stack(mats).astype(np.float64)   # [24, 3, 3]


_OCTAHEDRAL_ROTATIONS = _build_octahedral_rotations()


def apply_random_rotation(x: np.ndarray, rotation_set: str = "so3") -> np.ndarray:
    """Apply a single random rotation to acc (ch 0-2) and, if present, gyro (ch 3-5).

    Args:
        x            : [T, C] with C >= 3. Gyro (ch 3-5) is only rotated when
                       C >= 6 — acc-only sources (e.g. capture24, 3 channels)
                       are rotated on the 3 acc channels alone.
        rotation_set : "so3" (default) — uniformly random continuous SO(3)
                       rotation via unit quaternion.
                       "octahedral" — uniformly random pick from the 24
                       proper rotations of the cube (rotations that are
                       multiples of 90° about the coordinate axes: axis
                       permutations + sign flips with determinant +1).
                       "gravity_axis" — random rotation about the window's OWN
                       estimated gravity direction (mean acc); preserves the gravity
                       vector exactly in any sensor frame. Use this, not "yaw",
                       for sensor-frame data.
                       "yaw" / "gravity_preserving" — random rotation about the
                       vertical (Y, gravity) axis only; preserves the gravity
                       direction, so the DC/posture cue survives augmentation.

    The same rotation R is applied to both triaxial sensors so the relative
    geometry between acc and gyro is preserved.
    """
    if rotation_set == "octahedral":
        R = _OCTAHEDRAL_ROTATIONS[np.random.randint(len(_OCTAHEDRAL_ROTATIONS))]
    elif rotation_set in ("gravity_axis", "about_gravity"):
        # Rotate about THIS WINDOW's own gravity direction, estimated as the mean acc vector.
        # In a wrist SENSOR frame there is no fixed "up" axis -- gravity lands on a different axis
        # in every source (SONIC ch0, OCDetect ch1, Capture24 ch2, NHANES uniformly spread), and it
        # moves per window, which is exactly the posture signal. A fixed-axis yaw therefore does NOT
        # preserve gravity for most data; rotating about g-hat does, exactly, for every source.
        g = x[:, :3].mean(axis=0).astype(np.float64)
        n = np.linalg.norm(g)
        if n < 1e-6:
            R = np.eye(3)                      # gravity-free data (e.g. PADS): nothing to preserve
        else:
            k = g / n
            th = np.random.uniform(0.0, 2.0 * np.pi)
            K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
            R = np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)   # Rodrigues
    elif rotation_set in ("yaw", "gravity_preserving"):
        # Rotation about the vertical (gravity) axis only: keeps invariance to how the
        # watch sits around the wrist and to heading, but PRESERVES which way is down.
        # Required whenever the pipeline keeps the DC term (preprocess.normalize.method:
        # physical) -- octahedral rotation would randomise the gravity direction and train
        # the encoder to be posture-agnostic, which is the cue HAR needs most.
        th = np.random.uniform(0.0, 2.0 * np.pi)
        c, sn = np.cos(th), np.sin(th)
        R = np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]], dtype=np.float64)
    else:
        R = _random_rotation_matrix()
    return _apply_rotation(x, R)


# ---------------------------------------------------------------------------
# Coordinate-FRAME perturbation (NOT an augmentation)
# ---------------------------------------------------------------------------
# `random_rotation` above is a train-only augmentation: a fresh draw every
# __getitem__, applied only to the train split, meant to make the encoder
# orientation-invariant. `frame_perturb` is the opposite tool -- it models a
# sensor that is *mounted* in a different coordinate frame than the one the
# encoder was pretrained in, so it must be
#   (a) applied to EVERY split (train, val, test) alike -- the deployment frame
#       does not change between splits, and
#   (b) deterministic in `fixed` mode -- one rotation R per (seed, rotation_set)
#       for the entire dataset, so a probe measures frame MISMATCH rather than
#       augmentation noise.
# It is applied to the RAW window inside preprocess_window(), before filtering
# and normalisation, because that is where a real mounting rotation acts (a
# per-channel z-score does NOT commute with a rotation).

_FRAME_ROTATIONS: dict = {}


def frame_rotation_matrix(seed: int, rotation_set: str = "so3") -> np.ndarray:
    """The one fixed rotation this (seed, rotation_set) stands for. Cached."""
    key = (int(seed), str(rotation_set))
    R = _FRAME_ROTATIONS.get(key)
    if R is None:
        rng = np.random.default_rng(int(seed))
        if str(rotation_set) == "octahedral":
            R = _OCTAHEDRAL_ROTATIONS[int(rng.integers(len(_OCTAHEDRAL_ROTATIONS)))]
        else:
            R = _quat_to_matrix(rng.standard_normal(4))
        _FRAME_ROTATIONS[key] = R
    return R


def apply_frame_perturb(x: np.ndarray, cfg: Any) -> np.ndarray:
    """Re-express a raw window [T, C] in a different sensor-mounting frame.

    Config (top level, all splits)::

        frame_perturb:
          enabled: true
          mode: fixed          # "fixed" (default) | "per_window"
          rotation_set: so3    # "so3" (default) | "octahedral"
          seed: 7              # `fixed` mode only -- picks WHICH rotation

    fixed       : one R for the whole dataset -- "the watch is worn in an
                  unknown but consistent orientation". This is the condition a
                  frame-match ablation wants.
    per_window  : a fresh uniform draw per window in every split -- the harsher
                  "orientation is uncontrolled sample to sample" stress test.
    """
    fp = cfg_get(cfg, ["frame_perturb"], {}) or {}
    if not fp.get("enabled", False):
        return x
    rot_set = str(fp.get("rotation_set", "so3"))
    if str(fp.get("mode", "fixed")) == "per_window":
        return apply_random_rotation(x, rotation_set=rot_set)
    return _apply_rotation(x, frame_rotation_matrix(int(fp.get("seed", 0)), rot_set))


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

    # 1. Random rotation (acc ch0-2 + gyro ch3-5, same R). rotation_set:
    #    "so3" (default, continuous) | "octahedral" (24 proper cube rotations)
    rr_cfg = cfg_get(cfg, ["random_rotation"], {}) or {}
    if rr_cfg.get("enabled", False):
        out = apply_random_rotation(out, rotation_set=str(rr_cfg.get("rotation_set", "so3")))

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

