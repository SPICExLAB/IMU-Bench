"""Statistical feature tokens for the Mix-Token Transformer branch,
following Bonazzi et al. (arXiv:2605.04791) Table 5 (Appendix B).

Three groups, computed on the raw 1s/100-sample window at 100Hz:
- time-domain (per channel):    21 descriptors
- frequency-domain (per channel): 12 descriptors (top-k=3 dominant freqs)
- cross-channel (per pair / global): Pearson, Spearman, cosine, MI (per
  pair), PCA explained-variance ratios (global)

Token layout (the paper says features are "organized into a token
sequence ... positional encodings added to preserve ordering"; the exact
tokenization is not specified, so we use the natural one): one token per
(channel, group) for the per-channel groups + one token per cross-channel
descriptor type -> 6*2 + 5 = 17 tokens, each zero-padded to a common width
D_FEAT so a single linear encoder can embed them.

Everything here is pure numpy so it can be precomputed once per window and
cached — the features are deterministic functions of the raw signal.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew, spearmanr

NATIVE_HZ = 100.0
TOP_K = 3
# per-channel band edges (Hz) for low/mid/high band energy — the paper's
# exact edges are unpublished; these split the 0-50Hz Nyquist range at
# motion-relevant boundaries (gross posture / gesture dynamics / tremor+impact)
BANDS = [(0.0, 3.0), (3.0, 10.0), (10.0, 50.0)]


def _safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def time_domain(x: np.ndarray) -> np.ndarray:
    """x: [W] -> 21 features."""
    x = x.astype(np.float64)
    W = len(x)
    mean = x.mean()
    var = x.var()
    std = np.sqrt(var)
    mn, mx = x.min(), x.max()
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    rms = np.sqrt(np.mean(x**2))
    mad = np.median(np.abs(x - med))
    energy = np.sum(x**2)
    xc = x - mean
    zcr = np.mean(np.abs(np.diff(np.sign(xc))) > 0) if W > 1 else 0.0

    def _ac(lag):
        if W <= lag or var < 1e-12:
            return 0.0
        return float(np.dot(xc[:-lag], xc[lag:]) / (var * (W - lag)))

    ac1, ac2 = _ac(1), _ac(2)
    sk = float(skew(x)) if std > 1e-12 else 0.0
    ku = float(kurtosis(x)) if std > 1e-12 else 0.0
    # Hjorth
    d1 = np.diff(x)
    d2 = np.diff(d1)
    act = var
    mob = np.sqrt(_safe_div(d1.var(), var)) if W > 1 else 0.0
    mob_d = np.sqrt(_safe_div(d2.var(), d1.var())) if W > 2 else 0.0
    comp = _safe_div(mob_d, mob) if mob > 1e-12 else 0.0
    return np.array([mean, var, std, mn, mx, mx - mn, med, q25, q75, q75 - q25, rms, mad,
                     energy, zcr, ac1, ac2, sk, ku, act, mob, comp], dtype=np.float32)


def frequency_domain(x: np.ndarray, fs: float = NATIVE_HZ, prev_spec: np.ndarray | None = None):
    """x: [W] -> (12 features, power spectrum) ; spectral flux uses prev_spec
    (previous window's spectrum) when available, else 0."""
    x = x.astype(np.float64) - x.mean()
    W = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(W))) ** 2
    freqs = np.fft.rfftfreq(W, d=1.0 / fs)
    total = spec.sum() + 1e-12
    p = spec / total
    centroid = float(np.sum(freqs * p))
    entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))
    flux = float(np.sqrt(np.sum((np.sqrt(spec) - np.sqrt(prev_spec)) ** 2))) if prev_spec is not None else 0.0
    peak = float(freqs[int(np.argmax(spec))])
    bands = [float(spec[(freqs >= lo) & (freqs < hi)].sum() / total) for lo, hi in BANDS]
    cum = np.cumsum(p)
    rolloff = float(freqs[int(np.searchsorted(cum, 0.85))]) if len(cum) else 0.0
    gm = float(np.exp(np.mean(np.log(spec + 1e-12))))
    am = float(np.mean(spec) + 1e-12)
    flatness = gm / am
    top = np.argsort(spec)[::-1][:TOP_K]
    topf = [float(freqs[i]) for i in top] + [0.0] * (TOP_K - len(top))
    feats = np.array([centroid, entropy, flux, peak, *bands, rolloff, flatness, *topf], dtype=np.float32)
    return feats, spec


def cross_channel(X: np.ndarray):
    """X: [W, C] -> dict of 5 cross-channel descriptor vectors."""
    W, C = X.shape
    iu = np.triu_indices(C, k=1)
    Xc = X - X.mean(0, keepdims=True)
    sd = Xc.std(0) + 1e-8
    Z = Xc / sd
    pear = (Z.T @ Z / W)[iu]
    try:
        sp = spearmanr(X, axis=0).correlation
        sp = np.atleast_2d(sp)[iu] if C > 2 else np.array([float(sp)])
    except Exception:
        sp = np.zeros_like(pear)
    nrm = np.linalg.norm(X, axis=0) + 1e-8
    cos = ((X.T @ X) / np.outer(nrm, nrm))[iu]
    # mutual information via 2D histogram (16 bins), per pair
    mi = []
    for i, j in zip(*iu):
        h, _, _ = np.histogram2d(X[:, i], X[:, j], bins=16)
        pxy = h / (h.sum() + 1e-12)
        px = pxy.sum(1, keepdims=True)
        py = pxy.sum(0, keepdims=True)
        nz = pxy > 0
        mi.append(float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz] + 1e-12))))
    mi = np.asarray(mi)
    # PCA explained variance ratios
    cov = Xc.T @ Xc / max(W - 1, 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    evr = ev / (ev.sum() + 1e-12)
    return {
        "pearson": np.nan_to_num(pear).astype(np.float32),
        "spearman": np.nan_to_num(sp).astype(np.float32),
        "cosine": np.nan_to_num(cos).astype(np.float32),
        "mi": np.nan_to_num(mi).astype(np.float32),
        "pca_evr": np.nan_to_num(evr).astype(np.float32),
    }


N_TIME = 21
N_FREQ = 12


def window_tokens(X: np.ndarray, prev_X: np.ndarray | None = None) -> np.ndarray:
    """X: [W, C] raw window -> token matrix [n_tokens, D_FEAT].

    Tokens: C time-domain + C frequency-domain + 5 cross-channel.
    D_FEAT = max token width; shorter tokens are zero-padded on the right.
    """
    W, C = X.shape
    toks = []
    for c in range(C):
        toks.append(time_domain(X[:, c]))
    for c in range(C):
        prev_spec = None
        if prev_X is not None:
            _, prev_spec = frequency_domain(prev_X[:, c])
        f, _ = frequency_domain(X[:, c], prev_spec=prev_spec)
        toks.append(f)
    cc = cross_channel(X)
    for k in ["pearson", "spearman", "cosine", "mi", "pca_evr"]:
        toks.append(cc[k])
    D = max(len(t) for t in toks)
    out = np.zeros((len(toks), D), dtype=np.float32)
    for i, t in enumerate(toks):
        out[i, : len(t)] = t
    return out


def n_tokens(C: int = 6) -> int:
    return 2 * C + 5


def d_feat(C: int = 6) -> int:
    n_pairs = C * (C - 1) // 2
    return max(N_TIME, N_FREQ, n_pairs, C)
