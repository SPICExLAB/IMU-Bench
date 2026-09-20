"""Diversity statistics for a window subset -- the quantitative answer to
"what exactly does *coverage* mean here?" (EXPERIMENTS.md 11m).

Three complaints about the coverage number used in 11j/11l are addressed:

1. **"Covers 99 % of motion types" is a raw cluster count, so a subset that
   touches every mode once scores the same as the pool itself.** Fixed by
   reporting FREQUENCY-WEIGHTED effective type counts (exp of Shannon entropy,
   and inverse Simpson) alongside the raw count. A subset that hits 999/1000
   types but puts 90 % of its mass in one of them has an effective type count
   near 1, and the table shows that.

2. **"Motion type" was defined by our own encoder, so the diversity axis is
   circular.** Fixed upstream (02/03): the subsets are SELECTED with an
   encoder-free, rotation-invariant physical feature clustering (T-phys); the
   encoder clustering (T-enc) is only ever a read-out. Every statistic here
   takes the label array as an argument and is computed under both.

3. **"Diversity" conflates motion types with sources, sessions and subjects.**
   Fixed by reporting the source composition entropy separately, and by two
   cluster-free geometric statistics (participation-ratio effective rank and
   the Vendi score) that do not depend on any partition at all.

Long-tail weighting: `pool_mass_covered` weights each covered type by its
frequency IN THE POOL, so covering the tail is worth what the tail is actually
worth; `effective_types_shannon` weights by frequency in the SUBSET, so a
subset that only piles onto the head is penalised. Reporting both is the point
-- they answer different questions and 11j reported neither.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _probs(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    if tot <= 0:
        return counts
    p = counts / tot
    return p[p > 0]


def effective_count_shannon(counts: np.ndarray) -> float:
    """exp(H) -- the number of equally-frequent categories with the same entropy
    (Hill number of order 1). 1000 types used uniformly -> 1000; 1000 types with
    all the mass in one -> ~1."""
    p = _probs(counts)
    if p.size == 0:
        return 0.0
    return float(np.exp(-(p * np.log(p)).sum()))


def effective_count_simpson(counts: np.ndarray) -> float:
    """1 / sum p^2 (Hill number of order 2) -- more sensitive to the head than
    exp(H), so a gap between the two is itself evidence of a long tail."""
    p = _probs(counts)
    if p.size == 0:
        return 0.0
    return float(1.0 / np.square(p).sum())


def type_stats(subset_labels: np.ndarray, pool_counts: np.ndarray, k: int) -> Dict[str, float]:
    sub_counts = np.bincount(np.asarray(subset_labels, dtype=np.int64), minlength=k)
    present = sub_counts > 0
    pool_p = _probs_full(pool_counts)
    return {
        "types_present": int(present.sum()),
        "types_present_frac": float(present.sum()) / float(k),
        # long-tail-weighted coverage: how much of the POOL's probability mass
        # sits in types the subset touches at all
        "pool_mass_covered": float(pool_p[present].sum()),
        # frequency-weighted diversity OF THE SUBSET
        "effective_types_shannon": effective_count_shannon(sub_counts),
        "effective_types_simpson": effective_count_simpson(sub_counts),
        # how far the subset's type distribution is from the pool's
        "kl_subset_pool": _kl(sub_counts, pool_counts),
        "max_type_share": float(sub_counts.max() / max(1, sub_counts.sum())),
    }


def _probs_full(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    tot = counts.sum()
    return counts / tot if tot > 0 else counts


def _kl(sub_counts: np.ndarray, pool_counts: np.ndarray) -> float:
    """KL(subset || pool), nats. Pool is smoothed by 1 count so an unseen type
    cannot make this infinite."""
    q = _probs_full(np.asarray(pool_counts, dtype=np.float64) + 1.0)
    p = _probs_full(sub_counts)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def source_stats(idx: np.ndarray, ranges: Dict[str, tuple]) -> Dict[str, object]:
    counts, names = [], []
    for name, (lo, hi) in ranges.items():
        names.append(name)
        counts.append(int(((idx >= lo) & (idx < hi)).sum()))
    counts = np.asarray(counts)
    return {
        "source_counts": {n: int(c) for n, c in zip(names, counts)},
        "n_sources": int((counts > 0).sum()),
        "effective_sources": effective_count_shannon(counts),
    }


def participation_ratio(z: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2 of the feature covariance -- a cluster-free
    'effective number of dimensions' the data actually occupies."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(0, keepdims=True)
    lam = np.linalg.svd(z, compute_uv=False) ** 2
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    return float(lam.sum() ** 2 / np.square(lam).sum())


def vendi_score(z: np.ndarray, n_max: int = 4000, seed: int = 42) -> float:
    """exp of the von Neumann entropy of the normalised Gram matrix (Friedman &
    Dieng 2023) -- an 'effective number of distinct samples' that needs no
    clustering and no choice of K, so it cannot be gamed by the K we picked.
    Computed on a random sub-sample because it is O(n^3)."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) > n_max:
        z = z[np.random.default_rng(seed).choice(len(z), n_max, replace=False)]
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    # The n x n Gram matrix and the d x d covariance share their non-zero
    # spectrum, so for d << n take the singular values of z directly -- exactly
    # the same eigenvalues, O(n d^2) instead of O(n^3).
    if z.shape[1] < len(z):
        lam = (np.linalg.svd(z, compute_uv=False) ** 2) / len(z)
    else:
        lam = np.linalg.eigvalsh((z @ z.T) / len(z))
    lam = lam[lam > 1e-12]
    if lam.size == 0:
        return 0.0
    return float(np.exp(-(lam * np.log(lam)).sum()))


def coverage_of_pool(z_subset: np.ndarray, z_query: np.ndarray, seed: int = 42) -> Dict[str, float]:
    """How well the subset covers the pool: distance from a random pool sample
    to its nearest neighbour IN THE SUBSET (the 'fill distance'). This is the
    statistic that decides whether the unsampled 99 % is genuinely new data or
    repeats of what the subset already holds -- 11j answered that with a single
    median, this returns the quantiles so the tail is visible."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(np.asarray(z_subset, dtype=np.float32))
    d, _ = nn.kneighbors(np.asarray(z_query, dtype=np.float32))
    d = d[:, 0]
    return {
        "fill_median": float(np.median(d)),
        "fill_p90": float(np.percentile(d, 90)),
        "fill_p99": float(np.percentile(d, 99)),
        "fill_mean": float(d.mean()),
    }


def summarise(idx: np.ndarray,
              labels_phys: np.ndarray,
              labels_enc: np.ndarray,
              pool_counts_phys: np.ndarray,
              pool_counts_enc: np.ndarray,
              ranges: Dict[str, tuple],
              k: int,
              z_enc: Optional[np.ndarray] = None,
              z_query_idx: Optional[np.ndarray] = None,
              geom_sample: int = 4000,
              seed: int = 42) -> Dict[str, object]:
    """Everything above, for one arm."""
    idx = np.asarray(idx, dtype=np.int64)
    rng = np.random.default_rng(seed)
    out: Dict[str, object] = {"n_windows": int(len(idx)),
                              "hours": float(len(idx) * 5.12 / 3600.0)}
    out["phys"] = type_stats(labels_phys[idx], pool_counts_phys, k)
    out["enc"] = type_stats(labels_enc[idx], pool_counts_enc, k)
    out.update(source_stats(idx, ranges))
    if z_enc is not None:
        sub = idx if len(idx) <= geom_sample else rng.choice(idx, geom_sample, replace=False)
        zs = np.asarray(z_enc[np.sort(sub)], dtype=np.float32)
        out["participation_ratio_enc"] = participation_ratio(zs)
        out["vendi_enc"] = vendi_score(zs, n_max=geom_sample, seed=seed)
        if z_query_idx is not None:
            zq = np.asarray(z_enc[np.sort(z_query_idx)], dtype=np.float32)
            out["pool_coverage"] = coverage_of_pool(zs, zq, seed=seed)
    return out
