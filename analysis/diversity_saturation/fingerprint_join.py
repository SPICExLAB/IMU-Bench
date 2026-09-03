"""fingerprint_join.py -- join a rebuilt dataset onto a cached window block by CONTENT.

The naive assumption -- that rebuilding a source reproduces the cached window
sequence in the same order -- is FALSE for realdata: a fresh build keeps exactly
the same 93,510 windows but emits them in a different order (5/200 index-probes
matched). Any label join done positionally would silently scramble the labels.

So join on the windows themselves. Each window is fingerprinted by hashing its
raw float32 bytes; the cache is fingerprinted the same way; the resulting map is
a proven permutation or the join is refused.

A fingerprint collision would be a false join, so `build_index` asserts the
cache's fingerprints are unique before anything is matched.
"""
from __future__ import annotations

import hashlib

import numpy as np


def fingerprint(x: np.ndarray) -> bytes:
    """16-byte digest of a window's exact float32 bytes."""
    return hashlib.blake2b(np.ascontiguousarray(x, dtype=np.float32).tobytes(),
                           digest_size=16).digest()


def build_index(cache, channels=None, chunk=20_000, verbose=True):
    """fingerprint -> row index, over every row of a cached [N, C, T] block.

    `channels` restricts hashing to a channel slice (capture24's cache rows are
    6ch with 3 zero-filled gyro channels, while its live dataset yields 3ch).
    """
    n = len(cache)
    idx, dup = {}, 0
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        block = np.asarray(cache[s:e], dtype=np.float32)
        if channels is not None:
            block = block[:, channels]
        for i in range(len(block)):
            fp = fingerprint(block[i])
            if fp in idx:
                dup += 1
            else:
                idx[fp] = s + i
        if verbose and (s // chunk) % 5 == 0:
            print(f"      [fp] indexed {e}/{n}", flush=True)
    assert dup == 0, (
        f"{dup} duplicate window fingerprints in the cache -- content join is "
        "ambiguous, refusing to proceed")
    return idx


def join(fps, cache_index, name="join"):
    """Map a list of rebuilt-window fingerprints to cache rows.

    Returns (positions, matched_mask). Raises if the match is not a clean
    bijection, since a partial join would mislabel whatever it missed.
    """
    pos = np.full(len(fps), -1, dtype=np.int64)
    for j, fp in enumerate(fps):
        p = cache_index.get(fp)
        if p is not None:
            pos[j] = p
    matched = pos >= 0
    rate = float(matched.mean())
    print(f"      [{name}] {matched.sum()}/{len(fps)} windows matched by content "
          f"({100 * rate:.2f}%)", flush=True)
    assert rate > 0.999, (
        f"only {100 * rate:.2f}% of rebuilt windows found in the cache -- the "
        "rebuild is not reproducing cached content, refusing to join")
    assert len(np.unique(pos[matched])) == int(matched.sum()), \
        "two rebuilt windows mapped to the same cache row -- refusing to join"
    return pos, matched
