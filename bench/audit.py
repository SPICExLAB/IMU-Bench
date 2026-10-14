#!/usr/bin/env python3
"""Audit a dataset release for the same person under several participant ids, before splitting it.

    imu-bench audit <root> --subject-regex 'id_(\\d+)' [--glob '**/*.csv']
    imu-bench audit <root> --subject-regex 'id_(\\d+)' --acc-cols AccX AccY AccZ     # + mirrored / rotated copies

1. Exact copies: every file is hashed; files with identical bytes under different participant ids link those ids.
2. Near copies (with --acc-cols): the accelerometer magnitude is unchanged by mirroring or rotating the sensor
   axes, so two recordings whose standardised magnitude correlates above --threshold (default 0.999) at some
   small lag are the same motion. Candidates are found by a zero-lag correlation of length-normalised
   signatures within files of (nearly) equal length, then confirmed with a max-lag correlation.

Linked ids are merged into identity groups (connected components). A participant-disjoint split must be made over
these groups, not over the release's ids. This is the procedure behind the OpenWatch audit in the paper
(50 ids -> 43 identities; 35 of 35 augmented ids are mirrored copies).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _subject(rel: str, rx: re.Pattern) -> Optional[str]:
    m = rx.search(rel)
    return m.group(1) if m else None


def _components(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> List[List[str]]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = collections.defaultdict(list)
    for n in nodes:
        groups[find(n)].append(n)
    return sorted((sorted(g, key=_natural) for g in groups.values() if len(g) > 1), key=lambda g: _natural(g[0]))


def _natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _read_acc(path: str, cols: Sequence[str]) -> Optional[np.ndarray]:
    try:
        if path.endswith(".npy"):
            x = np.load(path)
            return np.asarray(x[:, [int(c) for c in cols]], dtype=np.float64)
        import pandas as pd
        return pd.read_csv(path, usecols=list(cols))[list(cols)].to_numpy(dtype=np.float64)
    except (KeyError, ValueError, IndexError, OSError):
        return None


def _signature(acc: np.ndarray) -> Optional[np.ndarray]:
    m = np.linalg.norm(acc, axis=1)
    m = m - m.mean()
    s = m.std()
    return m / s if s > 1e-9 else None


def _maxlag_corr(a: np.ndarray, b: np.ndarray, maxlag: int) -> float:
    """max over |lag| <= maxlag of mean(a[t] * b[t - lag]) on the overlap (both standardised)."""
    from scipy.signal import correlate
    full = correlate(a, b, mode="full", method="fft")                 # lag = k - (len(b) - 1)
    lags = np.arange(-(len(b) - 1), len(a))
    n = np.minimum(len(a), len(b) + lags) - np.maximum(0, lags)
    ok = (np.abs(lags) <= maxlag) & (n >= 50)
    return float((full[ok] / n[ok]).max()) if ok.any() else -2.0


def audit(root: str, subject_regex: str, pattern: str = "**/*.csv", acc_cols: Optional[Sequence[str]] = None,
          threshold: float = 0.999, maxlag: int = 60, length_tol: float = 0.02, sig_len: int = 256) -> Dict:
    import glob as _glob
    rx = re.compile(subject_regex)
    files = []
    for p in sorted(_glob.glob(os.path.join(root, pattern), recursive=True)):
        rel = os.path.relpath(p, root)
        sid = _subject(rel, rx)
        if sid is not None and os.path.isfile(p):
            files.append((rel, sid))
    subjects = sorted({s for _, s in files}, key=_natural)

    by_hash: Dict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    for rel, sid in files:
        with open(os.path.join(root, rel), "rb") as fh:
            by_hash[hashlib.sha256(fh.read()).hexdigest()].append((rel, sid))
    dup_groups = [v for v in by_hash.values() if len(v) > 1]
    exact_edges = sorted({(a, b) for v in dup_groups for a in {s for _, s in v} for b in {s for _, s in v} if a < b})
    n_unique = len(by_hash)

    near_edges, near_pairs = [], []
    if acc_cols:
        sigs = []
        _t0 = time.time()
        for _k, (h, v) in enumerate(by_hash.items()):     # one representative per exact-duplicate group
            if _k and _k % 1000 == 0:
                print(f"[audit] read {_k}/{len(by_hash)} files ({time.time() - _t0:.0f} s)", file=sys.stderr, flush=True)
            rel, sid = v[0]
            acc = _read_acc(os.path.join(root, rel), acc_cols)
            sg = _signature(acc) if acc is not None and len(acc) >= 50 else None
            if sg is not None:
                sigs.append((rel, sid, sg))
        sigs.sort(key=lambda t: len(t[2]))
        lens = np.array([len(t[2]) for t in sigs])
        grid = np.linspace(0, 1, sig_len)
        Z = np.stack([np.interp(grid, np.linspace(0, 1, len(t[2])), t[2]) for t in sigs]) if sigs else np.zeros((0, sig_len))
        Z = (Z - Z.mean(1, keepdims=True)) / (Z.std(1, keepdims=True) + 1e-9)
        seen = set()
        print(f"[audit] comparing {len(sigs)} signatures", file=sys.stderr, flush=True)
        for i in range(len(sigs)):
            hi = np.searchsorted(lens, lens[i] * (1 + length_tol), side="right")
            if hi <= i + 1:
                continue
            c = Z[i + 1:hi] @ Z[i] / sig_len
            for k in np.nonzero(c > threshold - 0.02)[0]:
                j = i + 1 + int(k)
                a, b = sigs[i], sigs[j]
                if a[1] == b[1]:
                    continue
                r = _maxlag_corr(a[2], b[2], maxlag)
                if r > threshold:
                    near_pairs.append(dict(a=a[0], b=b[0], corr=round(r, 5)))
                    key = tuple(sorted((a[1], b[1]), key=_natural))
                    if key not in seen:
                        seen.add(key)
                        near_edges.append(key)

    exact_groups = _components(subjects, exact_edges)
    all_groups = _components(subjects, list(exact_edges) + near_edges)
    n_ident = len(subjects) - sum(len(g) - 1 for g in all_groups)
    return dict(
        root=os.path.abspath(root), files=len(files), unique_files=n_unique, participant_ids=len(subjects),
        exact_duplicate_files=sum(len(v) for v in dup_groups), exact_duplicate_groups=len(dup_groups),
        exact_identity_groups=exact_groups, identity_groups=all_groups, distinct_identities=n_ident,
        near_duplicate_pairs=near_pairs[:500], near_duplicate_pairs_total=len(near_pairs),
        duplicate_file_groups=[[rel for rel, _ in v] for v in dup_groups][:500],
    )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="imu-bench audit", description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--subject-regex", required=True, help="regex on the file's path relative to root; group 1 = participant id")
    ap.add_argument("--glob", default="**/*.csv")
    ap.add_argument("--acc-cols", nargs=3, default=None, help="accelerometer columns (CSV names or .npy indices)")
    ap.add_argument("--threshold", type=float, default=0.999)
    ap.add_argument("--json", default=None, help="write the full report here")
    a = ap.parse_args(argv)
    rep = audit(a.root, a.subject_regex, a.glob, a.acc_cols, a.threshold)
    print(f"[audit] {rep['files']} files ({rep['unique_files']} distinct contents) under {rep['participant_ids']} participant ids")
    print(f"[audit] exact duplicates: {rep['exact_duplicate_files']} files in {rep['exact_duplicate_groups']} groups; "
          f"ids sharing files: {rep['exact_identity_groups'] or 'none'}")
    if a.acc_cols:
        print(f"[audit] near copies (|acc| corr > {a.threshold}): {rep['near_duplicate_pairs_total']} file pairs across ids")
    print(f"[audit] distinct identities: {rep['distinct_identities']} of {rep['participant_ids']} ids; "
          f"merge before splitting: {rep['identity_groups'] or 'nothing'}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"[audit] wrote {a.json}")


if __name__ == "__main__":
    main()
