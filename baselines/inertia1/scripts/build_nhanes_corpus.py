"""Build the Inertia-1 pretraining corpus from raw NHANES PAX archives.

Reproduces the preprocessing described in Inertia-1 (Xu et al. 2026, App. A):
    raw 80 Hz triaxial wrist accelerometry (g units)
      -> 4th-order Butterworth low-pass, 10 Hz cutoff (zero-phase)
      -> downsample to 20 Hz
      -> fixed 30 s windows (600 samples), triaxial, NO normalisation
The paper stores 10-minute parquet segments and cuts 30 s windows from them;
we cut the windows directly (non-overlapping, per hourly file) and store them
as fp16 [M, 3, 600] arrays, one .npy per participant.

Sub-sampling: Inertia-1 pretrained ~170k steps x 1024 windows = 174M window
samples over ~2.8M NHANES hours (~336M windows), i.e. ~0.5 epoch. We keep
EVERY participant but a deterministic hashed 1/3 of the hourly files (all
hours of day represented), ~110M windows -> ~1.6 epochs at the same budget.

Usage (SLURM array): --shard-idx $SLURM_ARRAY_TASK_ID --num-shards N
Per-participant outputs are idempotent (skipped if present) so a killed task
can simply be re-run.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import io
import json
import re
import tarfile
import time
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
from scipy.signal import butter, sosfiltfilt

NATIVE_HZ = 80
TARGET_HZ = 20
WINDOW_SEC = 30
WIN = WINDOW_SEC * TARGET_HZ  # 600
DECIM = NATIVE_HZ // TARGET_HZ  # 4
_TS_RE = re.compile(r"\.(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-\d{3}-P\d+\.sensor\.csv$")
_SOS = butter(4, 10.0, btype="low", fs=NATIVE_HZ, output="sos")
_CONVERT = pacsv.ConvertOptions(
    include_columns=["X", "Y", "Z"],
    column_types={"X": pa.float32(), "Y": pa.float32(), "Z": pa.float32()},
)


def list_archives(root: str, cycles=("pax_g", "pax_h")) -> List[str]:
    out = []
    for c in cycles:
        d = os.path.join(root, c)
        out.extend(sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tar.bz2")))
    return out


def seqn_of(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def _hour_key(name: str) -> Optional[str]:
    m = _TS_RE.search(name)
    if not m:
        return None
    y, mo, d, h, _mi, _s = m.groups()
    return f"{y}-{mo}-{d}-{h}"


def keep_hour(seqn: str, hour_key: str, keep_frac: float) -> bool:
    """Deterministic pseudo-random per-(participant,hour) selection."""
    if keep_frac >= 1.0:
        return True
    hsh = hashlib.md5(f"{seqn}|{hour_key}".encode()).digest()
    u = int.from_bytes(hsh[:4], "little") / 2**32
    return u < keep_frac


def process_member_bytes(b: bytes) -> Optional[np.ndarray]:
    """One hourly CSV -> fp16 windows [M, 3, 600] at 20 Hz (or None)."""
    try:
        tab = pacsv.read_csv(io.BytesIO(b), convert_options=_CONVERT)
    except Exception:
        return None
    if tab.num_rows < WIN * DECIM:
        return None
    x = np.stack([tab.column(c).to_numpy(zero_copy_only=False) for c in ("X", "Y", "Z")], axis=1)
    x = x.astype(np.float64, copy=False)
    if not np.isfinite(x).all():
        good = np.isfinite(x).all(axis=1)
        x = x[good]
        if len(x) < WIN * DECIM:
            return None
    y = sosfiltfilt(_SOS, x, axis=0)[::DECIM]  # [n20, 3]
    n = (len(y) // WIN) * WIN
    if n == 0:
        return None
    w = y[:n].reshape(-1, WIN, 3).transpose(0, 2, 1)  # [M, 3, 600]
    # No non-wear filtering (Inertia-1 describes none): keep sleep/stillness
    # (real 24 h wear data). Only drop windows whose RAW signal is exactly
    # constant on all axes (dead sensor / idle-sleep fill values).
    raw_w = x[: n * DECIM].reshape(-1, WIN * DECIM, 3)
    rv = raw_w.var(axis=1).max(axis=1)
    w = w[rv > 0]
    if len(w) == 0:
        return None
    return w.astype(np.float16)


def process_participant(path: str, keep_frac: float) -> tuple[np.ndarray, int, int]:
    seqn = seqn_of(path)
    chunks, n_sel, n_all = [], 0, 0
    with tarfile.open(path, mode="r|bz2") as tf:  # single streaming decompression pass
        for m in tf:
            if not m.name.endswith(".sensor.csv"):
                continue
            n_all += 1
            hk = _hour_key(m.name)
            if hk is None or not keep_hour(seqn, hk, keep_frac):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            w = process_member_bytes(f.read())
            if w is not None:
                chunks.append(w)
                n_sel += 1
    if chunks:
        return np.concatenate(chunks, 0), n_sel, n_all
    return np.empty((0, 3, WIN), np.float16), n_sel, n_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhanes-root", default="/projects/b53017/vbc1591/nhanes_pax")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--keep-frac", type=float, default=1 / 3)
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N participants of shard")
    args = ap.parse_args()

    pdir = os.path.join(args.out, "participants")
    os.makedirs(pdir, exist_ok=True)
    archives = list_archives(args.nhanes_root)
    mine = archives[args.shard_idx :: args.num_shards]
    if args.limit:
        mine = mine[: args.limit]
    print(f"[shard {args.shard_idx}/{args.num_shards}] {len(mine)} participants", flush=True)
    t0 = time.time()
    for i, path in enumerate(mine):
        seqn = seqn_of(path)
        out_npy = os.path.join(pdir, f"{seqn}.npy")
        out_meta = os.path.join(pdir, f"{seqn}.json")
        if os.path.exists(out_meta):
            continue
        t1 = time.time()
        try:
            w, n_sel, n_all = process_participant(path, args.keep_frac)
            err = None
        except Exception as e:  # corrupt archive etc.
            w, n_sel, n_all, err = np.empty((0, 3, WIN), np.float16), 0, 0, repr(e)
        tmp = out_npy + ".tmp.npy"
        np.save(tmp, w)
        os.replace(tmp, out_npy)
        meta = {"seqn": seqn, "cycle": os.path.basename(os.path.dirname(path)), "n_windows": int(len(w)),
                "hours_selected": n_sel, "hours_total": n_all, "error": err, "sec": round(time.time() - t1, 1)}
        with open(out_meta, "w") as f:
            json.dump(meta, f)
        print(f"  [{i+1}/{len(mine)}] {seqn}: {len(w)} win from {n_sel}/{n_all} h in {meta['sec']}s "
              f"(elapsed {(time.time()-t0)/60:.1f} min)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
