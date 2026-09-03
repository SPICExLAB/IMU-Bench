#!/usr/bin/env python
"""Stitch NHANES cache shards into the single .npy pair the loader expects.

build_nhanes_shard.py writes nhanes_{split}_{x,mask}.shardKKKofNNN.npy; the
cache loader (_load_cached_source) wants nhanes_{split}_{x,mask}.npy. This
concatenates them in shard order.

Streams via memmap on BOTH sides -- output opened with open_memmap and each
shard read with mmap_mode='r', copied in bounded row blocks -- so peak RAM is
one block (~a few hundred MB), never the whole array. That is the entire point
of sharding: nothing in this pipeline should ever need the full ~800 GB
resident again.

Usage:
    python scripts/merge_nhanes_shards.py --out-dir <cache_dir> [--split train] [--keep-shards]
"""
from __future__ import annotations

import argparse, glob, logging, os, re

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("merge_nhanes_shards")

_TAG = re.compile(r"\.shard(\d+)of(\d+)\.npy$")
BLOCK = 65536   # rows per copy block


def merge(out_dir: str, split: str, keep: bool) -> None:
    xs = sorted(glob.glob(os.path.join(out_dir, f"nhanes_{split}_x.shard*of*.npy")),
                key=lambda p: int(_TAG.search(p).group(1)))
    if not xs:
        logger.warning("[%s] no shards found, nothing to merge", split); return
    masks = [p.replace("_x.shard", "_mask.shard") for p in xs]
    missing = [m for m in masks if not os.path.exists(m)]
    if missing:
        raise FileNotFoundError(f"[{split}] shard(s) missing their mask file: {missing}")

    expected = int(_TAG.search(xs[0]).group(2))
    got = sorted(int(_TAG.search(p).group(1)) for p in xs)
    if got != list(range(expected)):
        raise RuntimeError(
            f"[{split}] incomplete shard set: found {got} but expected 0..{expected-1}. "
            "Refusing to merge a partial cache -- rerun the missing shard jobs first."
        )

    shapes = [np.load(p, mmap_mode="r").shape for p in xs]
    tail = shapes[0][1:]
    if any(s[1:] != tail for s in shapes):
        raise RuntimeError(f"[{split}] shards disagree on window shape: {set(s[1:] for s in shapes)}")
    n_total = sum(s[0] for s in shapes)
    logger.info("[%s] merging %d shards -> %s rows, window shape %s", split, len(xs), f"{n_total:,}", tail)

    out_x = os.path.join(out_dir, f"nhanes_{split}_x.npy")
    out_m = os.path.join(out_dir, f"nhanes_{split}_mask.npy")
    tmp_x, tmp_m = out_x + ".tmp.npy", out_m + ".tmp.npy"
    mask_tail = np.load(masks[0], mmap_mode="r").shape[1:]

    x_mm = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.float32, shape=(n_total,) + tail)
    m_mm = np.lib.format.open_memmap(tmp_m, mode="w+", dtype=bool, shape=(n_total,) + mask_tail)
    pos = 0
    for xp, mp in zip(xs, masks):
        xa = np.load(xp, mmap_mode="r"); ma = np.load(mp, mmap_mode="r")
        if len(xa) != len(ma):
            raise RuntimeError(f"{xp}: x has {len(xa)} rows but mask has {len(ma)}")
        for s in range(0, len(xa), BLOCK):
            e = min(s + BLOCK, len(xa))
            x_mm[pos + s:pos + e] = xa[s:e]
            m_mm[pos + s:pos + e] = ma[s:e]
        pos += len(xa)
        logger.info("[%s]   %s -> %s/%s rows", split, os.path.basename(xp), f"{pos:,}", f"{n_total:,}")
        del xa, ma
    del x_mm, m_mm

    os.replace(tmp_x, out_x); os.replace(tmp_m, out_m)
    logger.info("[%s] wrote %s (%.2f GB) and %s", split, out_x, os.path.getsize(out_x) / 1e9, out_m)

    if not keep:
        for p in xs + masks:
            os.remove(p)
        logger.info("[%s] removed %d shard files", split, len(xs) + len(masks))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    ap.add_argument("--keep-shards", action="store_true",
                    help="keep the per-shard files after merging (default: delete to reclaim disk)")
    a = ap.parse_args()
    for sp in (["train", "val"] if a.split == "both" else [a.split]):
        merge(a.out_dir, sp, a.keep_shards)


if __name__ == "__main__":
    main()
