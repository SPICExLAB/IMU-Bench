#!/usr/bin/env python
"""Build ONE shard of the NHANES cache.

Why sharding: the single-job build holds every collected window in RAM before
writing (see nhanes_wrist_loader._collect_windows), so a 185,000h NHANES needs
~800 GB and only the handful of 1TB nodes can run it -- which is what left the
build queued for hours. Splitting the participant list into N shards makes each
job's footprint ~(total/N), so shards fit ordinary nodes, run CONCURRENTLY
(collapsing the ~4h serial read to ~4h/N), and can use the dedicated b53017
partition. merge_nhanes_shards.py then stitches them into the single
nhanes_{train,val}_{x,mask}.npy pair the cache loader expects.

The split is byte-compatible with the unsharded path: it reproduces
make_nhanes_loaders_random's own deterministic file selection and train/val
split, then takes a round-robin slice, so shard k of N always sees the same
participants regardless of how many shards are used elsewhere.

Usage (one job per shard):
    python scripts/build_nhanes_shard.py --config configs/base.yaml \
        --model-config configs/vit1d_ssl_mae_stage1_acconly_flu.yaml \
        --out-dir <cache_dir> --shard-idx K --num-shards N [--split train|val]
"""
from __future__ import annotations

import argparse, logging, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_nhanes_shard")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, required=True)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    assert 0 <= args.shard_idx < args.num_shards, "shard-idx out of range"

    from scripts.build_multidata_cache import _cache_split, _cfg_without_augmentation
    from imu_lm.data.combined_mae_multidata_loader import AccOnlyPadTo6ChDataset, DEFAULT_ROOTS
    from imu_lm.data.nhanes_wrist_loader import (
        NhanesDataset, _collect_windows, select_nhanes_files,
    )
    from imu_lm.data.windowing import compute_T_and_hop

    cfg = _cfg_without_augmentation(deep_update(load_yaml(args.config), load_yaml(args.model_config)))
    mm = cfg_get(cfg, ["objective", "mae_multidata_varlen_altattn_flu"], {}) or {}
    src = (mm.get("sources", {}) or {}).get("nhanes", {}) or {}

    root = src.get("root", DEFAULT_ROOTS["nhanes_root"])
    seed = int(src.get("seed", 42))
    splits = tuple(src.get("splits", [0.9, 0.1]))
    target_hz = int(src.get("target_hz") or 50)

    win_sec = float(cfg_get(cfg, ["windowing", "window_seconds"], 10.24))
    hop_sec = win_sec * float(cfg_get(cfg, ["windowing", "window_hop_ratio"], 0.5))

    # --- reproduce make_nhanes_loaders_random's file selection + split exactly ---
    max_files = src.get("max_files")
    if src.get("sample_fraction") is not None:
        total = len(select_nhanes_files(root, max_files=None))
        frac = max(1, round(total * float(src["sample_fraction"])))
        max_files = min(int(max_files), frac) if max_files else frac
    all_files = select_nhanes_files(root, max_files=int(max_files) if max_files else None, seed=seed)
    perm = np.random.default_rng(seed).permutation(len(all_files))
    n_train = int(round(splits[0] * len(all_files)))
    files = ([all_files[i] for i in perm[:n_train]] if args.split == "train"
             else [all_files[i] for i in perm[n_train:]])

    # --- this shard's slice + its share of the window budget ---
    shard_files = files[args.shard_idx::args.num_shards]
    hours_key = "max_hours" if args.split == "train" else "max_hours_val"
    max_windows = None
    if src.get(hours_key) is not None:
        total_windows = int(float(src[hours_key]) * 3600.0 / hop_sec)
        max_windows = total_windows // args.num_shards
    logger.info("shard %d/%d [%s]: %d of %d participant files, budget=%s windows",
                args.shard_idx, args.num_shards, args.split, len(shard_files), len(files),
                f"{max_windows:,}" if max_windows else "uncapped")
    if not shard_files:
        logger.info("shard is empty, nothing to do"); return

    T, hop = compute_T_and_hop(cfg)
    X = _collect_windows(shard_files, target_hz, T, hop, max_windows=max_windows)
    logger.info("shard %d: collected %s windows (%.1f GB)", args.shard_idx, f"{len(X):,}", X.nbytes / 1e9)

    ds = AccOnlyPadTo6ChDataset(NhanesDataset(X, cfg=cfg, split_name=args.split), "nhanes")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"shard{args.shard_idx:03d}of{args.num_shards:03d}"
    _cache_split(
        ds,
        os.path.join(args.out_dir, f"nhanes_{args.split}_x.{tag}.npy"),
        os.path.join(args.out_dir, f"nhanes_{args.split}_mask.{tag}.npy"),
        "nhanes", f"{args.split}/{tag}", args.batch_size, args.num_workers,
    )


if __name__ == "__main__":
    # REQUIRED: _collect_windows uses multiprocessing "spawn"; without this
    # guard each child re-executes the module (fork bomb -- cost a 45 min job
    # slot on 2026-08-01).
    main()
