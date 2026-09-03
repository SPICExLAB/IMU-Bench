"""build_multidata_cache.py
One-time (or periodic) preprocessing pass for the combined MAE multidata
pipeline (``imu_lm.data.combined_mae_multidata_loader``).

Builds every enabled source in
``objective.mae_multidata_varlen_altattn_flu.sources`` via the *exact same*
per-source loaders used at training time (so caching can never silently
diverge from what a live run would produce), then writes each source's full
train/val windows to ``.npy`` files under an output directory:

    {out_dir}/{source}_{split}_x.npy       float32 [N, 6, T_max]  (memmapped write)
    {out_dir}/{source}_{split}_mask.npy    bool    [N, 6]

Point a future training config at the cache with::

    objective:
      mae_multidata_varlen_altattn_flu:
        cache_dir: /projects/b53017/vbc1591/mae_multidata_cache

and ``make_combined_mae_multidata_loaders`` will load straight from these
files for any source that has them, skipping all raw parsing / SMPL FK /
windowing. Sources without a cache still fall back to the normal builder, so
you can cache incrementally (e.g. just the slow synthetic sources first).

Augmentation (random_rotation, gaussian_noise, ...) is deliberately NOT baked
into the cache: every per-source loader calls apply_augment() inside its own
__getitem__, so caching with augmentation left on would freeze one random
draw (e.g. one of the 24 octahedral rotations) into each window forever,
instead of a fresh draw every epoch. This script builds datasets with every
augmentation forced off (see _cfg_without_augmentation) so the cache holds
clean, deterministically-preprocessed windows; CachedWindowDataset applies
apply_augment() itself at read time, using whatever augmentation config the
training run actually asks for.

The cached windows are the FULL T_max window (same length the live builders
produce) — the per-batch variable-length crop still happens at collate time,
so training behavior (variable T per batch) is unchanged versus running
without a cache.

Usage
-----
    python scripts/build_multidata_cache.py \\
        --config configs/base.yaml \\
        --model-config configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml \\
        --out-dir /projects/b53017/vbc1591/mae_multidata_cache

    # Only rebuild specific sources (e.g. the slow synthetic ones):
    python scripts/build_multidata_cache.py ... --sources motionx humanml3d sonic

Resuming after an interrupted run (SLURM walltime, crash, Ctrl-C, ...)
-----------------------------------------------------------------------
- A source/split whose FINAL ``.npy`` files already exist is skipped entirely
  (logged and moved past) — re-running the whole script after it finished is
  a fast no-op.
- A source/split that was interrupted mid-write left a ``*_train_x.tmp.npy``
  (partial memmap) and a ``*_train_x.progress.json`` (how far it got) behind.
  On the next run, that split resumes from the recorded position instead of
  starting over — the dataset is still rebuilt (raw parsing / SMPL FK is not
  itself checkpointed), but no already-cached window is recomputed or
  rewritten, and the progress bar's ``initial=`` reflects the resume point.
- Progress is persisted after every batch (a few bytes of JSON), so at most
  one in-flight batch is ever lost to a hard kill.

Notes
-----
- Run this on a compute node (sbatch), not the login node: several sources
  (motionx/humanml3d/sonic) run SMPL forward kinematics per clip and
  OCDetect/capture24/realdata read tens of GB from a shared filesystem.
- Memory: the per-sample tensor is written directly into a memory-mapped
  output .npy (``np.lib.format.open_memmap``), so RAM usage stays bounded by
  a few batches at a time, not the whole source — safe even for the largest
  sources (realdata, OCDetect, capture24) at full scale.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional, Tuple

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imu_lm.data.combined_mae_multidata_loader import ALL_SOURCES, _BUILDERS
from imu_lm.utils.helpers import cfg_get, deep_update, load_yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_multidata_cache")

# Every stochastic augmentation imu_lm.data.augmentations.transform.apply_augment
# knows about. Every one of the 11 per-source loaders calls apply_augment inside
# __getitem__ — so if the cache is built with these left enabled, each window's
# ONE randomly-drawn augmentation (e.g. one of the 24 octahedral rotations) gets
# baked into the .npy file permanently, and every subsequent training epoch would
# see that same frozen draw instead of a fresh one. Caching must only ever see
# deterministic preprocessing (impute/filter/normalize); CachedWindowDataset
# applies apply_augment itself, fresh, on every __getitem__ call at train time.
_AUGMENTATION_KEYS = (
    "random_rotation", "gaussian_noise", "jitter", "bias_drift", "sensor_dropout",
    "time_crop", "frequency_mask", "gain_jitter", "axis_bias", "time_warp",
)


def _cfg_without_augmentation(cfg: dict) -> dict:
    """Deep copy of cfg with every stochastic augmentation forced off, so the
    per-source builders below produce clean (preprocessed-only) cached windows."""
    cfg = copy.deepcopy(cfg)
    for key in _AUGMENTATION_KEYS:
        block = cfg.get(key)
        if isinstance(block, dict):
            block["enabled"] = False
    return cfg


def _collate_with_count(batch):
    """Stack (x, channel_valid_mask) pairs, dropping any None windows (a few
    sources' __getitem__ can return None for a bad window). Also returns the
    PRE-filter batch length, so the caller can advance its "dataset items
    consumed" counter correctly even when some items were dropped — that
    counter (not the post-filter row count) is what resuming skips by."""
    n_items = len(batch)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, n_items
    xs = torch.stack([b[0] for b in batch], 0)
    masks = torch.stack([b[1] for b in batch], 0)
    T = xs.shape[-1]
    # 3rd element (when present) is a [T] bool temporal-validity mask from a short clip;
    # persist it compactly as a per-window valid length.
    vlens = torch.tensor([int(b[2].sum()) if len(b) > 2 else T for b in batch], dtype=torch.int32)
    return xs, masks, vlens, n_items


def _tmp_paths(out_x: str, out_mask: str) -> Tuple[str, str, str]:
    # NOTE: np.save() silently appends ".npy" to any path that doesn't already
    # end with it — so these temp names must themselves end in ".npy", or a
    # later os.replace(tmp_path, out_path) would look for a file that was
    # actually written one ".npy" further along.
    tmp_x = out_x[:-len(".npy")] + ".tmp.npy"
    tmp_mask = out_mask[:-len(".npy")] + ".tmp.npy"
    tmp_tlen = out_x[:-len("_x.npy")] + "_tmask.tmp.npy"
    progress_path = tmp_x[:-len(".npy")] + ".progress.json"
    return tmp_x, tmp_mask, tmp_tlen, progress_path


def _write_progress(progress_path: str, items_consumed: int, rows_written: int) -> None:
    tmp = progress_path + ".swp"
    with open(tmp, "w") as f:
        json.dump({"items_consumed": items_consumed, "rows_written": rows_written}, f)
    os.replace(tmp, progress_path)


def _cache_split(ds: Optional[Dataset], out_x: str, out_mask: str,
                  name: str, split: str, batch_size: int, num_workers: int) -> None:
    if ds is None:
        logger.info("[%s/%s] no data, skipping", name, split)
        return
    if os.path.exists(out_x) and os.path.exists(out_mask):
        logger.info("[%s/%s] cache already exists (%s), skipping — delete it to rebuild", name, split, out_x)
        return

    n = len(ds)
    if n == 0:
        logger.info("[%s/%s] empty dataset, skipping", name, split)
        return

    tmp_x, tmp_mask, tmp_tlen, progress_path = _tmp_paths(out_x, out_mask)

    items_consumed = 0
    rows_written = 0
    x_mm: Optional[np.memmap] = None
    mask_mm: Optional[np.memmap] = None
    tlen_mm: Optional[np.memmap] = None

    if os.path.exists(tmp_x) and os.path.exists(tmp_mask) and os.path.exists(progress_path):
        try:
            with open(progress_path) as f:
                prog = json.load(f)
            resume_items, resume_rows = int(prog["items_consumed"]), int(prog["rows_written"])
            x_cand = np.lib.format.open_memmap(tmp_x, mode="r+")
            mask_cand = np.lib.format.open_memmap(tmp_mask, mode="r+")
            valid = (x_cand.shape[0] == n and mask_cand.shape[0] == n
                     and 0 <= resume_rows <= resume_items <= n)
            if valid and os.path.exists(tmp_tlen):
                tlen_cand = np.lib.format.open_memmap(tmp_tlen, mode="r+")
                valid = tlen_cand.shape[0] == n
            else:
                valid = False
            if valid:
                x_mm, mask_mm, tlen_mm = x_cand, mask_cand, tlen_cand
                items_consumed, rows_written = resume_items, resume_rows
                logger.info("[%s/%s] resuming from a previous run: %d/%d dataset items already "
                            "consumed (%d windows written)", name, split, items_consumed, n, rows_written)
            else:
                logger.warning("[%s/%s] found a partial cache but it doesn't match this dataset "
                                "(shape/counter mismatch) — restarting from scratch", name, split)
        except Exception:
            logger.warning("[%s/%s] could not read progress file, restarting from scratch",
                            name, split, exc_info=True)

    if items_consumed >= n:
        logger.info("[%s/%s] previous run already consumed all %d items, finalizing", name, split, n)
    else:
        remaining_ds = Subset(ds, range(items_consumed, n)) if items_consumed > 0 else ds
        loader = DataLoader(
            remaining_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=_collate_with_count, drop_last=False,
        )

        t0 = time.time()
        pbar = tqdm(total=n, initial=items_consumed, desc=f"[{name}/{split}]", unit="win", dynamic_ncols=True)
        for xs, masks, vlens, n_items in loader:
            if x_mm is None:
                # First batch of a fresh (non-resumed) run: now we know T_max/C.
                assert xs is not None, f"[{name}/{split}] first batch was entirely None windows — cannot infer shape"
                T_max, C = xs.shape[-1], xs.shape[1]
                x_mm = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.float32, shape=(n, C, T_max))
                mask_mm = np.lib.format.open_memmap(tmp_mask, mode="w+", dtype=bool, shape=(n, C))
                tlen_mm = np.lib.format.open_memmap(tmp_tlen, mode="w+", dtype=np.int32, shape=(n,))
                tlen_mm[:] = T_max
            if xs is not None:
                b = xs.shape[0]
                x_mm[rows_written:rows_written + b] = xs.numpy().astype(np.float32)
                mask_mm[rows_written:rows_written + b] = masks.numpy().astype(bool)
                tlen_mm[rows_written:rows_written + b] = vlens.numpy().astype(np.int32)
                rows_written += b
            items_consumed += n_items
            _write_progress(progress_path, items_consumed, rows_written)
            pbar.update(n_items)
        pbar.close()
        logger.info("[%s/%s] %d/%d items consumed, %d windows written (%.1fs)",
                    name, split, items_consumed, n, rows_written, time.time() - t0)

    if x_mm is None:
        logger.warning("[%s/%s] dataloader yielded nothing despite len(ds)=%d, skipping", name, split, n)
        return

    x_mm.flush()
    mask_mm.flush()
    tlen_mm.flush()
    has_short = bool((np.asarray(tlen_mm[:rows_written]) < x_mm.shape[-1]).any())
    del x_mm, mask_mm, tlen_mm

    if rows_written != n:
        # Some windows were None throughout the run (dropped by the collate) —
        # shrink both arrays to the actual written count. Write to a separate
        # path first, then swap in: overwriting a file in place while it may
        # still be mmap'd elsewhere is not safe.
        for tmp in (tmp_x, tmp_mask, tmp_tlen):
            trimmed = tmp[:-len(".npy")] + ".trim.npy"
            np.save(trimmed, np.load(tmp, mmap_mode="r")[:rows_written])
            os.replace(trimmed, tmp)

    os.replace(tmp_x, out_x)
    os.replace(tmp_mask, out_mask)
    out_tlen = out_x[:-len("_x.npy")] + "_tmask.npy"
    if has_short:
        os.replace(tmp_tlen, out_tlen)     # per-window valid length (int32 [N]); loader expands to [T] bool
        logger.info("[%s/%s] short clips present -> wrote %s", name, split, out_tlen)
    else:
        os.remove(tmp_tlen)                # all windows full-length: no temporal mask needed
    if os.path.exists(progress_path):
        os.remove(progress_path)
    size_gb = os.path.getsize(out_x) / 1e9
    logger.info("[%s/%s] done: %d windows, %.2f GB -> %s", name, split, rows_written, size_gb, out_x)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="base config yaml (e.g. configs/base.yaml)")
    ap.add_argument("--model-config", required=True,
                    help="model config yaml (e.g. configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml)")
    ap.add_argument("--out-dir", required=True, help="cache output directory")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="only cache these sources (default: every enabled source in the config)")
    ap.add_argument("--batch-size", type=int, default=256, help="collection batch size (not the training batch size)")
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    cfg = deep_update(base_cfg, model_cfg)

    mm_cfg = cfg_get(cfg, ["objective", "mae_multidata_varlen_altattn_flu"], {}) or {}
    sources_cfg = mm_cfg.get("sources", {}) or {}
    vl_max = int(cfg_get(cfg, ["data", "variable_length", "max_samples"], 512))

    # Build datasets with augmentation forced off (see _cfg_without_augmentation
    # docstring) — the cache stores clean windows; augmentation is applied fresh
    # at read time by CachedWindowDataset instead.
    build_cfg = _cfg_without_augmentation(cfg)

    names = args.sources if args.sources else ALL_SOURCES
    logger.info("Caching sources: %s  ->  %s", names, args.out_dir)

    for name in names:
        if name not in _BUILDERS:
            logger.warning("unknown source '%s' (valid: %s), skipping", name, ALL_SOURCES)
            continue
        src_cfg = sources_cfg.get(name, {}) or {}
        if not bool(src_cfg.get("enabled", True)):
            logger.info("[%s] disabled in config, skipping (pass --sources to force)", name)
            continue

        train_out = os.path.join(args.out_dir, f"{name}_train_x.npy")
        val_out = os.path.join(args.out_dir, f"{name}_val_x.npy")
        train_mask_out = os.path.join(args.out_dir, f"{name}_train_mask.npy")
        val_mask_out = os.path.join(args.out_dir, f"{name}_val_mask.npy")
        if os.path.exists(train_out) and os.path.exists(val_out):
            logger.info("[%s] both train and val cache already complete, skipping entirely "
                        "(pass --sources to force a rebuild after deleting the .npy files)", name)
            continue

        logger.info("=== building %s (raw parsing / SMPL FK / windowing) ===", name)
        try:
            train_ds, val_ds = _BUILDERS[name](build_cfg, src_cfg, vl_max)
        except Exception:
            logger.exception("[%s] builder failed, skipping", name)
            continue

        _cache_split(train_ds, train_out, train_mask_out, name, "train", args.batch_size, args.num_workers)
        _cache_split(val_ds, val_out, val_mask_out, name, "val", args.batch_size, args.num_workers)

    logger.info("Done. Set objective.mae_multidata_varlen_altattn_flu.cache_dir: %s in your config "
                "to use this cache next time.", args.out_dir)


if __name__ == "__main__":
    main()
