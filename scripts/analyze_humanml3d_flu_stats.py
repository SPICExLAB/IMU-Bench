"""
Analyze sequence-length statistics for HumanML3D FLU dataset clips.

Usage:
    python scripts/analyze_humanml3d_flu_stats.py \
        --root /home/yiquan/HumanML3D \
        [--splits train val test] \
        [--patch-size 16] \
        [--t-min 64] \
        [--t-max 512] \
        [--out-dir data_vis/humanml3d_flu_stats]

Outputs (saved to --out-dir):
    stats_<split>.json           per-split summary statistics
    seq_length_dist.png          histogram + CDF of sequence durations
    seq_length_per_split.png     violin/box comparison across splits
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_HML3D_FPS  = 20.0
_TARGET_FPS = 50.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snap_T(T_raw: int, patch_size: int, T_min: int, T_max: int):
    """Replicate imu_lm/data/humanml3d_dataset._snap_T."""
    if T_raw < T_min:
        return None
    T_snapped = (min(T_raw, T_max) // patch_size) * patch_size
    return max(T_snapped, patch_size)


def collect_lengths(
    root: str,
    split: str,
    patch_size: int,
    T_min: int,
    T_max: int,
) -> dict:
    """Walk joints/*.npy files listed in split.txt; record raw and snapped lengths."""
    split_file = pjoin(root, "HumanML3D", f"{split}.txt")
    if not os.path.exists(split_file):
        logger.warning("Split file not found: %s — skipping", split_file)
        return {}

    with open(split_file) as f:
        clip_ids = [l.strip() for l in f if l.strip()]

    raw_lengths   = []   # frames at 20 Hz (as stored in .npy)
    final_lengths = []   # snapped lengths at 50 Hz (what the Dataset returns)
    skipped_short = 0
    skipped_miss  = 0

    for cid in clip_ids:
        joints_path = pjoin(root, "joints", f"{cid}.npy")
        if not os.path.exists(joints_path):
            skipped_miss += 1
            continue

        # Load just the shape — avoid reading all data
        arr = np.load(joints_path, allow_pickle=True)
        N_20hz = arr.shape[0]  # frames at original 20 Hz

        # Resample to 50 Hz (same formula as _resample_joints)
        dur   = (N_20hz - 1) / _HML3D_FPS
        N_50hz = int(dur * _TARGET_FPS) + 1

        T = _snap_T(N_50hz, patch_size, T_min, T_max)

        raw_lengths.append(N_50hz)
        if T is None:
            skipped_short += 1
            final_lengths.append(None)
        else:
            final_lengths.append(T)

    loaded_mask  = [t is not None for t in final_lengths]
    loaded_T     = np.array([t for t in final_lengths if t is not None], dtype=float)
    loaded_dur   = loaded_T / _TARGET_FPS   # seconds

    raw_arr = np.array(raw_lengths, dtype=float)

    result = {
        "split":         split,
        "total_clips":   len(clip_ids),
        "missing_joints": skipped_miss,
        "skipped_too_short": skipped_short,
        "loaded_clips":  int(loaded_mask.count(True)),
        "patch_size":    patch_size,
        "T_min":         T_min,
        "T_max":         T_max,
        # Raw 50-Hz lengths (before snap / clamp)
        "raw_dur_sec": {
            "min":    float(raw_arr.min() / _TARGET_FPS),
            "max":    float(raw_arr.max() / _TARGET_FPS),
            "mean":   float(raw_arr.mean() / _TARGET_FPS),
            "median": float(np.median(raw_arr) / _TARGET_FPS),
            "p10":    float(np.percentile(raw_arr, 10) / _TARGET_FPS),
            "p25":    float(np.percentile(raw_arr, 25) / _TARGET_FPS),
            "p75":    float(np.percentile(raw_arr, 75) / _TARGET_FPS),
            "p90":    float(np.percentile(raw_arr, 90) / _TARGET_FPS),
            "p95":    float(np.percentile(raw_arr, 95) / _TARGET_FPS),
            "p99":    float(np.percentile(raw_arr, 99) / _TARGET_FPS),
            "std":    float(raw_arr.std() / _TARGET_FPS),
            "total_hours": float(raw_arr.sum() / _TARGET_FPS / 3600),
        },
        # Snapped / clamped lengths (what the Dataset actually sees)
        "snapped_dur_sec": {
            "min":    float(loaded_dur.min()),
            "max":    float(loaded_dur.max()),
            "mean":   float(loaded_dur.mean()),
            "median": float(np.median(loaded_dur)),
            "p10":    float(np.percentile(loaded_dur, 10)),
            "p25":    float(np.percentile(loaded_dur, 25)),
            "p75":    float(np.percentile(loaded_dur, 75)),
            "p90":    float(np.percentile(loaded_dur, 90)),
            "p95":    float(np.percentile(loaded_dur, 95)),
            "p99":    float(np.percentile(loaded_dur, 99)),
            "std":    float(loaded_dur.std()),
            "total_hours": float(loaded_dur.sum() / 3600),
        },
        # Discrete value counts (snapped — useful for bar chart)
        "snapped_value_counts": {
            str(int(v)): int(c)
            for v, c in zip(*np.unique(loaded_T.astype(int), return_counts=True))
        },
        "_raw_50hz_lengths":  [int(x) for x in raw_arr],
        "_snapped_50hz_lengths": [int(t) if t is not None else -1 for t in final_lengths],
    }
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_hist_cdf(results: dict[str, dict], out_dir: str, target_fps: float):
    """Single figure: histogram of raw durations + CDF, one subplot per split."""
    splits = list(results.keys())
    n = len(splits)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, split in enumerate(splits):
        r   = results[split]
        raw = np.array(r["_raw_50hz_lengths"]) / target_fps   # seconds

        # Histogram
        ax_h = axes[i][0]
        ax_h.hist(raw, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
        ax_h.axvline(r["raw_dur_sec"]["mean"],   color="red",    linestyle="--", label=f"mean {r['raw_dur_sec']['mean']:.2f}s")
        ax_h.axvline(r["raw_dur_sec"]["median"], color="orange", linestyle="--", label=f"median {r['raw_dur_sec']['median']:.2f}s")
        ax_h.set_xlabel("Duration (s)")
        ax_h.set_ylabel("Count")
        ax_h.set_title(f"[{split}] Raw sequence duration — {r['total_clips']} clips")
        ax_h.legend(fontsize=8)

        # CDF
        ax_c = axes[i][1]
        sorted_raw = np.sort(raw)
        cdf = np.arange(1, len(sorted_raw) + 1) / len(sorted_raw)
        ax_c.plot(sorted_raw, cdf, color="steelblue")
        for p, pct in [(0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (0.9, "90%"), (0.95, "95%")]:
            v = np.quantile(sorted_raw, p)
            ax_c.axvline(v, linestyle=":", color="grey", alpha=0.7)
            ax_c.text(v, p + 0.02, f"{pct}\n{v:.1f}s", fontsize=6, ha="center", color="grey")
        ax_c.set_xlabel("Duration (s)")
        ax_c.set_ylabel("CDF")
        ax_c.set_title(f"[{split}] CDF of sequence duration")
        ax_c.set_ylim(0, 1.05)

    fig.tight_layout()
    out_path = pjoin(out_dir, "seq_length_dist.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_snapped_bar(results: dict[str, dict], out_dir: str, target_fps: float):
    """Bar chart of snapped (Dataset-visible) length value counts."""
    splits = list(results.keys())
    all_vals: set = set()
    for r in results.values():
        all_vals.update(int(k) for k in r["snapped_value_counts"])
    all_vals = sorted(all_vals)
    labels   = [f"{v/target_fps:.2f}s\n({v})" for v in all_vals]

    fig, axes = plt.subplots(1, len(splits), figsize=(max(8, 2.5 * len(all_vals)), 4), sharey=False)
    if len(splits) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for i, split in enumerate(splits):
        r      = results[split]
        counts = [r["snapped_value_counts"].get(str(v), 0) for v in all_vals]
        axes[i].bar(labels, counts, color=colors[i % len(colors)], edgecolor="white")
        axes[i].set_title(f"[{split}] Snapped lengths (loaded: {r['loaded_clips']})")
        axes[i].set_xlabel("Snapped duration (T frames)")
        axes[i].set_ylabel("Clip count")
        axes[i].tick_params(axis="x", labelsize=7)

    fig.tight_layout()
    out_path = pjoin(out_dir, "seq_length_snapped_bar.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_violin_across_splits(results: dict[str, dict], out_dir: str, target_fps: float):
    """Violin plot of raw durations across splits side by side."""
    splits = list(results.keys())
    if len(splits) < 2:
        return  # not meaningful with only one split

    data = [np.array(results[s]["_raw_50hz_lengths"]) / target_fps for s in splits]

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(splits)), 5))
    parts = ax.violinplot(data, positions=range(len(splits)), showmedians=True, showextrema=True)
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(splits)
    ax.set_ylabel("Duration (s)")
    ax.set_title("Sequence duration distribution across splits")

    # Overlay means
    for i, d in enumerate(data):
        ax.scatter([i], [d.mean()], color="red", zorder=5, s=30, label="mean" if i == 0 else "")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = pjoin(out_dir, "seq_length_violin_splits.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HumanML3D FLU dataset statistics")
    parser.add_argument("--root",       default="/home/yiquan/HumanML3D",
                        help="HumanML3D workspace root (contains joints/, HumanML3D/)")
    parser.add_argument("--splits",     nargs="+", default=["train", "val", "test"],
                        help="Dataset splits to analyse (default: train val test)")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--t-min",      type=int, default=64,
                        help="T_min in samples at 50 Hz (default 64 = 1.28 s)")
    parser.add_argument("--t-max",      type=int, default=512,
                        help="T_max in samples at 50 Hz (default 512 = 10.24 s)")
    parser.add_argument("--out-dir",    default="data_vis/humanml3d_flu_stats")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results: dict[str, dict] = {}
    for split in args.splits:
        logger.info("Collecting lengths for split: %s ...", split)
        r = collect_lengths(
            root       = args.root,
            split      = split,
            patch_size = args.patch_size,
            T_min      = args.t_min,
            T_max      = args.t_max,
        )
        if not r:
            continue
        results[split] = r

        # Print summary table
        rd = r["raw_dur_sec"]
        sd = r["snapped_dur_sec"]
        print(f"\n{'='*60}")
        print(f"Split: {split}  |  total={r['total_clips']}  loaded={r['loaded_clips']}  "
              f"skipped_short={r['skipped_too_short']}  missing={r['missing_joints']}")
        print(f"{'─'*60}")
        print(f"{'Metric':<18} {'Raw 50Hz dur':>15} {'Snapped dur':>15}")
        print(f"{'─'*60}")
        for key in ["min", "p10", "p25", "median", "mean", "p75", "p90", "p95", "p99", "max", "std"]:
            print(f"  {key:<16} {rd[key]:>14.3f}s {sd[key]:>14.3f}s")
        print(f"  {'total_hours':<16} {rd['total_hours']:>14.3f}h {sd['total_hours']:>14.3f}h")
        print(f"{'='*60}")

        # Snapped value counts
        vc = r["snapped_value_counts"]
        print(f"  Snapped length distribution (T → count):")
        for k, v in sorted(vc.items(), key=lambda x: int(x[0])):
            dur_s = int(k) / _TARGET_FPS
            bar   = "█" * min(40, v // max(1, max(int(c) for c in vc.values()) // 40))
            print(f"    T={k:>4} ({dur_s:.2f}s): {v:>5}  {bar}")

        # Save json (without large raw arrays for readability)
        save_r = {k: v for k, v in r.items() if not k.startswith("_")}
        with open(pjoin(args.out_dir, f"stats_{split}.json"), "w") as f:
            json.dump(save_r, f, indent=2)

    if not results:
        logger.error("No results — check --root and --splits.")
        return

    # Plots
    plot_hist_cdf(results, args.out_dir, _TARGET_FPS)
    plot_snapped_bar(results, args.out_dir, _TARGET_FPS)
    plot_violin_across_splits(results, args.out_dir, _TARGET_FPS)

    logger.info("Done. Outputs in: %s", args.out_dir)


if __name__ == "__main__":
    main()
