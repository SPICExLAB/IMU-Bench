"""analyze_imu_stats.py
======================
Compare wrist IMU statistics between:

  A) Nymeria FLU synthetic IMU  (nymeria_wrist_loader_FLU)
  B) Real-data parquet IMU      (imu_lm/data/loaders.py  +  base.yaml sensor_columns)

Outputs
-------
  • Console table: per-channel mean / std / min / max / percentiles
  • Console table: signal-level (magnitude) stats + power spectral density peak
  • PNG plots saved to --out-dir:
      - channel_distributions.png   — per-channel histogram + KDE overlay
      - magnitude_distributions.png — ||acc|| and ||gyro|| distributions
      - correlation_heatmap.png     — inter-channel Pearson correlation
      - psd.png                     — mean PSD per channel (Welch)

Usage
-----
    python scripts/analyze_imu_stats.py \\
        --config        configs/base.yaml \\
        --nymeria-dir   /mnt/nas2/processed_data_test_body_Ty \\
        --parquet       /path/to/data.parquet \\
        [--nymeria-n    50]          # max Nymeria pkl files to sample
        [--parquet-n    50000]       # max real-data rows to sample
        [--fps          50]          # sampling rate (for PSD)
        [--out-dir      data_vis/imu_stats]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Channel names
# ─────────────────────────────────────────────────────────────────────────────
ACC_COLS  = ["acc_F",  "acc_L",  "acc_U"]
GYRO_COLS = ["gyro_F", "gyro_L", "gyro_U"]
ALL_COLS  = ACC_COLS + GYRO_COLS      # matches loader output order: acc then gyro


# ─────────────────────────────────────────────────────────────────────────────
# Data gathering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_nymeria_flu_samples(
    pkl_dir: str,
    max_files: int,
    fps: int,
    window_size: int = 128,
) -> np.ndarray:
    """Return per-window z-score normalised [M, 6] float32 array from Nymeria FLU sessions.

    Each session is sliced into non-overlapping windows of ``window_size`` frames
    and the same per-window z-score preprocessing applied to the parquet data is
    applied here, so both datasets are compared under identical conditions.
    """
    from imu_lm.data.nymeria_wrist_loader_FLU import _load_nymeria_session_flu
    from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window

    _preproc_cfg: dict = {
        "preprocess": {
            "impute":    {"enabled": False},
            "filter":    {"enabled": False},
            "normalize": {"enabled": True, "method": "zscore", "eps": 1e-6},
        }
    }

    pkls = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    if not pkls:
        raise FileNotFoundError(f"No *.pkl files found in {pkl_dir}")
    if max_files < len(pkls):
        random.seed(42)
        pkls = random.sample(pkls, max_files)

    logger.info("Loading %d Nymeria FLU pkl files …", len(pkls))
    stats = PreprocessStats()
    processed_windows: list[np.ndarray] = []
    for p in tqdm(pkls, unit="pkl"):
        result = _load_nymeria_session_flu(p, target_hz=fps)
        if result is None:
            continue
        X, _ = result   # X: [N, 6]  (raw, within one session)
        n_windows = len(X) // window_size
        for i in range(n_windows):
            win = X[i * window_size : (i + 1) * window_size].copy()
            out = preprocess_window(win, _preproc_cfg, stats)  # [6, T] or None
            if out is not None:
                processed_windows.append(out.T)  # [T, 6]

    if not processed_windows:
        raise RuntimeError("No Nymeria sessions loaded.")
    data = np.concatenate(processed_windows, axis=0)   # [M, 6]
    logger.info(
        "Nymeria FLU: %d frames from %d windows (z-score per window)",
        len(data), len(processed_windows),
    )
    return data


def _load_parquet_wrist_samples(
    parquet_path: str,
    sensor_cols: list[str],
    max_rows: int,
    cfg: dict | None = None,
    window_size: int = 128,
) -> np.ndarray:
    """Return preprocessed [M, 6] float32 rows from a parquet file.

    Applies the same preprocessing pipeline as training:
      impute → lowpass filter → z-score normalize (per window)

    The z-score normalisation removes the per-window DC offset, which
    effectively removes the gravity component from accelerometer channels.
    Data is sliced into ``window_size``-frame windows, each window is
    preprocessed independently (matching WindowDataset behaviour), then all
    windows are concatenated and returned as flat frames.

    Args:
        parquet_path:  Path to parquet file/directory.
        sensor_cols:   Column names (must be 6 channels: 3 acc + 3 gyro).
        max_rows:      Approximate cap on output rows (sampled after windowing).
        cfg:           Training config dict; if None a minimal default is used
                       that enables z-score normalization only.
        window_size:   Window length in frames (default 128 = 2.56 s @ 50 Hz).
    """
    try:
        import pyarrow.dataset as pds
    except ImportError as e:
        raise ImportError("pyarrow is required for parquet loading") from e

    from imu_lm.data.augmentations.preprocess import PreprocessStats, preprocess_window

    # Minimal fallback config: only z-score normalize (removes gravity offset)
    _default_cfg: dict = {
        "preprocess": {
            "impute":    {"enabled": False},
            "filter":    {"enabled": False},
            "normalize": {"enabled": True, "method": "zscore", "eps": 1e-6},
        }
    }
    preproc_cfg = cfg if cfg is not None else _default_cfg

    logger.info("Scanning parquet: %s", parquet_path)
    dataset = pds.dataset(parquet_path, format="parquet")

    # Read only the 6 sensor columns
    available = dataset.schema.names
    missing = [c for c in sensor_cols if c not in available]
    if missing:
        raise ValueError(
            f"Parquet is missing columns: {missing}\n"
            f"Available: {available[:20]} ..."
        )

    # Identify session-grouping columns so we window within sessions only
    _dataset_col = (cfg or {}).get("data", {}).get("dataset_column", "dataset")
    _subject_col = (cfg or {}).get("data", {}).get("subject_column", "subject_id")
    _session_col = (cfg or {}).get("data", {}).get("session_column", "session_id")
    session_cols_present = [
        c for c in [_dataset_col, _subject_col, _session_col]
        if c in available
    ]
    load_cols = sensor_cols + [c for c in session_cols_present if c not in sensor_cols]

    # ── Stream unique session keys without loading sensor data ──────────────
    # Scanning only the 3 key columns is cheap even for 700 M-row parquets.
    logger.info("Discovering unique sessions (streaming key columns) …")
    key_scanner = dataset.scanner(columns=session_cols_present, batch_size=500_000)
    seen_keys: set = set()
    session_keys: list = []
    for batch in key_scanner.to_batches():
        df_keys = batch.to_pandas()
        for key in df_keys.drop_duplicates().itertuples(index=False):
            t = tuple(key)
            if t not in seen_keys:
                seen_keys.add(t)
                session_keys.append(dict(zip(session_cols_present, t)))

    logger.info("Found %d unique sessions; windowing within each …", len(session_keys))

    # ── Process one session at a time — constant memory ─────────────────────
    stats = PreprocessStats()
    processed_windows: list[np.ndarray] = []
    rng = np.random.default_rng(42)

    for sess in tqdm(session_keys, unit="session"):
        if len(processed_windows) * window_size >= max_rows * 4:
            # Collected enough raw windows; stop early to save time.
            break

        # Build per-session filter
        filt = None
        for col, val in sess.items():
            cond = pds.field(col) == val
            filt = cond if filt is None else (filt & cond)

        tbl = dataset.to_table(columns=load_cols, filter=filt)
        if len(tbl) == 0:
            continue
        raw = tbl.select(sensor_cols).to_pandas().to_numpy(dtype=np.float32)  # [N, 6]

        n_windows = len(raw) // window_size
        for i in range(n_windows):
            win = raw[i * window_size : (i + 1) * window_size].copy()  # [T, 6]
            out = preprocess_window(win, preproc_cfg, stats)             # [6, T] or None
            if out is not None:
                processed_windows.append(out.T)  # back to [T, 6]

    if not processed_windows:
        raise RuntimeError("All parquet windows were dropped during preprocessing.")

    data = np.concatenate(processed_windows, axis=0)  # [M, 6]

    if len(data) > max_rows:
        idx = np.random.default_rng(42).choice(len(data), size=max_rows, replace=False)
        idx.sort()
        data = data[idx]

    logger.info(
        "Real-data parquet: %d frames (from %d windows, %d dropped by preprocess)",
        len(data), len(processed_windows), stats.windows_dropped_impute,
    )
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _channel_stats(data: np.ndarray, col_names: list[str]) -> dict:
    """Compute per-channel descriptive stats. Returns dict of arrays."""
    stats = {}
    for i, name in enumerate(col_names):
        v = data[:, i]
        v = v[np.isfinite(v)]
        stats[name] = {
            "mean":  float(np.mean(v)),
            "std":   float(np.std(v)),
            "min":   float(np.min(v)),
            "max":   float(np.max(v)),
            "p5":    float(np.percentile(v,  5)),
            "p25":   float(np.percentile(v, 25)),
            "p50":   float(np.percentile(v, 50)),
            "p75":   float(np.percentile(v, 75)),
            "p95":   float(np.percentile(v, 95)),
            "rms":   float(np.sqrt(np.mean(v**2))),
        }
    return stats


def _magnitude_stats(data: np.ndarray, label: str) -> dict:
    """Compute ||acc|| and ||gyro|| stats."""
    acc_mag  = np.linalg.norm(data[:, :3], axis=1)
    gyro_mag = np.linalg.norm(data[:, 3:], axis=1)
    out = {}
    for name, mag in [("||acc||", acc_mag), ("||gyro||", gyro_mag)]:
        out[name] = {
            "mean": float(np.mean(mag)),
            "std":  float(np.std(mag)),
            "p5":   float(np.percentile(mag,  5)),
            "p50":  float(np.percentile(mag, 50)),
            "p95":  float(np.percentile(mag, 95)),
        }
    return out


def _print_stats(stats: dict, title: str):
    cols = ["mean", "std", "min", "p5", "p25", "p50", "p75", "p95", "max", "rms"]
    hdr  = f"{'channel':<12}" + "".join(f"{c:>9}" for c in cols)
    sep  = "─" * len(hdr)
    print(f"\n{title}")
    print(sep)
    print(hdr)
    print(sep)
    for ch, s in stats.items():
        row = f"{ch:<12}" + "".join(f"{s[c]:>9.3f}" for c in cols)
        print(row)
    print(sep)


def _print_magnitude_stats(stats: dict, title: str):
    cols = ["mean", "std", "p5", "p50", "p95"]
    hdr  = f"{'signal':<12}" + "".join(f"{c:>9}" for c in cols)
    sep  = "─" * len(hdr)
    print(f"\n{title}")
    print(sep)
    print(hdr)
    print(sep)
    for ch, s in stats.items():
        row = f"{ch:<12}" + "".join(f"{s[c]:>9.3f}" for c in cols)
        print(row)
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {"Nymeria FLU": "#4C72B0", "Real Parquet": "#DD8452"}


def _plot_channel_distributions(datasets: dict[str, np.ndarray],
                                  col_names: list[str], out_path: str):
    """One subplot per channel; histogram + KDE for each dataset."""
    from scipy.stats import gaussian_kde

    n_cols = len(col_names)
    fig, axes = plt.subplots(2, n_cols // 2, figsize=(4 * (n_cols // 2), 7), sharey=False)
    axes = axes.flatten()

    for ci, (name, ax) in enumerate(zip(col_names, axes)):
        for label, data in datasets.items():
            v = data[:, ci]
            v = v[np.isfinite(v)]
            # clip to 1–99th percentile for display
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            v_clip  = v[(v >= lo) & (v <= hi)]
            color   = _COLORS.get(label, "gray")

            ax.hist(v_clip, bins=80, density=True, alpha=0.35,
                    color=color, label=label)
            try:
                xs = np.linspace(lo, hi, 300)
                kde = gaussian_kde(v_clip, bw_method=0.15)
                ax.plot(xs, kde(xs), color=color, linewidth=1.8)
            except Exception:
                pass

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("value")
        ax.grid(True, alpha=0.3)
        if ci == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Per-Channel Distributions (clipped 1–99 %ile)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", out_path)


def _plot_magnitude_distributions(datasets: dict[str, np.ndarray], out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, (slice_, title) in zip(axes, [
        (slice(None, 3), "||acc||  (m/s²)"),
        (slice(3, None), "||gyro||  (rad/s)"),
    ]):
        for label, data in datasets.items():
            mag = np.linalg.norm(data[:, slice_], axis=1)
            lo, hi = np.percentile(mag, 1), np.percentile(mag, 99)
            mag_clip = mag[(mag >= lo) & (mag <= hi)]
            color = _COLORS.get(label, "gray")
            ax.hist(mag_clip, bins=80, density=True, alpha=0.45,
                    color=color, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("magnitude")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("IMU Magnitude Distributions", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", out_path)


def _plot_correlation_heatmap(datasets: dict[str, np.ndarray],
                               col_names: list[str], out_path: str):
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, datasets.items()):
        # subsample for speed
        idx = np.random.choice(len(data), min(len(data), 20_000), replace=False)
        corr = np.corrcoef(data[idx].T)   # [6, 6]

        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(col_names))); ax.set_xticklabels(col_names, rotation=45)
        ax.set_yticks(range(len(col_names))); ax.set_yticklabels(col_names)
        ax.set_title(label, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(col_names)):
            for j in range(len(col_names)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    plt.suptitle("Inter-Channel Pearson Correlation", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", out_path)


def _plot_psd(datasets: dict[str, np.ndarray],
              col_names: list[str], fps: int, out_path: str):
    from scipy.signal import welch

    n_cols = len(col_names)
    fig, axes = plt.subplots(2, n_cols // 2, figsize=(4 * (n_cols // 2), 7))
    axes = axes.flatten()

    for ci, (name, ax) in enumerate(zip(col_names, axes)):
        for label, data in datasets.items():
            # Welch on a random slice of up to 100_000 consecutive frames
            v = data[:, ci]
            if len(v) > 100_000:
                start = np.random.randint(0, len(v) - 100_000)
                v = v[start : start + 100_000]
            v = v[np.isfinite(v)]
            f, pxx = welch(v, fs=fps, nperseg=min(512, len(v) // 4))
            color = _COLORS.get(label, "gray")
            ax.semilogy(f, pxx, color=color, label=label, linewidth=1.5)

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD")
        ax.set_xlim(0, fps / 2)
        ax.grid(True, which="both", alpha=0.3)
        if ci == 0:
            ax.legend(fontsize=8)

    plt.suptitle(f"Power Spectral Density (Welch, fs={fps} Hz)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", out_path)


def _plot_temporal_sample(datasets: dict[str, np.ndarray],
                           col_names: list[str], fps: int,
                           out_path: str, n_seconds: float = 10.0):
    """Plot a short temporal trace for each dataset side-by-side."""
    n_frames = int(n_seconds * fps)
    n_cols_plot = len(col_names)
    n_ds = len(datasets)

    fig, axes = plt.subplots(n_cols_plot, n_ds,
                              figsize=(7 * n_ds, 2.2 * n_cols_plot),
                              sharex="col")

    if n_ds == 1:
        axes = axes[:, np.newaxis]  # ensure 2D indexing

    t = np.arange(n_frames) / fps

    for di, (label, data) in enumerate(datasets.items()):
        start = max(0, len(data) // 3)   # pick a clip from the middle third
        end   = min(start + n_frames, len(data))
        clip  = data[start:end]

        for ci, name in enumerate(col_names):
            ax = axes[ci, di]
            ax.plot(t[:len(clip)], clip[:, ci],
                    color=_COLORS.get(label, "gray"), linewidth=0.8)
            ax.set_ylabel(name, fontsize=9)
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.set_title(label, fontsize=11)
            if ci == n_cols_plot - 1:
                ax.set_xlabel("Time (s)")

    plt.suptitle(f"Temporal Trace — first {n_seconds:.0f} s from middle of sequence", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Wrist IMU data-statistics analysis")
    ap.add_argument("--config",       default="configs/base.yaml",
                    help="Base config YAML (needed for parquet sensor_columns)")
    ap.add_argument("--nymeria-dir",  default=None,
                    help="Directory of Nymeria *.pkl files")
    ap.add_argument("--parquet",      default=None,
                    help="Path to real-data parquet file/directory (loaders.py source)")
    ap.add_argument("--nymeria-n",    type=int,  default=50,
                    help="Max number of Nymeria pkl files to sample")
    ap.add_argument("--parquet-n",    type=int,  default=100_000,
                    help="Max number of parquet rows to sample")
    ap.add_argument("--fps",          type=int,  default=50)
    ap.add_argument("--out-dir",      default="data_vis/imu_stats")
    args = ap.parse_args()

    if args.nymeria_dir is None and args.parquet is None:
        ap.error("Provide at least --nymeria-dir or --parquet (or both).")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    datasets: dict[str, np.ndarray] = {}

    if args.nymeria_dir:
        datasets["Nymeria FLU"] = _load_nymeria_flu_samples(
            args.nymeria_dir, args.nymeria_n, args.fps
        )

    if args.parquet:
        from imu_lm.utils.helpers import load_yaml
        cfg = load_yaml(args.config)
        sensor_cols = cfg.get("data", {}).get("sensor_columns",
                      ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        datasets["Real Parquet"] = _load_parquet_wrist_samples(
            args.parquet, sensor_cols, args.parquet_n,
            cfg=cfg,
        )

    # ── Console statistics ─────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("  IMU DATA STATISTICS  (units: acc = m/s²,  gyro = rad/s)")
    print("═" * 80)

    for label, data in datasets.items():
        ch_stats  = _channel_stats(data, ALL_COLS)
        mag_stats = _magnitude_stats(data, label)

        _print_stats(ch_stats,  f"[{label}]  Per-channel stats  (N={len(data):,} frames)")
        _print_magnitude_stats(mag_stats, f"[{label}]  Magnitude stats")

    # ── Comparison summary (if both sources available) ─────────────────────
    if len(datasets) == 2:
        labels = list(datasets.keys())
        d0, d1 = datasets[labels[0]], datasets[labels[1]]
        print(f"\n{'─'*60}")
        print(f"  COMPARISON  {labels[0]}  vs  {labels[1]}")
        print(f"{'─'*60}")
        hdr = f"{'channel':<12}  {'mean diff':>10}  {'std ratio':>10}  {'rms ratio':>10}"
        print(hdr)
        print("─" * len(hdr))
        for ci, ch in enumerate(ALL_COLS):
            s0 = _channel_stats(d0, ALL_COLS)[ch]
            s1 = _channel_stats(d1, ALL_COLS)[ch]
            diff  = s1["mean"] - s0["mean"]
            sratio = s1["std"]  / (s0["std"]  + 1e-9)
            rratio = s1["rms"]  / (s0["rms"]  + 1e-9)
            print(f"{ch:<12}  {diff:>+10.4f}  {sratio:>10.3f}  {rratio:>10.3f}")
        print("─" * len(hdr))

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating plots …")

    _plot_channel_distributions(
        datasets, ALL_COLS,
        os.path.join(args.out_dir, "channel_distributions.png"),
    )
    _plot_magnitude_distributions(
        datasets,
        os.path.join(args.out_dir, "magnitude_distributions.png"),
    )
    _plot_correlation_heatmap(
        datasets, ALL_COLS,
        os.path.join(args.out_dir, "correlation_heatmap.png"),
    )
    _plot_psd(
        datasets, ALL_COLS, args.fps,
        os.path.join(args.out_dir, "psd.png"),
    )
    _plot_temporal_sample(
        datasets, ALL_COLS, args.fps,
        os.path.join(args.out_dir, "temporal_trace.png"),
    )

    print(f"\nAll outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
