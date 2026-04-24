"""Compare original 50 Hz real parquet IMU data against its 50→16→50 Hz degraded version.

Produces two output directories:
  <out_dir>/time/      – time-domain overlay plots (one per sensor channel)
  <out_dir>/freq/      – power spectral density overlay plots (one per channel)
  <out_dir>/diff/      – absolute difference signal (one per channel)
  <out_dir>/summary.png – grid of all channels × {time, PSD, diff}

Usage
-----
    python scripts/vis_realdata_16hz_degradation.py \\
        --parquet  /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \\
        --dataset  pamap2 \\               # any dataset in the parquet
        [--subject  <subject_id>]  \\      # optional filter
        [--session  <session_id>]  \\      # optional filter; first session used if omitted
        [--n-seconds  10]          \\      # how many seconds to plot (default 10)
        [--out-dir  data_vis/realdata_16hz_comparison]
"""

from __future__ import annotations

import argparse
import logging
import os
from math import gcd

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pyarrow.dataset as ds
import pyarrow.compute as pc
from scipy.signal import resample_poly, welch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_SOURCE_HZ = 50
_DEGRADED_HZ = 16
_SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
_CHANNEL_LABELS = ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
_CHANNEL_UNITS  = ["m/s²"] * 3 + ["rad/s"] * 3


# ---------------------------------------------------------------------------
# Core degradation  (mirrors realdata_16hz_to_50hz_loader.py)
# ---------------------------------------------------------------------------

def _degrade(X: np.ndarray, source_hz: int = _SOURCE_HZ,
             low_hz: int = _DEGRADED_HZ) -> np.ndarray:
    """50 Hz → 16 Hz → 50 Hz via anti-aliased polyphase resampling."""
    g_down = gcd(source_hz, low_hz)
    X_low = resample_poly(X, low_hz // g_down, source_hz // g_down, axis=0).astype(np.float32)

    g_up = gcd(low_hz, source_hz)
    X_out = resample_poly(X_low, source_hz // g_up, low_hz // g_up, axis=0).astype(np.float32)

    return X_out[: len(X)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session(
    parquet_path: str,
    dataset_name: str,
    subject_id: str | None,
    session_id: str | None,
    n_seconds: int,
) -> tuple[np.ndarray, str]:
    """Return X [N, 6] float32 for one session from the parquet, capped at n_seconds."""
    pa_ds = ds.dataset(parquet_path, format="parquet")

    filt = pc.field("dataset") == dataset_name
    if subject_id:
        filt = filt & (pc.field("subject_id") == subject_id)
    if session_id:
        filt = filt & (pc.field("session_id") == session_id)

    load_cols = _SENSOR_COLS + ["subject_id", "session_id", "timestamp_ns"]
    table = pa_ds.to_table(columns=load_cols, filter=filt)
    if len(table) == 0:
        raise RuntimeError(f"No rows found for dataset={dataset_name!r} subject={subject_id!r} session={session_id!r}")

    df = table.to_pandas().sort_values("timestamp_ns").reset_index(drop=True)

    # If multiple sessions returned (no session filter), take the first one
    first_sess = df["session_id"].iloc[0]
    df = df[df["session_id"] == first_sess].reset_index(drop=True)

    sid = f"{df['subject_id'].iloc[0]}/{first_sess}"
    logger.info("Loaded session %s/%s  rows=%d", dataset_name, sid, len(df))

    max_rows = n_seconds * _SOURCE_HZ
    X = df[_SENSOR_COLS].to_numpy(dtype=np.float32)[:max_rows]
    return X, sid


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_time(ax: plt.Axes, t: np.ndarray, orig: np.ndarray, deg: np.ndarray,
               label: str, unit: str):
    ax.plot(t, orig, color="#1f77b4", lw=1.0, alpha=0.85, label="50 Hz (original)")
    ax.plot(t, deg,  color="#d62728", lw=1.0, alpha=0.75, label="50→16→50 Hz", linestyle="--")
    ax.set_ylabel(f"{label}\n[{unit}]", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right")


def _plot_psd(ax: plt.Axes, orig: np.ndarray, deg: np.ndarray,
              label: str, unit: str, fs: int = _SOURCE_HZ):
    f_o, p_o = welch(orig, fs=fs, nperseg=min(256, len(orig)))
    f_d, p_d = welch(deg,  fs=fs, nperseg=min(256, len(deg)))
    ax.semilogy(f_o, p_o, color="#1f77b4", lw=1.2, label="50 Hz (original)")
    ax.semilogy(f_d, p_d, color="#d62728", lw=1.2, label="50→16→50 Hz", linestyle="--")
    ax.axvline(_DEGRADED_HZ / 2, color="orange", lw=1.0, linestyle=":", label=f"Nyquist @ {_DEGRADED_HZ/2:.0f} Hz")
    ax.set_ylabel(f"PSD\n[{unit}²/Hz]", fontsize=7)
    ax.set_xlabel("Frequency [Hz]", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper right")


def _plot_diff(ax: plt.Axes, t: np.ndarray, orig: np.ndarray, deg: np.ndarray,
               label: str, unit: str):
    diff = np.abs(orig - deg)
    ax.plot(t, diff, color="#2ca02c", lw=0.9)
    ax.set_ylabel(f"|Δ| {label}\n[{unit}]", fontsize=7)
    ax.tick_params(labelsize=7)
    rms = float(np.sqrt(np.mean(diff**2)))
    ax.set_title(f"RMS error: {rms:.4f} {unit}", fontsize=7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare 50 Hz vs 50→16→50 Hz IMU data")
    parser.add_argument("--parquet",    required=True,  help="Path to IMULM master parquet")
    parser.add_argument("--dataset",    required=True,  help="Dataset name (e.g. pamap2)")
    parser.add_argument("--subject",    default=None,   help="Subject ID filter (optional)")
    parser.add_argument("--session",    default=None,   help="Session ID filter (optional)")
    parser.add_argument("--n-seconds",  type=int, default=10, help="Seconds to visualize (default 10)")
    parser.add_argument("--out-dir",    default="data_vis/realdata_16hz_comparison")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "time"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "freq"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "diff"), exist_ok=True)

    # --- load ---
    X_orig, session_label = load_session(
        args.parquet, args.dataset, args.subject, args.session, args.n_seconds
    )
    X_deg = _degrade(X_orig)
    N = len(X_orig)
    t = np.arange(N) / _SOURCE_HZ   # seconds

    logger.info("Signal length: %d samples (%.1f s)  after degradation: %d samples",
                N, N / _SOURCE_HZ, len(X_deg))
    logger.info("Nyquist of degraded signal: %.1f Hz", _DEGRADED_HZ / 2)

    n_ch = len(_SENSOR_COLS)

    # -----------------------------------------------------------------------
    # Per-channel individual plots
    # -----------------------------------------------------------------------
    for ci, (col, label, unit) in enumerate(zip(_SENSOR_COLS, _CHANNEL_LABELS, _CHANNEL_UNITS)):
        orig_ch = X_orig[:, ci]
        deg_ch  = X_deg[:, ci]

        # Time domain
        fig, ax = plt.subplots(figsize=(10, 2.8))
        _plot_time(ax, t, orig_ch, deg_ch, label, unit)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_title(f"{args.dataset}  |  {session_label}  |  {label}  —  time domain", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "time", f"{col}_time.png"), dpi=150)
        plt.close(fig)

        # PSD
        fig, ax = plt.subplots(figsize=(7, 3))
        _plot_psd(ax, orig_ch, deg_ch, label, unit)
        ax.set_title(f"{args.dataset}  |  {session_label}  |  {label}  —  PSD", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "freq", f"{col}_psd.png"), dpi=150)
        plt.close(fig)

        # Diff
        fig, ax = plt.subplots(figsize=(10, 2.5))
        _plot_diff(ax, t, orig_ch, deg_ch, label, unit)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.set_title(f"{args.dataset}  |  {session_label}  |  {label}  —  absolute error", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "diff", f"{col}_diff.png"), dpi=150)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Summary grid  (n_ch rows × 3 cols: time | PSD | diff)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 2.8 * n_ch))
    fig.suptitle(
        f"50 Hz vs 50→{_DEGRADED_HZ}→50 Hz  |  {args.dataset}  |  {session_label}\n"
        f"(Nyquist of degraded signal = {_DEGRADED_HZ / 2:.0f} Hz)",
        fontsize=11, y=1.01,
    )
    gs = gridspec.GridSpec(n_ch, 3, figure=fig, hspace=0.55, wspace=0.35)

    for ci, (col, label, unit) in enumerate(zip(_SENSOR_COLS, _CHANNEL_LABELS, _CHANNEL_UNITS)):
        orig_ch = X_orig[:, ci]
        deg_ch  = X_deg[:, ci]

        ax_t = fig.add_subplot(gs[ci, 0])
        _plot_time(ax_t, t, orig_ch, deg_ch, label, unit)
        if ci == 0:
            ax_t.set_title("Time domain", fontsize=8)
        if ci == n_ch - 1:
            ax_t.set_xlabel("Time [s]", fontsize=7)

        ax_f = fig.add_subplot(gs[ci, 1])
        _plot_psd(ax_f, orig_ch, deg_ch, label, unit)
        if ci == 0:
            ax_f.set_title("Power Spectral Density", fontsize=8)

        ax_d = fig.add_subplot(gs[ci, 2])
        _plot_diff(ax_d, t, orig_ch, deg_ch, label, unit)
        if ci == 0:
            ax_d.set_title("Absolute error |orig − degraded|", fontsize=8)
        if ci == n_ch - 1:
            ax_d.set_xlabel("Time [s]", fontsize=7)

    # Global legend
    handles, lbls = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=8, framealpha=0.9)

    out_path = os.path.join(args.out_dir, "summary.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Summary grid saved → %s", out_path)

    # -----------------------------------------------------------------------
    # Print per-channel RMS error table
    # -----------------------------------------------------------------------
    print("\n=== Per-channel RMS error (original vs degraded) ===")
    print(f"{'Channel':<10}  {'Unit':<8}  {'RMS error':>12}  {'Rel error (%)':>14}")
    print("-" * 52)
    for ci, (col, label, unit) in enumerate(zip(_SENSOR_COLS, _CHANNEL_LABELS, _CHANNEL_UNITS)):
        orig_ch = X_orig[:, ci]
        deg_ch  = X_deg[:, ci]
        rms_err  = float(np.sqrt(np.mean((orig_ch - deg_ch) ** 2)))
        rms_orig = float(np.sqrt(np.mean(orig_ch ** 2))) + 1e-9
        rel_err  = 100.0 * rms_err / rms_orig
        print(f"{label:<10}  {unit:<8}  {rms_err:>12.5f}  {rel_err:>13.2f}%")
    print()


if __name__ == "__main__":
    main()
