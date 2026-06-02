#!/usr/bin/env python3
"""Compare spectral content across all IMU data sources.

Shows mean PSD and band energy fractions for:
  - Realdata (parquet): pamap2, recofit, samosa, shoaib, ut_watch, utd_mhad, wisdm
  - OCDetect: wrist IMU during daily activities
  - Nymeria real: real wrist IMU (FLU frame)
  - Nymeria syn: synthesized from mocap xsens_pose (finite-diff kinematics)

Also saves a second figure showing how a single window is decomposed into
frequency bands via FFT/IFFT, to illustrate what each band contains.

Usage:
    MPLBACKEND=Agg python scripts/vis_dataset_frequency.py \
        --parquet       /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
        --ocdetect-dir  /mnt/nas2/yiquan_data/preprocessed \
        --nymeria-dir   /mnt/nas2/processed_data_body_Ty_rwrist \
        --n-per-ds      300 \
        --out           data_vis/dataset_frequency_comparison.png
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

WIN = 128
FS  = 50.0

REALDATA_SETS = ["pamap2", "recofit", "samosa", "shoaib", "ut_watch", "utd_mhad", "wisdm"]
REAL_IMU_COLS  = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
OCD_IMU_COLS   = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]

BANDS       = [(0, 1), (1, 3), (3, 8), (8, 25)]
BAND_LABELS = ["0–1 Hz\n(gravity/tilt)", "1–3 Hz\n(walk/run)", "3–8 Hz\n(fast motion)", "8–25 Hz\n(vibration)"]
BAND_COLORS = ["#e74c3c", "#e67e22", "#27ae60", "#2980b9"]

GROUP_COLORS = {
    "pamap2":       "#1f77b4",
    "recofit":      "#ff7f0e",
    "samosa":       "#2ca02c",
    "shoaib":       "#d62728",
    "ut_watch":     "#9467bd",
    "utd_mhad":     "#8c564b",
    "wisdm":        "#e377c2",
    "ocdetect":     "#bcbd22",
    "nymeria_real": "#17becf",
    "nymeria_syn":  "#7f7f7f",
}


# ── Spectral utilities ────────────────────────────────────────────────────────

def mean_psd(windows: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """windows [N, C, T] → (freqs [F], mean_power [C, F])."""
    T     = windows.shape[2]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    power = np.abs(np.fft.rfft(windows, axis=2)) ** 2   # [N, C, F]
    return freqs, power.mean(axis=0)                     # [C, F]


def band_energy_fraction(windows: np.ndarray, fs: float) -> np.ndarray:
    """windows [N, C, T] → [N, n_bands] fraction of total energy in each band.

    Uses SUM over bins (total energy), not mean (power density).
    Each band covers different numbers of bins:
      WIN=128, FS=50 → 2.56 bins/Hz
      0–1 Hz:  ~3 bins,  1–3 Hz: ~5 bins,  3–8 Hz: ~13 bins,  8–25 Hz: ~44 bins
    """
    T     = windows.shape[2]
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    power = np.abs(np.fft.rfft(windows, axis=2)) ** 2   # [N, C, F]
    power_ch_mean = power.mean(axis=1)                   # [N, F] mean over channels

    band_e = np.zeros((len(windows), len(BANDS)), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band_e[:, i] = power_ch_mean[:, mask].sum(axis=1)   # total energy in band

    total = band_e.sum(axis=1, keepdims=True) + 1e-8
    return band_e / total


def decompose_by_band(signal_1d: np.ndarray, fs: float) -> dict:
    """Decompose a 1D signal into frequency bands via FFT/IFFT.

    Returns dict with 'original', 'bands' list, 'band_fractions', 'freqs', 'power'.
    """
    T     = len(signal_1d)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    F     = np.fft.rfft(signal_1d)
    power = np.abs(F) ** 2
    total_power = power.sum() + 1e-12

    bands_out = []
    fractions = []
    for lo, hi in BANDS:
        mask    = (freqs >= lo) & (freqs < hi)
        F_band  = np.zeros_like(F)
        F_band[mask] = F[mask]
        band_signal  = np.fft.irfft(F_band, n=T)
        fraction     = power[mask].sum() / total_power
        bands_out.append(band_signal)
        fractions.append(float(fraction))

    return {
        "original":       signal_1d,
        "bands":          bands_out,          # list of [T] arrays
        "band_fractions": fractions,          # fraction of total energy
        "freqs":          freqs,
        "power":          power,
    }


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_realdata(parquet_path: str, n_per_ds: int, rng) -> dict[str, np.ndarray]:
    """Returns {dataset_name: [N, 6, WIN]}."""
    out = {}
    needed = ["dataset", "session_id", "timestamp_ns"] + REAL_IMU_COLS
    for ds in REALDATA_SETS:
        df = pd.read_parquet(parquet_path, columns=needed,
                             filters=[("dataset", "==", ds)])
        if df.empty:
            print(f"  [realdata] {ds}: not found, skipping")
            continue
        df   = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)
        vals = df[REAL_IMU_COLS].values.T.astype(np.float32)  # [6, L]
        L    = vals.shape[1]
        if L < WIN:
            continue
        n       = min(n_per_ds, L - WIN + 1)
        starts  = rng.choice(L - WIN + 1, size=n, replace=False)
        windows = np.stack([vals[:, s:s+WIN] for s in starts])
        out[ds] = windows
        print(f"  [realdata] {ds}: {n} windows")
    return out


def load_ocdetect(data_dir: str, n_per_subj: int, rng) -> np.ndarray:
    """Returns [N, 6, WIN]."""
    files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.csv")))
    if not files:
        print(f"  [OCDetect] no files found in {data_dir}")
        return np.empty((0, 6, WIN), dtype=np.float32)

    from collections import defaultdict
    subj_files: dict[str, list] = defaultdict(list)
    for f in files:
        subj_id = "_".join(os.path.basename(f).split("_")[:2])
        subj_files[subj_id].append(f)

    all_wins = []
    for subj, sfiles in sorted(subj_files.items()):
        frames = []
        rows_needed = n_per_subj * WIN * 3
        for fpath in sfiles[:3]:
            try:
                frames.append(pd.read_csv(fpath, usecols=OCD_IMU_COLS, nrows=rows_needed))
            except Exception:
                continue
            if sum(len(f) for f in frames) >= rows_needed:
                break
        if not frames:
            continue
        df   = pd.concat(frames, ignore_index=True)
        vals = df[OCD_IMU_COLS].values.T.astype(np.float32)
        L    = vals.shape[1]
        if L < WIN:
            continue
        n      = min(n_per_subj, L - WIN + 1)
        starts = rng.choice(L - WIN + 1, size=n, replace=False)
        all_wins.extend(vals[:, s:s+WIN] for s in starts)

    if not all_wins:
        return np.empty((0, 6, WIN), dtype=np.float32)
    out = np.stack(all_wins)
    print(f"  [OCDetect] {len(out)} windows from {len(subj_files)} subjects")
    return out


def _synth_imu_from_pkl(d: dict, add_gravity: bool = True) -> np.ndarray | None:
    """Synthesize acc+gyro [T, 6] from xsens_pose via finite differences."""
    try:
        xsens = d["gt_data"]["xsens_pose"]
        if hasattr(xsens, "numpy"):
            xsens = xsens.numpy()
        xsens = np.array(xsens, dtype=np.float64)    # [T, 24, 4, 4]
        T_total = xsens.shape[0]
        if T_total < 3:
            return None

        pos = xsens[:, 9, :3, 3]    # [T, 3] right wrist world position
        rot = xsens[:, 9, :3, :3]   # [T, 3, 3] R_world_from_wrist

        fps = float(d.get("pose_sampling_rate_hz", 50.0))

        # Acceleration via central difference (m/s²)
        acc_world = np.zeros_like(pos)
        acc_world[1:-1] = (pos[2:] - 2*pos[1:-1] + pos[:-2]) * (fps ** 2)
        acc_world[0]    = acc_world[1]
        acc_world[-1]   = acc_world[-2]
        if add_gravity:
            acc_world[:, 1] += 9.81   # Y-up convention

        # Angular velocity via small-angle approximation on incremental rotation
        gyro_world = np.zeros((T_total, 3), dtype=np.float64)
        for i in range(T_total - 1):
            dR = rot[i+1] @ rot[i].T
            gyro_world[i] = np.array([
                dR[2, 1] - dR[1, 2],
                dR[0, 2] - dR[2, 0],
                dR[1, 0] - dR[0, 1],
            ]) * 0.5 * fps
        gyro_world[-1] = gyro_world[-2]

        imu = np.concatenate([acc_world, gyro_world], axis=1).astype(np.float32)
        return imu
    except Exception:
        return None


def load_nymeria(pkl_dir: str, n_per_file: int, rng,
                 mode: str = "real") -> np.ndarray:
    """mode='real' reads wrist_imu_data; mode='syn' synthesizes from xsens_pose."""
    files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    if not files:
        print(f"  [nymeria_{mode}] no pkl files in {pkl_dir}")
        return np.empty((0, 6, WIN), dtype=np.float32)

    indices = rng.choice(len(files), size=min(30, len(files)), replace=False)
    files   = [files[i] for i in indices]

    all_wins = []
    for fpath in files:
        try:
            d = pickle.load(open(fpath, "rb"))
        except Exception:
            continue

        if mode == "real":
            if not d.get("has_wrist_imu", False):
                continue
            acc  = np.array(d["wrist_imu_data"]["accel"], dtype=np.float32)
            gyro = np.array(d["wrist_imu_data"]["gyro"],  dtype=np.float32)
            imu  = np.concatenate([acc, gyro], axis=1)
        else:
            imu = _synth_imu_from_pkl(d)
            if imu is None:
                continue

        L    = imu.shape[0]
        if L < WIN:
            continue
        vals   = imu.T
        n      = min(n_per_file, L - WIN + 1)
        starts = rng.choice(L - WIN + 1, size=n, replace=False)
        all_wins.extend(vals[:, s:s+WIN] for s in starts)

    if not all_wins:
        return np.empty((0, 6, WIN), dtype=np.float32)
    out = np.stack(all_wins)
    print(f"  [nymeria_{mode}] {len(out)} windows from {len(files)} files")
    return out


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(groups: dict[str, np.ndarray], out_path: str):
    """Main comparison figure: PSD + heatmap + bar chart."""
    freqs = np.fft.rfftfreq(WIN, d=1.0 / FS)
    names = [n for n in groups if len(groups[n]) > 0]

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("IMU Spectral Content Comparison Across Dataset Sources",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax_acc  = fig.add_subplot(gs[0, 0:2])
    ax_gyro = fig.add_subplot(gs[0, 2])

    band_data: dict[str, np.ndarray] = {}

    def smooth(x, w=3):
        return np.convolve(x, np.ones(w) / w, mode="same")

    for name in names:
        windows = groups[name]
        color   = GROUP_COLORS.get(name, "#333333")
        _, psd  = mean_psd(windows, FS)               # [6, F]
        ax_acc.semilogy(freqs, smooth(psd[:3].mean(0)), color=color, lw=1.5,
                        label=name, alpha=0.85)
        ax_gyro.semilogy(freqs, smooth(psd[3:].mean(0)), color=color, lw=1.5,
                         label=name, alpha=0.85)
        band_data[name] = band_energy_fraction(windows, FS).mean(axis=0)

    for ax, title in [(ax_acc,  "Accelerometer PSD (mean over acc x/y/z)"),
                      (ax_gyro, "Gyroscope PSD (mean over gyro x/y/z)")]:
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel("Power (log scale)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for v in [1, 3, 8]:
            ax.axvline(v, color="k", lw=0.5, ls="--", alpha=0.4)
        ax.set_xlim(0, 25)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    ax_acc.legend(fontsize=7, ncol=2, framealpha=0.7, loc="upper right")

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 0:2])
    heat_mat = np.array([band_data[n] for n in names])
    im = ax_heat.imshow(heat_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(BANDS)))
    ax_heat.set_xticklabels([b.replace("\n", " ") for b in BAND_LABELS], fontsize=9)
    ax_heat.set_yticks(range(len(names)))
    ax_heat.set_yticklabels(names, fontsize=8)
    ax_heat.set_title("Mean spectral energy fraction per band",
                      fontsize=10, fontweight="bold")
    for i, name in enumerate(names):
        for j in range(len(BANDS)):
            ax_heat.text(j, i, f"{heat_mat[i,j]:.2f}", ha="center", va="center",
                         fontsize=8, color="white" if heat_mat[i,j] > 0.6 else "black")
    plt.colorbar(im, ax=ax_heat, fraction=0.03).ax.tick_params(labelsize=7)

    # Bar chart: 0-1 Hz fraction
    ax_bar = fig.add_subplot(gs[1, 2])
    low_fracs = [band_data[n][0] for n in names]
    colors    = [GROUP_COLORS.get(n, "#333") for n in names]
    bars = ax_bar.barh(names, low_fracs, color=colors, edgecolor="k", lw=0.5)
    ax_bar.set_xlabel("Fraction of energy in 0–1 Hz", fontsize=9)
    ax_bar.set_title("Low-frequency dominance\n(high = gravity-dominated)",
                     fontsize=10, fontweight="bold")
    ax_bar.set_xlim(0, 1)
    ax_bar.axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_bar.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, low_fracs):
        ax_bar.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=8)
    ax_bar.tick_params(labelsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


def plot_fft_decomposition(groups: dict[str, np.ndarray], out_path: str):
    """Show how one representative window from each source decomposes into bands."""
    names = [n for n in groups if len(groups[n]) > 0]
    n_src = len(names)
    t     = np.arange(WIN) / FS   # time axis in seconds

    # Each source gets one row of subplots: original + 4 band components
    n_cols = 5   # original + 4 bands
    fig, axes = plt.subplots(n_src, n_cols,
                             figsize=(n_cols * 3.2, n_src * 2.2 + 1),
                             squeeze=False)
    fig.suptitle("FFT Band Decomposition — acc_x channel (one window per source)\n"
                 "Sum of all bands = original signal exactly",
                 fontsize=11, fontweight="bold")

    col_titles = ["Original\n(acc_x)"] + [f"{lo}–{hi} Hz" for lo, hi in BANDS]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, fontweight="bold",
                               color=BAND_COLORS[col - 1] if col > 0 else "black")

    for row, name in enumerate(names):
        windows = groups[name]
        # pick the window with most dynamic range (most interesting)
        acc_ch = windows[:, 0, :]   # [N, T] acc_x channel
        chosen = acc_ch[np.ptp(acc_ch, axis=1).argmax()]  # window with largest range

        decomp = decompose_by_band(chosen, FS)
        color  = GROUP_COLORS.get(name, "#333")

        # Col 0: original
        axes[row, 0].plot(t, decomp["original"], color=color, lw=1.0)
        axes[row, 0].set_ylabel(name, fontsize=8, fontweight="bold", color=color)

        # Cols 1-4: each band
        for col, (band_sig, frac) in enumerate(
                zip(decomp["bands"], decomp["band_fractions"]), start=1):
            axes[row, col].plot(t, band_sig, color=BAND_COLORS[col - 1], lw=1.0)
            axes[row, col].set_title(f"{frac*100:.1f}% of energy",
                                     fontsize=7, pad=2, color=BAND_COLORS[col - 1])

        # Uniform y-scale per row: use original signal range
        y_max = np.abs(decomp["original"]).max() * 1.2 + 1e-6
        for col in range(n_cols):
            axes[row, col].set_ylim(-y_max, y_max)
            axes[row, col].tick_params(labelsize=6)
            axes[row, col].grid(alpha=0.3)
            if row < n_src - 1:
                axes[row, col].set_xticklabels([])

    for col in range(n_cols):
        axes[-1, col].set_xlabel("Time (s)", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet",      default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--ocdetect-dir", default="/mnt/nas2/yiquan_data/preprocessed")
    p.add_argument("--nymeria-dir",  default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--n-per-ds",     type=int, default=300,
                   help="Windows to sample per dataset/source")
    p.add_argument("--no-realdata",  action="store_true")
    p.add_argument("--no-ocdetect",  action="store_true")
    p.add_argument("--no-nymeria",   action="store_true")
    p.add_argument("--out", default="data_vis/dataset_frequency_comparison.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(42)

    groups: dict[str, np.ndarray] = {}

    if not args.no_realdata and os.path.exists(args.parquet):
        print("Loading realdata ...")
        groups.update(load_realdata(args.parquet, args.n_per_ds, rng))
    elif not args.no_realdata:
        print(f"[warn] parquet not found: {args.parquet}")

    if not args.no_ocdetect and os.path.exists(args.ocdetect_dir):
        print("Loading OCDetect ...")
        groups["ocdetect"] = load_ocdetect(
            args.ocdetect_dir, max(1, args.n_per_ds // 22), rng)
    elif not args.no_ocdetect:
        print(f"[warn] OCDetect dir not found: {args.ocdetect_dir}")

    if not args.no_nymeria and os.path.exists(args.nymeria_dir):
        print("Loading Nymeria real ...")
        groups["nymeria_real"] = load_nymeria(
            args.nymeria_dir, max(1, args.n_per_ds // 30), rng, mode="real")
        print("Loading Nymeria syn ...")
        groups["nymeria_syn"] = load_nymeria(
            args.nymeria_dir, max(1, args.n_per_ds // 30), rng, mode="syn")
    elif not args.no_nymeria:
        print(f"[warn] Nymeria dir not found: {args.nymeria_dir}")

    if not groups:
        sys.exit("No data loaded.")

    # Summary table
    print("\n── Band energy fractions (corrected: sum over bins) ──")
    print(f"  {'Dataset':>14}  {'0-1Hz':>7}  {'1-3Hz':>7}  {'3-8Hz':>7}  {'8-25Hz':>7}  {'N':>5}")
    print("  " + "-" * 58)
    for name, windows in groups.items():
        if len(windows) == 0:
            continue
        frac = band_energy_fraction(windows, FS).mean(axis=0)
        n_bins = [((np.fft.rfftfreq(WIN, 1/FS) >= lo) & (np.fft.rfftfreq(WIN, 1/FS) < hi)).sum()
                  for lo, hi in BANDS]
        print(f"  {name:>14}  {frac[0]:7.3f}  {frac[1]:7.3f}  {frac[2]:7.3f}  {frac[3]:7.3f}  {len(windows):5d}")

    print(f"\n  Bins per band at WIN={WIN}, FS={FS}:")
    freqs = np.fft.rfftfreq(WIN, 1/FS)
    for lo, hi in BANDS:
        n = ((freqs >= lo) & (freqs < hi)).sum()
        print(f"    {lo}–{hi} Hz: {n} bins")

    # Plots
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    plot_comparison(groups, args.out)

    decomp_path = args.out.replace(".png", "_decomposition.png")
    plot_fft_decomposition(groups, decomp_path)


if __name__ == "__main__":
    main()
