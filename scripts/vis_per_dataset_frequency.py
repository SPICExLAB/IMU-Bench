#!/usr/bin/env python3
"""Per-dataset frequency distribution: mean PSD ± std and band energy fractions.

One row per dataset. Columns:
  1. Accelerometer mean PSD ± 1 std (log scale)
  2. Gyroscope mean PSD ± 1 std (log scale)
  3. Band energy fraction bar chart

Usage:
    MPLBACKEND=Agg python scripts/vis_per_dataset_frequency.py \
        --parquet       /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
        --ocdetect-dir  /mnt/nas2/yiquan_data/preprocessed \
        --nymeria-dir   /mnt/nas2/processed_data_body_Ty_rwrist \
        --n-per-ds      300 \
        --out           data_vis/freq_dist_per_dataset.png
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
REAL_IMU_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
OCD_IMU_COLS  = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]

BANDS       = [(0, 1), (1, 3), (3, 8), (8, 25)]
BAND_LABELS = ["0–1 Hz", "1–3 Hz", "3–8 Hz", "8–25 Hz"]
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

def compute_psd(windows: np.ndarray, fs: float):
    """windows [N, C, T] → (freqs [F], mean_psd [C, F], std_psd [C, F])."""
    power = np.abs(np.fft.rfft(windows, axis=2)) ** 2   # [N, C, F]
    freqs = np.fft.rfftfreq(windows.shape[2], d=1.0 / fs)
    return freqs, power.mean(axis=0), power.std(axis=0)


def compute_band_fractions(windows: np.ndarray, fs: float) -> np.ndarray:
    """windows [N, C, T] → [n_bands] mean energy fraction using sum over bins."""
    freqs = np.fft.rfftfreq(windows.shape[2], d=1.0 / fs)
    power = np.abs(np.fft.rfft(windows, axis=2)) ** 2   # [N, C, F]
    power_ch = power.mean(axis=1)                        # [N, F] mean over channels
    band_e = np.zeros((len(windows), len(BANDS)), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS):
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            band_e[:, i] = power_ch[:, mask].sum(axis=1)
    total = band_e.sum(axis=1, keepdims=True) + 1e-8
    fracs = band_e / total
    return fracs.mean(axis=0), fracs.std(axis=0)   # [n_bands] mean and std


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_realdata(parquet_path: str, n_per_ds: int, rng) -> dict[str, np.ndarray]:
    out = {}
    needed = ["dataset", "session_id", "timestamp_ns"] + REAL_IMU_COLS
    for ds in REALDATA_SETS:
        df = pd.read_parquet(parquet_path, columns=needed,
                             filters=[("dataset", "==", ds)])
        if df.empty:
            continue
        df   = df.sort_values(["session_id", "timestamp_ns"]).reset_index(drop=True)
        vals = df[REAL_IMU_COLS].values.T.astype(np.float32)
        L    = vals.shape[1]
        if L < WIN:
            continue
        n      = min(n_per_ds, L - WIN + 1)
        starts = rng.choice(L - WIN + 1, size=n, replace=False)
        out[ds] = np.stack([vals[:, s:s+WIN] for s in starts])
        print(f"  {ds}: {n} windows")
    return out


def load_ocdetect(data_dir: str, n_per_subj: int, rng) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(data_dir, "OCDetect_*.csv")))
    if not files:
        return np.empty((0, 6, WIN), dtype=np.float32)
    from collections import defaultdict
    subj_files: dict = defaultdict(list)
    for f in files:
        subj_files["_".join(os.path.basename(f).split("_")[:2])].append(f)
    all_wins = []
    for subj, sfiles in sorted(subj_files.items()):
        frames, rows_needed = [], n_per_subj * WIN * 3
        for fpath in sfiles[:3]:
            try:
                frames.append(pd.read_csv(fpath, usecols=OCD_IMU_COLS, nrows=rows_needed))
            except Exception:
                continue
            if sum(len(f) for f in frames) >= rows_needed:
                break
        if not frames:
            continue
        vals = pd.concat(frames, ignore_index=True)[OCD_IMU_COLS].values.T.astype(np.float32)
        L    = vals.shape[1]
        if L < WIN:
            continue
        n      = min(n_per_subj, L - WIN + 1)
        starts = rng.choice(L - WIN + 1, size=n, replace=False)
        all_wins.extend(vals[:, s:s+WIN] for s in starts)
    if not all_wins:
        return np.empty((0, 6, WIN), dtype=np.float32)
    out = np.stack(all_wins)
    print(f"  ocdetect: {len(out)} windows")
    return out


def load_nymeria(pkl_dir: str, n_per_file: int, rng, mode: str = "real") -> np.ndarray:
    files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    if not files:
        return np.empty((0, 6, WIN), dtype=np.float32)
    idx   = rng.choice(len(files), size=min(30, len(files)), replace=False)
    files = [files[i] for i in idx]
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
            imu = _synth_imu(d)
            if imu is None:
                continue
        L = imu.shape[0]
        if L < WIN:
            continue
        vals   = imu.T
        n      = min(n_per_file, L - WIN + 1)
        starts = rng.choice(L - WIN + 1, size=n, replace=False)
        all_wins.extend(vals[:, s:s+WIN] for s in starts)
    if not all_wins:
        return np.empty((0, 6, WIN), dtype=np.float32)
    out = np.stack(all_wins)
    print(f"  nymeria_{mode}: {len(out)} windows")
    return out


def _synth_imu(d: dict) -> np.ndarray | None:
    try:
        xsens = d["gt_data"]["xsens_pose"]
        if hasattr(xsens, "numpy"):
            xsens = xsens.numpy()
        xsens = np.array(xsens, dtype=np.float64)
        if xsens.shape[0] < 3:
            return None
        pos = xsens[:, 9, :3, 3]
        rot = xsens[:, 9, :3, :3]
        fps = float(d.get("pose_sampling_rate_hz", 50.0))
        acc = np.zeros_like(pos)
        acc[1:-1] = (pos[2:] - 2*pos[1:-1] + pos[:-2]) * fps**2
        acc[0]    = acc[1];   acc[-1] = acc[-2]
        acc[:, 1] += 9.81
        gyro = np.zeros_like(pos)
        for i in range(len(pos) - 1):
            dR = rot[i+1] @ rot[i].T
            gyro[i] = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]]) * 0.5 * fps
        gyro[-1] = gyro[-2]
        return np.concatenate([acc, gyro], axis=1).astype(np.float32)
    except Exception:
        return None


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_per_dataset(groups: dict[str, np.ndarray], out_path: str):
    names = [n for n, w in groups.items() if len(w) > 0]
    n_ds  = len(names)
    freqs = np.fft.rfftfreq(WIN, d=1.0 / FS)

    fig = plt.figure(figsize=(16, 2.8 * n_ds + 1))
    fig.suptitle("Per-Dataset IMU Frequency Distribution\n"
                 "Left: Accelerometer PSD  |  Centre: Gyroscope PSD  |  Right: Band energy fractions",
                 fontsize=12, fontweight="bold", y=1.001)

    gs = gridspec.GridSpec(n_ds, 3, figure=fig, hspace=0.55, wspace=0.35,
                           width_ratios=[2, 2, 1.2])

    def smooth(x, w=3):
        return np.convolve(x, np.ones(w) / w, mode="same")

    for row, name in enumerate(names):
        windows = groups[name]
        color   = GROUP_COLORS.get(name, "#333333")
        freqs_out, psd_mean, psd_std = compute_psd(windows, FS)
        frac_mean, frac_std          = compute_band_fractions(windows, FS)

        for col, (ch_slice, ch_label) in enumerate([
                (slice(0, 3), "Accelerometer (acc x/y/z)"),
                (slice(3, 6), "Gyroscope (gyro x/y/z)"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            m  = psd_mean[ch_slice].mean(axis=0)   # mean over channels
            s  = psd_std[ch_slice].mean(axis=0)

            ax.semilogy(freqs_out, smooth(m),     color=color, lw=1.8)
            ax.fill_between(freqs_out,
                            np.maximum(smooth(m - s), 1e-10),
                            smooth(m + s),
                            color=color, alpha=0.2)

            # Band boundary lines
            for v in [1, 3, 8]:
                ax.axvline(v, color="k", lw=0.5, ls="--", alpha=0.35)

            ax.set_xlim(0, 25)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25)

            if row == 0:
                ax.set_title(ch_label, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(name, fontsize=9, fontweight="bold", color=color,
                              labelpad=4)
            if row == n_ds - 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=8)
            else:
                ax.set_xticklabels([])

            # Annotate band fractions on the PSD plot
            band_x = [0.5, 2.0, 5.5, 16.5]
            y_ann  = ax.get_ylim()[0] * 10
            for bx, frac, bc in zip(band_x, frac_mean, BAND_COLORS):
                ax.text(bx, m.max() * 0.3, f"{frac*100:.0f}%",
                        fontsize=6.5, ha="center", color=bc, fontweight="bold")

        # Band fraction bar chart (col 2)
        ax_bar = fig.add_subplot(gs[row, 2])
        bars = ax_bar.barh(BAND_LABELS, frac_mean * 100,
                           xerr=frac_std * 100,
                           color=BAND_COLORS, edgecolor="k", lw=0.4,
                           error_kw={"elinewidth": 0.8, "capsize": 2})
        ax_bar.set_xlim(0, 105)
        ax_bar.tick_params(labelsize=7)
        ax_bar.grid(axis="x", alpha=0.3)
        ax_bar.set_xlabel("% of energy", fontsize=7)
        for bar, val, std in zip(bars, frac_mean * 100, frac_std * 100):
            ax_bar.text(val + std + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=6.5)
        if row == 0:
            ax_bar.set_title("Band energy fraction\n(mean ± std)", fontsize=9, fontweight="bold")
        if row < n_ds - 1:
            ax_bar.set_xticklabels([])
            ax_bar.set_xlabel("")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet",      default="/mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet")
    p.add_argument("--ocdetect-dir", default="/mnt/nas2/yiquan_data/preprocessed")
    p.add_argument("--nymeria-dir",  default="/mnt/nas2/processed_data_body_Ty_rwrist")
    p.add_argument("--n-per-ds",     type=int, default=300)
    p.add_argument("--no-realdata",  action="store_true")
    p.add_argument("--no-ocdetect",  action="store_true")
    p.add_argument("--no-nymeria",   action="store_true")
    p.add_argument("--out",          default="data_vis/freq_dist_per_dataset.png")
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(42)
    groups: dict[str, np.ndarray] = {}

    if not args.no_realdata and os.path.exists(args.parquet):
        print("Loading realdata ...")
        groups.update(load_realdata(args.parquet, args.n_per_ds, rng))

    if not args.no_ocdetect and os.path.exists(args.ocdetect_dir):
        print("Loading OCDetect ...")
        groups["ocdetect"] = load_ocdetect(
            args.ocdetect_dir, max(1, args.n_per_ds // 22), rng)

    if not args.no_nymeria and os.path.exists(args.nymeria_dir):
        print("Loading Nymeria ...")
        groups["nymeria_real"] = load_nymeria(
            args.nymeria_dir, max(1, args.n_per_ds // 30), rng, mode="real")
        groups["nymeria_syn"] = load_nymeria(
            args.nymeria_dir, max(1, args.n_per_ds // 30), rng, mode="syn")

    if not groups:
        sys.exit("No data loaded.")

    print("\n── Band energy fractions ──")
    print(f"  {'Dataset':>14}  {'0-1Hz':>7}  {'1-3Hz':>7}  {'3-8Hz':>7}  {'8-25Hz':>7}  N")
    print("  " + "-" * 58)
    for name, windows in groups.items():
        if len(windows) == 0:
            continue
        m, s = compute_band_fractions(windows, FS)
        print(f"  {name:>14}  {m[0]:7.3f}  {m[1]:7.3f}  {m[2]:7.3f}  {m[3]:7.3f}  {len(windows)}")

    plot_per_dataset(groups, args.out)


if __name__ == "__main__":
    main()
