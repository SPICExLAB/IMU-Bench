#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

IMU_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]


def load_ocdetect(path: str):
    cols = ["timestamp", "ignore", "compulsive"] + IMU_COLS
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=cols, low_memory=False)

    # Android -> FLU
    # F = +X, L = -Y, U = +Z
    acc_f = df["acc x"].to_numpy(dtype=np.float32)
    acc_l = -df["acc y"].to_numpy(dtype=np.float32)
    acc_u = df["acc z"].to_numpy(dtype=np.float32)

    gyro_f = df["gyro x"].to_numpy(dtype=np.float32)
    gyro_l = -df["gyro y"].to_numpy(dtype=np.float32)
    gyro_u = df["gyro z"].to_numpy(dtype=np.float32)

    t_ns = df["timestamp"].to_numpy(dtype=np.int64)
    t_s = (t_ns - t_ns[0]) / 1e9 if len(t_ns) else np.array([], dtype=np.float32)

    ignore = df["ignore"].to_numpy(dtype=np.int32)
    comp = df["compulsive"].fillna(0.0).to_numpy(dtype=np.float32)

    return t_s, (acc_f, acc_l, acc_u), (gyro_f, gyro_l, gyro_u), ignore, comp


def plot_sequence(path: str, out_path: str, dpi: int = 150):
    t_s, acc, gyro, ignore, comp = load_ocdetect(path)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True, constrained_layout=True)

    # Acc
    axes[0].plot(t_s, acc[0], label="acc_F", lw=0.8)
    axes[0].plot(t_s, acc[1], label="acc_L", lw=0.8)
    axes[0].plot(t_s, acc[2], label="acc_U", lw=0.8)
    axes[0].set_ylabel("m/s²")
    axes[0].set_title("OCDetect IMU sequence (FLU)")
    axes[0].legend(loc="upper right", ncol=3)

    # Gyro
    axes[1].plot(t_s, gyro[0], label="gyro_F", lw=0.8)
    axes[1].plot(t_s, gyro[1], label="gyro_L", lw=0.8)
    axes[1].plot(t_s, gyro[2], label="gyro_U", lw=0.8)
    axes[1].set_ylabel("rad/s")
    axes[1].legend(loc="upper right", ncol=3)

    # Labels/mask
    axes[2].plot(t_s, comp, label="compulsive", lw=0.9)
    axes[2].plot(t_s, (ignore != 0).astype(np.float32), label="ignore!=0", lw=0.9, alpha=0.8)
    axes[2].set_ylabel("flag")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right", ncol=2)

    for ax in axes:
        ax.grid(alpha=0.25)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def default_out_path(in_path: str):
    p = Path(in_path)
    stem = p.stem
    return str(Path("data_vis") / f"{stem}_imu_sequence_flu.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to OCDetect CSV/parquet")
    ap.add_argument("--out", default=None, help="Output PNG path")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    out = args.out or default_out_path(args.input)
    plot_sequence(args.input, out, dpi=args.dpi)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()