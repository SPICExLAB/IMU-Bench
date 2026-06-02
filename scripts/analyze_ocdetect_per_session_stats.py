"""analyze_ocdetect_per_session_stats.py

Per-session (CSV file) statistics for the first N OCDetect files.
IMU is transformed to FLU frame (same as OCDetect_dataloader.py).
Reports mean and variance of each channel, using only ignore==0 samples.

Usage:
    python scripts/analyze_ocdetect_per_session_stats.py [--n 20] [--data-dir PATH]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

DATA_DIR = "/mnt/nas2/yiquan_data/preprocessed"
IMU_COLS  = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]
FLU_NAMES = ["acc_F", "acc_L", "acc_U", "gyro_F", "gyro_L", "gyro_U"]


def load_flu(path: str):
    """Load one CSV, apply Android→FLU transform, return (imu_flu [N,6], ignore [N])."""
    df = pd.read_csv(
        path,
        usecols=["ignore"] + IMU_COLS,
        dtype={c: "float32" for c in IMU_COLS},
        low_memory=False,
    )
    imu = df[IMU_COLS].values.astype(np.float64)
    # Android → FLU: F = -X_android, L = -Y_android, U = +Z_android
    imu[:, 0] *= -1   # acc_x  → acc_F  (F = -X)
    imu[:, 1] *= -1   # acc_y  → acc_L  (L = -Y)
    imu[:, 3] *= -1   # gyro_x → gyro_F
    imu[:, 4] *= -1   # gyro_y → gyro_L
    ignore = df["ignore"].values.astype(np.int32)
    return imu, ignore


def session_stats(imu: np.ndarray, ignore: np.ndarray) -> dict:
    usable = imu[ignore == 0]
    stats = {
        "n_total"  : len(imu),
        "n_usable" : len(usable),
        "pct_usable": 100.0 * len(usable) / max(len(imu), 1),
    }
    for i, ch in enumerate(FLU_NAMES):
        vals = usable[:, i] if len(usable) > 0 else np.array([float("nan")])
        stats[f"{ch}_mean"] = float(vals.mean())
        stats[f"{ch}_var"]  = float(vals.var())
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",        type=int, default=20,   help="Number of files to analyse")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "OCDetect_*.csv")))[: args.n]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {args.data_dir}")

    rows = []
    for path in files:
        fname = os.path.basename(path)
        print(f"  loading {fname} …")
        imu, ignore = load_flu(path)
        s = session_stats(imu, ignore)
        s["file"] = fname
        rows.append(s)

    df = pd.DataFrame(rows)

    # ── pretty-print ─────────────────────────────────────────────────────────
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 120)
    print(f"  OCDetect – per-session IMU statistics (FLU frame)  |  first {len(df)} files")
    print("=" * 120)

    # Coverage
    print("\n-- Data coverage --")
    cov_cols = ["file", "n_total", "n_usable", "pct_usable"]
    print(df[cov_cols].to_string(index=False))

    # Mean per channel
    mean_cols = ["file"] + [f"{ch}_mean" for ch in FLU_NAMES]
    print("\n-- Channel means (m/s² for acc, rad/s for gyro) --")
    print(df[mean_cols].to_string(index=False))

    # Variance per channel
    var_cols  = ["file"] + [f"{ch}_var"  for ch in FLU_NAMES]
    print("\n-- Channel variances --")
    print(df[var_cols].to_string(index=False))

    # Save
    out = os.path.join(os.path.dirname(__file__), "ocdetect_session_stats.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
