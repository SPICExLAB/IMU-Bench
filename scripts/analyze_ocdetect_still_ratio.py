"""analyze_ocdetect_still_ratio.py

Analyzes the OCDetect dataset to measure the ratio of "hand still" windows
vs "active" windows.

"Still" definitions used (reported separately):
  1. Low-variance: per-window std of ALL 6 channels < acc_std_thresh AND gyro_std_thresh
     → Best proxy for "no dynamic motion" (gravity is DC offset, so low-std = truly still)
  2. Low-magnitude gyro: window-mean |gyro| < gyro_mean_thresh (no rotation)
  3. Literally near-zero: ALL 6 channel means |val| < near_zero_thresh
     → Catches sensor-off / all-zero artifacts (acc would be ~9.8 if the sensor is on)

Also reports per-subject breakdown and the compulsive-label distribution within each category.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
DATA_DIR = "/mnt/nas2/yiquan_data/preprocessed"
WINDOW_SAMPLES = 128   # 2.56 s @ 50 Hz
HOP_SAMPLES    = 64

# Thresholds (tune to your needs)
ACC_STD_THRESH    = 0.05   # m/s²  – std per channel considered "static"
GYRO_STD_THRESH   = 0.02   # rad/s – std per channel considered "static"
GYRO_MEAN_THRESH  = 0.05   # rad/s – mean |gyro| considered "no rotation"
NEAR_ZERO_THRESH  = 0.5    # any unit – literally near-zero for ALL 6 channels

IMU_COLS = ["acc x", "acc y", "acc z", "gyro x", "gyro y", "gyro z"]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_file(path: str):
    """Returns (imu [N,6], compulsive [N], ignore [N])."""
    df = pd.read_csv(
        path,
        usecols=["ignore", "compulsive"] + IMU_COLS,
        dtype={c: "float32" for c in IMU_COLS},
        low_memory=False,
    )
    imu        = df[IMU_COLS].values.astype(np.float32)
    compulsive = df["compulsive"].fillna(0.0).values.astype(np.float32)
    ignore     = df["ignore"].values.astype(np.int32)
    return imu, compulsive, ignore


def slide_windows(imu, compulsive, ignore):
    """Slide non-overlapping-gap windows; returns (wins [M,6,W], labels [M])."""
    usable  = (ignore == 0)
    changes = np.diff(usable.astype(np.int8), prepend=0, append=0)
    starts  = np.where(changes ==  1)[0]
    ends    = np.where(changes == -1)[0]

    X_parts, y_parts = [], []
    for s, e in zip(starts, ends):
        seg_len = int(e - s)
        if seg_len < WINDOW_SAMPLES:
            continue
        seg_imu  = imu[s:e]
        seg_comp = compulsive[s:e]
        n_wins   = (seg_len - WINDOW_SAMPLES) // HOP_SAMPLES + 1
        idxs     = np.arange(n_wins) * HOP_SAMPLES
        idx_mat  = idxs[:, None] + np.arange(WINDOW_SAMPLES)
        wins     = seg_imu[idx_mat]                    # [M, W, 6]
        labels   = (seg_comp[idx_mat] >= 1.0).any(axis=1).astype(np.int8)
        X_parts.append(wins.transpose(0, 2, 1))        # [M, 6, W]
        y_parts.append(labels)

    if not X_parts:
        return np.empty((0, 6, WINDOW_SAMPLES), np.float32), np.empty(0, np.int8)
    return np.concatenate(X_parts, 0), np.concatenate(y_parts, 0)


def classify_windows(wins):
    """
    wins: [M, 6, W]  (acc[0:3], gyro[3:6])

    Returns dict of bool arrays [M]:
      - low_var_still   : all 6 channel stds below threshold (truly static)
      - low_gyro_mean   : mean |gyro| below threshold (no rotation, acc free)
      - literally_zero  : all 6 channel means literally near zero (artifact check)
    """
    acc  = wins[:, :3, :]   # [M, 3, W]
    gyro = wins[:, 3:, :]   # [M, 3, W]

    acc_std  = acc.std(axis=2)   # [M, 3]
    gyro_std = gyro.std(axis=2)  # [M, 3]
    gyro_mean_abs = np.abs(gyro.mean(axis=2))  # [M, 3]
    all_mean_abs  = np.abs(wins.mean(axis=2))  # [M, 6]

    low_var_still  = (
        (acc_std  < ACC_STD_THRESH).all(axis=1) &
        (gyro_std < GYRO_STD_THRESH).all(axis=1)
    )
    low_gyro_mean  = (gyro_mean_abs < GYRO_MEAN_THRESH).all(axis=1)
    literally_zero = (all_mean_abs  < NEAR_ZERO_THRESH).all(axis=1)

    return {
        "low_var_still" : low_var_still,
        "low_gyro_mean" : low_gyro_mean,
        "literally_zero": literally_zero,
    }


def subject_id(path: str) -> int:
    import re
    m = re.match(r"OCDetect_(\d+)_", os.path.basename(path))
    return int(m.group(1))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "OCDetect_*.csv")))
    if not files:
        sys.exit(f"No CSV files found under {DATA_DIR}")
    print(f"Found {len(files)} CSV files")

    # Per-subject accumulators
    rows = []

    all_wins_list, all_labels_list = [], []
    all_flags_list = {k: [] for k in ("low_var_still", "low_gyro_mean", "literally_zero")}

    for path in tqdm(files, desc="Processing files", unit="file"):
        subj = subject_id(path)
        try:
            imu, comp, ign = load_file(path)
        except Exception as e:
            print(f"  [skip] {os.path.basename(path)}: {e}")
            continue

        wins, labels = slide_windows(imu, comp, ign)
        if len(wins) == 0:
            continue

        flags = classify_windows(wins)

        n = len(wins)
        n_pos = int(labels.sum())

        rows.append({
            "subject"          : subj,
            "file"             : os.path.basename(path),
            "n_windows"        : n,
            "n_compulsive"     : n_pos,
            "pct_compulsive"   : 100.0 * n_pos / n,
            "n_low_var_still"  : int(flags["low_var_still"].sum()),
            "pct_low_var_still": 100.0 * flags["low_var_still"].mean(),
            "n_low_gyro_mean"  : int(flags["low_gyro_mean"].sum()),
            "pct_low_gyro_mean": 100.0 * flags["low_gyro_mean"].mean(),
            "n_literally_zero" : int(flags["literally_zero"].sum()),
            "pct_literally_zero": 100.0 * flags["literally_zero"].mean(),
        })

        all_wins_list.append(wins)
        all_labels_list.append(labels)
        for k in all_flags_list:
            all_flags_list[k].append(flags[k])

    if not rows:
        sys.exit("No windows collected — check data_dir and ignore masks")

    df_files = pd.DataFrame(rows)

    # ── Global summary ──────────────────────────────────────────────────────
    all_labels = np.concatenate(all_labels_list)
    all_flags  = {k: np.concatenate(v) for k, v in all_flags_list.items()}
    all_wins   = np.concatenate(all_wins_list)

    N = len(all_labels)

    print("\n" + "=" * 70)
    print(f"  GLOBAL SUMMARY  ({N:,} windows across {len(files)} files)")
    print("=" * 70)

    print(f"\n  Compulsive-episode windows  : {int(all_labels.sum()):>8,}  ({100*all_labels.mean():.2f}%)")
    print(f"  Background windows          : {int((all_labels==0).sum()):>8,}  ({100*(all_labels==0).mean():.2f}%)")

    print(f"\n  -- 'Still' definitions --")
    for key, label, note in [
        ("low_var_still",
         "Low-variance still  (acc_std<{:.2f}, gyro_std<{:.2f})".format(ACC_STD_THRESH, GYRO_STD_THRESH),
         "Best proxy for genuinely static hand"),
        ("low_gyro_mean",
         "Low-gyro-mean       (|gyro mean|<{:.2f} rad/s)".format(GYRO_MEAN_THRESH),
         "No rotation; acc may still wiggle"),
        ("literally_zero",
         "Literally near-zero (all 6 mean |val|<{:.1f})".format(NEAR_ZERO_THRESH),
         "Catches sensor-off / all-zero artifacts"),
    ]:
        flag = all_flags[key]
        n_still  = int(flag.sum())
        n_active = int((~flag).sum())
        pct_still = 100.0 * flag.mean()
        # compulsive within still vs active
        comp_in_still  = int(all_labels[flag].sum())
        comp_in_active = int(all_labels[~flag].sum())
        print(f"\n  [{label}]")
        print(f"    {note}")
        print(f"    Still   : {n_still:>8,}  ({pct_still:.2f}%)")
        print(f"    Active  : {n_active:>8,}  ({100-pct_still:.2f}%)")
        print(f"    Compulsive in still  : {comp_in_still:>6,}  ({100*comp_in_still/(n_still+1e-9):.2f}%)")
        print(f"    Compulsive in active : {comp_in_active:>6,}  ({100*comp_in_active/(n_active+1e-9):.2f}%)")

    # ── Channel statistics when still vs active ─────────────────────────────
    print("\n" + "-" * 70)
    print("  Channel statistics: still (low-var) vs active")
    print("-" * 70)
    still_mask  = all_flags["low_var_still"]
    active_mask = ~still_mask
    ch_names = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    print(f"  {'channel':<10}  {'still_mean':>12}  {'still_std':>10}  {'active_mean':>12}  {'active_std':>10}")
    for i, ch in enumerate(ch_names):
        if still_mask.sum() > 0:
            sm = all_wins[still_mask,  i, :].mean()
            ss = all_wins[still_mask,  i, :].std()
        else:
            sm = ss = float("nan")
        am = all_wins[active_mask, i, :].mean()
        as_ = all_wins[active_mask, i, :].std()
        print(f"  {ch:<10}  {sm:>12.4f}  {ss:>10.4f}  {am:>12.4f}  {as_:>10.4f}")

    # ── Per-subject summary ──────────────────────────────────────────────────
    df_subj = (
        df_files.groupby("subject")
        .agg(
            n_files       =("file",             "count"),
            n_windows     =("n_windows",         "sum"),
            n_compulsive  =("n_compulsive",       "sum"),
            n_low_var     =("n_low_var_still",    "sum"),
            n_low_gyro    =("n_low_gyro_mean",    "sum"),
            n_literal_zero=("n_literally_zero",   "sum"),
        )
        .reset_index()
    )
    df_subj["pct_compulsive"]    = 100.0 * df_subj["n_compulsive"]   / df_subj["n_windows"]
    df_subj["pct_low_var_still"] = 100.0 * df_subj["n_low_var"]      / df_subj["n_windows"]
    df_subj["pct_low_gyro_mean"] = 100.0 * df_subj["n_low_gyro"]     / df_subj["n_windows"]
    df_subj["pct_literally_zero"]= 100.0 * df_subj["n_literal_zero"] / df_subj["n_windows"]

    print("\n" + "=" * 70)
    print("  PER-SUBJECT BREAKDOWN")
    print("=" * 70)
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(df_subj[[
        "subject", "n_files", "n_windows",
        "pct_compulsive", "pct_low_var_still", "pct_low_gyro_mean", "pct_literally_zero"
    ]].to_string(index=False))

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir = Path(__file__).parent
    df_files.to_csv(out_dir / "ocdetect_still_ratio_per_file.csv",    index=False)
    df_subj.to_csv(out_dir  / "ocdetect_still_ratio_per_subject.csv", index=False)
    print(f"\n  Saved per-file CSV   → {out_dir}/ocdetect_still_ratio_per_file.csv")
    print(f"  Saved per-subject CSV→ {out_dir}/ocdetect_still_ratio_per_subject.csv")

    # ── Threshold sweep ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  THRESHOLD SWEEP  (low-variance definition)")
    print("  How the still ratio changes with different std thresholds")
    print("=" * 70)
    acc_stds  = all_wins[:, :3, :].std(axis=2)   # [M, 3]
    gyro_stds = all_wins[:, 3:, :].std(axis=2)   # [M, 3]
    print(f"  {'acc_thresh':>12}  {'gyro_thresh':>12}  {'still%':>8}  {'active%':>8}")
    for a_th in [0.02, 0.05, 0.10, 0.20, 0.50]:
        for g_th in [0.01, 0.02, 0.05, 0.10]:
            mask = (acc_stds < a_th).all(1) & (gyro_stds < g_th).all(1)
            print(f"  {a_th:>12.2f}  {g_th:>12.2f}  {100*mask.mean():>8.2f}  {100*(~mask).mean():>8.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
