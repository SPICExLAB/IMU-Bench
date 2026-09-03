"""Convert the raw HHAR (Heterogeneity Human Activity Recognition) dataset
(Stisen et al., SenSys 2015) into the project's master-parquet HAR schema.

Source: https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition
("Activity recognition exp/Watch_accelerometer.csv",
"Activity recognition exp/Watch_gyroscope.csv"). We use the *watch* streams
(not phone) to match the wrist-worn convention used by most other HAR
sources in this repo, and drop the phone accelerometer/gyroscope files.

Accelerometer and gyroscope are recorded as two independent, asynchronous
streams per (User, Device) -- there is no shared row index -- so each
(User, Device) group is aligned via a nearest-timestamp asof-merge on
Creation_Time (the device's own sensor clock; Arrival_Time is coarser and not
monotonic under sort, so it is not used for ordering). The activity label
("gt" column, "nan" = no activity) is attached per accelerometer sample; runs
of consecutive samples sharing the same label become one session, which is
then resampled to the canonical 50 Hz.

Units: x/y/z are already m/s^2 (accelerometer) and rad/s (gyroscope) --
Android SensorEvent convention -- verified empirically (accel magnitude ~9.8
at rest; gyro near-zero at rest, ~1.5 rad/s mean during walk), so no unit
conversion is applied.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CANONICAL_SAMPLE_RATE_HZ = 50.0
ASOF_TOLERANCE_NS = 100_000_000  # 100ms; native rates are ~50-200 Hz per device
MIN_SEGMENT_SAMPLES = 4


def resample_to_canonical(x: np.ndarray, t_native_s: np.ndarray, target_hz: float) -> np.ndarray:
    duration_s = t_native_s[-1] - t_native_s[0]
    n_out = max(2, int(np.floor(duration_s * target_hz)) + 1)
    t_target = t_native_s[0] + np.arange(n_out) / target_hz
    out = np.empty((n_out, x.shape[1]), dtype=np.float64)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(t_target, t_native_s, x[:, c])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-root",
        default="/projects/b53017/vbc1591/HHAR/activity_recognition_exp/Activity recognition exp",
    )
    ap.add_argument("--out", default="/projects/b53017/vbc1591/hhar_dataset.parquet")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    acc = pd.read_csv(raw_root / "Watch_accelerometer.csv")
    gyro = pd.read_csv(raw_root / "Watch_gyroscope.csv")

    rows = []
    n_sessions = 0
    for (user, device), acc_g in acc.groupby(["User", "Device"]):
        gyro_g = gyro[(gyro["User"] == user) & (gyro["Device"] == device)]
        if gyro_g.empty:
            continue

        acc_g = acc_g.sort_values("Creation_Time").reset_index(drop=True)
        gyro_g = gyro_g.sort_values("Creation_Time").reset_index(drop=True)

        merged = pd.merge_asof(
            acc_g[["Creation_Time", "x", "y", "z", "gt"]],
            gyro_g[["Creation_Time", "x", "y", "z"]].rename(columns={"x": "gx", "y": "gy", "z": "gz"}),
            on="Creation_Time",
            direction="nearest",
            tolerance=ASOF_TOLERANCE_NS,
        ).dropna(subset=["gx", "gy", "gz"])

        gt = merged["gt"].astype(str).values
        valid = gt != "nan"
        if not valid.any():
            continue

        change = np.flatnonzero(gt[1:] != gt[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(gt)]))
        for seg_idx, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
            label = gt[s]
            if label == "nan" or (e - s) < MIN_SEGMENT_SAMPLES:
                continue

            seg = merged.iloc[s:e]
            t_native_s = seg["Creation_Time"].to_numpy(dtype=np.float64) / 1e9
            acc_arr = seg[["x", "y", "z"]].to_numpy(dtype=np.float64)
            gyro_arr = seg[["gx", "gy", "gz"]].to_numpy(dtype=np.float64)

            acc_rs = resample_to_canonical(acc_arr, t_native_s, CANONICAL_SAMPLE_RATE_HZ).astype(np.float32)
            gyro_rs = resample_to_canonical(gyro_arr, t_native_s, CANONICAL_SAMPLE_RATE_HZ).astype(np.float32)
            n = acc_rs.shape[0]

            timestamp_ns = (np.arange(n, dtype=np.int64) * round(1e9 / CANONICAL_SAMPLE_RATE_HZ))
            session_id = f"{device}_{label}_seg{seg_idx}"

            rows.append(
                pd.DataFrame(
                    {
                        "dataset": "hhar",
                        "subject_id": user,
                        "session_id": session_id,
                        "timestamp_ns": timestamp_ns,
                        "acc_x": acc_rs[:, 0],
                        "acc_y": acc_rs[:, 1],
                        "acc_z": acc_rs[:, 2],
                        "gyro_x": gyro_rs[:, 0],
                        "gyro_y": gyro_rs[:, 1],
                        "gyro_z": gyro_rs[:, 2],
                        "global_activity_id": 0,  # filled in below
                        "global_activity_label": label,
                        "dataset_activity_id": 0,  # filled in below
                        "dataset_activity_label": label,
                    }
                )
            )
            n_sessions += 1

    df = pd.concat(rows, ignore_index=True)

    activity_names = sorted(df["dataset_activity_label"].unique())
    activity_to_id = {name: i + 1 for i, name in enumerate(activity_names)}
    df["dataset_activity_id"] = df["dataset_activity_label"].map(activity_to_id).astype("int32")
    df["global_activity_id"] = df["dataset_activity_id"]

    schema = pa.schema(
        [
            ("dataset", pa.string()),
            ("subject_id", pa.string()),
            ("session_id", pa.string()),
            ("timestamp_ns", pa.int64()),
            ("acc_x", pa.float32()),
            ("acc_y", pa.float32()),
            ("acc_z", pa.float32()),
            ("gyro_x", pa.float32()),
            ("gyro_y", pa.float32()),
            ("gyro_z", pa.float32()),
            ("global_activity_id", pa.int32()),
            ("global_activity_label", pa.string()),
            ("dataset_activity_id", pa.int32()),
            ("dataset_activity_label", pa.string()),
        ]
    )
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, args.out)

    print(f"sessions={n_sessions} subjects={df['subject_id'].nunique()} total_rows={len(df)}")
    print(f"activities ({len(activity_to_id)}): {activity_to_id}")
    print(df.groupby("dataset_activity_label")["session_id"].nunique())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
