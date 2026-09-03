"""Convert the raw MHEALTH dataset (Banos et al., UGR) into the project's
master-parquet HAR schema.

Source: https://archive.ics.uci.edu/dataset/319/mhealth+dataset
("MHEALTHDATASET/mHealth_subject<N>.log", tab-separated, 24 columns, no
header, already sampled at the canonical 50 Hz). Each row holds simultaneous
readings from three Shimmer2 sensors (chest, left-ankle, right-lower-arm) plus
an activity label (0 = null/no activity). We use the right-lower-arm
accelerometer + gyroscope (columns 15-20, 1-indexed per the README) as the
project's single wrist/forearm-analog 6-channel stream -- matching the
wrist-worn convention used by most other HAR sources in this repo -- and drop
the chest ECG, magnetometer channels, and ankle sensor.

Output columns/dtypes match IMULM_master_dataset_v2.parquet exactly. No
resampling needed: MHEALTH is natively 50 Hz.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CANONICAL_SAMPLE_RATE_HZ = 50.0

# README section 2 (L1..L12); label 0 = null class, dropped.
ACTIVITY_NAMES = {
    1: "standing_still",
    2: "sitting_relaxing",
    3: "lying_down",
    4: "walking",
    5: "climbing_stairs",
    6: "waist_bends_forward",
    7: "frontal_elevation_of_arms",
    8: "knees_bending",
    9: "cycling",
    10: "jogging",
    11: "running",
    12: "jump_front_and_back",
}

# 0-indexed column positions for the right-lower-arm sensor + label.
ARM_ACC_COLS = [14, 15, 16]
ARM_GYRO_COLS = [17, 18, 19]
LABEL_COL = 23


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/projects/b53017/vbc1591/MHEALTH/MHEALTHDATASET")
    ap.add_argument("--out", default="/projects/b53017/vbc1591/mhealth_dataset.parquet")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    log_files = sorted(raw_root.glob("mHealth_subject*.log"))

    rows = []
    for fpath in log_files:
        subject_id = fpath.stem.replace("mHealth_", "")  # "subject1", "subject10", ...
        arr = np.loadtxt(fpath, dtype=np.float64)
        labels = arr[:, LABEL_COL].astype(np.int64)

        # Segment into contiguous same-label runs; drop the null class (0).
        change = np.flatnonzero(np.diff(labels) != 0) + 1
        bounds = np.concatenate(([0], change, [len(labels)]))
        for seg_idx, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
            lbl = int(labels[s])
            if lbl == 0 or lbl not in ACTIVITY_NAMES:
                continue
            n = e - s
            if n < 2:
                continue

            acc = arr[s:e, ARM_ACC_COLS].astype(np.float32)
            gyro_dps = arr[s:e, ARM_GYRO_COLS].astype(np.float64)
            gyro = np.deg2rad(gyro_dps).astype(np.float32)  # deg/s -> rad/s

            timestamp_ns = (np.arange(n, dtype=np.int64) * round(1e9 / CANONICAL_SAMPLE_RATE_HZ))
            name = ACTIVITY_NAMES[lbl]

            rows.append(
                pd.DataFrame(
                    {
                        "dataset": "mhealth",
                        "subject_id": subject_id,
                        "session_id": f"{subject_id}_seg{seg_idx}_{name}",
                        "timestamp_ns": timestamp_ns,
                        "acc_x": acc[:, 0],
                        "acc_y": acc[:, 1],
                        "acc_z": acc[:, 2],
                        "gyro_x": gyro[:, 0],
                        "gyro_y": gyro[:, 1],
                        "gyro_z": gyro[:, 2],
                        "global_activity_id": lbl,
                        "global_activity_label": name,
                        "dataset_activity_id": lbl,
                        "dataset_activity_label": name,
                    }
                )
            )

    df = pd.concat(rows, ignore_index=True)
    df["global_activity_id"] = df["global_activity_id"].astype("int32")
    df["dataset_activity_id"] = df["dataset_activity_id"].astype("int32")

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

    print(f"subjects={df['subject_id'].nunique()} sessions={df['session_id'].nunique()} total_rows={len(df)}")
    print(df.groupby("dataset_activity_label")["session_id"].nunique())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
