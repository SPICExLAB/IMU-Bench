"""Convert the raw WHARF dataset (wrist-worn, single tri-axial accelerometer,
32 Hz, 6-bit coded) into the project's master-parquet HAR schema.

Source: https://github.com/fulviomas/WHARF ("WHARF Data Set/Data/<HMP>/*.txt").
Each file holds whitespace-separated raw accelerometer samples in [0, 63],
coded per MANUAL.txt as [0; 63] -> [-1.5g; +1.5g]. WHARF has no gyroscope, so
gyro_x/y/z are written as NaN (matching the existing accelerometer-only
sources in the master dataset, e.g. capture24).

Output columns/dtypes match IMULM_master_dataset_v2.parquet exactly:
dataset, subject_id, session_id, timestamp_ns, acc_x/y/z (float32),
gyro_x/y/z (float32), global_activity_id/label, dataset_activity_id/label.

Every source is resampled to the pipeline's canonical 50 Hz (see
imu_lm.data.windowing.CANONICAL_SAMPLE_RATE_HZ) since filter_window()
hardcodes fs=50 for the Butterworth design.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_SAMPLE_RATE_HZ = 32.0
CANONICAL_SAMPLE_RATE_HZ = 50.0
G_TO_MS2 = 9.80665

FILENAME_RE = re.compile(
    r"^Accelerometer-(?P<start_time>[\d-]+)-(?P<hmp>[a-z_]+)-(?P<volunteer>[mf]\d+)\.txt$"
)


def decode_accel_g(raw: np.ndarray) -> np.ndarray:
    """[0; 63] -> [-1.5g; +1.5g], per MANUAL.txt section 4.2."""
    return -1.5 + (raw.astype(np.float64) / 63.0) * 3.0


def resample_to_canonical(x: np.ndarray, native_hz: float, target_hz: float) -> np.ndarray:
    n = x.shape[0]
    duration_s = (n - 1) / native_hz
    t_native = np.arange(n) / native_hz
    n_out = max(2, int(np.floor(duration_s * target_hz)) + 1)
    t_target = np.arange(n_out) / target_hz
    out = np.empty((n_out, x.shape[1]), dtype=np.float64)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(t_target, t_native, x[:, c])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="/projects/b53017/vbc1591/WHARF/WHARF Data Set/Data")
    ap.add_argument("--out", default="/projects/b53017/vbc1591/wharf_dataset.parquet")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    hmp_dirs = sorted(
        d for d in raw_root.iterdir() if d.is_dir() and d.name != "MODELS"
    )
    activity_names = sorted(d.name.lower() for d in hmp_dirs)
    activity_to_id = {name: i + 1 for i, name in enumerate(activity_names)}

    rows = []
    n_files = 0
    n_skipped = 0
    for hmp_dir in hmp_dirs:
        for fpath in sorted(hmp_dir.glob("*.txt")):
            m = FILENAME_RE.match(fpath.name)
            if not m:
                print(f"WARNING: filename didn't match convention, skipping: {fpath}")
                n_skipped += 1
                continue

            raw = np.loadtxt(fpath, dtype=np.int64)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape[0] < 2 or raw.shape[1] != 3:
                n_skipped += 1
                continue

            accel_g = decode_accel_g(raw)
            accel_ms2 = resample_to_canonical(accel_g, RAW_SAMPLE_RATE_HZ, CANONICAL_SAMPLE_RATE_HZ) * G_TO_MS2
            n = accel_ms2.shape[0]

            hmp = hmp_dir.name.lower()
            volunteer = m.group("volunteer")
            session_id = fpath.stem  # unique per trial file

            timestamp_ns = (np.arange(n, dtype=np.int64) * round(1e9 / CANONICAL_SAMPLE_RATE_HZ))

            rows.append(
                pd.DataFrame(
                    {
                        "dataset": "wharf",
                        "subject_id": volunteer,
                        "session_id": session_id,
                        "timestamp_ns": timestamp_ns,
                        "acc_x": accel_ms2[:, 0].astype(np.float32),
                        "acc_y": accel_ms2[:, 1].astype(np.float32),
                        "acc_z": accel_ms2[:, 2].astype(np.float32),
                        "gyro_x": np.float32(np.nan),
                        "gyro_y": np.float32(np.nan),
                        "gyro_z": np.float32(np.nan),
                        "global_activity_id": activity_to_id[hmp],
                        "global_activity_label": hmp,
                        "dataset_activity_id": activity_to_id[hmp],
                        "dataset_activity_label": hmp,
                    }
                )
            )
            n_files += 1

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

    print(f"files_used={n_files} files_skipped={n_skipped}")
    print(f"total_rows={len(df)} sessions={df['session_id'].nunique()} subjects={df['subject_id'].nunique()}")
    print(f"activities ({len(activity_to_id)}): {activity_to_id}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
