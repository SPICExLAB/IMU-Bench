"""Convert the raw PADS (Parkinson's Disease Smartwatch) dataset into the
project's master-parquet HAR schema, so the existing probe / finetune /
scratch pipelines can be pointed at it unchanged.

Source: https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/
(Varghese et al., "Parkinson's Disease Smartwatch dataset (PADS)").

Layout of the raw release:
    patients/patient_<sid>.json      subject metadata incl. "condition"
    movement/observation_<sid>.json  session manifest (sampling_rate, channels,
                                     record_name, device_location, file_name)
    movement/timeseries/<sid>_<Task>_<Left|Right>Wrist.txt
        comma-separated, 7 columns:
            Time, Accelerometer_X/Y/Z, Gyroscope_X/Y/Z
        1024 rows (~10 s) per movement task, 2048 (~20 s) for "Relaxed".

Unlike HAR datasets in this repo, the label here is *subject-level* (clinical
condition), not a per-sample activity. Every window of every recording of a
subject therefore carries that subject's condition, and splits MUST be grouped
by subject (configs/base.yaml already sets splits.group_key: subject_id) --
grouping by session would put the same patient's other recordings in train and
leak the label outright.

Each (subject, movement task, wrist) recording becomes one session:
    session_id = "<Task>_<LeftWrist|RightWrist>"
so the two wrists are independent single-IMU sessions (matching the
wrist-worn, single-device convention of the rest of the repo) and the task name
stays recoverable for per-task analysis.

Units: the release stores accelerometer in g and gyroscope in rad/s (per the
"units" field of each observation record). Accelerometer is converted to m/s^2
to match the repo convention (HHAR/PAMAP2/... are all m/s^2); gyroscope is
already rad/s and passes through unchanged.

IMPORTANT -- the accelerometer is GRAVITY-FREE. Measured per-axis means are
~0 and |a| averages 0.05-0.33 g across recordings, i.e. these are Apple Watch
CoreMotion *userAcceleration* samples, not raw accelerometer readings (the
release's own run_preprocessing.py only strips residual low-frequency drift on
top of that). So PADS carries no gravity-direction cue, unlike the
gravity-bearing corpora this repo pretrains on -- expect a distribution shift
for a frozen encoder, and read probe numbers with that in mind.

The first ~0.5 s of every recording contains the watch's start-of-task
notification vibration; the release drops it (`data[:, 48:]` at 100 Hz) and so
does this script by default (--trim-start-seconds).

Sampling: the manifest declares 100 Hz but the Time column shows ~102 Hz with
jitter, so every channel is linearly interpolated onto the canonical 50 Hz grid
using the recorded timestamps rather than assuming a nominal rate.

Label schemes (--label-scheme):
    dd3        0=Healthy, 1=Parkinson's, 2=Differential Diagnosis (everything
               else: Other Movement Disorders, Essential Tremor, Atypical
               Parkinsonism, Multiple Sclerosis). This reproduces the `label`
               column the release itself ships in preprocessed/file_list.csv
               and is the dataset's headline differential-diagnosis benchmark.
    pd_binary  0=Healthy, 1=Parkinson's; all other conditions dropped. This is
               the original paper's "PD vs. Healthy" binary problem.
    pd_vs_dd   0=Differential Diagnosis (Other Movement Disorders / Essential
               Tremor / Atypical Parkinsonism / Multiple Sclerosis),
               1=Parkinson's; Healthy dropped. This is the original paper's
               "PD vs. Differential Diagnosis" binary problem -- the harder of
               the two, since both arms have tremor.
    condition6 one class per raw "condition" string (6 classes).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CANONICAL_SAMPLE_RATE_HZ = 50.0
G_TO_MS2 = 9.80665
MIN_SEGMENT_SAMPLES = 64          # >= 1 window at 2.56 s / 50 Hz would be 128;
                                  # keep short recordings, windowing drops them
DATASET_NAME = "pads"

# Column order inside the .txt files (channel list is identical for every
# record in the release; asserted against the manifest below).
EXPECTED_CHANNELS = [
    "Time",
    "Accelerometer_X",
    "Accelerometer_Y",
    "Accelerometer_Z",
    "Gyroscope_X",
    "Gyroscope_Y",
    "Gyroscope_Z",
]

# dd3: the release's own 3-class differential-diagnosis grouping.
DD3_ID = {"Healthy": 0, "Parkinson's": 1}
DD3_LABEL = {0: "Healthy", 1: "Parkinson", 2: "DifferentialDiagnosis"}
DD3_OTHER_ID = 2

PD_BINARY_ID = {"Healthy": 0, "Parkinson's": 1}
PD_BINARY_LABEL = {0: "Healthy", 1: "Parkinson"}

# pd_vs_dd: Parkinson's against the other movement disorders (Healthy dropped).
PD_VS_DD_LABEL = {0: "DifferentialDiagnosis", 1: "Parkinson"}


def resample_to_canonical(x: np.ndarray, t_native_s: np.ndarray, target_hz: float) -> np.ndarray:
    duration_s = t_native_s[-1] - t_native_s[0]
    n_out = max(2, int(np.floor(duration_s * target_hz)) + 1)
    t_target = t_native_s[0] + np.arange(n_out) / target_hz
    out = np.empty((n_out, x.shape[1]), dtype=np.float64)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(t_target, t_native_s, x[:, c])
    return out


def condition_to_label(condition: str, scheme: str) -> tuple[int, str] | None:
    """Map a raw "condition" string to (id, name); None = drop this subject."""
    if scheme == "dd3":
        cid = DD3_ID.get(condition, DD3_OTHER_ID)
        return cid, DD3_LABEL[cid]
    if scheme == "pd_binary":
        if condition not in PD_BINARY_ID:
            return None
        cid = PD_BINARY_ID[condition]
        return cid, PD_BINARY_LABEL[cid]
    if scheme == "pd_vs_dd":
        if condition == "Healthy":
            return None
        cid = 1 if condition == "Parkinson's" else 0
        return cid, PD_VS_DD_LABEL[cid]
    if scheme == "condition6":
        return None  # assigned after the scan, once all strings are known
    raise ValueError(f"unknown label scheme: {scheme}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-root",
        default="/projects/b53017/vbc1591/physionet.org/files/parkinsons-disease-smartwatch/1.0.0",
    )
    ap.add_argument("--out", default="/projects/b53017/vbc1591/pads_dataset.parquet")
    ap.add_argument(
        "--label-scheme", default="dd3",
        choices=["dd3", "pd_binary", "pd_vs_dd", "condition6"],
    )
    ap.add_argument(
        "--tasks",
        default=None,
        help="comma-separated movement tasks to keep (default: all, e.g. "
             "Relaxed,RelaxedTask,StretchHold,HoldWeight,DrinkGlas,CrossArms,"
             "LiftHold,PointFinger,TouchNose,TouchIndex,Entrainment)",
    )
    ap.add_argument(
        "--wrists", default="LeftWrist,RightWrist",
        help="comma-separated device locations to keep",
    )
    ap.add_argument(
        "--trim-start-seconds", type=float, default=0.5,
        help="drop this much from the start of every recording -- the watch's "
             "start-of-task notification vibration (the release drops 48 "
             "samples @100 Hz); 0 disables",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    movement_dir = raw_root / "movement"
    keep_tasks = set(args.tasks.split(",")) if args.tasks else None
    keep_wrists = set(args.wrists.split(","))

    patient_files = sorted((raw_root / "patients").glob("patient_*.json"))
    if not patient_files:
        raise FileNotFoundError(f"no patient JSONs under {raw_root / 'patients'}")

    rows = []
    n_sessions = 0
    n_missing_files = 0
    n_short = 0
    skipped_subjects: list[str] = []
    subj_condition: dict[str, str] = {}
    subj_label: dict[str, tuple[int, str]] = {}

    for pf in patient_files:
        patient = json.loads(pf.read_text())
        sid = str(patient["id"])
        condition = str(patient["condition"])

        lbl = condition_to_label(condition, args.label_scheme)
        if lbl is None and args.label_scheme != "condition6":
            skipped_subjects.append(sid)
            continue
        subj_condition[sid] = condition
        if lbl is not None:
            subj_label[sid] = lbl

        obs_path = movement_dir / f"observation_{sid}.json"
        if not obs_path.exists():
            n_missing_files += 1
            continue
        obs = json.loads(obs_path.read_text())

        for session in obs.get("session", []):
            task = str(session["record_name"])
            if keep_tasks is not None and task not in keep_tasks:
                continue
            for record in session.get("records", []):
                wrist = str(record["device_location"])
                if wrist not in keep_wrists:
                    continue
                if list(record["channels"]) != EXPECTED_CHANNELS:
                    raise ValueError(
                        f"unexpected channel layout for {sid}/{task}/{wrist}: "
                        f"{record['channels']}"
                    )
                units = list(record["units"])
                if units[1:4] != ["g", "g", "g"] or units[4:7] != ["rad/s"] * 3:
                    raise ValueError(
                        f"unexpected units for {sid}/{task}/{wrist}: {units}"
                    )

                txt_path = movement_dir / record["file_name"]
                if not txt_path.exists():
                    n_missing_files += 1
                    continue

                arr = np.loadtxt(txt_path, delimiter=",", dtype=np.float64)
                if arr.ndim != 2 or arr.shape[1] != len(EXPECTED_CHANNELS):
                    raise ValueError(f"bad shape {arr.shape} in {txt_path}")
                if arr.shape[0] < MIN_SEGMENT_SAMPLES:
                    n_short += 1
                    continue

                t_native_s = arr[:, 0]
                # Time must be increasing for np.interp; the release is ordered
                # but jittered, so guard rather than assume.
                if not np.all(np.diff(t_native_s) > 0):
                    order = np.argsort(t_native_s, kind="stable")
                    arr = arr[order]
                    t_native_s = arr[:, 0]
                    uniq = np.concatenate(([True], np.diff(t_native_s) > 0))
                    arr, t_native_s = arr[uniq], t_native_s[uniq]
                    if arr.shape[0] < MIN_SEGMENT_SAMPLES:
                        n_short += 1
                        continue

                if args.trim_start_seconds > 0:
                    keep = t_native_s >= (t_native_s[0] + args.trim_start_seconds)
                    if keep.sum() < MIN_SEGMENT_SAMPLES:
                        n_short += 1
                        continue
                    arr, t_native_s = arr[keep], t_native_s[keep]

                imu = resample_to_canonical(
                    arr[:, 1:7], t_native_s, CANONICAL_SAMPLE_RATE_HZ
                ).astype(np.float32)
                imu[:, 0:3] *= G_TO_MS2      # g -> m/s^2; gyro already rad/s
                n = imu.shape[0]

                timestamp_ns = np.arange(n, dtype=np.int64) * round(
                    1e9 / CANONICAL_SAMPLE_RATE_HZ
                )
                rows.append(
                    pd.DataFrame(
                        {
                            "dataset": DATASET_NAME,
                            "subject_id": sid,
                            "session_id": f"{task}_{wrist}",
                            "timestamp_ns": timestamp_ns,
                            "acc_x": imu[:, 0],
                            "acc_y": imu[:, 1],
                            "acc_z": imu[:, 2],
                            "gyro_x": imu[:, 3],
                            "gyro_y": imu[:, 4],
                            "gyro_z": imu[:, 5],
                            "global_activity_id": 0,        # filled in below
                            "global_activity_label": condition,
                            "dataset_activity_id": 0,       # filled in below
                            "dataset_activity_label": condition,
                        }
                    )
                )
                n_sessions += 1

    if not rows:
        raise RuntimeError("no sessions produced -- is the download complete?")

    df = pd.concat(rows, ignore_index=True)

    if args.label_scheme == "condition6":
        names = sorted(df["dataset_activity_label"].unique())
        cond_to_id = {name: i for i, name in enumerate(names)}
        df["dataset_activity_id"] = df["dataset_activity_label"].map(cond_to_id).astype("int32")
    else:
        id_by_subject = {s: v[0] for s, v in subj_label.items()}
        name_by_subject = {s: v[1] for s, v in subj_label.items()}
        df["dataset_activity_id"] = df["subject_id"].map(id_by_subject).astype("int32")
        df["dataset_activity_label"] = df["subject_id"].map(name_by_subject)
        cond_to_id = {
            name_by_subject[s]: id_by_subject[s] for s in sorted(name_by_subject)
        }
    df["global_activity_id"] = df["dataset_activity_id"]
    df["global_activity_label"] = df["dataset_activity_label"]

    # Sort so each session's rows are contiguous and time-ordered: the loader's
    # streaming session index assumes that layout.
    df = df.sort_values(["subject_id", "session_id", "timestamp_ns"], kind="stable").reset_index(drop=True)

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

    n_subj = df["subject_id"].nunique()
    dur_h = len(df) / CANONICAL_SAMPLE_RATE_HZ / 3600.0
    print(f"label_scheme={args.label_scheme} classes={cond_to_id}")
    print(
        f"sessions={n_sessions} subjects={n_subj} rows={len(df)} "
        f"duration={dur_h:.2f} h @ {CANONICAL_SAMPLE_RATE_HZ:g} Hz"
    )
    print(f"missing/unavailable recordings={n_missing_files} too-short={n_short} "
          f"subjects-dropped-by-scheme={len(skipped_subjects)}")
    print("\nsubjects per class:")
    print(df.groupby("dataset_activity_label")["subject_id"].nunique())
    print("\nsessions per class:")
    print(df.groupby("dataset_activity_label")[["subject_id", "session_id"]]
            .apply(lambda g: g.groupby(["subject_id", "session_id"]).ngroups))
    print("\nsessions per movement task:")
    print(df.assign(task=df["session_id"].str.rsplit("_", n=1).str[0])
            .groupby("task")[["subject_id", "session_id"]]
            .apply(lambda g: g.groupby(["subject_id", "session_id"]).ngroups))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
