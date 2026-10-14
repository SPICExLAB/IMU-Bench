#!/usr/bin/env python3
"""SmartFallMM (SmartFall Group, Texas State University) -> the HAR parquet schema, right-wrist Meta sensor.

    python scripts/prepare_smartfallmm_dataset.py [--out /projects/b53017/vbc1591/smartfallmm_dataset.parquet]

Source: https://github.com/txst-cs-smartfall/SmartFallMM-Dataset (commit c538ee0c). The wrist MetaMotion sensor is
worn on the RIGHT wrist and the Huawei watch on the left (Haque et al., Sensors 24(19):6235, 2024), so the Meta
sensor is the stream that matches IMU-Bench's right-wrist input. Only the young group performs falls (activities
10-14, onto an air mattress); the older group records activities of daily living only and is not used here.

What this script does, and why (see analysis/smartfallmm/01_inspect_meta_wrist.py for the numbers):
  * pairs accelerometer and gyroscope files by (subject, activity, trial) and aligns them BY TIMESTAMP, never by
    row: the two streams start at different instants and about 18 % of young trials do not overlap at all;
  * keeps a trial only if both streams are natively ~50 Hz (45-55 Hz): about a quarter of the early recordings are
    25 Hz accelerometer / 32 Hz gyroscope, which we do not upsample into the benchmark;
  * resamples both streams linearly onto one uniform 50 Hz grid over their overlap, and drops a trial whose overlap is
    shorter than one 2.56 s window or contains a gap longer than 0.2 s;
  * converts units: accelerometer g -> m/s^2, gyroscope deg/s -> rad/s.
The sensor was configured at +-2 g and +-125 deg/s, so both streams clip (the gyroscope in ~15 % of samples); the
clipping is a property of the release and is left as recorded.

Columns follow the other activity parquets. `dataset_activity_id` is the binary fall label (1 = ADL, 2 = fall);
`global_activity_id` keeps the 14-way activity id of the release.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

ROOT = "/projects/b53017/vbc1591/SmartFallMM-Dataset"
PAT = re.compile(r"S(\d+)A(\d+)T(\d+)\.csv$")
ACT = {1: "drinking_water", 2: "pick_up_object", 3: "jacket_on_off", 4: "stepping_up", 5: "sweeping_floor",
       6: "washing_hand", 7: "waving_hand", 8: "tug_walking", 9: "sit_stand", 10: "back_fall", 11: "front_fall",
       12: "left_fall", 13: "right_fall", 14: "rotate_fall"}
FALLS = {10, 11, 12, 13, 14}
HZ, G, D2R = 50.0, 9.80665, np.pi / 180.0


def load(path):
    try:
        df = pd.read_csv(path, header=None, dtype=str, on_bad_lines="skip")
    except Exception:
        return None
    if df.shape[1] >= 6:
        tcol, xyz = 1, [3, 4, 5]
    elif df.shape[1] == 4:
        tcol, xyz = 0, [1, 2, 3]
    else:
        return None
    t = pd.to_datetime(df[tcol], errors="coerce")
    x = df[xyz].apply(pd.to_numeric, errors="coerce")
    ok = t.notna() & x.notna().all(axis=1)
    if ok.sum() < 5:
        return None
    t = t[ok].astype("int64").values / 1e9; x = x[ok].values.astype(np.float64)
    o = np.argsort(t, kind="stable"); t, x = t[o], x[o]
    keep = np.concatenate([[True], np.diff(t) > 0])          # duplicate timestamps break interpolation
    return t[keep], x[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/projects/b53017/vbc1591/smartfallmm_dataset.parquet")
    ap.add_argument("--group", default="young")
    a = ap.parse_args()
    da, dg = f"{ROOT}/{a.group}/accelerometer/meta_wrist", f"{ROOT}/{a.group}/gyroscope/meta_wrist"
    names = sorted({f for f in os.listdir(da) if PAT.search(f)} & {f for f in os.listdir(dg) if PAT.search(f)})
    drop = {"unparsable": 0, "not_50hz": 0, "short_overlap": 0, "gap": 0}
    frames, kept = [], []
    for f in names:
        s, act, tr = map(int, PAT.search(f).groups())
        ac, gy = load(f"{da}/{f}"), load(f"{dg}/{f}")
        if ac is None or gy is None:
            drop["unparsable"] += 1; continue
        rate = lambda t: (len(t) - 1) / (t[-1] - t[0]) if t[-1] > t[0] else 0.0
        if not (45 <= rate(ac[0]) <= 55 and 45 <= rate(gy[0]) <= 55):
            drop["not_50hz"] += 1; continue
        t0, t1 = max(ac[0][0], gy[0][0]), min(ac[0][-1], gy[0][-1])
        if t1 - t0 < 2.56:
            drop["short_overlap"] += 1; continue
        ina, ing = (ac[0] >= t0 - 0.05) & (ac[0] <= t1 + 0.05), (gy[0] >= t0 - 0.05) & (gy[0] <= t1 + 0.05)
        if max(np.diff(ac[0][ina]).max(initial=0), np.diff(gy[0][ing]).max(initial=0)) > 0.2:
            drop["gap"] += 1; continue
        grid = t0 + np.arange(int(np.floor((t1 - t0) * HZ)) + 1) / HZ
        A = np.stack([np.interp(grid, ac[0], ac[1][:, k]) for k in range(3)], 1) * G
        W = np.stack([np.interp(grid, gy[0], gy[1][:, k]) for k in range(3)], 1) * D2R
        fall = act in FALLS
        frames.append(pd.DataFrame({
            "dataset": "smartfallmm", "subject_id": f"{a.group}_S{s:02d}", "session_id": f"{a.group}_S{s:02d}A{act:02d}T{tr:02d}",
            "timestamp_ns": (grid * 1e9).astype(np.int64),
            "acc_x": A[:, 0].astype(np.float32), "acc_y": A[:, 1].astype(np.float32), "acc_z": A[:, 2].astype(np.float32),
            "gyro_x": W[:, 0].astype(np.float32), "gyro_y": W[:, 1].astype(np.float32), "gyro_z": W[:, 2].astype(np.float32),
            "global_activity_id": np.int32(act), "global_activity_label": ACT[act],
            "dataset_activity_id": np.int32(2 if fall else 1), "dataset_activity_label": "fall" if fall else "adl"}))
        kept.append((s, act, tr, len(grid) / HZ, fall))
    df = pd.concat(frames, ignore_index=True)
    for c in ("global_activity_id", "dataset_activity_id"):
        df[c] = df[c].astype(np.int32)
    df.to_parquet(a.out, index=False)
    k = pd.DataFrame(kept, columns=["subject", "activity", "trial", "seconds", "fall"])
    print(f"paired files {len(names)} | kept {len(k)} trials | dropped {drop}")
    print(f"subjects {k.subject.nunique()} | fall trials {int(k.fall.sum())} ({k[k.fall].seconds.sum() / 60:.1f} min) | ADL trials {int((~k.fall).sum())} ({k[~k.fall].seconds.sum() / 60:.1f} min) | total {k.seconds.sum() / 3600:.2f} h")
    per = k.groupby("subject").agg(trials=("trial", "size"), falls=("fall", "sum"))
    print("trials per subject: min", per.trials.min(), "median", int(per.trials.median()), "max", per.trials.max(), "| subjects without any fall:", per[per.falls == 0].index.tolist())
    print("trials per activity:", k.groupby("activity").size().to_dict())
    print(f"wrote {a.out} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
