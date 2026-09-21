#!/usr/bin/env python3
"""SmartFallMM (Texas State SmartFall group) -> the benchmark's canonical parquet.

    python scripts/prepare_smartfallmm.py        # writes /projects/b53017/vbc1591/smartfallmm_dataset.parquet

Source: https://github.com/txst-cs-smartfall/SmartFallMM-Dataset (commit c538ee0c, 2026-08-20), cloned to
/projects/b53017/vbc1591/SmartFallMM-Dataset. "(c) SmartFall Group, Texas State University. All rights reserved";
use requires acknowledgment. We do not redistribute it.

Device. Only the RIGHT-WRIST MbientLab Meta sensor (`meta_wrist`): accelerometer + gyroscope, regular 20 ms
timestamps. The placement is stated by the authors ("A wrist meta sensor was placed on the right wrist", Haque et
al., Sensors 24(19):6235, 2024) and matches the repository's placement figure. The left-wrist Huawei watch is not
used: its timestamps are batched and irregular (median effective rate 14.5 Hz accelerometer / 8.9 Hz gyroscope
against a nominal 32 Hz), so it cannot be resampled by time.

Known defect of this device, kept and disclosed rather than hidden: the sensor ranges were +-2 g and +-125 deg/s, so
both streams SATURATE. In the young group the gyroscope sits at its limit for a median 14 % of the samples of a
trial (49 % for hand waving) and in 95-99 % of all trials; the accelerometer clips in 92 % of fall trials (about
1 % of their samples, the impact). Every encoder sees the same clipped signal.

Pairing. Accelerometer and gyroscope files are paired by (subject, activity, trial) AND by time: both are read on
their local-time clock, the overlapping interval is kept, and both are linearly interpolated onto one 50 Hz grid --
never concatenated by row. Two file layouts occur (6 columns with epoch/ISO/elapsed, 4 columns with a local time
string). Trials are dropped when the streams overlap for less than one 2.56 s window (18 % of young trials have
disjoint clocks), when a gap inside the overlap exceeds 0.5 s, or when the accelerometer file does not look like an
accelerometer (median norm outside 0.7-1.3 g; a few files of the older group hold gyroscope values). 25 Hz / 32 Hz
recordings of the first sessions are interpolated to 50 Hz like the rest.

Units follow hhar_dataset.parquet: acc in m/s^2 (g * 9.80665, gravity included), gyro in rad/s (deg/s * pi/180).

Labels. `dataset_activity_id` is the dataset's activity 1-14; `global_activity_id` is the binary fall-detection label
(1 = activity of daily living, 2 = fall; activities 10-14 are back/front/left/right/rotate falls). Every row of a
trial carries the trial's label: the files are the authors' manually trimmed activity segments.

Groups. The 31 young participants (falls + ADLs) are dataset `smartfallmm`; the older participants performed ADLs
only and are written as a separate dataset `smartfallmm_old`, so a task that trains on `smartfallmm` never sees
them. They can serve as a false-alarm check; they cannot validate fall detection in older adults.
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = "/projects/b53017/vbc1591/SmartFallMM-Dataset"
OUT = "/projects/b53017/vbc1591/smartfallmm_dataset.parquet"
G, HZ, MIN_OVERLAP, MAX_GAP = 9.80665, 50, 2.56, 0.5
PAT = re.compile(r"S(\d+)A(\d+)T(\d+)\.csv$")
ACT = {1: "drinking water", 2: "picking up object", 3: "putting on/taking off jacket", 4: "stepping up", 5: "sweeping floor",
       6: "washing hands", 7: "waving hand", 8: "TUG test/walking", 9: "sit/stand", 10: "back fall", 11: "front fall",
       12: "left fall", 13: "right fall", 14: "rotate fall"}
FALLS = {10, 11, 12, 13, 14}


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
    t = t[ok].astype("int64").values / 1e9
    x = x[ok].values.astype(np.float64)
    o = np.argsort(t, kind="stable")
    t, x = t[o], x[o]
    keep = np.concatenate([[True], np.diff(t) > 0])          # duplicated timestamps break interpolation
    return t[keep], x[keep]


def main():
    frames, log = [], []
    for grp, ds in (("young", "smartfallmm"), ("old", "smartfallmm_old")):
        A = {f for f in os.listdir(f"{ROOT}/{grp}/accelerometer/meta_wrist") if PAT.search(f)}
        Gy = {f for f in os.listdir(f"{ROOT}/{grp}/gyroscope/meta_wrist") if PAT.search(f)}
        for f in sorted(A | Gy):
            s, a, tr = map(int, PAT.search(f).groups())
            why = None
            ac = load(f"{ROOT}/{grp}/accelerometer/meta_wrist/{f}") if f in A else None
            gy = load(f"{ROOT}/{grp}/gyroscope/meta_wrist/{f}") if f in Gy else None
            if ac is None or gy is None:
                why = "unpaired or unparsable"
            elif a not in ACT:
                why = "unknown activity id"
            else:
                (ta, xa), (tg, xg) = ac, gy
                t0, t1 = max(ta[0], tg[0]), min(ta[-1], tg[-1])
                if t1 - t0 < MIN_OVERLAP:
                    why = "acc/gyro overlap < 2.56 s"
                elif not 0.7 <= np.median(np.linalg.norm(xa, axis=1)) <= 1.3:
                    why = "accelerometer file is not in g"
                else:
                    ia, ig = (ta >= t0 - 0.1) & (ta <= t1 + 0.1), (tg >= t0 - 0.1) & (tg <= t1 + 0.1)
                    if max(np.diff(ta[ia]).max(initial=0), np.diff(tg[ig]).max(initial=0)) > MAX_GAP:
                        why = "gap > 0.5 s inside the overlap"
            log.append(dict(group=grp, file=f, subject=s, activity=a, trial=tr, dropped=why))
            if why:
                continue
            n = int(np.floor((t1 - t0) * HZ)) + 1
            grid = t0 + np.arange(n) / HZ
            acc = np.stack([np.interp(grid, ta, xa[:, k]) for k in range(3)], 1) * G
            gyr = np.stack([np.interp(grid, tg, xg[:, k]) for k in range(3)], 1) * (np.pi / 180.0)
            fall = a in FALLS
            frames.append(pd.DataFrame({
                "dataset": ds, "subject_id": f"{grp}_S{s:02d}", "session_id": f"{grp}_S{s:02d}_A{a:02d}_T{tr:02d}",
                "timestamp_ns": (np.arange(n) * (1e9 / HZ)).astype(np.int64),
                "acc_x": acc[:, 0].astype(np.float32), "acc_y": acc[:, 1].astype(np.float32), "acc_z": acc[:, 2].astype(np.float32),
                "gyro_x": gyr[:, 0].astype(np.float32), "gyro_y": gyr[:, 1].astype(np.float32), "gyro_z": gyr[:, 2].astype(np.float32),
                "global_activity_id": np.int32(2 if fall else 1), "global_activity_label": "fall" if fall else "adl",
                "dataset_activity_id": np.int32(a), "dataset_activity_label": ACT[a]}))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT, index=False)
    lg = pd.DataFrame(log)
    lg.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis", "smartfallmm", "build_log.csv"), index=False)
    print(f"wrote {OUT}: {len(df):,} rows, {df.session_id.nunique():,} trials, {len(df) / HZ / 3600:.2f} h")
    for ds, g in df.groupby("dataset"):
        tr = g.drop_duplicates("session_id")
        print(f"  {ds}: {g.subject_id.nunique()} participants, {len(tr)} trials ({int((tr.global_activity_id == 2).sum())} falls, "
              f"{int((tr.global_activity_id == 1).sum())} ADLs), {len(g) / HZ / 3600:.2f} h; windows of 2.56 s at 50 % hop ~ "
              f"{int(sum(max(0, (n - 128) // 64 + 1) for n in g.groupby('session_id').size()))}")
        print("    trials per activity:", tr.groupby("dataset_activity_id").size().to_dict())
    print("dropped:", lg[lg.dropped.notna()].groupby(["group", "dropped"]).size().to_dict())


if __name__ == "__main__":
    main()
