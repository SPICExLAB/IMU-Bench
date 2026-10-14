#!/usr/bin/env python
"""Convert the user-study smartwatch recordings into the HAR parquet schema used by the bench.

    python scripts/prepare_userstudy_watch.py [--src DIR] [--out PARQUET]

Source: /projects/b53017/vbc1591/User_study_watch/p{1..10}/P<n><activity>_<yy-mm-dd>_<hh-mm-ss>_motion(1).txt,
one recording of one activity per file, ~50 Hz, whitespace-separated, 21 columns (CoreMotion-style):
  0 timestamp (s) | 1-3 user acceleration (g, gravity removed) | 4-6 gravity (g) | 7-9 rotation rate (rad/s)
  10-12 magnetic field (uT) | 13-15 attitude (rad) | 16-19 quaternion | 20 device uptime (s)
The file at the top level (Aidan_...) has no activity label and is skipped.

Output rows follow hhar_dataset.parquet: acc = (user acceleration + gravity) * 9.80665 m/s^2, gyro = rotation
rate in rad/s, timestamp_ns from column 0.

Split (user decision 2026-09-18: fine-grained classes, split in time within each recording): the first 80 %
of every recording's samples are training data and the last 20 % validation, with one 2.56 s window (128
samples) dropped at the boundary so no window straddles it. The split is carried in session_id as
"<recording>__train" / "<recording>__val" (splits.split_by_session_suffix). Subjects therefore appear in both
splits -- this task is NOT subject-disjoint.

Labels: the file names carry 69 raw activity names. Only spelling variants and phrasings of the same activity
are merged (see MERGE); every other activity keeps its own class -> 60 classes. The raw -> class map is written
next to the parquet as userstudy_watch_labels.json.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

SRC = "/projects/b53017/vbc1591/User_study_watch"
OUT = "/projects/b53017/vbc1591/userstudy_watch_dataset.parquet"
G = 9.80665
GAP = 128            # one 2.56 s window at 50 Hz
TRAIN_FRAC = 0.8

# raw (lower-cased) -> readable class name. Spelling variants / same-activity phrasings share a class.
MERGE = {
    "stiringcoffee": "stirring coffee", "stiringcoffe": "stirring coffee", "stircoffe": "stirring coffee",
    "wavinggoodbye": "waving goodbye",
    "typing": "typing on keyboard", "typingonkeyboard": "typing on keyboard", "typingonkeybiard": "typing on keyboard",
    "usingmouse": "using mouse", "usingmousecomputer": "using mouse",
    "handsanitizer": "applying hand sanitizer", "applyhandstanitizer": "applying hand sanitizer",
    "drawingaircircle": "drawing circle in air", "drawcircleinair": "drawing circle in air",
    "unscrewwaterbottle": "unscrewing bottle", "unscrewbottle": "unscrewing bottle",
}
READABLE = {
    "airheart": "air heart", "basketballshooting": "basketball shooting", "clickingpen": "clicking pen",
    "crumblingpaper": "crumpling paper", "drawfigureeight": "drawing figure eight", "drawing": "drawing",
    "eatingwithchopsticks": "eating with chopsticks", "filingnails": "filing nails", "fistbump": "fist bump",
    "fistpump": "fist pump", "flickingbatmittonracket": "flicking badminton racket", "foldingpaper": "folding paper",
    "handgesturesequence": "hand gesture sequence", "karatechop": "karate chop", "massagegiving": "giving massage",
    "opencloselaptop": "opening and closing laptop", "opengatorade": "opening Gatorade bottle",
    "openingairpodscase": "opening AirPods case", "organizefridge": "organizing fridge", "phonescrolling": "phone scrolling",
    "pointing": "pointing", "pouringwaterbetweencups": "pouring water between cups", "puttingonlotion": "putting on lotion",
    "readingjapanesemanga": "reading manga", "rockpaperscissors": "rock paper scissors", "rollingupposter": "rolling up poster",
    "scratchingnails": "scratching with nails", "screwmasonjar": "screwing mason jar", "setting3dprinter": "setting up 3D printer",
    "shadowboxing": "shadow boxing", "shuffledeckofcards": "shuffling cards", "snapping": "finger snapping",
    "takingphoneoutofpocket": "taking phone out of pocket", "tearinguppaper": "tearing paper", "tieshoelace": "tying shoelace",
    "turningbookpages": "turning book pages", "twistingscrewdriver": "twisting screwdriver", "unbuttonjacket": "unbuttoning jacket",
    "unpluggingusbcable": "unplugging USB cable", "unscrewlipbalm": "unscrewing lip balm", "usingstyalus": "using stylus",
    "wavehello": "waving hello", "whiteboardwiping": "wiping whiteboard", "windingcord": "winding cord",
    "windingrope": "winding rope", "wipingglass": "wiping glass", "wipinglaptop": "wiping laptop",
    "wrirtingonwhiteboard": "writing on whiteboard", "writing": "writing", "writingwithmarker": "writing with marker",
    "writingwithpen": "writing with pen", "zippingsleepingbag": "zipping sleeping bag", "zipunzipbag": "zipping bag",
}


def raw_activity(path):
    m = re.match(r"^P\d+(.+?)_\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_motion", os.path.basename(path))
    if not m:
        raise ValueError(f"unexpected file name: {path}")
    return m.group(1).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.src, "p*", "*.txt")))
    raw = {f: raw_activity(f) for f in files}
    cls_of = {r: MERGE.get(r, READABLE.get(r)) for r in set(raw.values())}
    missing = sorted(r for r, c in cls_of.items() if c is None)
    if missing:
        raise SystemExit(f"no class name for raw labels: {missing}")
    classes = sorted(set(cls_of.values()))
    cid = {c: i for i, c in enumerate(classes)}
    frames, recs = [], []
    for f in files:
        pid = os.path.basename(os.path.dirname(f))
        x = np.loadtxt(f, usecols=range(10))
        x = x[np.argsort(x[:, 0], kind="stable")]
        acc = ((x[:, 1:4] + x[:, 4:7]) * G).astype(np.float32)
        gyr = x[:, 7:10].astype(np.float32)
        ts = np.round(x[:, 0] * 1e9).astype(np.int64)
        n = len(x); cut = int(n * TRAIN_FRAC)
        parts = {"train": slice(0, cut), "val": slice(cut + GAP, n)}
        c = cls_of[raw[f]]
        rec_id = f"{pid}_{raw[f]}"
        for part, sl in parts.items():
            k = len(range(n)[sl])
            frames.append(pd.DataFrame({
                "dataset": "userstudy_watch", "subject_id": pid, "session_id": f"{rec_id}__{part}",
                "timestamp_ns": ts[sl], "acc_x": acc[sl, 0], "acc_y": acc[sl, 1], "acc_z": acc[sl, 2],
                "gyro_x": gyr[sl, 0], "gyro_y": gyr[sl, 1], "gyro_z": gyr[sl, 2],
                "global_activity_id": np.int32(cid[c]), "global_activity_label": c,
                "dataset_activity_id": np.int32(cid[c]), "dataset_activity_label": c}))
            recs.append({"recording": rec_id, "part": part, "samples": k, "seconds": round(k / 50.0, 1), "class": c})
    df = pd.concat(frames, ignore_index=True)
    for col in ("global_activity_id", "dataset_activity_id"):
        df[col] = df[col].astype(np.int32)
    df.to_parquet(a.out, index=False)
    meta = {"source": a.src, "recordings": len(files), "subjects": sorted({os.path.basename(os.path.dirname(f)) for f in files}),
            "classes": classes, "raw_to_class": {r: cls_of[r] for r in sorted(cls_of)},
            "split": "per recording: first 80% train, last 20% val, 128-sample gap", "parts": recs}
    json.dump(meta, open(os.path.join(os.path.dirname(a.out), "userstudy_watch_labels.json"), "w"), indent=1)
    rs = pd.DataFrame(recs)
    print(f"wrote {a.out}: {len(df):,} rows, {len(files)} recordings, {len(classes)} classes, "
          f"{df.subject_id.nunique()} subjects")
    print("seconds per part:", rs.groupby("part").seconds.agg(["sum", "min", "median"]).round(1).to_dict())


if __name__ == "__main__":
    main()
