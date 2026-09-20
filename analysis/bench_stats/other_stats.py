"""Subjects / hours for the non-master-parquet IMU-Bench datasets -> analysis/bench_stats/other_stats.json.

* HHAR, MHEALTH, PADS: their own parquets (same schema as the master); hours = sum of session spans.
* Nymeria: duration_seconds stored in every processed session pickle; participants parsed from file names.
* OpenWatch: raw CSVs at 100 Hz; subjects mapped to canonical identities (openwatch_loader), '(n)' copies dropped.
* HumanML3D: official split lists; duration = new_joints frames / 20 fps.
"""
import glob, json, os, pickle, re, sys
import numpy as np, pyarrow.parquet as pq

sys.path.insert(0, "/gpfs/home/vbc1591/IMU-SSL")
from imu_lm.data.openwatch_loader import canonical_subject, _DUPLICATE_SUFFIX_RE

OUT = {}

def parquet_stats(path):
    f = pq.ParquetFile(path); ses = {}; cls = {}
    for i in range(f.metadata.num_row_groups):
        t = f.read_row_group(i, columns=["dataset", "subject_id", "session_id", "timestamp_ns", "dataset_activity_id"]).to_pandas()
        for k, (mn, mx) in t.groupby(["dataset", "subject_id", "session_id"], observed=True)["timestamp_ns"].agg(["min", "max"]).iterrows():
            c = ses.get(k); ses[k] = [mn, mx] if c is None else [min(c[0], mn), max(c[1], mx)]
        for d, ids in t[t.dataset_activity_id >= 0].groupby("dataset", observed=True)["dataset_activity_id"]:
            cls.setdefault(d, set()).update(np.unique(ids).tolist())
    out = {}
    for (d, s, _), (mn, mx) in ses.items():
        o = out.setdefault(d, {"subjects": set(), "sessions": 0, "hours": 0.0})
        o["subjects"].add(s); o["sessions"] += 1; o["hours"] += (mx - mn) / 3.6e12
    for d, o in out.items():
        o["subjects"] = len(o["subjects"]); o["hours"] = round(o["hours"], 2); o["classes"] = len(cls.get(d, ()))
    return out

for name in ["hhar", "mhealth", "pads"]:
    OUT[name] = parquet_stats(f"/projects/b53017/vbc1591/{name}_dataset.parquet"); print(name, OUT[name], flush=True)

nym = {}
for split, root in [("train", "/projects/b53017/vbc1591/processed_data_body_Ty_rwrist"),
                    ("val", "/projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist")]:
    files = sorted(glob.glob(os.path.join(root, "*.pkl"))); secs = 0.0; subj = set()
    for p in files:
        m = re.match(r"^\d+_s\d+_([a-z]+_[a-z]+)_act", os.path.basename(p)); subj.add(m.group(1) if m else os.path.basename(p))
        with open(p, "rb") as fh:
            secs += float(pickle.load(fh)["duration_seconds"])
    nym[split] = {"sessions": len(files), "participants": sorted(subj), "hours": round(secs / 3600, 2)}
    print("nymeria", split, len(files), len(subj), round(secs / 3600, 2), flush=True)
tr, va = set(nym["train"]["participants"]), set(nym["val"]["participants"])
OUT["nymeria"] = {"train": {k: (len(v) if k == "participants" else v) for k, v in nym["train"].items()},
                  "val": {k: (len(v) if k == "participants" else v) for k, v in nym["val"].items()},
                  "participants_total": len(tr | va), "val_participants_also_in_train": len(tr & va)}

ow_root = "/projects/b53017/vbc1591/openwatch_raw/open_watch_without_augmentations"
subj = set(); clips = 0; rows = 0
for p in glob.glob(os.path.join(ow_root, "id_*", "*", "*.csv")):
    sid_s = p.split(os.sep)[-3].replace("id_", "")
    if not sid_s.isdigit() or _DUPLICATE_SUFFIX_RE.search(os.path.splitext(os.path.basename(p))[0]):
        continue
    subj.add(canonical_subject(int(sid_s))); clips += 1
    with open(p) as fh:
        rows += sum(1 for _ in fh) - 1
OUT["openwatch"] = {"canonical_subjects": len(subj), "clips": clips, "hours": round(rows / 100 / 3600, 3)}
print("openwatch", OUT["openwatch"], flush=True)

h = "/projects/b53017/vbc1591/HumanML3D/HumanML3D"; hm = {}
for split in ["train", "val", "test"]:
    ids = [l.strip() for l in open(os.path.join(h, f"{split}.txt")) if l.strip()]; frames = 0; n = 0
    for i in ids:
        p = os.path.join(h, "new_joints", i + ".npy")
        if os.path.exists(p):
            frames += np.load(p, mmap_mode="r").shape[0]; n += 1
    hm[split] = {"listed": len(ids), "with_joints": n, "hours": round(frames / 20 / 3600, 2)}
OUT["humanml3d"] = hm; print("humanml3d", hm, flush=True)

json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "other_stats.json"), "w"), indent=1)
