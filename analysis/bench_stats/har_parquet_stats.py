"""Per-dataset stats of the HAR master parquet for the IMU-Bench task table.

subjects / sessions / hours / classes per `dataset`. Hours = sum over sessions of
(last - first timestamp), so native sampling rate does not matter. Classes = distinct
non-negative dataset_activity_id. Streams row groups; writes analysis/bench_stats/har_parquet_stats.json.
"""
import json, os, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq

P = "/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet"
f = pq.ParquetFile(P)
ses = {}      # (dataset, subject, session) -> [min_ts, max_ts]
cls = {}
for i in range(f.metadata.num_row_groups):
    t = f.read_row_group(i, columns=["dataset", "subject_id", "session_id", "timestamp_ns", "dataset_activity_id"]).to_pandas()
    g = t.groupby(["dataset", "subject_id", "session_id"], observed=True)["timestamp_ns"].agg(["min", "max"])
    for k, (mn, mx) in g.iterrows():
        cur = ses.get(k)
        ses[k] = [mn, mx] if cur is None else [min(cur[0], mn), max(cur[1], mx)]
    for d, ids in t[t.dataset_activity_id >= 0].groupby("dataset", observed=True)["dataset_activity_id"]:
        cls.setdefault(d, set()).update(np.unique(ids).tolist())
    if i % 50 == 0:
        print(f"row group {i}/{f.metadata.num_row_groups}", flush=True)
out = {}
for (d, s, se), (mn, mx) in ses.items():
    o = out.setdefault(d, {"subjects": set(), "sessions": 0, "hours": 0.0})
    o["subjects"].add(s); o["sessions"] += 1; o["hours"] += (mx - mn) / 3.6e12
for d, o in out.items():
    o["subjects"] = len(o["subjects"]); o["hours"] = round(o["hours"], 2); o["classes"] = len(cls.get(d, ()))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "har_parquet_stats.json"), "w"), indent=1)
for d, o in sorted(out.items()):
    print(d, o)
