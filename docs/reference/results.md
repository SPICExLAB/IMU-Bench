# Result files

One JSON per cell at `bench/results/<task>/<encoder>__<protocol>__s<seed>.json`, written by the task's scorer:

```json
{
  "task": "microgesture_openwatch",
  "encoder": "mae_5src",
  "protocol": "probe",
  "seed": 42,
  "metrics": {"clip_macro_f1": 0.647, "clip_acc": 0.678, "window_macro_f1": 0.494, "window_acc": 0.536},
  "primary_metric": "clip_macro_f1",
  "eval_split": "test",
  "run": "bench_microgesture_openwatch__mae_5src__probe__s42",
  "checkpoint": "runs/.../checkpoints/best.pt",
  "checkpoint_step": 1200,
  "selection": {"split": "val", "monitor": "macro_f1", "mode": "max"},
  "job_ids": [7212525],
  "provenance": {"legacy_run": false, "scorer": "bench.score.microgesture", "scored_at": "..."},
  "status": "ok",
  "notes": "",
  "schema_version": 1
}
```

| Field | Meaning |
|---|---|
| `metrics` | every metric the scorer computes; the leaderboard reads `primary_metric` |
| `eval_split` | `test` where a test split exists (gesture, retrieval), otherwise the held-out validation split the card defines |
| `checkpoint_step` | the step selected by the card's rule on the validation split |
| `provenance` | scorer, time, and whether the run predates the harness (`legacy_run`) |

`imu-bench leaderboard` collects these files into `bench/leaderboard_tracks.json` (per track and protocol: rows,
cells, ranks) and the Markdown/HTML tables. The released result files, the registry and the leaderboard data are
mirrored in the dataset [YiquanLi/IMU-Bench-results](https://huggingface.co/datasets/YiquanLi/IMU-Bench-results),
whose `results_long.csv` lists every cell as one row.
