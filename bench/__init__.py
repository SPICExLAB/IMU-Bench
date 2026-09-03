"""IMU-Bench: controlled multi-task benchmark + leaderboard for this repo.

Layout
------
bench/registry/tasks.yaml      task cards: dataset, split, unified input/output spec, metrics, scorer
bench/registry/encoders.yaml   submissions: pretrained encoders (+ external specialist baselines)
bench/registry/protocols.yaml  adaptation recipes: probe / finetune / scratch
bench/schema.py                the one result-JSON contract every scorer writes
bench/make_config.py           task x encoder x protocol -> materialised training yaml
bench/launch.py                write + sbatch a train->score chain for a cell (or --fill-missing)
bench/score/*.py               one scorer adapter per task family (wraps existing scripts)
bench/harvest.py               import legacy runs / analysis JSONs into bench/results/
bench/leaderboard.py           bench/results/**.json -> LEADERBOARD.md (+ html)
bench/results/<task>/<encoder>__<protocol>__s<seed>.json
"""
