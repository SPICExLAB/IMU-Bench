# CLI

`imu-bench` is installed by `pip install -e .` (entry point `bench.cli:main`).

| Command | Does |
|---|---|
| `imu-bench tasks` | lists the ten task cards: dataset, split, input, metric |
| `imu-bench eval --encoder <id or card.yaml> [--tasks …] [--protocols probe finetune] [--seed 42] [--slurm] [--force] [--dry-run]` | registers a card file under `bench/registry/encoders.d/`, then trains and scores every requested cell, here or as SLURM chains |
| `imu-bench check <id>` | completeness (20 cells), protocol fidelity (trained configs identical to the task cards), card completeness, track eligibility |
| `imu-bench package <id> [--out submission] [--allow-incomplete]` | lays out `results/`, `registry/<id>.yaml`, `configs/` and `check.json` for the leaderboard pull request |
| `imu-bench leaderboard [--site index.html] [--readme README.md] [--quiet]` | regenerates `bench/leaderboard_tracks.json` and the two track tables; optionally rewrites the website and README blocks |
| `imu-bench audit <dir> --subject-regex '<pattern>' [--glob '**/*.csv'] [--acc-cols AccX AccY AccZ] [--threshold 0.999]` | finds duplicated, mirrored or rotated participants in a dataset release |
| `imu-bench fetch-weights [ids…]` | downloads the released weights of the open-track models to where their cards expect them |

The lower-level modules remain available: `python -m bench.make_config`, `python -m bench.launch`,
`python -m bench.score`, `python -m bench.leaderboard` (see [Reproducing the paper](../guides/reproducing.md)).

Environment variables:

| Variable | Meaning |
|---|---|
| `IMUBENCH_DATA_ROOT` | directory holding the datasets in the layout of [Datasets](../guides/datasets.md); every dataset path in the cards is rewritten to it |
| `IMUBENCH_VENV` | environment directory activated by generated SLURM scripts (default `imu-bench/`) |
| `BENCH_PARTITION`, `BENCH_ACCOUNT` | SLURM partition and account for `--slurm` |
