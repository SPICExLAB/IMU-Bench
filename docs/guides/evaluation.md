# Evaluation

## Running cells

```bash
imu-bench eval --encoder my_encoder.yaml                        # registers the card, then all 20 cells, here
imu-bench eval --encoder my_encoder --tasks har_hhar pd_pads --protocols probe     # a subset while developing
imu-bench eval --encoder my_encoder --slurm                     # one SLURM train -> score chain per cell
imu-bench eval --encoder my_encoder --force                     # rerun cells that already have a result
imu-bench eval --encoder my_encoder --dry-run                   # list what would run
```

Each cell trains with the task card's recipe and scores the selected checkpoint once:

| Artefact | Path |
|---|---|
| run directory (checkpoints, logs) | `runs/bench_<task>__<encoder>__<protocol>__s42/` |
| generated config | `bench/generated/configs/bench_<task>__<encoder>__<protocol>__s42.yaml` |
| result file | `bench/results/<task>/<encoder>__<protocol>__s42.json` |

Model selection uses the validation split only; the evaluation split is scored once, by the task's scorer in
`bench/score/`. The primary seed is 42 for every row of the leaderboard.

## Protocols

| Protocol | Encoder | Decoder | Enters the ranking |
|---|---|---|---|
| `probe` | frozen | the card's light decoder, trained | yes |
| `finetune` | trainable, from the pretrained weights | the same decoder | yes |
| `scratch` | trainable, random init | the same decoder | reference row |

The decoders are deliberately light: a linear head for classification, width-192 attention decoders with one to four
layers for the kinematic tasks, a projection for retrieval. The paper shows that a deeper decoder narrows or reverses
the measured advantage of pretraining, which is why decoder capacity is part of the card and not of the submission.

## Checking a submission

```bash
imu-bench check my_encoder
```

reports whether all twenty cells exist, whether every trained config is identical to its task card (decoder,
learning rate, budget, split, selection rule), whether the card is complete (`name`, `params_m`, `pretrain_data`,
`normalize_method`) and which track the submission qualifies for. A cell that changed the recipe is listed outside
the ranking.

## Reading results

```bash
imu-bench leaderboard            # bench/leaderboard_tracks.json + the two track tables
```

A row is shown only when all ten tasks are scored under both protocols. Rows are ordered by mean rank over the seven
columns (arm pose, full-body pose, odometry, gesture, IMU–text, Parkinson's, mean activity F1 over the four sets);
ties share a rank, the best value per column is marked at the printed precision, and reference rows
(`control_random`, `control_raw`, `scratch`, released external models) are printed but never ranked.
The format of a result file is in [Result files](../reference/results.md).
