# IMU-Bench — controlled multi-task benchmark & leaderboard

> The paper's ten tasks, the six compared objectives and the commands to run a cell are in the top-level README.md; the notes below are the harness's operational detail and predate the final task list.

A SWE-Bench-style layer over this repo: fixed task instances, one submission
format (an encoder + an adaptation protocol), one scorer per task, one result
JSON schema, and a leaderboard generated only from those JSONs.

```
bench/registry/tasks.yaml       task cards (dataset, split, unified I/O, metric, selection, scorer, templates)
bench/registry/encoders.yaml    submissions (pretrained ViT1D encoders + external specialist baselines)
bench/registry/protocols.yaml   probe / finetune / scratch
bench/make_config.py            task × encoder × protocol × seed  →  bench/generated/configs/<run>.yaml
bench/launch.py                 sbatch train→score chains, score-only jobs, --fill-missing
bench/score/                    one scorer per task family → bench/results/<task>/<enc>__<proto>__s<seed>.json
bench/leaderboard.py            results → LEADERBOARD.md (+ bench/leaderboard.json)
bench/generated/submissions.tsv every sbatch call (job id, kind, cell, run, script)
```

## The six tasks (v1)

| id | dataset | input `[C, T]` | output | selection | reported | primary |
|---|---|---|---|---|---|---|
| `arm_pose_nymeria` | Nymeria real wrist, 867/101 sessions | `[6,512]` 10.24 s @50 Hz | `[32, 5, 6]` rot6d, arm chain | val loss (6D-MSE) | val | pos err (cm) ↓ |
| `full_body_pose_nymeria` | Nymeria real wrist | `[6,512]` | `[32, 23, 6]` rot6d | val MPJPE | val | MPJPE (mm) ↓ |
| `odometry_nymeria_worldframe` | Nymeria real wrist → pelvis | `[6,128]` 2.56 s | `[8, 4, 3]` velocity, canonical world frame | val loss (vel-MSE) | val | ATE (m) ↓ |
| `microgesture_openwatch` | OpenWatch, posture-stratified 33/5/5 subj | `[6,128]` 1 s @100 Hz (zero-padded) | `[6]` logits → k=3 clip vote | val window macro-F1 | **test** | clip macro-F1 ↑ |
| `language_humanml3d` | HumanML3D / UniMTS_data, official val/test | `[6,512]` wrap-padded | `[512]` CLIP-space embedding | fixed 8k budget, `latest.pt` | **test** | R@1 @ G=100 (mean i2t/t2i) ↑ |
| `pd_pads` | PADS, 375/94 subjects | `[6,128]` 2.56 s (gravity-free) | `[3]` logits → subject mean-prob | val window bal-acc | val | subject bal-acc ↑ |

All inputs are a single wrist IMU in the FLU frame, acc m/s² (+gravity except PADS), gyro rad/s, z-scored per window; the acc-only ablation sees `x[:3]`.

## What "controlled" means here

Within a task every encoder is trained by the **same generated config** (one template per protocol, LR/steps/patience/monitor from the task card's `recipe`), the same seed, the same split, the same checkpoint rule, and scored by the same scorer. Between tasks the recipe differs — that is a property of the task, not of the encoder.

Encoders (all ViT1D 384/12/6, 21M): `mae_11src` (ours), `mae_5src`, `mae_noaltattn`, `mae_acconly`, `focal`, `simmtm`, `relcon`, plus `scratch`. External rows (`mobileposer`, `tlio`, `airio`, `unimts*`, `mixtoken`, `normwear`) are harvested from their own scripts and shown in italics; they never enter the encoder ranking.

## Daily use

```bash
source TransfHAR_env/bin/activate
python -m bench.launch --fill-missing --dry-run      # what would run
python -m bench.launch --fill-missing                # submit everything missing (train+score or score-only)
python -m bench.launch --fill-missing --tasks language_humanml3d --seeds 43 44   # replicate seeds
python -m bench.launch --cell pd_pads focal probe    # one cell
python -m bench.score  --task pd_pads --encoder focal --protocol probe [--run <existing run>]   # (re)score
python -m bench.leaderboard                          # regenerate LEADERBOARD.md
```

A cell's run is `runs/bench_<task>__<encoder>__<protocol>__s<seed>/`. Legacy runs whose config is recipe-equivalent are mapped in `tasks.yaml: legacy_runs` and scored into the same cell with `provenance.legacy_run = true` (shown as `†`).

## Adding a task / encoder

* **Encoder:** add an entry to `encoders.yaml` (run dir, `alt_attention`, `channels`, `shared_kernel`) — nothing else.
* **Task:** add a card to `tasks.yaml` (templates for probe/finetune/scratch, `recipe`, metrics with direction, `scorer`) and a `bench/score/<family>.py` exposing `score(run_dir, ckpt_path, task) -> (metrics, info)`.

## Known v1 caveats (also printed at the bottom of LEADERBOARD.md)

* Pose / odometry / PD report the **val** split they select on (no test split exists yet for the fixed Nymeria/PADS partitions). Microgesture and language report a real test split.
* Full-body MPJPE is best-of-curve on val (noise-mined; ~±1-2 mm); the finetune LR (1e-4) is the untuned template value for every encoder — the tuned 1e-3 row from the LR sweep is not part of the controlled grid.
* Language: single-seed sd ≈ 0.7 MRR; use `--seeds 43 44`.

## HAR (added 2026-08-17)

Six activity-recognition datasets — WISDM, UTD-MHAD, PAMAP2, RecoFit, HHAR, MHEALTH — are
registered as tasks `har_<dataset>` (family `har`). They enter the **main tables as ONE column,
`har_mean` = mean window macro-F1 over the six sets** (present only when all six are scored), so
HAR does not swamp the mean rank; per-dataset numbers are in the coverage grid and the per-task
detail tables. Recipe = the Experiments_table §3 configs (probe lr 1e-3 / finetune lr 1e-4,
val_loss early stopping, 2.56 s windows, subject-disjoint 20 % val split). Scorer
`bench.score.har` **re-evaluates** the selected checkpoint on the val split with pooled predictions
(runs before 2026-07-18 logged per-batch-averaged F1 — the log is not trusted). DeepConvLSTM
(supervised, same windows/split) is the external reference row.
