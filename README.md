# IMU-Bench

**Benchmarking Wrist IMU Representations Across Kinematic, Semantic and Clinical Tasks**

Yiquan Li\*, Aidan Bradshaw\*, Chenfeng Gao, Karan Ahuja (Northwestern University; \* equal contribution)

Project website and leaderboard: **https://spicexlab.github.io/imubench/** · Paper: arXiv (link to follow)

IMU-Bench is a benchmark of **ten tasks from eight datasets** for studying what a single wrist IMU
(accelerometer + gyroscope, six channels) representation carries across kinematic, semantic and clinical
targets, and how supervision for one task transfers to the others. This repository holds the registry-driven
harness that produced every number in the paper, the task cards and generated per-cell configs, the scorers,
the per-cell result files, the paper table and figure generators, and the pretraining recipe of **InrtlWrist**,
the masked-autoencoder reference encoder.

## The benchmark

| Task (id in `bench/registry/tasks.yaml`) | Dataset | Output | Primary metric | Split |
|---|---|---|---|---|
| Arm pose `arm_pose_nymeria` | Nymeria real right-wrist IMU | 5 right-arm joint rotations per 0.32 s patch | position error (cm) ↓ | recording session |
| Full-body pose `full_body_pose_nymeria` | Nymeria | 23 SMPL joint rotations per patch | root-relative MPJPE (mm) ↓ | recording session |
| Inertial odometry `odometry_nymeria_local` | Nymeria | pelvis velocity in the wrist frame, integrated to a trajectory | ATE (m) ↓ | recording session |
| Microgesture `microgesture_openwatch` | OpenWatch | 5 gestures + background, voted per clip | clip macro-F1 ↑ | subject (43 canonical ids) |
| IMU–text retrieval `language_humanml3d` | HumanML3D, simulated wrist IMU | alignment with CLIP ViT-B/32 text embeddings | mean Recall@1, galleries of 100 ↑ | official test split |
| Parkinson's disease `pd_pads` | PADS | healthy / Parkinson's / differential diagnosis per participant | participant macro-F1 ↑ | subject-disjoint (469 participants) |
| Activity `har_hhar`, `har_mhealth`, `har_prism` | HHAR (6), MHEALTH (12), PRISM (5) | 2.56 s window label | macro-F1 ↑ | subject-disjoint |
| Activity `har_userwatch` | TransfHAR (60) | 2.56 s window label | macro-F1 ↑ | 80/20 in time |

Every task is defined by one **task card** (data, split rule, input window, decoder, optimisation recipe,
selection rule, scorer) and every model by one **encoder card** (checkpoint, architecture, input
normalisation, pretraining corpus). Three protocols share the decoder, optimiser, budget, split and
selection rule within a task: **frozen probe**, **fine-tune**, **from scratch**.

**Encoders compared** (all one 12-layer, width-384 ViT over 0.32 s patches, 21 M parameters, one corpus of
197,094 unlabeled windows from UT-Watch, WISDM, RecoFit, PAMAP2, SAMoSA, Shoaib, OCDetect, EatingSpeed and
wrist motion simulated from Motion-X++ and BONES-SEED):

| id in `encoders.yaml` | objective |
|---|---|
| `mae_5src` | **InrtlWrist** (ours): masked reconstruction with alternating frame / global attention |
| `limubert_5src` | LIMU-BERT, masked reconstruction |
| `simmtm_5src` | SimMTM, reconstruction + contrastive |
| `focal_5src` | FOCAL, contrastive + orthogonality + temporal losses |
| `relcon_5src` | RelCon, relative contrastive loss |
| `sslwear_5src` | SSL-Wearables, transformation prediction |

plus the InrtlWrist ablations (plain attention, accelerometer only), the controls (random frozen encoder,
raw input with zero layers, same backbone from scratch), the released NormWear checkpoint and the retrained
task-specific systems (MobilePoser, IMUPoser, TLIO, Air-IO, Mix-Token, UniMTS wrist-only, DeepConvLSTM,
the PADS BOSS + CatBoost pipeline). See `bench/registry/encoders.yaml`.

## Repository layout

```
bench/registry/tasks.yaml       task cards (dataset, split, input, decoder, recipe, selection, metrics, scorer)
bench/registry/encoders.yaml    encoder cards (run directory / checkpoint, architecture flags, corpus)
bench/registry/protocols.yaml   probe / finetune / scratch
bench/make_config.py            task x encoder x protocol x seed -> bench/generated/configs/<run>.yaml
bench/launch.py                 SLURM submission of train -> score chains (see "Running a cell" for non-SLURM use)
bench/score/                    one scorer per task family -> bench/results/<task>/<encoder>__<protocol>__s<seed>.json
bench/results/                  the per-cell result files behind the paper (metrics, selected checkpoint, provenance)
bench/leaderboard.py            results -> LEADERBOARD.md (+ bench/leaderboard.json)
analysis/paper_tables/          generators of every paper table (main table, per-task tables, audit, ablations)
analysis/plot_*.py              paper figures (radar, cross-task transfer, scaling grid)
paper_results/                  the tidy CSVs behind each figure and table; `python paper_results/build.py` regenerates all
configs/                        pretraining and downstream configs, incl. the InrtlWrist recipe
imu_lm/                         models (ViT1D, decoders, baseline objectives), data loaders, splits, metrics
scripts/                        run_pretrain.py (the trainer used for every cell), cache building, evaluation scripts
baselines/                      task-specific systems: patches for third-party repos (THIRD_PARTY.md), Mix-Token and PADS ports
launch/                         the SLURM scripts used for the paper's runs
```

## Setup

```bash
python -m venv TransfHAR_env && source TransfHAR_env/bin/activate
pip install -r requirements.txt      # install the CUDA build of PyTorch first if needed
```

Python ≥ 3.9. Every run in the paper used one NVIDIA A100 or H100.

## Data

The datasets are not redistributed. Download each from its source, then point the harness at them:
`dataset.root` in `bench/registry/tasks.yaml` for the evaluation sets and the `paths` block of
`configs/base.yaml` for the pretraining caches (the committed values are the paths of our cluster).

* Evaluation: Nymeria (https://projectaria.com/datasets/nymeria/), OpenWatch, HumanML3D with the UniMTS
  simulated-IMU release, PADS (PhysioNet), HHAR, MHEALTH, PRISM. TransfHAR, the 60-activity smartwatch
  set recorded for this paper, will be released separately. Request the download links from https://projectaria.com/datasets/nymeria/ (the per-user signed URL files are not redistributed).
* Pretraining: UT-Watch, WISDM, RecoFit, PAMAP2, SAMoSA, Shoaib, OCDetect, EatingSpeed, Motion-X++ and
  BONES-SEED (`scripts/download/download_bones_seed.py`). Caches are built with
  `scripts/build_multidata_cache.py --config configs/base.yaml --model-config <pretrain config> --out-dir <cache dir>`.
* The full-body pose scorer needs the SMPL model file; see `assets/smpl/README.md`.

All inputs are one wrist, six channels in a forward–left–up frame (acc m/s², gyro rad/s), z-scored per
channel and window; the loaders and unit checks live in `imu_lm/data/`.

## Pretraining InrtlWrist

```bash
python scripts/run_pretrain.py --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic
```

The five baseline objectives use the same corpus and backbone with `imu_lm/objectives/ssl_vit_baselines.py`
(`scripts/run_ssl_baseline.py`); their configs and launch scripts are under `configs/` and `launch/pretrain/`.
Pretrained checkpoints will be published alongside the paper.

## Running a cell

A cell is one task × encoder × protocol × seed. Its config is generated from the registry, trained with the
same trainer as pretraining, and scored by the task's scorer:

```bash
python -m bench.make_config --task pd_pads --encoder mae_5src --protocol probe --seed 42
python scripts/run_pretrain.py --config configs/base.yaml \
    --model-config bench/generated/configs/bench_pd_pads__mae_5src__probe__s42.yaml \
    --run-name bench_pd_pads__mae_5src__probe__s42
python -m bench.score --task pd_pads --encoder mae_5src --protocol probe --seed 42
python -m bench.leaderboard                       # regenerate LEADERBOARD.md from bench/results/
```

On a SLURM cluster `python -m bench.launch --cell pd_pads mae_5src probe` submits the train → score chain and
`python -m bench.launch --fill-missing --dry-run` lists every cell that has no result yet.
The generated configs of every cell in the paper are committed under `bench/generated/configs/`.

**Adding an encoder** is one entry in `encoders.yaml` (run directory or checkpoint, attention flag, channels).
**Adding a task** is one card in `tasks.yaml` plus a scorer in `bench/score/` exposing
`score(run_dir, ckpt_path, task) -> (metrics, info)`.

## Reproducing the paper's tables and figures

```bash
python paper_results/build.py        # every table (.tex) and figure, plus the CSVs in paper_results/
python analysis/paper_tables/make_main_table.py      # Table 2 / the leaderboard columns
python analysis/paper_tables/make_task_tables.py     # the per-task appendix tables
python analysis/plot_main_radar.py                   # Figure 2
python analysis/plot_transferability_5src.py         # Figure 3
```

Each result file in `bench/results/<task>/` records the task, encoder, protocol, seed, all metrics, the
evaluation split, the selected checkpoint step and the provenance of the run, so every printed number can be
traced to one file.

## Task-specific baselines

Third-party systems are not vendored. `baselines/THIRD_PARTY.md` lists the upstream repository and commit for
each, and `baselines/patches/` holds the patch that adapts it to the benchmark's inputs and splits (Nymeria
trainers for IMUPoser, SimplePoser, TLIO and Air-IO; the UniMTS wrist-only retrieval model; NormWear).
Mix-Token and the PADS pipeline are ported in `baselines/mixtoken/` and `baselines/pd/`.

## Citation

```bibtex
@article{li2026imubench,
  title   = {IMU-Bench: Benchmarking Wrist IMU Representations Across Kinematic, Semantic and Clinical Tasks},
  author  = {Li, Yiquan and Bradshaw, Aidan and Gao, Chenfeng and Ahuja, Karan},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Licence

The code in this repository is released for research use; a licence file will be added with the public
release. The datasets, the SMPL model and the third-party baseline repositories keep their own licences.
