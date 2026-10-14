# IMU-Bench

**Benchmarking Wrist IMU Representations Across Kinematic, Semantic and Clinical Tasks**

Yiquan Li\*, Aidan Bradshaw\*, Chenfeng Gao, Karan Ahuja (Northwestern University; \* equal contribution)

[![Project page](https://img.shields.io/badge/Project%20page-spicexlab.github.io%2Fimubench-1d6f8f)](https://spicexlab.github.io/imubench/)
[![Leaderboard](https://img.shields.io/badge/%F0%9F%A4%97%20Space-Leaderboard-blue)](https://huggingface.co/spaces/YiquanLi/IMU-Bench)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-InrtlWrist-blue)](https://huggingface.co/YiquanLi/InrtlWrist)
[![Results](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Results%20%26%20submissions-blue)](https://huggingface.co/datasets/YiquanLi/IMU-Bench-results)
[![Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Collection-IMU--Bench-blue)](https://huggingface.co/collections/YiquanLi/imu-bench-6acf98435ad7fb7337edd21f)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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
python -m venv imu-bench && source imu-bench/bin/activate
pip install -r requirements.txt      # install the CUDA build of PyTorch first if needed
```

Python ≥ 3.9. Every run in the paper used one NVIDIA A100 or H100.

The SLURM scripts that `python -m bench.launch` generates activate `imu-bench/` by default; set `IMUBENCH_VENV` to use another environment directory.

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

The benchmark fixes the **raw input**: one wrist, six channels in a forward–left–up frame (acc m/s², gyro rad/s)
at 50 Hz, the released windows and splits. How an encoder normalises that input is part of the submission, not of
the protocol: the encoder card declares it (`normalize_method: zscore | physical | raw`) and the harness feeds every
probe and fine-tune of that encoder the same way. The released encoders use per-window z-scoring; physical-unit and
raw inputs are supported as well. The loaders and unit checks live in `imu_lm/data/`.

## Pretraining InrtlWrist

```bash
python scripts/run_pretrain.py --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic
```

The five baseline objectives use the same corpus and backbone with `imu_lm/objectives/ssl_vit_baselines.py`
(`scripts/run_ssl_baseline.py`); their configs and launch scripts are under `configs/` and `launch/pretrain/`.

### Pretrained InrtlWrist

The reference checkpoint (144,000 updates on the five-source corpus, the `mae_5src` row of every table) is on the
Hugging Face Hub: **https://huggingface.co/YiquanLi/InrtlWrist** (encoder + reconstruction decoder + config,
93 MB). `scripts/load_inrtlwrist.py` builds the encoder and loads it:

```python
import torch
from huggingface_hub import hf_hub_download
from scripts.load_inrtlwrist import load_inrtlwrist, zscore

enc = load_inrtlwrist(hf_hub_download("YiquanLi/InrtlWrist", "inrtlwrist_mae_5src_step144k.pt"))
x = zscore(torch.randn(2, 6, 512))        # [B, 6, T]: acc x/y/z (m/s^2) then gyro x/y/z (rad/s), 50 Hz, T <= 512
z = enc(x)                                # [B, 384] pooled embedding (what the frozen probes read)
tokens, patches = enc.forward_tokens(x)   # [B, 6 * T/16, 384] patch tokens (what the kinematic decoders read)
```

To evaluate it with the harness, point the `mae_5src` card in `bench/registry/encoders.yaml` at a run directory
whose `checkpoints/best.pt` is this file.

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

## Submitting to the leaderboard

Results, task cards and splits live in the Hugging Face dataset
[YiquanLi/IMU-Bench-results](https://huggingface.co/datasets/YiquanLi/IMU-Bench-results); the leaderboard page is
generated from those files. To add an encoder:

1. Register it as a card in `bench/registry/encoders.yaml` (run directory or checkpoint, `alt_attention`, `channels`).
2. Run the ten task cards under the released protocols; every cell writes `bench/results/<task>/<encoder>__<protocol>__s<seed>.json`.
3. Open a pull request to the dataset with your result files and your encoder card:

```python
from huggingface_hub import HfApi
HfApi().upload_folder(
    repo_id="YiquanLi/IMU-Bench-results", repo_type="dataset",
    folder_path="my_submission",   # my_submission/results/<task>/<encoder>__<protocol>__s42.json + my_submission/registry/<encoder>.yaml
    create_pr=True, commit_message="Add <encoder> to the leaderboard",
)
```

Cells enter the ranking only when they come from the released task cards (same decoder, recipe, split, selection
rule and scorer); systems trained under their own recipes are listed for orientation.

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

The code, task cards, generated configs and result files in this repository are released under the MIT License
(see `LICENSE`), as is the InrtlWrist checkpoint. The datasets, the SMPL model and the third-party baseline
repositories keep their own licences.
