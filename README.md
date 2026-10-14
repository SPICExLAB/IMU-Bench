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

New here? **[USER_GUIDE.md](USER_GUIDE.md)** takes you from a clone to a leaderboard submission.

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
raw input with zero layers, same backbone from scratch), the released HarNet and NormWear checkpoints and our
Inertia-1 reproduction (open track), and the retrained
task-specific systems (MobilePoser, IMUPoser, TLIO, Air-IO, Mix-Token, UniMTS wrist-only, DeepConvLSTM,
the PADS BOSS + CatBoost pipeline). See `bench/registry/encoders.yaml`.

## Repository layout

```
bench/registry/tasks.yaml       task cards (dataset, split, input, decoder, recipe, selection, metrics, scorer)
bench/registry/encoders.yaml    encoder cards (run directory / checkpoint, architecture flags, corpus)
bench/registry/protocols.yaml   probe / finetune / scratch
bench/registry/tracks.yaml      the controlled and open leaderboard tracks: rules, rows, watch-list
bench/cli.py                    the `imu-bench` command (tasks, eval, check, package, leaderboard)
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
examples/plugin/                an encoder card + builder for running any PyTorch model through the task cards
```

## Setup

```bash
git clone https://github.com/SPICExLAB/IMU-Bench && cd IMU-Bench && pip install -e .
imu-bench tasks                       # the ten task cards
```

Python ≥ 3.9; install the CUDA build of PyTorch first if needed. Every run in the paper used one NVIDIA A100 or
H100. The harness works inside the clone (task templates in `configs/`, runs in `runs/`, results in
`bench/results/`), which is why the install is editable. The SLURM scripts that `imu-bench eval --slurm`
generates activate `imu-bench/` by default; set `IMUBENCH_VENV` to use another environment directory.

## Leaderboard

<!-- tracks:start (generated by python -m bench.leaderboard_tracks; do not edit by hand) -->
<!-- tracks:end -->

`imu-bench leaderboard` regenerates these tables, `bench/leaderboard_tracks.json` and the website table from the
result files. Every cell's secondary metrics are in `bench/results/<task>/<encoder>__<protocol>__s42.json`.

## Five checks for any wrist-IMU evaluation

These hold whichever model leads the board. Each comes from a problem we found while building the benchmark
(paper, appendices on the data audit and on decoder capacity).

1. **Hash a release before splitting it by participant.** OpenWatch's 50 participant ids are 43 people:
   byte-identical recordings appear under different ids, and all 35 motion-augmented ids are wrist-mirrored copies
   of real participants. The release's own split places a copy of training participant 46 in its test set as id 48.
   IMU-Bench splits over the 43 canonical identities. To run the same check on any release:
   `imu-bench audit <dir> --subject-regex '/id_(\d+)/' --acc-cols AccX AccY AccZ` groups files by content hash and
   finds mirrored or rotated copies through the accelerometer magnitude. On OpenWatch it returns the 43 identities
   and links all 35 augmented ids to their sources in about 90 s.
2. **Check the pretraining pool against every evaluation set.** Common activity pools bundle WISDM, PAMAP2, RecoFit,
   SAMoSA, Shoaib and UT-Watch, so those sets are not used for evaluation. An earlier corpus of ours contained the
   Nymeria and HumanML3D training splits and OpenWatch recordings whose training part included all five gesture test
   participants. Open-track encoder cards declare their pretraining data so that they can be checked the same way.
3. **Run a never-trained encoder through the same decoder.** On full-body pose, pretraining beats a never-trained
   encoder by 9.7 mm with a linear readout and 7.1 mm with the benchmark decoder, but by only 0.3 mm once four
   temporal layers are added; a 12-layer odometry decoder lets the untrained encoder beat every pretrained one. The
   task cards fix light decoders, and the leaderboard prints the decoder-alone rows (`control_random`,
   `control_raw`) in every column.
4. **Verify sensor units for every source.** EatingSpeed stores the gyroscope in deg/s (a factor of 57 from rad/s)
   and OpenWatch stores raw counts (4,096 per g). Per-window z-scoring hides such errors; physical-unit pipelines do
   not. The harness converts units on load and rejects a source whose still windows have a median gyroscope
   magnitude above 0.3 rad/s.
5. **Know when an official split is partly transductive.** The official HumanML3D split cuts some AMASS recordings
   into several clips: 437 of its 2,192 test clips share frames with a training or validation clip. This affects
   every encoder equally, but retrieval scores on this split are partly transductive.

## Evaluate your own model

Any PyTorch encoder can be run through the ten task cards. Write a builder that returns a module mapping
`[B, C, T]` wrist IMU (acc m/s² then gyro rad/s, forward–left–up frame) to tokens `[B, N, D]`, a pooled
`[B, D]`, or both, and describe it in an encoder card:

```yaml
# my_encoder.yaml
my_encoder:
  name: My encoder (Doe et al., 2026)
  external_encoder: plugin
  entry: my_package.imubench:build        # build(weights=<local path>, **kwargs) -> nn.Module with .embed_dim
  weights: hf://my-org/my-model/encoder.pt
  channels: 6                             # 3 for accelerometer-only models
  model_hz: 50                            # the harness resamples to this rate
  acc_units: m/s2                         # or g
  normalize_method: zscore                # what the harness applies first: zscore | physical | raw
  params_m: 21
  pretrain_data: what the model was pretrained on, for the overlap audit
  weights_url: https://huggingface.co/my-org/my-model
  track: open
```

```bash
export IMUBENCH_DATA_ROOT=/data/imubench               # see DATASETS.md
imu-bench eval --encoder my_encoder.yaml               # all ten tasks, frozen probe + fine-tune, here
imu-bench eval --encoder my_encoder --slurm            # or one SLURM job per cell
imu-bench check my_encoder                             # complete? same recipe as the task cards? which track?
```

`examples/plugin/` wraps InrtlWrist this way and reproduces its native embedding exactly; the interface is
documented in `imu_lm/models/baselines/plugin_encoder.py`. The released models from other groups on the open track
(HarNet5, HarNet10, NormWear, ImageBind's IMU branch, UniMTS, PRIMUS) use the same interface or an adapter in
`imu_lm/models/baselines/`, and their cards are in `bench/registry/encoders.d/`. To rerun any of them:

```bash
imu-bench fetch-weights imagebind_imu          # official weights (+ pinned upstream code) where the card expects them
imu-bench eval --encoder imagebind_imu
```

## Data

The datasets are not redistributed. [DATASETS.md](DATASETS.md) gives the access steps and licence of each,
the files the harness reads and the directory layout; set `IMUBENCH_DATA_ROOT` to that directory and every
dataset path in the task cards is rewritten to it. No dataset needs a signed data-use agreement; Nymeria, AMASS,
SMPL and OpenWatch need a free registration. DATASETS.md also lists the current gaps (the retrieval task's
simulated IMU, part of PrISM, TransfHAR).

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

**Adding an encoder** is one entry in `encoders.yaml` (run directory or checkpoint, attention flag, channels,
`normalize_method`: how the encoder wants the fixed raw input scaled).
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
generated from those files. There are two tracks, with the same task cards and scorers:

* **Open track**: any architecture, size, corpus and objective. Weights must be public (or released with the
  submission), and the pretraining corpus must not contain recordings from the evaluation splits; declare it in
  `pretrain_data`.
* **Controlled track**: pretrain only on the released corpus, with the shared 21 M ViT backbone and trainer, at
  most 160,000 updates. Only the objective differs. Use this track to compare objectives.

```bash
imu-bench check my_encoder                       # 20/20 cells, recipe identical to the task cards, card complete
imu-bench package my_encoder --out submission/   # results + encoder card + the generated configs
```

```python
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="YiquanLi/IMU-Bench-results", repo_type="dataset", folder_path="submission",
                      create_pr=True, commit_message="Add my_encoder")
```

A cell counts only if it comes from the released task cards (same raw input, decoder, recipe, split, selection
rule and scorer); `imu-bench check` compares each trained config with the card. Input normalisation and any other
preprocessing inside the encoder are the submitter's choice and are declared in the card. Systems trained under
their own task recipes are listed for orientation and are not ranked.

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
