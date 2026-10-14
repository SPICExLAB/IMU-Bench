# IMU-Bench user guide

IMU-Bench evaluates one wrist-IMU encoder on ten tasks from eight datasets, under task cards that fix everything
except the encoder. This guide takes you from a clone to a leaderboard submission. It follows the practices that made
SWE-bench's leaderboard trustworthy, and says plainly which of them IMU-Bench already has and which are still planned.

| SWE-bench practice | IMU-Bench today | Status |
|---|---|---|
| One evaluation harness that everyone runs | `imu-bench eval`: the same trainer, decoders, splits, selection rule and scorers for every encoder | available |
| A fixed submission format (`predictions.jsonl`) | an **encoder card** (YAML) plus an encoder reachable through the plugin interface; results are one JSON per cell | available |
| Leaderboard entries carry their artefacts and can be re-run | `imu-bench package` lays out results, card and the exact training configs; submissions are pull requests to a public dataset repository | available |
| Verified subset: instances checked by hand | every evaluation set went through a participant-duplicate audit and a corpus-overlap audit before splitting (`imu-bench audit`, DATASETS.md) | available |
| Lite subset for fast iteration | `--tasks`/`--protocols` subsets of the ten cards; a named Lite subset with a one-GPU-hour budget | planned |
| Containerised runs | an editable install with pinned requirements; an official Docker image | planned |
| Maintainers re-run submissions | the leaderboard is regenerated only from merged result files; spot re-runs of public-weight submissions | planned |

The ten tasks, the metrics and the protocols are described in the paper and in the [README](README.md); this guide
is about running and submitting.

## 1. Install

```bash
git clone https://github.com/SPICExLAB/IMU-Bench && cd IMU-Bench
python -m venv imu-bench && source imu-bench/bin/activate
pip install -e .                      # install the CUDA build of PyTorch first if pip picks the CPU one
imu-bench tasks                       # prints the ten task cards: dataset, split, input, metric
```

Python 3.9 or newer, one GPU. The install is editable because the harness runs inside the clone: task templates in
`configs/`, generated per-cell configs in `bench/generated/configs/`, training runs in `runs/`, results in `bench/results/`.

## 2. Data

No dataset is redistributed. [DATASETS.md](DATASETS.md) lists, for each evaluation set, where to get it, its
licence, the preparation script and the directory layout. Put everything under one directory and point the harness
at it:

```bash
export IMUBENCH_DATA_ROOT=/data/imubench
```

Every dataset path in the task cards is rewritten to that directory. Nymeria, AMASS, SMPL and OpenWatch need a
free registration; nothing needs a signed agreement. DATASETS.md also lists the current gaps: a third party can run
seven of the ten tasks end to end today (arm pose, full-body pose, odometry, microgesture, Parkinson's, HHAR,
MHEALTH); IMU–text retrieval needs the simulated IMU regenerated with the UniMTS pipeline, PrISM is partly public,
and TransfHAR is being released.

## 3. What a cell is, and what is fixed

A **cell** is one task × encoder × protocol × seed. For every cell the task card fixes:

* the **raw input**: one wrist, six channels (accelerometer x/y/z in m/s², gyroscope x/y/z in rad/s) in a
  forward–left–up frame at 50 Hz, the window length, the stride, the splits;
* the **decoder** (a linear head for classification, a light attention decoder for pose and odometry, a projection
  for retrieval), its optimiser, learning rate, training budget and early-stopping rule;
* the **selection rule** (which validation metric picks the checkpoint) and the **scorer**.

The encoder is the only free part. Two protocols enter the leaderboard: **frozen probe** (encoder fixed, decoder
trained) and **fine-tune** (everything trained from the pretrained weights). Training from scratch is a reference row.
How the encoder normalises the raw input is part of the encoder, not of the protocol: the card declares it
(`normalize_method: zscore | physical | raw`) and the harness applies it identically in every cell of that encoder.

## 4. Submission modes

### 4.1 Encoder plugin (the standard mode)

Any PyTorch module can be evaluated. Write a builder and describe the encoder in a card.

```python
# my_package/imubench.py
import torch.nn as nn

class MyEncoder(nn.Module):
    embed_dim = 768                       # required: an integer attribute

    def forward(self, x):                 # x: [B, C, T] float, acc x/y/z (m/s²) then gyro x/y/z (rad/s), card's model_hz
        ...
        return {"tokens": tokens,         # [B, N, D] over time, used by the pose and odometry decoders
                "pooled": pooled}         # [B, D], read by the classification and retrieval heads
        # returning only [B, D], or only [B, N, D], is also accepted

def build(weights=None, **kwargs) -> nn.Module:
    model = MyEncoder(**kwargs)
    if weights: model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model
```

```yaml
# my_encoder.yaml  (one encoder per file; the key is the encoder id)
my_encoder:
  name: My encoder (Doe et al., 2026)
  external_encoder: plugin
  entry: my_package.imubench:build        # imported once per job; the clone root is on sys.path
  weights: hf://my-org/my-model/encoder.pt  # or a local path; downloaded with huggingface_hub
  channels: 6                             # 3 for accelerometer-only models (the gyroscope is dropped)
  model_hz: 50                            # the harness resamples 50 Hz input to this rate (anti-aliased)
  acc_units: m/s2                         # or g
  normalize_method: zscore                # zscore | physical | raw, applied before your model
  params_m: 85
  pretrain_data: Capture-24 (all), UK Biobank (train)   # exact corpus, for the overlap audit
  weights_url: https://huggingface.co/my-org/my-model
  track: open
```

The harness resamples, scales units, masks channels and, for models that return a single vector, repeats it over the
token grid so that the kinematic decoders still run. `examples/plugin/inrtlwrist_plugin.{py,yaml}` wraps the reference
encoder this way and reproduces its native scores; copy it as a template. The full contract is documented in
`imu_lm/models/baselines/plugin_encoder.py`.

### 4.2 Built-in architectures

An encoder trained with this repository's trainer (the shared ViT1D of the controlled track) is registered with a
run directory or checkpoint instead of a plugin entry; see `bench/registry/encoders.yaml` for the cards of the
released encoders. This is the mode for controlled-track submissions.

### 4.3 Embeddings instead of an encoder

Not supported, by design. The fine-tune protocol needs the encoder's weights, the kinematic probes need per-patch
tokens, and a frozen-probe-only entry would be ranked against rows that ran both protocols. If your model cannot be
released, release the weights to the maintainers privately with the submission; a frozen-probe-only "embeddings"
mode for the global tasks is on the roadmap and will be listed outside the ranking.

## 5. Run the evaluation

```bash
imu-bench eval --encoder my_encoder.yaml                       # registers the card, then all 20 cells, here
imu-bench eval --encoder my_encoder --tasks har_hhar pd_pads --protocols probe   # a subset while developing
imu-bench eval --encoder my_encoder --slurm                    # one SLURM train -> score chain per cell
imu-bench eval --encoder my_encoder --force                    # rerun cells that already have a result
```

Each cell writes `runs/bench_<task>__<encoder>__<protocol>__s42/` (checkpoints, logs) and one result file
`bench/results/<task>/<encoder>__<protocol>__s42.json` with every metric, the selected checkpoint step, the selection
rule and the provenance of the run. The scorer evaluates the selected checkpoint once; model selection uses the
validation split only. The primary seed is 42, the same for every row on the leaderboard.

Built-in reference encoders to try first: `control_random` (the backbone, never trained), `control_raw` (raw input
through a random projection), `scratch`, and `mae_5src` (InrtlWrist). If a reference row does not reproduce the
released number, check the data preparation before trusting a run of your own model.

## 6. Check and read your results

```bash
imu-bench check my_encoder          # 20/20 cells present? trained configs identical to the task cards? card complete? which track?
imu-bench leaderboard               # regenerates bench/leaderboard_tracks.json and the two track tables
```

`check` compares every trained config with the card's recipe, so a cell that changed the decoder, learning rate,
budget or split is reported and will not be ranked. The leaderboard shows a row only when all ten tasks are scored
under both protocols; rows are ordered by mean rank over the seven columns (arm pose, full-body pose, odometry,
gesture, IMU–text, Parkinson's, mean activity F1), ties share a rank, and reference rows are printed but never
ranked.

## 7. Tracks

Both tracks use the same task cards, protocols and scorers; they differ in what the encoder may be.

* **Controlled track**: pretrain only on the released corpus (197,094 unlabeled wrist windows built with
  `scripts/build_multidata_cache.py`), with the shared 21 M-parameter ViT backbone and trainer, at most 160,000
  updates, checkpoint selected on the corpus validation split. Only the objective is free. This track answers which
  pretraining objective gives the most useful representation.
* **Open track**: any architecture, size, corpus and objective. Weights must be public or released with the
  submission. This track answers which encoder anyone can download is best.

## 8. Submit to the leaderboard

Submissions are pull requests to the public results repository, the way SWE-bench collects experiments.

```bash
imu-bench package my_encoder --out submission/   # results/, registry/my_encoder.yaml, configs/, check.json
```

```python
from huggingface_hub import HfApi
HfApi().upload_folder(repo_id="YiquanLi/IMU-Bench-results", repo_type="dataset",
                      folder_path="submission", create_pr=True, commit_message="Add my_encoder")
```

What happens next:

1. We run `imu-bench check` on the pull request. A missing cell, a changed recipe or an incomplete card is
   reported on the PR.
2. We check the declared pretraining corpus against the evaluation sets (section 9). A row whose corpus contains
   evaluation recordings is listed as "overlap" and left out of the ranking of the affected tasks.
3. For open-track submissions with public weights we re-run a sample of cells; a mismatch beyond the printed
   precision is reported on the PR.
4. After merging, the leaderboard on the website and in the Space is regenerated from the merged files. Nothing on
   it is typed by hand.

## 9. Data declaration and leakage

Every card must list the pretraining corpus (`pretrain_data`), dataset by dataset, with the split used. The
evaluation sets of the ten tasks must not be in it. Two checks run on each submission and are available to you
before submitting:

* **Name check.** Common activity-recognition pools bundle WISDM, PAMAP2, RecoFit, SAMoSA, Shoaib and UT-Watch;
  those sets are therefore not evaluation tasks here, and the four activity tasks (HHAR, MHEALTH, PrISM, TransfHAR)
  are held out of the released corpus. Declare bundles by their member datasets.
* **Participant audit.** `imu-bench audit <dir> --subject-regex '<id pattern>' --acc-cols AccX AccY AccZ` hashes a
  release's files and links participant ids that hold identical or mirrored recordings. Run it before splitting any
  dataset of your own by participant; on OpenWatch it reduces 50 ids to 43 people.

The README's "Five checks for any wrist-IMU evaluation" explains the problems behind these rules.

## 10. Reproducibility

* One seed (42) per cell; the released result files record the checkpoint step and the config that produced each
  number, and `bench/generated/configs/` holds every config of the paper's cells.
* Differences within the measurement noise are not separated: the paper reports the noise band of the kinematic
  tasks, and bold on the leaderboard marks ties at the printed precision.
* Decoder capacity changes frozen-probe rankings; the cards fix light decoders, and the `control_random` and
  `control_raw` rows show what the decoder alone achieves in every column.

## 11. Frequently asked questions

**My model only uses the accelerometer.** Set `channels: 3`; the gyroscope channels are dropped before your
model. The row is marked accelerometer-only.

**My model expects 30 Hz input and 10 s windows.** Set `model_hz: 30`; the harness resamples. Window lengths are
fixed by the task cards (1 s to 10.24 s); a model that needs longer context sees the task window only.

**My model normalises its own input.** Set `normalize_method: raw` and do it inside `forward`.

**Can I tune the probe?** No. The decoder, learning rate, budget and selection rule are part of the task card; a
cell that changes them is listed outside the ranking. Fine-tuning is the protocol for showing what the encoder can
reach with adaptation.

**I only ran some tasks.** `imu-bench package --allow-incomplete` still lays the results out and the PR is
accepted, but the row is not ranked until all twenty cells exist.

**How long does a full run take?** On one A100, a few hours for the twenty cells of a 21 M-parameter encoder;
the kinematic fine-tunes dominate.

## 12. Roadmap

* A Lite subset (one task per family) with a one-GPU-hour budget, for development.
* An official Docker image with pinned dependencies, and the leaderboard requirement to run inside it.
* Spot re-runs of every open-track submission with public weights, reported on the PR.
* A frozen-probe-only embeddings mode for the global tasks, listed outside the ranking.
* The Nymeria preprocessing converter, the TransfHAR release and the regenerated HumanML3D simulation.
