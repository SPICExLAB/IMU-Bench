# IMU-Bench

**Benchmarking wrist IMU representations across kinematic, semantic and clinical tasks.**

IMU-Bench evaluates one wrist-IMU encoder on ten tasks from eight datasets: arm and full-body pose, inertial
odometry, microgesture recognition, IMU–text retrieval, Parkinson's disease classification and activity recognition
on four sets. Every task is defined by a **task card** that fixes the raw input, the decoder, the training recipe, the
split, the selection rule and the scorer, so that the encoder is the only thing that differs between rows of the
leaderboard. The benchmark ships with **InrtlWrist**, a general-purpose inertial foundation model for wrist-worn
IMUs, and with the per-cell result files behind every number in the paper.

<div class="links-row" markdown>
[Website and leaderboard](https://spicexlab.github.io/imubench/){ .md-button }
[Code](https://github.com/SPICExLAB/IMU-Bench)
[InrtlWrist weights](https://huggingface.co/YiquanLi/InrtlWrist)
[Results and submissions](https://huggingface.co/datasets/YiquanLi/IMU-Bench-results)
[Paper (arXiv, soon)](https://spicexlab.github.io/imubench/paper/index.html)
</div>

## Overview

- **One harness, one protocol.** `imu-bench eval` trains and scores every encoder with the same decoders, splits,
  budgets and scorers; nothing on the leaderboard is typed by hand.
- **Two submission paths.** Plug any PyTorch encoder in through a small interface, or register an encoder trained
  with the shared backbone for the controlled track.
- **Audited data.** Every evaluation set was checked for duplicated participants and for overlap with common
  pretraining pools before it was split; submissions declare their corpus and are checked the same way.
- **Open results.** Each cell of the leaderboard is one JSON file with all metrics, the selected checkpoint and the
  exact config; submissions are pull requests that carry the same artefacts.

## Latest news

- **Oct 2026.** `imu-bench` command line, plugin encoders and the open track; InrtlWrist weights and the results
  dataset on the Hugging Face Hub; website and leaderboard live.
- **Sep 2026.** Paper submitted.

## Quick start

```bash
git clone https://github.com/SPICExLAB/IMU-Bench && cd IMU-Bench
python -m venv imu-bench && source imu-bench/bin/activate && pip install -e .
export IMUBENCH_DATA_ROOT=/data/imubench           # see Datasets
imu-bench tasks                                    # the ten task cards
imu-bench eval --encoder examples/plugin/inrtlwrist_plugin.yaml --tasks har_hhar --protocols probe
```

```python
import torch
from huggingface_hub import hf_hub_download
from scripts.load_inrtlwrist import load_inrtlwrist, zscore

enc = load_inrtlwrist(hf_hub_download("YiquanLi/InrtlWrist", "inrtlwrist_mae_5src_step144k.pt"))
z = enc(zscore(torch.randn(2, 6, 512)))            # [2, 384] pooled embedding of two 10.24 s wrist windows
```

## Documentation structure

- **Getting started:** [Tasks](tasks.md), [Installation](installation.md), [Quickstart](guides/quickstart.md), [Datasets](guides/datasets.md).
- **Guides:** [Submission modes](guides/submission-modes.md), [Evaluation](guides/evaluation.md),
  [Tracks and leaderboard submission](guides/submit.md), [Data audit and leakage](guides/audit.md),
  [Reproducing the paper](guides/reproducing.md).
- **Reference:** [CLI](reference/cli.md), [Encoder cards](reference/encoder-card.md),
  [Task cards](reference/task-cards.md), [Result files](reference/results.md).
- [FAQ](faq.md) and [Roadmap](roadmap.md).

## Available resources

| Resource | Where |
|---|---|
| Code, task cards, generated configs, scorers | [github.com/SPICExLAB/IMU-Bench](https://github.com/SPICExLAB/IMU-Bench) |
| InrtlWrist checkpoint (encoder + decoder + config) | [huggingface.co/YiquanLi/InrtlWrist](https://huggingface.co/YiquanLi/InrtlWrist) |
| Per-cell results, registry, splits, leaderboard data | [huggingface.co/datasets/YiquanLi/IMU-Bench-results](https://huggingface.co/datasets/YiquanLi/IMU-Bench-results) |
| Website and leaderboard (also mirrored as a Space) | [spicexlab.github.io/imubench](https://spicexlab.github.io/imubench/) · [Space](https://huggingface.co/spaces/YiquanLi/IMU-Bench) |

## Citation

```bibtex
@article{li2026imubench,
  title   = {IMU-Bench: Benchmarking Wrist IMU Representations Across Kinematic, Semantic and Clinical Tasks},
  author  = {Li, Yiquan and Bradshaw, Aidan and Gao, Chenfeng and Ahuja, Karan},
  journal = {arXiv preprint},
  year    = {2026}
}
```
