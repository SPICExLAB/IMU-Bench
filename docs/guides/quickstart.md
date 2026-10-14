# Quickstart

Four steps from a clone to a first scored cell.

## 1. Install and point the harness at your data

```bash
git clone https://github.com/SPICExLAB/IMU-Bench && cd IMU-Bench
python -m venv imu-bench && source imu-bench/bin/activate && pip install -e .
export IMUBENCH_DATA_ROOT=/data/imubench      # layout in Datasets
imu-bench tasks
```

## 2. Run a reference encoder on one task

```bash
imu-bench eval --encoder control_random --tasks har_hhar --protocols probe
```

`control_random` is the shared backbone with random weights, never trained: the number it reaches is what the
decoder alone achieves. The cell writes `runs/bench_har_hhar__control_random__probe__s42/` and
`bench/results/har_hhar/control_random__probe__s42.json`.

## 3. Run InrtlWrist through the plugin interface

```bash
imu-bench eval --encoder examples/plugin/inrtlwrist_plugin.yaml --tasks har_hhar pd_pads --protocols probe
```

The card points at the released checkpoint on the Hugging Face Hub; the harness downloads it. The scores should
match the released result files for these cells, which verifies the data preparation.

## 4. Run your own encoder

Copy `examples/plugin/inrtlwrist_plugin.py` and `.yaml`, replace the model, then:

```bash
imu-bench eval --encoder my_encoder.yaml            # all ten tasks, both protocols
imu-bench check my_encoder                           # complete? protocol-faithful? which track?
imu-bench package my_encoder --out submission/       # ready for the leaderboard pull request
```

Details: [Submission modes](submission-modes.md), [Evaluation](evaluation.md), [Tracks and leaderboard submission](submit.md).
