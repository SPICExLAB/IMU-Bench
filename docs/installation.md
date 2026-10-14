# Installation

```bash
git clone https://github.com/SPICExLAB/IMU-Bench && cd IMU-Bench
python -m venv imu-bench && source imu-bench/bin/activate
pip install -e .
imu-bench tasks
```

Python 3.9 or newer and one GPU. Install the CUDA build of PyTorch first if `pip` picks the CPU wheel for your
platform; every run in the paper used one NVIDIA A100 or H100.

The install is **editable** because the harness runs inside the clone:

| Path | Holds |
|---|---|
| `configs/` | task templates and pretraining configs |
| `bench/registry/` | task cards, encoder cards, protocols, tracks |
| `bench/generated/configs/` | the generated config of every cell (the paper's are committed) |
| `runs/` | training runs: checkpoints and logs, one directory per cell |
| `bench/results/` | one result JSON per cell |

!!! note "SLURM"
    `imu-bench eval --slurm` submits one train → score chain per cell. The generated scripts activate `imu-bench/`
    by default; set `IMUBENCH_VENV` to use another environment directory, and `BENCH_PARTITION` / `BENCH_ACCOUNT`
    for your queue.

Next: [Quickstart](guides/quickstart.md), then [Datasets](guides/datasets.md).
