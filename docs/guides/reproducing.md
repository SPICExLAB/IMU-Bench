# Reproducing the paper

## Tables and figures

Every table and figure is generated from the result files:

```bash
python paper_results/build.py                          # every table (.tex), figure and tidy CSV
python analysis/paper_tables/make_main_table.py        # main results table
python analysis/paper_tables/make_task_tables.py       # per-task appendix tables
python analysis/plot_main_radar.py                     # radar figure
python analysis/plot_transferability_5src.py           # cross-task transfer heatmaps
```

`paper_results/` holds the tidy CSVs behind each element with the source file of every number.

## Pretraining InrtlWrist

```bash
python scripts/run_pretrain.py --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic
```

The five controlled-track baselines use the same corpus and backbone with `scripts/run_ssl_baseline.py`; their configs
and launch scripts are under `configs/` and `launch/pretrain/`. The released checkpoint (144,000 updates) is on the
Hub and loads with `scripts/load_inrtlwrist.py`:

```python
from huggingface_hub import hf_hub_download
from scripts.load_inrtlwrist import load_inrtlwrist, zscore
enc = load_inrtlwrist(hf_hub_download("YiquanLi/InrtlWrist", "inrtlwrist_mae_5src_step144k.pt"))
```

## Running a cell without the CLI

```bash
python -m bench.make_config --task pd_pads --encoder mae_5src --protocol probe --seed 42
python scripts/run_pretrain.py --config configs/base.yaml \
    --model-config bench/generated/configs/bench_pd_pads__mae_5src__probe__s42.yaml \
    --run-name bench_pd_pads__mae_5src__probe__s42
python -m bench.score --task pd_pads --encoder mae_5src --protocol probe --seed 42
```

## Qualitative figures

`scripts/vis_site_pose.py` and `scripts/vis_site_odometry.py` draw the skeleton and trajectory figures of the website
from the fine-tuned checkpoints and the scorer's own reconstruction.
