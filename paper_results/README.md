# paper_results — the numbers behind every figure and table of the ICLR paper

One folder per paper element. Each holds tidy CSVs (the exact numbers the paper prints), a copy of the
current rendered figure where there is one (`current_*`), and nothing else. Everything here is
**generated** — edit the generator, not the CSV.

```bash
cd /gpfs/home/vbc1591/IMU-SSL && source imu-bench/bin/activate
python paper_results/build.py      # re-runs every generator: paper .tex/.png AND these CSVs
```

Pending cells are empty in the CSVs (`status=pending`) and `--` in the paper; re-run `build.py`
whenever jobs land.

## Index

| Paper element (current number) | LaTeX label | Folder | Main file(s) | Generator |
|---|---|---|---|---|
| Figure 1, IMU-Bench overview | `fig:teaser` | `fig1_overview/` | `content.yaml` (what the figure must say), `current_fig_overview.tex` (TikZ) | hand-drawn |
| Table 1, task list | `tab:imubench_tasks` | `tab1_tasks/` | `tasks.csv` | `analysis/bench_stats/make_task_table.py` |
| Table 2, main results | `tab:main_results` | `tab2_main_results/` | `main_results_long.csv` (one row per cell, with source file), `main_results_{probe,finetune,scratch}_wide.csv` | `analysis/paper_tables/make_main_table.py` |
| Figure 2, Table 2 as a radar | `fig:main_radar` | `tab2_main_results/` | `radar_normalized.csv` (raw value, panel best and relative score per cell), `current_main_radar.{png,pdf}` | `analysis/plot_main_radar.py` |
| Figure 3, cross-task transfer | `fig:transferability_comparison` | `fig2_transfer/` | `transfer_long.csv`, `current_transfer5_{frozen,trainable}.png` | `analysis/plot_transferability_5src.py` |
| Figure 4, motion types x volume grid | `fig:grid_clean` | `fig3_scaling_grid/` | `grid_long.csv`, `current_grid_5src_nonleaky.{png,pdf}` | `analysis/diversity_saturation/33_plot_grid_nonleaky.py` |
| App. Table, pretraining details | `tab:pretrain_details` | `appendix_pretrain_details/` | `pretrain_details.csv` | `analysis/paper_tables/make_impl_tables.py` |
| App. Table, downstream protocols | `tab:downstream_protocols` | `appendix_downstream_protocols/` | `downstream_protocols.csv` | same |
| App. Table, OpenWatch audit | `tab:audit_openwatch` | `appendix_audit_openwatch/` | `audit_facts.csv`, `release_vs_canonical_split.csv` | `analysis/paper_tables/make_audit_table.py` |
| App. Table, per-task results (all metrics) | `tab:task_<task>` | `tab2_main_results/` | `per_task_all_metrics.csv` | `analysis/paper_tables/make_task_tables.py` |
| App. Table, selection-free control | `tab:noselection` | `appendix_selection/` | `selection_free.csv` (reported vs final-checkpoint value per cell) | `analysis/paper_tables/make_selection_table.py` |
| App. Table, decoder capacity + controls | `tab:decoder_controls` | `appendix_decoder_controls/` | `decoder_controls.csv` | `analysis/paper_tables/make_ablation_tables.py` |
| App. Table, corpus construction x budget | `tab:corpus_budget` | `appendix_corpus_budget/` | `corpus_budget_long.csv` | same |
| App. Figure, volume at full diversity | `fig:volume_scaling` | `appendix_volume_scaling/` | `volume_k1000.csv`, `motion_mode_coverage.csv`, `current_volume_scaling_5src.{png,pdf}` | `analysis/diversity_saturation/31_plot_volume_5src.py` |
| App. Table, pretraining corpus | `tab:pretraining_data_composition` | `appendix_pretrain_corpus/` | `pretrain_corpus.csv` | `analysis/bench_stats/make_pretrain_table.py` |

## Reading the CSVs

- `value` is in the metric's **raw units**: arm pose cm, full-body MPJPE mm, odometry ATE m, IMU-text
  Recall@1 in %, every F1 / accuracy in [0, 1] (the paper multiplies F1 by 100). `higher_is_better` says
  which way is good. `har_mean` averages HHAR, MHEALTH, PRISM and the user study.
- Encoder ids: `mae_5src` (MAE, alternating attention), `limubert_5src`, `simmtm_5src`, `focal_5src`,
  `relcon_5src`, `mae_5src_noaltattn` (MAE, plain attention = like-for-like with the four external
  objectives), `mae_5src_acconly`, `scratch`, controls `control_random` (random frozen encoder) and
  `control_raw` (0-layer encoder, raw input). Transfer sources `tx5_<task>` = MAE fine-tuned on that task.
- `fig3_scaling_grid/grid_long.csv` column `source` matters: activity, Parkinson's and gesture panels are
  benchmark probes; arm, full-body and odometry panels are the grid's lighter legacy probes (not comparable
  to Table 2); the loss panel is on the grid validation set.
- Quick plot:
  ```python
  import pandas as pd
  d = pd.read_csv("paper_results/tab2_main_results/main_results_long.csv")
  d[(d.protocol == "probe") & (d.status == "ok")].pivot(index="encoder_label", columns="task_label", values="value")
  ```

## Where the raw results live (for anything not exported here)

| What | Where |
|---|---|
| One scored cell (task x encoder x protocol) | `bench/results/<task>/<encoder>__<protocol>__s42.json` (metrics + provenance) |
| Per-sample predictions for bootstrap | `bench/results/_units/<task>/<encoder>__<protocol>__s42.npz` |
| Registries (task cards, encoder cards) | `bench/registry/tasks.yaml`, `bench/registry/encoders.yaml` |
| Training runs (checkpoints, logs) | `runs/bench_<task>__<encoder>__<protocol>__s42/`, pretraining runs `runs/vit1d_ssl_*` |
| Leaderboard page | `bench/leaderboard_paper.html` (`python -m bench.leaderboard_paper`) |
| Grid legacy probes + cell sizes | `analysis/diversity_saturation/out/grid_results.json` |
| Motion-mode coverage | `analysis/diversity_saturation/out/coverage_5src.json` |
| Corpus-ablation index files / composition | `/projects/b53017/vbc1591/diversity_saturation/corpus_ablation/`, `analysis/corpus_ablation/out/indices.json` |
| OpenWatch forensics | `analysis/openwatch_forensics/openwatch_contamination_maps.json` |
| Dataset statistics for Table 1 | `analysis/bench_stats/{har_parquet_stats,other_stats}.json` |
| Superseded (void) results, kept for the record | `bench/results/_superseded_*` (physical-input bug, world-frame odometry, legacy runs, small decoders) |
