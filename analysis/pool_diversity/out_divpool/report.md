# Pretraining-pool motion-diversity analysis (`/projects/b53017/vbc1591/mae_multidata_cache_raw`)

Shared acc-only descriptor space, PCA-32 (explained var 0.804), K = 2000 motion clusters fitted on a source-balanced sample; up to 200,000 full-length windows sampled per source. Subsets: {'nhanes': '/gpfs/home/vbc1591/IMU-SSL/analysis/subset_selection/nhanes/indices_balanced_54h.npy', 'capture24': '/gpfs/home/vbc1591/IMU-SSL/analysis/subset_selection/capture24/indices_balanced_60h.npy', 'ocdetect': '/gpfs/home/vbc1591/IMU-SSL/analysis/subset_selection/ocdetect/indices_balanced_300h.npy', 'eatingspeed': '/gpfs/home/vbc1591/IMU-SSL/analysis/subset_selection/eatingspeed/indices_balanced_150h.npy'}

| source | hours (train) | gyro | clusters covered | exclusive clusters (share of own windows) | norm. entropy | effective clusters | max cluster share | intensity median / p90 [m/s²] | h for 95 % of own clusters (random / balanced) |
|---|---|---|---|---|---|---|---|---|---|
| capture24 | 60 | no | 1635/2000 (81.8 %) | 0 (0.0 %) | 0.9116 | 1021.8 | 0.73 % | 0.403 / 2.869 | 30.0 / 3.0 |
| eatingspeed | 150 | yes | 1662/2000 (83.1 %) | 46 (10.4 %) | 0.876 | 779.2 | 0.66 % | 0.473 / 2.676 | 150.0 / 3.0 |
| motionx | 72 | yes | 799/2000 (40.0 %) | 1 (0.9 %) | 0.7735 | 357.5 | 2.16 % | 5.067 / 15.797 | 72.08 / 14.42 |
| nhanes | 54 | no | 1647/2000 (82.3 %) | 2 (0.1 %) | 0.8999 | 934.5 | 0.38 % | 1.105 / 3.284 | 54.4 / 2.72 |
| ocdetect | 300 | yes | 1662/2000 (83.1 %) | 6 (0.8 %) | 0.8858 | 839.3 | 0.52 % | 1.89 / 4.086 | 150.0 / 3.0 |
| sonic | 216 | yes | 1089/2000 (54.4 %) | 97 (19.2 %) | 0.807 | 461.3 | 1.15 % | 4.759 / 13.546 | 216.02 / 10.8 |
| totalcapture | 1 | yes | 169/2000 (8.5 %) | 0 (0.0 %) | 0.6077 | 101.4 | 7.38 % | 3.065 / 6.419 | 0.66 / 0.33 |

Greedy source order by marginal new motion clusters:

| rank | source | new clusters | cumulative coverage |
|---|---|---|---|
| 1 | eatingspeed | 1662 | 83.1 % |
| 2 | sonic | 295 | 97.9 % |
| 3 | ocdetect | 38 | 99.8 % |
| 4 | nhanes | 4 | 100.0 % |
| 5 | motionx | 1 | 100.0 % |
| 6 | capture24 | 0 | 100.0 % |
| 7 | totalcapture | 0 | 100.0 % |

Overlap: share of ROW source's clusters also covered by COL source:

| | capture24 | eatingspeed | motionx | nhanes | ocdetect | sonic | totalcapture |
|---|---|---|---|---|---|---|---|
| capture24 | 1.00 | 0.94 | 0.41 | 0.95 | 0.93 | 0.49 | 0.10 |
| eatingspeed | 0.92 | 1.00 | 0.39 | 0.91 | 0.89 | 0.48 | 0.09 |
| motionx | 0.83 | 0.81 | 1.00 | 0.86 | 0.93 | 0.69 | 0.14 |
| nhanes | 0.95 | 0.92 | 0.42 | 1.00 | 0.92 | 0.51 | 0.10 |
| ocdetect | 0.91 | 0.89 | 0.45 | 0.92 | 1.00 | 0.53 | 0.10 |
| sonic | 0.74 | 0.73 | 0.50 | 0.77 | 0.81 | 1.00 | 0.09 |
| totalcapture | 0.94 | 0.91 | 0.65 | 0.96 | 0.98 | 0.55 | 1.00 |

Figures: saturation_curves.png, overlap_matrix.png, intensity_by_source.png, cluster_share_by_source.png; full numbers in summary.json
