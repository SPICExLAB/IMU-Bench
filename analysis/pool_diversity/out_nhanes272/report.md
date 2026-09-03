# Pretraining-pool motion-diversity analysis (`/projects/b53017/vbc1591/mae_multidata_cache_raw`)

Shared acc-only descriptor space, PCA-32 (explained var 0.815), K = 2000 motion clusters fitted on a source-balanced sample; up to 200,000 full-length windows sampled per source. Subsets: {'nhanes': 'analysis/nhanes_diversity/out/indices_balanced_272h.npy'}

| source | hours (train) | gyro | clusters covered | exclusive clusters (share of own windows) | norm. entropy | effective clusters | max cluster share | intensity median / p90 [m/s²] | h for 95 % of own clusters (random / balanced) |
|---|---|---|---|---|---|---|---|---|---|
| capture24 | 3,122 | no | 1779/2000 (88.9 %) | 0 (0.0 %) | 0.9132 | 1034.0 | 0.53 % | 0.115 / 2.201 | 1560.79 / 31.22 |
| eatingspeed | 409 | yes | 1790/2000 (89.5 %) | 10 (2.2 %) | 0.8762 | 780.2 | 0.76 % | 0.316 / 2.172 | 204.57 / 4.09 |
| motionx | 72 | yes | 743/2000 (37.1 %) | 0 (0.0 %) | 0.7477 | 294.0 | 2.46 % | 5.067 / 15.797 | 72.08 / 14.42 |
| nhanes | 272 | no | 1886/2000 (94.3 %) | 3 (0.1 %) | 0.9204 | 1091.8 | 0.34 % | 1.095 / 3.285 | 136.0 / 2.72 |
| ocdetect | 1,782 | yes | 1586/2000 (79.3 %) | 0 (0.0 %) | 0.8845 | 831.4 | 0.41 % | 1.395 / 3.232 | 891.08 / 17.82 |
| sonic | 216 | yes | 1134/2000 (56.7 %) | 42 (12.5 %) | 0.8032 | 448.3 | 0.80 % | 4.759 / 13.546 | 216.02 / 10.8 |
| totalcapture | 1 | yes | 159/2000 (8.0 %) | 0 (0.0 %) | 0.5913 | 89.5 | 8.89 % | 3.065 / 6.419 | 0.66 / 0.33 |

Greedy source order by marginal new motion clusters:

| rank | source | new clusters | cumulative coverage |
|---|---|---|---|
| 1 | nhanes | 1886 | 94.3 % |
| 2 | sonic | 92 | 98.9 % |
| 3 | eatingspeed | 19 | 99.9 % |
| 4 | capture24 | 3 | 100.0 % |
| 5 | motionx | 0 | 100.0 % |
| 6 | ocdetect | 0 | 100.0 % |
| 7 | totalcapture | 0 | 100.0 % |

Overlap: share of ROW source's clusters also covered by COL source:

| | capture24 | eatingspeed | motionx | nhanes | ocdetect | sonic | totalcapture |
|---|---|---|---|---|---|---|---|
| capture24 | 1.00 | 0.96 | 0.37 | 0.99 | 0.86 | 0.54 | 0.09 |
| eatingspeed | 0.95 | 1.00 | 0.36 | 0.98 | 0.84 | 0.53 | 0.08 |
| motionx | 0.90 | 0.87 | 1.00 | 0.96 | 0.92 | 0.71 | 0.13 |
| nhanes | 0.93 | 0.93 | 0.38 | 1.00 | 0.83 | 0.55 | 0.08 |
| ocdetect | 0.97 | 0.94 | 0.43 | 0.99 | 1.00 | 0.56 | 0.10 |
| sonic | 0.85 | 0.83 | 0.47 | 0.92 | 0.78 | 1.00 | 0.07 |
| totalcapture | 0.99 | 0.95 | 0.60 | 1.00 | 0.97 | 0.54 | 1.00 |

Figures: saturation_curves.png, overlap_matrix.png, intensity_by_source.png, cluster_share_by_source.png; full numbers in summary.json
