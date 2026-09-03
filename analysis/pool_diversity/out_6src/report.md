# Pretraining-pool motion-diversity analysis (`/projects/b53017/vbc1591/mae_multidata_cache_raw`)

Shared acc-only descriptor space, PCA-32 (explained var 0.952), K = 2000 motion clusters fitted on a source-balanced sample; up to 200,000 full-length windows sampled per source.

| source | hours (train) | gyro | clusters covered | exclusive clusters (share of own windows) | norm. entropy | effective clusters | max cluster share | intensity median / p90 [m/s²] | h for 95 % of own clusters (random / balanced) |
|---|---|---|---|---|---|---|---|---|---|
| capture24 | 3,122 | no | 1648/2000 (82.4 %) | 0 (0.0 %) | 0.9012 | 943.9 | 0.49 % | 0.115 / 2.201 | 1560.79 / 31.22 |
| eatingspeed | 409 | yes | 1645/2000 (82.2 %) | 76 (12.4 %) | 0.8714 | 752.6 | 0.84 % | 0.316 / 2.172 | 204.57 / 4.09 |
| motionx | 75 | yes | 769/2000 (38.5 %) | 14 (13.4 %) | 0.7758 | 363.8 | 2.41 % | 5.75 / 51.44 | 75.43 / 15.09 |
| ocdetect | 1,782 | yes | 1466/2000 (73.3 %) | 5 (0.2 %) | 0.8645 | 714.2 | 0.46 % | 1.393 / 3.236 | 891.08 / 17.82 |
| sonic | 216 | yes | 1026/2000 (51.3 %) | 139 (23.5 %) | 0.8285 | 543.2 | 0.82 % | 4.766 / 13.598 | 216.16 / 10.81 |
| totalcapture | 1 | yes | 136/2000 (6.8 %) | 0 (0.0 %) | 0.5675 | 74.7 | 5.86 % | 3.065 / 6.419 | 0.66 / 0.33 |

Greedy source order by marginal new motion clusters:

| rank | source | new clusters | cumulative coverage |
|---|---|---|---|
| 1 | capture24 | 1648 | 82.4 % |
| 2 | sonic | 250 | 94.9 % |
| 3 | eatingspeed | 80 | 98.9 % |
| 4 | motionx | 17 | 99.8 % |
| 5 | ocdetect | 5 | 100.0 % |
| 6 | totalcapture | 0 | 100.0 % |

Overlap: share of ROW source's clusters also covered by COL source:

| | capture24 | eatingspeed | motionx | ocdetect | sonic | totalcapture |
|---|---|---|---|---|---|---|
| capture24 | 1.00 | 0.93 | 0.40 | 0.87 | 0.47 | 0.08 |
| eatingspeed | 0.93 | 1.00 | 0.38 | 0.83 | 0.44 | 0.08 |
| motionx | 0.86 | 0.81 | 1.00 | 0.82 | 0.65 | 0.13 |
| ocdetect | 0.97 | 0.93 | 0.43 | 1.00 | 0.46 | 0.09 |
| sonic | 0.76 | 0.71 | 0.48 | 0.66 | 1.00 | 0.06 |
| totalcapture | 0.99 | 0.96 | 0.71 | 0.96 | 0.43 | 1.00 |

Figures: saturation_curves.png, overlap_matrix.png, intensity_by_source.png, cluster_share_by_source.png; full numbers in summary.json
