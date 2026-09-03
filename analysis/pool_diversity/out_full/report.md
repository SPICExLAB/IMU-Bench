# Pretraining-pool motion-diversity analysis (`/projects/b53017/vbc1591/mae_multidata_cache_raw`)

Shared acc-only descriptor space, PCA-32 (explained var 0.819), K = 2000 motion clusters fitted on a source-balanced sample; up to 200,000 full-length windows sampled per source.

| source | hours (train) | gyro | clusters covered | exclusive clusters (share of own windows) | norm. entropy | effective clusters | max cluster share | intensity median / p90 [m/s²] | h for 95 % of own clusters (random / balanced) |
|---|---|---|---|---|---|---|---|---|---|
| capture24 | 3,122 | no | 1785/2000 (89.2 %) | 0 (0.0 %) | 0.9093 | 1003.9 | 0.55 % | 0.115 / 2.201 | 1560.79 / 31.22 |
| eatingspeed | 409 | yes | 1782/2000 (89.1 %) | 20 (3.7 %) | 0.8774 | 787.8 | 0.80 % | 0.316 / 2.172 | 204.57 / 4.09 |
| motionx | 72 | yes | 733/2000 (36.6 %) | 1 (0.1 %) | 0.7516 | 302.7 | 2.14 % | 5.067 / 15.797 | 72.08 / 14.42 |
| nhanes | 4,000 | no | 1817/2000 (90.8 %) | 1 (0.0 %) | 0.9115 | 1020.9 | 0.35 % | 0.759 / 2.577 | 2000.0 / 40.0 |
| ocdetect | 1,782 | yes | 1565/2000 (78.2 %) | 0 (0.0 %) | 0.8846 | 831.9 | 0.50 % | 1.396 / 3.226 | 891.08 / 17.82 |
| sonic | 216 | yes | 1170/2000 (58.5 %) | 67 (17.8 %) | 0.8058 | 457.0 | 0.85 % | 4.759 / 13.546 | 216.02 / 10.8 |
| totalcapture | 1 | yes | 151/2000 (7.5 %) | 0 (0.0 %) | 0.57 | 76.1 | 7.59 % | 3.065 / 6.419 | 0.66 / 0.33 |

Greedy source order by marginal new motion clusters:

| rank | source | new clusters | cumulative coverage |
|---|---|---|---|
| 1 | nhanes | 1817 | 90.8 % |
| 2 | sonic | 146 | 98.2 % |
| 3 | eatingspeed | 35 | 99.9 % |
| 4 | motionx | 2 | 100.0 % |
| 5 | capture24 | 0 | 100.0 % |
| 6 | ocdetect | 0 | 100.0 % |
| 7 | totalcapture | 0 | 100.0 % |

Overlap: share of ROW source's clusters also covered by COL source:

| | capture24 | eatingspeed | motionx | nhanes | ocdetect | sonic | totalcapture |
|---|---|---|---|---|---|---|---|
| capture24 | 1.00 | 0.95 | 0.37 | 0.98 | 0.86 | 0.56 | 0.08 |
| eatingspeed | 0.95 | 1.00 | 0.35 | 0.97 | 0.83 | 0.54 | 0.08 |
| motionx | 0.91 | 0.86 | 1.00 | 0.93 | 0.91 | 0.73 | 0.13 |
| nhanes | 0.96 | 0.95 | 0.38 | 1.00 | 0.84 | 0.56 | 0.08 |
| ocdetect | 0.98 | 0.95 | 0.43 | 0.98 | 1.00 | 0.58 | 0.09 |
| sonic | 0.86 | 0.83 | 0.46 | 0.88 | 0.78 | 1.00 | 0.08 |
| totalcapture | 0.97 | 0.95 | 0.61 | 0.99 | 0.97 | 0.64 | 1.00 |

Figures: saturation_curves.png, overlap_matrix.png, intensity_by_source.png, cluster_share_by_source.png; full numbers in summary.json
