# NHANES motion-diversity analysis (capture24, cache `/projects/b53017/vbc1591/mae_multidata_cache_raw`)

Candidate pool: 2,194,855 windows = 3,122 h (strided evenly over all train participants). Budget: 60 h = 42,188 windows = 1.92 % of the pool (0.01 % of the ~2.72 M h on disk). K = 2000 motion clusters in PCA-32 descriptor space (explained var 0.84).

| selection | hours | clusters covered | norm. entropy | effective clusters | rare clusters kept | NHANES-exclusive kept | NN dist median / p95 | max cluster share |
|---|---|---|---|---|---|---|---|---|
| full_pool | 3121.6 | 2000/2000 (100.0 %) | 0.9827 | 1752.9 | 100.0 % | 100.0 % | 0.0 / 0.0 | 0.19 % |
| random | 60.0 | 1994/2000 (99.7 %) | 0.9789 | 1703.4 | 99.4 % | 100.0 % | 0.478 / 7.383 | 0.21 % |
| strided | 60.0 | 1998/2000 (99.9 %) | 0.9805 | 1724.4 | 99.8 % | 100.0 % | 0.48 / 7.482 | 0.19 % |
| balanced | 60.0 | 2000/2000 (100.0 %) | 0.9999 | 1998.6 | 100.0 % | 100.0 % | 0.478 / 7.262 | 0.05 % |
| balanced_exclusive2x | 60.0 | 2000/2000 (100.0 %) | 0.9985 | 1977.8 | 100.0 % | 100.0 % | 0.472 / 7.228 | 0.10 % |

Coverage curve (fraction of clusters covered vs hours kept):

| hours | random | strided | balanced |
|---|---|---|---|
| 0.3 | 0.0955 | 0.1015 | 0.1095 |
| 1.6 | 0.3955 | 0.404 | 0.5485 |
| 3.1 | 0.6155 | 0.627 | 1.0 |
| 15.6 | 0.9615 | 0.969 | 1.0 |
| 31.2 | 0.9915 | 0.995 | 1.0 |
| 62.4 | 1.0 | 0.998 | 1.0 |
| 156.1 | 1.0 | 1.0 | 1.0 |
| 312.2 | 1.0 | 1.0 | 1.0 |
| 624.3 | 1.0 | 1.0 | 1.0 |
| 1560.8 | 1.0 | 1.0 | 1.0 |

Other sources' coverage of the NHANES cluster space (40k windows each): eatingspeed 1693/2000, motionx 557/2000, nhanes 1762/2000, ocdetect 1233/2000, totalcapture 171/2000. NHANES-exclusive clusters: 55/2000 holding 2.9 % of NHANES windows.

Files: `indices_balanced_{H}h.npy` (use as `sources.nhanes.subset_index_file`), coverage_curve.png, cluster_occupancy.png, intensity_hist.png, pca_scatter.png, summary.json
