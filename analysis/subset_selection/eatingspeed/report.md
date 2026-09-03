# NHANES motion-diversity analysis (eatingspeed, cache `/projects/b53017/vbc1591/mae_multidata_cache_raw`)

Candidate pool: 287,678 windows = 409 h (strided evenly over all train participants). Budget: 150 h = 105,469 windows = 36.66 % of the pool (0.01 % of the ~2.72 M h on disk). K = 2000 motion clusters in PCA-32 descriptor space (explained var 0.84).

| selection | hours | clusters covered | norm. entropy | effective clusters | rare clusters kept | NHANES-exclusive kept | NN dist median / p95 | max cluster share |
|---|---|---|---|---|---|---|---|---|
| full_pool | 409.1 | 2000/2000 (100.0 %) | 0.9826 | 1752.9 | 100.0 % | 100.0 % | 0.0 / 0.0 | 0.47 % |
| random | 150.0 | 2000/2000 (100.0 %) | 0.9818 | 1742.2 | 100.0 % | 100.0 % | 0.12 / 5.934 | 0.46 % |
| strided | 150.0 | 2000/2000 (100.0 %) | 0.9819 | 1743.1 | 100.0 % | 100.0 % | 0.108 / 5.815 | 0.47 % |
| balanced | 150.0 | 2000/2000 (100.0 %) | 0.9994 | 1990.9 | 100.0 % | 100.0 % | 0.084 / 5.114 | 0.05 % |
| balanced_exclusive2x | 150.0 | 2000/2000 (100.0 %) | 0.9957 | 1936.0 | 100.0 % | 100.0 % | 0.088 / 5.24 | 0.09 % |

Coverage curve (fraction of clusters covered vs hours kept):

| hours | random | strided | balanced |
|---|---|---|---|
| 0.4 | 0.129 | 0.135 | 0.1435 |
| 2.0 | 0.483 | 0.495 | 0.719 |
| 4.1 | 0.72 | 0.72 | 1.0 |
| 8.2 | 0.893 | 0.8985 | 1.0 |
| 20.5 | 0.9825 | 0.9875 | 1.0 |
| 40.9 | 0.998 | 0.9985 | 1.0 |
| 81.8 | 1.0 | 1.0 | 1.0 |
| 204.6 | 1.0 | 1.0 | 1.0 |

Other sources' coverage of the NHANES cluster space (40k windows each): capture24 1705/2000, motionx 574/2000, nhanes 1531/2000, ocdetect 1110/2000, totalcapture 165/2000. NHANES-exclusive clusters: 210/2000 holding 11.3 % of NHANES windows.

Files: `indices_balanced_{H}h.npy` (use as `sources.nhanes.subset_index_file`), coverage_curve.png, cluster_occupancy.png, intensity_hist.png, pca_scatter.png, summary.json
