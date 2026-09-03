# NHANES motion-diversity analysis (ocdetect, cache `/projects/b53017/vbc1591/mae_multidata_cache_raw`)

Candidate pool: 1,253,077 windows = 1,782 h (strided evenly over all train participants). Budget: 300 h = 210,938 windows = 16.83 % of the pool (0.01 % of the ~2.72 M h on disk). K = 2000 motion clusters in PCA-32 descriptor space (explained var 0.76).

| selection | hours | clusters covered | norm. entropy | effective clusters | rare clusters kept | NHANES-exclusive kept | NN dist median / p95 | max cluster share |
|---|---|---|---|---|---|---|---|---|
| full_pool | 1782.2 | 2000/2000 (100.0 %) | 0.9805 | 1724.2 | 100.0 % | 100.0 % | 0.0 / 0.0 | 0.19 % |
| random | 300.0 | 2000/2000 (100.0 %) | 0.98 | 1717.6 | 100.0 % | 100.0 % | 2.601 / 6.44 | 0.20 % |
| strided | 300.0 | 2000/2000 (100.0 %) | 0.98 | 1718.0 | 100.0 % | 100.0 % | 2.587 / 6.425 | 0.19 % |
| balanced | 300.0 | 2000/2000 (100.0 %) | 0.9999 | 1997.9 | 100.0 % | 100.0 % | 2.533 / 5.905 | 0.05 % |
| balanced_exclusive2x | 300.0 | 2000/2000 (100.0 %) | 0.9989 | 1983.1 | 100.0 % | 100.0 % | 2.538 / 5.884 | 0.10 % |

Coverage curve (fraction of clusters covered vs hours kept):

| hours | random | strided | balanced |
|---|---|---|---|
| 0.9 | 0.257 | 0.261 | 0.313 |
| 1.8 | 0.4425 | 0.4365 | 0.6265 |
| 8.9 | 0.877 | 0.8715 | 1.0 |
| 17.8 | 0.9665 | 0.963 | 1.0 |
| 35.6 | 0.991 | 0.9935 | 1.0 |
| 89.1 | 0.999 | 1.0 | 1.0 |
| 178.2 | 1.0 | 1.0 | 1.0 |
| 356.4 | 1.0 | 1.0 | 1.0 |
| 891.1 | 1.0 | 1.0 | 1.0 |

Other sources' coverage of the NHANES cluster space (40k windows each): capture24 1750/2000, eatingspeed 1553/2000, motionx 620/2000, nhanes 1745/2000, totalcapture 205/2000. NHANES-exclusive clusters: 52/2000 holding 1.6 % of NHANES windows.

Files: `indices_balanced_{H}h.npy` (use as `sources.nhanes.subset_index_file`), coverage_curve.png, cluster_occupancy.png, intensity_hist.png, pca_scatter.png, summary.json
