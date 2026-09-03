# NHANES motion-diversity analysis (nhanes, cache `/projects/b53017/vbc1591/mae_multidata_cache_raw`)

Candidate pool: 2,812,500 windows = 4,000 h (strided evenly over all train participants). Budget: 272 h = 191,250 windows = 6.80 % of the pool (0.01 % of the ~2.72 M h on disk). K = 2000 motion clusters in PCA-32 descriptor space (explained var 0.81).

| selection | hours | clusters covered | norm. entropy | effective clusters | rare clusters kept | NHANES-exclusive kept | NN dist median / p95 | max cluster share |
|---|---|---|---|---|---|---|---|---|
| full_pool | 4000.0 | 2000/2000 (100.0 %) | 0.984 | 1770.8 | 100.0 % | 100.0 % | 0.0 / 0.0 | 0.21 % |
| random | 272.0 | 1999/2000 (100.0 %) | 0.9832 | 1760.4 | 99.9 % | 100.0 % | 1.959 / 6.492 | 0.23 % |
| strided | 272.0 | 2000/2000 (100.0 %) | 0.9834 | 1762.5 | 100.0 % | 100.0 % | 1.969 / 6.53 | 0.21 % |
| balanced | 272.0 | 2000/2000 (100.0 %) | 1.0 | 1999.6 | 100.0 % | 100.0 % | 1.902 / 6.177 | 0.05 % |
| balanced_exclusive2x | 272.0 | 2000/2000 (100.0 %) | 0.9998 | 1997.5 | 100.0 % | 100.0 % | 1.902 / 6.177 | 0.10 % |

Coverage curve (fraction of clusters covered vs hours kept):

| hours | random | strided | balanced |
|---|---|---|---|
| 0.4 | 0.129 | 0.1305 | 0.1405 |
| 2.0 | 0.4835 | 0.472 | 0.703 |
| 4.0 | 0.7045 | 0.6885 | 1.0 |
| 20.0 | 0.9755 | 0.9775 | 1.0 |
| 40.0 | 0.9945 | 0.9945 | 1.0 |
| 80.0 | 0.9995 | 0.9985 | 1.0 |
| 200.0 | 0.9995 | 1.0 | 1.0 |
| 400.0 | 1.0 | 1.0 | 1.0 |
| 800.0 | 1.0 | 1.0 | 1.0 |
| 2000.0 | 1.0 | 1.0 | 1.0 |

Other sources' coverage of the NHANES cluster space (40k windows each): capture24 1934/2000, eatingspeed 1457/2000, motionx 672/2000, ocdetect 1754/2000, sonic 1090/2000, totalcapture 198/2000. NHANES-exclusive clusters: 6/2000 holding 0.2 % of NHANES windows.

Files: `indices_balanced_{H}h.npy` (use as `sources.nhanes.subset_index_file`), coverage_curve.png, cluster_occupancy.png, intensity_hist.png, pca_scatter.png, summary.json
