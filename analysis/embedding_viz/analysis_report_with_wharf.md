# SSL embedding space: cross-dataset / cross-activity analysis

N=10729 windows, k=10 cosine neighbors, 384-d frozen MAE embeddings.

## 1. kNN neighborhood purity

- same-DATASET fraction: **0.737** (chance 0.070)
- same-ACTIVITY fraction: **0.535** (chance 0.046)

Dataset purity is 10.5x chance; activity purity is 11.5x chance.

## 2. Silhouette (cosine)

- by dataset: **0.002**
- by activity (n>=20): **-0.284**

## 3. Cross-dataset activity retrieval

Same-activity fraction among 10 NN restricted to OTHER datasets
(activities present in >=2 datasets, n>=20):

| activity | n | #datasets | cross-ds same-act@10 | chance |
|---|---|---|---|---|
| jogging | 164 | 3 | 0.496 | 0.015 |
| cycling | 341 | 4 | 0.440 | 0.032 |
| walking | 555 | 7 | 0.419 | 0.052 |
| standing | 342 | 5 | 0.289 | 0.032 |
| writing | 155 | 3 | 0.270 | 0.014 |
| running | 122 | 3 | 0.239 | 0.011 |
| stairs down | 259 | 4 | 0.231 | 0.024 |
| vacuum cleaning | 127 | 3 | 0.230 | 0.012 |
| eating | 353 | 4 | 0.219 | 0.033 |
| cooking | 1222 | 4 | 0.216 | 0.114 |
| clapping | 148 | 4 | 0.206 | 0.014 |
| brushing teeth | 179 | 4 | 0.188 | 0.017 |
| stairs up | 352 | 5 | 0.184 | 0.033 |
| sitting | 344 | 5 | 0.167 | 0.032 |
| typing | 132 | 3 | 0.155 | 0.012 |
| arm exercise | 219 | 2 | 0.102 | 0.020 |
| drinking | 251 | 5 | 0.102 | 0.023 |
| folding | 81 | 2 | 0.094 | 0.008 |
| soccer | 57 | 2 | 0.089 | 0.005 |
| catch | 78 | 2 | 0.060 | 0.007 |
| jumping | 44 | 3 | 0.030 | 0.004 |
| wiping | 80 | 2 | 0.025 | 0.007 |
| tools | 280 | 2 | 0.021 | 0.026 |
| dishes | 83 | 2 | 0.018 | 0.008 |
| grooming | 180 | 3 | 0.016 | 0.017 |
| lying | 123 | 2 | 0.007 | 0.011 |
| washing hands | 74 | 2 | 0.004 | 0.007 |
| basketball | 79 | 2 | 0.001 | 0.007 |
| phone use | 92 | 2 | 0.000 | 0.009 |

## 4. Leave-one-dataset-out kNN activity classification

(shared activities only; majority-class baseline in parens)

| held-out dataset | n test | kNN acc | majority baseline |
|---|---|---|---|
| hhar | 800 | 0.404 | 0.201 |
| mhealth | 500 | 0.448 | 0.120 |
| pamap2 | 464 | 0.347 | 0.136 |
| prism | 781 | 0.444 | 0.915 |
| recofit | 334 | 0.410 | 0.572 |
| samosa | 800 | 0.239 | 0.268 |
| shoaib | 669 | 0.511 | 0.108 |
| ut_watch | 721 | 0.298 | 0.319 |
| utd_mhad | 116 | 0.259 | 0.276 |
| wharf | 575 | 0.063 | 0.228 |
| wisdm | 756 | 0.225 | 0.221 |
