# SSL embedding space: cross-dataset / cross-activity analysis

N=9929 windows, k=10 cosine neighbors, 384-d frozen MAE embeddings.

## 1. kNN neighborhood purity

- same-DATASET fraction: **0.716** (chance 0.076)
- same-ACTIVITY fraction: **0.521** (chance 0.048)

Dataset purity is 9.5x chance; activity purity is 10.9x chance.

## 2. Silhouette (cosine)

- by dataset: **-0.058**
- by activity (n>=20): **-0.201**

## 3. Cross-dataset activity retrieval

Same-activity fraction among 10 NN restricted to OTHER datasets
(activities present in >=2 datasets, n>=20):

| activity | n | #datasets | cross-ds same-act@10 | chance |
|---|---|---|---|---|
| walking | 424 | 6 | 0.542 | 0.043 |
| jogging | 164 | 3 | 0.496 | 0.017 |
| cycling | 341 | 4 | 0.440 | 0.034 |
| standing | 342 | 5 | 0.289 | 0.034 |
| writing | 155 | 3 | 0.270 | 0.016 |
| brushing teeth | 127 | 3 | 0.263 | 0.013 |
| eating | 278 | 3 | 0.257 | 0.028 |
| running | 122 | 3 | 0.239 | 0.012 |
| stairs down | 227 | 3 | 0.235 | 0.023 |
| stairs up | 278 | 4 | 0.231 | 0.028 |
| vacuum cleaning | 127 | 3 | 0.230 | 0.013 |
| cooking | 1155 | 3 | 0.228 | 0.116 |
| clapping | 148 | 4 | 0.206 | 0.015 |
| sitting | 344 | 5 | 0.167 | 0.035 |
| typing | 132 | 3 | 0.155 | 0.013 |
| drinking | 172 | 4 | 0.138 | 0.017 |
| arm exercise | 219 | 2 | 0.102 | 0.022 |
| folding | 81 | 2 | 0.094 | 0.008 |
| soccer | 57 | 2 | 0.089 | 0.006 |
| catch | 78 | 2 | 0.060 | 0.008 |
| jumping | 44 | 3 | 0.030 | 0.004 |
| wiping | 80 | 2 | 0.025 | 0.008 |
| tools | 280 | 2 | 0.021 | 0.028 |
| grooming | 141 | 2 | 0.021 | 0.014 |
| dishes | 83 | 2 | 0.018 | 0.008 |
| lying | 123 | 2 | 0.007 | 0.012 |
| washing hands | 74 | 2 | 0.004 | 0.007 |
| basketball | 79 | 2 | 0.001 | 0.008 |

## 4. Leave-one-dataset-out kNN activity classification

(shared activities only; majority-class baseline in parens)

| held-out dataset | n test | kNN acc | majority baseline |
|---|---|---|---|
| hhar | 800 | 0.404 | 0.201 |
| mhealth | 500 | 0.448 | 0.120 |
| pamap2 | 464 | 0.349 | 0.136 |
| prism | 781 | 0.449 | 0.915 |
| recofit | 334 | 0.410 | 0.572 |
| samosa | 800 | 0.240 | 0.268 |
| shoaib | 669 | 0.528 | 0.108 |
| ut_watch | 655 | 0.328 | 0.351 |
| utd_mhad | 116 | 0.259 | 0.276 |
| wisdm | 756 | 0.231 | 0.221 |
