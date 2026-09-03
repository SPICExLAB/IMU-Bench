# SSL embedding space: cross-dataset / cross-activity analysis

N=9368 windows, k=10 cosine neighbors, 384-d frozen MAE embeddings.

## 1. kNN neighborhood purity

- same-DATASET fraction: **0.699** (chance 0.080)
- same-ACTIVITY fraction: **0.498** (chance 0.050)

Dataset purity is 8.7x chance; activity purity is 10.0x chance.

## 2. Silhouette (cosine)

- by dataset: **-0.085**
- by activity (n>=20): **-0.216**

## 3. Cross-dataset activity retrieval

Same-activity fraction among 10 NN restricted to OTHER datasets
(activities present in >=2 datasets, n>=20):

| activity | n | #datasets | cross-ds same-act@10 | chance |
|---|---|---|---|---|
| jogging | 158 | 3 | 0.539 | 0.017 |
| walking | 442 | 6 | 0.500 | 0.047 |
| cycling | 318 | 4 | 0.459 | 0.034 |
| standing | 348 | 5 | 0.270 | 0.037 |
| writing | 141 | 3 | 0.255 | 0.015 |
| eating | 256 | 3 | 0.252 | 0.027 |
| stairs down | 220 | 3 | 0.248 | 0.023 |
| stairs up | 283 | 4 | 0.248 | 0.030 |
| cooking | 1159 | 3 | 0.239 | 0.124 |
| brushing teeth | 122 | 3 | 0.236 | 0.013 |
| running | 121 | 3 | 0.232 | 0.013 |
| clapping | 149 | 4 | 0.216 | 0.016 |
| vacuum cleaning | 135 | 3 | 0.199 | 0.014 |
| sitting | 356 | 5 | 0.165 | 0.038 |
| typing | 144 | 3 | 0.149 | 0.015 |
| drinking | 183 | 4 | 0.135 | 0.020 |
| folding | 91 | 2 | 0.122 | 0.010 |
| catch | 78 | 2 | 0.108 | 0.008 |
| soccer | 58 | 2 | 0.102 | 0.006 |
| washing hands | 77 | 2 | 0.049 | 0.008 |
| jumping | 47 | 3 | 0.045 | 0.005 |
| grooming | 163 | 3 | 0.034 | 0.017 |
| wiping | 80 | 2 | 0.029 | 0.009 |
| dishes | 74 | 2 | 0.026 | 0.008 |
| tools | 278 | 2 | 0.018 | 0.030 |
| lying | 138 | 2 | 0.007 | 0.015 |
| basketball | 79 | 2 | 0.005 | 0.008 |

## 4. Leave-one-dataset-out kNN activity classification

(shared activities only; majority-class baseline in parens)

| held-out dataset | n test | kNN acc | majority baseline |
|---|---|---|---|
| hhar | 800 | 0.395 | 0.190 |
| mhealth | 500 | 0.470 | 0.120 |
| pamap2 | 485 | 0.295 | 0.161 |
| prism | 800 | 0.411 | 0.891 |
| recofit | 150 | 0.347 | 0.633 |
| samosa | 800 | 0.241 | 0.269 |
| shoaib | 679 | 0.504 | 0.110 |
| ut_watch | 640 | 0.341 | 0.361 |
| utd_mhad | 88 | 0.148 | 0.364 |
| wisdm | 756 | 0.259 | 0.221 |
