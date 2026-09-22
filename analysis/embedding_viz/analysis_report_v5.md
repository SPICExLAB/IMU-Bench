# SSL embedding space, reference 5-source encoder

Encoder: `runs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic/checkpoints/best.pt`

N=9368 windows, k=10 cosine neighbours, 384-d frozen embeddings. Windows, preprocessing and caps are v4's.

## 1. kNN neighbourhood purity

- same-DATASET fraction: **0.639** (chance 0.080, 8.0x)
- same-ACTIVITY fraction: **0.469** (chance 0.050, 9.4x)

## 2. Silhouette (cosine)

- by dataset: **-0.076**
- by activity (n>=20): **-0.201**

## 3. Isolation by exposure and signal origin (the control)

Mean same-dataset fraction among the 10 nearest neighbours, by cell.
If isolation tracked pretraining exposure, the two 'held out' cells
would be high and the two 'in pool' cells low.

| cell | n | same-dataset@10 |
|---|---|---|
| real, in pool | 4888 | 0.580 |
| real, held out | 3058 | 0.584 |
| mocap-synth, in pool | 1069 | 0.969 |
| mocap-synth, held out | 353 | 0.947 |

Activity purity among labelled windows: in-pool datasets 0.424, held-out datasets 0.567.

## 4. Per-dataset isolation

| dataset | n | same-dataset@10 | chance | cell |
|---|---|---|---|---|
| utd_mhad | 88 | 0.993 | 0.009 | real, in pool |
| sonic | 800 | 0.983 | 0.085 | mocap-synth, in pool |
| mhealth | 658 | 0.970 | 0.070 | real, held out |
| humanml3d | 353 | 0.947 | 0.038 | mocap-synth, held out |
| motionx | 269 | 0.925 | 0.029 | mocap-synth, in pool |
| wisdm | 800 | 0.731 | 0.085 | real, in pool |
| recofit | 800 | 0.623 | 0.085 | real, in pool |
| shoaib | 800 | 0.623 | 0.085 | real, in pool |
| hhar | 800 | 0.576 | 0.085 | real, held out |
| samosa | 800 | 0.499 | 0.085 | real, in pool |
| pamap2 | 800 | 0.490 | 0.085 | real, in pool |
| ut_watch | 800 | 0.469 | 0.085 | real, in pool |
| prism | 800 | 0.468 | 0.085 | real, held out |
| nymeria | 800 | 0.391 | 0.085 | real, held out |

## 5. Cross-dataset activity retrieval

Same-activity fraction among 10 NN restricted to OTHER datasets
(activities present in >=2 datasets, n>=20):

| activity | n | #datasets | cross-ds same-act@10 | chance |
|---|---|---|---|---|
| cycling | 318 | 4 | 0.487 | 0.034 |
| running | 121 | 3 | 0.396 | 0.013 |
| walking | 442 | 6 | 0.366 | 0.047 |
| jogging | 158 | 3 | 0.332 | 0.017 |
| stairs down | 220 | 3 | 0.314 | 0.023 |
| brushing teeth | 122 | 3 | 0.268 | 0.013 |
| stairs up | 283 | 4 | 0.266 | 0.030 |
| standing | 348 | 5 | 0.251 | 0.037 |
| eating | 256 | 3 | 0.247 | 0.027 |
| writing | 141 | 3 | 0.238 | 0.015 |
| vacuum cleaning | 135 | 3 | 0.224 | 0.014 |
| cooking | 1159 | 3 | 0.161 | 0.124 |
| drinking | 183 | 4 | 0.145 | 0.020 |
| clapping | 149 | 4 | 0.123 | 0.016 |
| sitting | 356 | 5 | 0.116 | 0.038 |
| jumping | 47 | 3 | 0.081 | 0.005 |
| typing | 144 | 3 | 0.058 | 0.015 |
| washing hands | 77 | 2 | 0.051 | 0.008 |
| catch | 78 | 2 | 0.041 | 0.008 |
| wiping | 80 | 2 | 0.036 | 0.009 |
| grooming | 163 | 3 | 0.034 | 0.017 |
| soccer | 58 | 2 | 0.028 | 0.006 |
| dishes | 74 | 2 | 0.024 | 0.008 |
| folding | 91 | 2 | 0.024 | 0.010 |
| basketball | 79 | 2 | 0.019 | 0.008 |
| lying | 138 | 2 | 0.005 | 0.015 |
| tools | 278 | 2 | 0.005 | 0.030 |

## 6. Leave-one-dataset-out kNN activity classification

| held-out dataset | in pretraining pool | n test | kNN acc | majority baseline |
|---|---|---|---|---|
| hhar | no | 800 | 0.344 | 0.190 |
| mhealth | no | 500 | 0.450 | 0.120 |
| pamap2 | yes | 485 | 0.297 | 0.161 |
| prism | no | 800 | 0.401 | 0.891 |
| recofit | yes | 150 | 0.260 | 0.633 |
| samosa | yes | 800 | 0.242 | 0.269 |
| shoaib | yes | 679 | 0.408 | 0.110 |
| ut_watch | yes | 640 | 0.309 | 0.361 |
| utd_mhad | yes | 88 | 0.000 | 0.364 |
| wisdm | yes | 756 | 0.261 | 0.221 |
