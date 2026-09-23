# SSL embedding space, reference 5-source encoder -- v6 (fine labels)

Encoder: `runs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic/checkpoints/best.pt`

N=12961 windows from 18 sources, 9139 of them activity-labelled across **261 distinct activities**; k=10 cosine neighbours on 384-d frozen embeddings.

## 1. kNN neighbourhood purity

- same-DATASET: **0.632** (chance 0.058, 10.9x)
- same-ACTIVITY: **0.388** (chance 0.017, 22.6x)
- same-FAMILY: **0.656** (chance 0.145, 4.5x)

## 2. Silhouette (cosine)

- by dataset: **-0.056**
- by activity (n>=20): **-0.170**

## 3. Isolation by exposure and signal origin (the control)

| cell | n | same-dataset@10 |
|---|---|---|
| real, in pool | 7049 | 0.577 |
| real, held out | 4490 | 0.617 |
| mocap-synth, in pool | 1069 | 0.961 |
| mocap-synth, held out | 353 | 0.942 |

Activity purity among labelled windows: in-pool 0.403, held-out 0.366.

The same control with the wrap-padded sources dropped, because a clip padded out to 10.24 s can be isolated by its own periodicity:

| cell | n | same-dataset@10 |
|---|---|---|
| real, in pool | 6400 | 0.536 |
| real, held out | 3690 | 0.563 |
| mocap-synth, in pool | 1069 | 0.961 |
| mocap-synth, held out | 353 | 0.942 |

Dropped: openwatch, utd_mhad.

**Robustness of the exposure contrast.** held-out minus in-pool, real sources only, padded sources already dropped: **+0.027**. Leaving out one source at a time:

| source left out | held-out minus in-pool |
|---|---|
| mhealth | -0.060 |
| eatingspeed | -0.001 |
| samosa | +0.012 |
| ut_watch | +0.013 |
| userstudy_watch | +0.014 |
| pamap2 | +0.019 |
| hhar | +0.028 |
| recofit | +0.031 |
| shoaib | +0.036 |
| wisdm | +0.052 |
| ocdetect | +0.057 |
| prism | +0.070 |
| nymeria | +0.090 |

Range -0.060 to +0.090; changes sign under leave-one-out: **yes**. For comparison, the mocap-synth minus real contrast is +0.364.

## 4. Per-dataset isolation, with the padding control

`wrap` is the mean factor a source's windows were wrap-padded by to reach 10.24 s. A source that is BOTH isolated and heavily padded is not evidence of anything about the representation.

| dataset | n | same-dataset@10 | chance | cell | wrap |
|---|---|---|---|---|---|
| sonic | 800 | 0.981 | 0.062 | mocap-synth, in pool | 1.00x |
| utd_mhad | 649 | 0.980 | 0.050 | real, in pool | 2.91x |
| mhealth | 658 | 0.968 | 0.051 | real, held out | 1.00x |
| humanml3d | 353 | 0.942 | 0.027 | mocap-synth, held out | 1.00x |
| motionx | 269 | 0.899 | 0.021 | mocap-synth, in pool | 1.00x |
| openwatch | 800 | 0.865 | 0.062 | real, held out | 3.41x |
| ocdetect | 800 | 0.744 | 0.062 | real, in pool | 1.00x |
| wisdm | 800 | 0.708 | 0.062 | real, in pool | 1.00x |
| userstudy_watch | 632 | 0.628 | 0.049 | real, held out | 1.00x |
| shoaib | 800 | 0.595 | 0.062 | real, in pool | 1.00x |
| recofit | 800 | 0.565 | 0.062 | real, in pool | 1.00x |
| hhar | 800 | 0.560 | 0.062 | real, held out | 1.04x |
| pamap2 | 800 | 0.474 | 0.062 | real, in pool | 1.00x |
| ut_watch | 800 | 0.438 | 0.062 | real, in pool | 1.00x |
| samosa | 800 | 0.427 | 0.062 | real, in pool | 1.15x |
| prism | 800 | 0.407 | 0.062 | real, held out | 1.00x |
| nymeria | 800 | 0.338 | 0.062 | real, held out | 1.00x |
| eatingspeed | 800 | 0.335 | 0.062 | real, in pool | 1.00x |

## 5. Cross-dataset retrieval, by family

| family | n | #datasets | cross-ds same-family@10 | chance |
|---|---|---|---|---|
| locomotion | 766 | 6 | 0.512 | 0.059 |
| cycling | 341 | 4 | 0.466 | 0.026 |
| stairs | 549 | 5 | 0.415 | 0.042 |
| household | 2069 | 5 | 0.325 | 0.160 |
| hand/desk | 1516 | 7 | 0.311 | 0.117 |
| posture | 834 | 5 | 0.285 | 0.064 |
| gesture | 1463 | 3 | 0.263 | 0.113 |
| sport | 1252 | 8 | 0.165 | 0.097 |
| object handling | 349 | 1 | n/a | 0.027 |

n/a: only one dataset carries this family, so there is nothing to retrieve from and the score is 0 by construction.

## 6. Cross-dataset retrieval, by activity (>=2 datasets, n>=20)

| activity | n | #datasets | cross-ds same-act@10 | chance |
|---|---|---|---|---|
| cycling | 341 | 4 | 0.466 | 0.026 |
| running | 122 | 3 | 0.350 | 0.009 |
| jogging | 164 | 3 | 0.346 | 0.013 |
| walking | 424 | 6 | 0.304 | 0.033 |
| stairs down | 227 | 3 | 0.285 | 0.018 |
| eating | 278 | 3 | 0.251 | 0.021 |
| stairs up | 278 | 4 | 0.249 | 0.021 |
| writing | 161 | 4 | 0.245 | 0.012 |
| vacuum cleaning | 127 | 3 | 0.232 | 0.010 |
| standing | 342 | 5 | 0.218 | 0.026 |
| brushing teeth | 127 | 3 | 0.217 | 0.010 |
| drinking | 172 | 4 | 0.127 | 0.013 |
| clapping | 154 | 5 | 0.101 | 0.012 |
| sitting | 344 | 5 | 0.093 | 0.027 |
| cooking | 440 | 2 | 0.059 | 0.034 |
| typing | 132 | 3 | 0.039 | 0.010 |
| dishes | 83 | 2 | 0.027 | 0.006 |
| jumping | 44 | 3 | 0.025 | 0.003 |
| folding | 81 | 2 | 0.017 | 0.006 |
| wiping | 80 | 2 | 0.015 | 0.006 |
| grooming | 141 | 2 | 0.015 | 0.011 |
| washing hands | 74 | 2 | 0.011 | 0.006 |
| soccer | 57 | 2 | 0.011 | 0.004 |
| lying | 123 | 2 | 0.004 | 0.009 |
| catch | 78 | 2 | 0.003 | 0.006 |
| basketball | 79 | 2 | 0.003 | 0.006 |
| drawing | 40 | 2 | 0.000 | 0.003 |

## 7. Leave-one-dataset-out kNN activity classification

| held-out dataset | in pool | n test | kNN acc | majority |
|---|---|---|---|---|
| hhar | no | 800 | 0.384 | 0.201 |
| mhealth | no | 500 | 0.458 | 0.120 |
| pamap2 | yes | 464 | 0.347 | 0.136 |
| recofit | yes | 143 | 0.231 | 0.629 |
| samosa | yes | 586 | 0.259 | 0.358 |
| shoaib | yes | 669 | 0.537 | 0.108 |
| userstudy_watch | no | 14 | 0.429 | 0.571 |
| ut_watch | yes | 687 | 0.186 | 0.335 |
| utd_mhad | yes | 88 | 0.045 | 0.364 |
| wisdm | yes | 756 | 0.353 | 0.221 |
