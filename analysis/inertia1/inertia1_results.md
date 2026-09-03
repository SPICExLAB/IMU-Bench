# Inertia-1 external baseline — results on our 6 tasks

Inertia-1 (Xu et al. 2026, arXiv:2607.06617; github.com/yang-ai-lab/Inertia-1) ships **no pretrained weights** (GitHub issue #1 is an unanswered weights request), so both rows are OUR reproductions of their two robust objectives (paper §4.1/§4.3: PatchTST + AR-Transformer, small preset ≈8.5M params, 30 s @ 20 Hz triaxial accelerometer in g, no normalisation) pretrained with their recipe on NHANES (their pretraining corpus; raw archives on disk here). Downstream: identical data/split/window/decoder/optimiser to the sibling rows; only the encoder differs (acc-only input, raw g-scale, anti-aliased 50→20 Hz inside the adapter, tokens interpolated to the T/16 grid for pose/odometry decoders). See baselines/inertia1/ and imu_lm/models/baselines/inertia1_encoder.py.

## 0. Pretraining (our reproduction — no official Inertia-1 weights exist)

| Variant | run | steps | best val loss | status |
|---|---|---|---|---|
| Inertia-1 PatchTST-small (NHANES repro) | `inertia1_patchtst_small_nhanes` | 170000 | 0.01806 | done |
| Inertia-1 ART-small (NHANES repro) | `inertia1_art_small_nhanes` | 170000 | 0.02548 | done |

## 1. Arm pose (Nymeria real-wrist, 5 joints) — cf. Experiments_table.md §1

| Model | val_loss | pos_err (cm) | ang_err (deg) | status | run |
|---|---|---|---|---|---|
| Ours (MAE) | 0.01638 | 8.17 | 9.81 | ref | `vit1d_arm_pose_probe_nymeria_mae_multidata` |
| Ours (MAE) finetune | 0.01599 | 8.00 | 9.89 | ref | `vit1d_arm_pose_finetune_nymeria_mae_multidata` |
| FOCAL | 0.01916 | 8.73 | 10.75 | ref | `vit1d_arm_pose_probe_nymeria_focal_multidata` |
| SimMTM | 0.01813 | 8.56 | 10.39 | ref | `vit1d_arm_pose_probe_nymeria_simmtm_multidata` |
| Scratch | 0.01672 | 8.15 | 9.94 | ref | `vit1d_arm_pose_supervised_nymeria_altattn_scratch` |
| Ours (MAE, acc-only) | 0.01888 | — | — | ref | `vit1d_arm_pose_probe_nymeria_mae_acconly_multidata` |
| **Inertia-1 PatchTST-small (NHANES repro) probe** | 0.01755 | 8.43 | 10.26 | done | `vit1d_arm_pose_probe_nymeria_inertia1_patchtst` |
| **Inertia-1 PatchTST-small (NHANES repro) finetune** | 0.01769 | 8.55 | 10.46 | done | `vit1d_arm_pose_finetune_nymeria_inertia1_patchtst` |
| **Inertia-1 ART-small (NHANES repro) probe** | 0.01789 | 8.47 | 10.44 | done | `vit1d_arm_pose_probe_nymeria_inertia1_art` |
| **Inertia-1 ART-small (NHANES repro) finetune** | 0.01772 | 8.47 | 10.46 | done | `vit1d_arm_pose_finetune_nymeria_inertia1_art` |

## 2. Full-body pose (Nymeria real-wrist, 23 joints) — cf. §2 (mpjpe = best val mpjpe_mm)

| Model | mpjpe (mm) | @step | status | run |
|---|---|---|---|---|
| Ours (MAE) | 97.63 | 110000 | ref | `vit1d_full_body_pose_probe_nymeria_mae_multidata` |
| Ours (MAE) finetune | 91.95 | 12000 | ref | `vit1d_full_body_pose_finetune_nymeria_mae_multidata` |
| Ours (MAE, acc-only) | 103.37 | 56000 | ref | `vit1d_full_body_pose_probe_nymeria_mae_acconly_multidata` |
| RelCon | 102.47 | 32000 | ref | `vit1d_full_body_pose_probe_nymeria_relcon_multidata` |
| SimMTM | 103.48 | 50000 | ref | `vit1d_full_body_pose_probe_nymeria_simmtm_multidata` |
| Scratch | 92.43 | 370000 | ref | `vit1d_full_body_pose_supervised_nymeria_altattn_scratch` |
| **Inertia-1 PatchTST-small (NHANES repro) probe** | 100.48 | 100000 | done | `vit1d_full_body_pose_probe_nymeria_inertia1_patchtst` |
| **Inertia-1 PatchTST-small (NHANES repro) finetune** | 97.89 | 37000 | done | `vit1d_full_body_pose_finetune_nymeria_inertia1_patchtst` |
| **Inertia-1 ART-small (NHANES repro) probe** | 101.45 | 49000 | done | `vit1d_full_body_pose_probe_nymeria_inertia1_art` |
| **Inertia-1 ART-small (NHANES repro) finetune** | 96.04 | 19000 | done | `vit1d_full_body_pose_finetune_nymeria_inertia1_art` |

## 3. Odometry (Nymeria pelvis) — cf. §4 (local-velocity) and §4b (world-frame)

| Model | protocol | vel_mse | ATE (m) | drift % | status | source |
|---|---|---|---|---|---|---|
| Ours (MAE) | world-frame probe | 0.02086 | 0.0787 | 48.28 | ref | `data_vis/nymeria_odom_worldframe_probe_multidata/odom_probe.csv` |
| RelCon | local-vel probe | 0.02819 | 0.0867 | 55.29 | ref | `data_vis/nymeria_odom_relcon_probe_multidata/odom_probe.csv` |
| Ours (MAE) finetune | world-frame finetune | 0.01656 | 0.1293 | 34.08 | ref (§4b) | `vit1d_odom_worldframe_finetune_nymeria_mae_multidata` |
| TLIO / Air-IO | world-frame | 0.02419 / 0.02461 | 0.1510 / 0.1444 | 38.79 / 38.56 | ref (§4b) | `baselines/odometry/` |
| **Inertia-1 PatchTST-small (NHANES repro)** | world-frame probe | 0.02272 | 0.0862 | 64.74 | done | `data_vis/nymeria_odom_worldframe_probe_inertia1_patchtst/odom_probe.csv` |
| **Inertia-1 PatchTST-small (NHANES repro)** | local-vel probe | 0.02748 | 0.0830 | 60.84 | done | `data_vis/nymeria_odom_inertia1_patchtst_probe/odom_probe.csv` |
| **Inertia-1 PatchTST-small (NHANES repro)** | world-frame finetune | 0.02099 | 0.1572 | 41.68 | done | `data_vis/nymeria_odom_worldframe_finetune_inertia1_patchtst/odom_probe.csv` |
| **Inertia-1 ART-small (NHANES repro)** | world-frame probe | 0.02314 | 0.0911 | 64.69 | done | `data_vis/nymeria_odom_worldframe_probe_inertia1_art/odom_probe.csv` |
| **Inertia-1 ART-small (NHANES repro)** | local-vel probe | 0.02767 | 0.0864 | 60.86 | done | `data_vis/nymeria_odom_inertia1_art_probe/odom_probe.csv` |
| **Inertia-1 ART-small (NHANES repro)** | world-frame finetune | 0.02062 | 0.1539 | 39.90 | done | `data_vis/nymeria_odom_worldframe_finetune_inertia1_art/odom_probe.csv` |

## 4. Microgesture (OpenWatch, paper protocol: 1 s windows, k=3 clip voting) — cf. analysis/openwatch_fair_comparison.md

| Model | condition | split | Window F1 | Window Acc | Clip F1 (k=3) | Clip Acc (k=3) |
|---|---|---|---|---|---|---|
| Ours (MAE) probe | noaug | val | 0.561 | 0.577 | 0.674 | 0.676 |
| Ours (MAE) probe | noaug | test | 0.490 | 0.508 | 0.606 | 0.598 |
| Ours (MAE) probe | aug | val | 0.553 | 0.580 | 0.638 | 0.627 |
| Ours (MAE) probe | aug | test | 0.482 | 0.494 | 0.615 | 0.598 |
| Ours (MAE) finetune | noaug | val | 0.664 | 0.747 | 0.784 | 0.824 |
| Ours (MAE) finetune | noaug | test | 0.580 | 0.672 | 0.704 | 0.770 |
| Ours (MAE) finetune | aug | val | 0.704 | 0.778 | 0.783 | 0.833 |
| Ours (MAE) finetune | aug | test | 0.544 | 0.634 | 0.580 | 0.667 |
| **Inertia-1 PatchTST-small (NHANES repro) probe** | noaug | val | 0.486 | 0.515 | 0.542 | 0.569 |
| **Inertia-1 PatchTST-small (NHANES repro) probe** | noaug | test | 0.485 | 0.496 | 0.517 | 0.529 |
| **Inertia-1 PatchTST-small (NHANES repro) probe** | aug | val | 0.465 | 0.475 | 0.541 | 0.539 |
| **Inertia-1 PatchTST-small (NHANES repro) probe** | aug | test | 0.470 | 0.479 | 0.532 | 0.529 |
| **Inertia-1 PatchTST-small (NHANES repro) finetune** | noaug | val | 0.531 | 0.654 | 0.644 | 0.735 |
| **Inertia-1 PatchTST-small (NHANES repro) finetune** | noaug | test | 0.448 | 0.544 | 0.476 | 0.575 |
| **Inertia-1 PatchTST-small (NHANES repro) finetune** | aug | val | 0.535 | 0.631 | 0.662 | 0.735 |
| **Inertia-1 PatchTST-small (NHANES repro) finetune** | aug | test | 0.501 | 0.567 | 0.585 | 0.632 |
| **Inertia-1 ART-small (NHANES repro) probe** | noaug | val | 0.514 | 0.542 | 0.594 | 0.598 |
| **Inertia-1 ART-small (NHANES repro) probe** | noaug | test | 0.476 | 0.508 | 0.505 | 0.529 |
| **Inertia-1 ART-small (NHANES repro) probe** | aug | val | 0.500 | 0.520 | 0.571 | 0.569 |
| **Inertia-1 ART-small (NHANES repro) probe** | aug | test | 0.457 | 0.489 | 0.532 | 0.552 |
| **Inertia-1 ART-small (NHANES repro) finetune** | noaug | val | 0.544 | 0.647 | 0.661 | 0.735 |
| **Inertia-1 ART-small (NHANES repro) finetune** | noaug | test | 0.459 | 0.546 | 0.549 | 0.632 |
| **Inertia-1 ART-small (NHANES repro) finetune** | aug | val | 0.548 | 0.657 | 0.617 | 0.735 |
| **Inertia-1 ART-small (NHANES repro) finetune** | aug | test | 0.485 | 0.596 | 0.575 | 0.667 |

## 5. Language alignment (HumanML3D CLIP retrieval, val, wrist-only) — cf. §13d

**Gallery = 100** (R@1 / R@5 / R@10 / MRR, %)

| Model | IMU→Text | Text→IMU |
|---|---|---|
| Ours (MAE) finetune, i2t | 39.4 / 73.9 / 84.3 / 54.9 | 37.5 / 72.0 / 82.4 / 53.0 |
| Ours (MAE) frozen probe | 26.0 / 57.7 / 71.3 / 40.8 | 24.3 / 55.8 / 70.6 / 39.2 |
| Ours scratch, i2t | 21.4 / 52.8 / 67.2 / 36.2 | 20.6 / 50.6 / 65.1 / 34.9 |
| UniMTS official zero-shot (wrist) | 33.6 / 68.6 / 81.2 / 49.3 | 1.6 / 7.2 / 12.6 / 6.4 |
| **Inertia-1 PatchTST-small (NHANES repro) frozen probe** | 18.4 / 48.5 / 64.7 / 32.9 | 17.2 / 47.1 / 63.1 / 31.6 |
| **Inertia-1 PatchTST-small (NHANES repro) finetune, i2t** | 25.8 / 60.1 / 74.1 / 41.4 | 24.5 / 57.0 / 71.3 / 39.7 |
| **Inertia-1 ART-small (NHANES repro) frozen probe** | 20.6 / 51.0 / 67.1 / 34.9 | 20.2 / 49.1 / 65.5 / 34.2 |
| **Inertia-1 ART-small (NHANES repro) finetune, i2t** | 28.3 / 60.6 / 74.7 / 43.5 | 26.1 / 57.8 / 73.3 / 41.1 |

**Gallery = full** (R@1 / R@5 / R@10 / MRR, %)

| Model | IMU→Text | Text→IMU |
|---|---|---|
| Ours (MAE) finetune, i2t | 11.1 / 35.3 / 47.8 / 22.7 | 10.4 / 29.7 / 44.6 / 20.5 |
| Ours (MAE) frozen probe | 6.6 / 20.5 / 31.0 / 14.5 | 5.4 / 18.1 / 25.9 / 12.7 |
| Ours scratch, i2t | 4.8 / 16.4 / 27.1 / 11.6 | 4.7 / 15.4 / 23.8 / 11.2 |
| UniMTS official zero-shot (wrist) | 9.7 / 28.6 / 41.8 / 19.9 | 0.1 / 0.7 / 1.3 / 0.9 |
| **Inertia-1 PatchTST-small (NHANES repro) frozen probe** | 4.3 / 14.2 / 22.2 / 10.4 | 3.7 / 11.3 / 20.2 / 8.9 |
| **Inertia-1 PatchTST-small (NHANES repro) finetune, i2t** | 5.9 / 20.7 / 31.6 / 14.2 | 5.8 / 17.6 / 28.1 / 13.0 |
| **Inertia-1 ART-small (NHANES repro) frozen probe** | 4.8 / 16.5 / 25.1 / 11.5 | 5.6 / 14.9 / 21.6 / 11.1 |
| **Inertia-1 ART-small (NHANES repro) finetune, i2t** | 7.2 / 23.6 / 34.5 / 16.1 | 6.4 / 19.1 / 30.7 / 14.0 |

## 6. Parkinson's disease (PADS) — cf. §14 (window-level @ best-val-loss ckpt; subject-level via eval_pads_subject_level.py)

| Model | task | acc | bal_acc | macro_f1 | bal_acc @best-bal_acc ckpt | subject bal_acc (mean-prob) | status | run |
|---|---|---|---|---|---|---|---|---|
| Ours (MAE) | pd | 63.0714% | 38.9352% | 0.375 | 43.3675% | 34.8485% | ref | `vit1d_pd_probe_pads_mae` |
| Ours (MAE) finetune | pd | 63.7069% | 39.1872% | 0.378 | 45.3784% | 36.3636% | ref | `vit1d_pd_finetune_pads_mae` |
| SimMTM finetune | pd | 63.4212% | 40.1737% | 0.394 | 47.0629% | 37.8788% | ref | `vit1d_pd_finetune_pads_simmtm` |
| RelCon | pd | 62.2901% | 33.4448% | 0.257 | 35.4218% | 33.3333% | ref | `vit1d_pd_probe_pads_relcon` |
| Scratch | pd | 63.0189% | 35.9921% | 0.316 | 41.3427% | 33.3333% | ref | `vit1d_pd_supervised_pads_scratch` |
| Ours (MAE) | pdhc | 82.8672% | 52.1312% | 0.503 | 61.9570% | 50.0000% | ref | `vit1d_pdhc_probe_pads_mae` |
| Ours (MAE) | pddd | 74.5015% | 57.9870% | 0.576 | 60.9271% | 56.8182% | ref | `vit1d_pddd_probe_pads_mae` |
| **Inertia-1 PatchTST-small (NHANES repro) probe** | pd | 62.8207% | 34.3364% | 0.276 | 34.7407% | 33.3333% | done | `vit1d_pd_probe_pads_inertia1_patchtst` |
| **Inertia-1 PatchTST-small (NHANES repro) finetune** | pd | 62.8090% | 34.7510% | 0.287 | 39.5334% | 33.3333% | done | `vit1d_pd_finetune_pads_inertia1_patchtst` |
| **Inertia-1 PatchTST-small (NHANES repro) probe** | pdhc | 83.0143% | 50.2291% | 0.461 | 50.3339% | 50.0000% | done | `vit1d_pdhc_probe_pads_inertia1_patchtst` |
| **Inertia-1 PatchTST-small (NHANES repro) probe** | pddd | 72.3532% | 51.0570% | 0.441 | 51.6432% | 50.0000% | done | `vit1d_pddd_probe_pads_inertia1_patchtst` |
| **Inertia-1 ART-small (NHANES repro) probe** | pd | 62.2668% | 33.4408% | 0.257 | 35.1947% | 33.3333% | done | `vit1d_pd_probe_pads_inertia1_art` |
| **Inertia-1 ART-small (NHANES repro) finetune** | pd | 63.2560% | 37.2809% | 0.341 | 38.7016% | 36.3636% | done | `vit1d_pd_finetune_pads_inertia1_art` |
| **Inertia-1 ART-small (NHANES repro) probe** | pdhc | 83.0078% | 50.0769% | 0.457 | 50.2456% | 50.0000% | done | `vit1d_pdhc_probe_pads_inertia1_art` |
| **Inertia-1 ART-small (NHANES repro) probe** | pddd | 72.1117% | 50.9957% | 0.443 | 54.4386% | 50.0000% | done | `vit1d_pddd_probe_pads_inertia1_art` |

