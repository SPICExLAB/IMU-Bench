# SSL Benchmark: Ours (MAE) vs. FOCAL vs. SimMTM vs. RelCon vs. Scratch

## 1. Arm pose estimation (Nymeria real-wrist, 5 joints)

| Model | val_loss | pos_err (cm) | ang_err (deg) | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 0.01638 | 8.17 | **9.81** | ✅ | `vit1d_arm_pose_probe_nymeria_mae_multidata` |
| Ours (MAE) finetune | **0.01599** | **8.00** | 9.89 | ✅ | `vit1d_arm_pose_finetune_nymeria_mae_multidata` |
| Ours (MAE, no alt-attn) | 0.01685 | 8.17 | 9.97 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 0.018883 | — | — | ⏳ probe done (job 7461716), pos/ang err scoring pending | `vit1d_arm_pose_probe_nymeria_mae_acconly_multidata` |
| FOCAL | 0.01916 | 8.73 | 10.75 | ✅ | `vit1d_arm_pose_probe_nymeria_focal_multidata` |
| SimMTM | 0.01813 | 8.56 | 10.39 | ✅ | `vit1d_arm_pose_probe_nymeria_simmtm_multidata` |
| RelCon | 0.018257 | — | — | ⏳ probe done (job 7431077), pos/ang err scoring pending | `vit1d_arm_pose_probe_nymeria_relcon_multidata` |
| Scratch | 0.01672 | 8.15 | 9.94 | ✅ | `vit1d_arm_pose_supervised_nymeria_altattn_scratch` |
| MobilePoser* | — | 9.74 | 24.28 | ✅ | `baselines/full_body_pose/SimplePoser` |
| Inertia-1 PatchTST-small (NHANES repro)† | 0.01755 | 8.43 | 10.26 | ✅ (job 9567105) | `vit1d_arm_pose_probe_nymeria_inertia1_patchtst` |
| Inertia-1 PatchTST-small (NHANES repro)† finetune | 0.01769 | 8.55 | 10.46 | ✅ (job 9567106) | `vit1d_arm_pose_finetune_nymeria_inertia1_patchtst` |
| Inertia-1 ART-small (NHANES repro)† | 0.01789 | 8.47 | 10.44 | ✅ (job 9567131) | `vit1d_arm_pose_probe_nymeria_inertia1_art` |
| Inertia-1 ART-small (NHANES repro)† finetune | 0.01772 | 8.47 | 10.46 | ✅ (job 9567132) | `vit1d_arm_pose_finetune_nymeria_inertia1_art` |

## 1b. Arm pose estimation (TotalCapture DIP, 5 joints) — pos_err/ang_err not available, see EXPERIMENTS.md

| Model | val_loss | pos_err (cm) | ang_err (deg) | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 0.019950 | — | — | ✅ | `vit1d_arm_pose_probe_totalcapture_mae_multidata` |
| Ours (MAE) finetune | 0.020179 | — | — | ✅ | `vit1d_arm_pose_finetune_totalcapture_mae_multidata` |
| Ours (MAE, no alt-attn) | **0.018941** | — | — | ✅ | `vit1d_arm_pose_probe_totalcapture_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 0.021213 | — | — | ✅ | `vit1d_arm_pose_probe_totalcapture_mae_acconly_multidata` |
| FOCAL | 0.024580 | — | — | ✅ | `vit1d_arm_pose_probe_totalcapture_focal_multidata` |
| SimMTM | 0.021552 | — | — | ✅ | `vit1d_arm_pose_probe_totalcapture_simmtm_multidata` |
| RelCon | 0.019861 | — | — | ✅ | `vit1d_arm_pose_probe_totalcapture_relcon_multidata` |
| Scratch | 0.020045 | — | — | ✅ | `vit1d_arm_pose_supervised_totalcapture_altattn_scratch` |

## 2. Full-body pose estimation (Nymeria real-wrist, 23 joints)

| Model | mpjpe (mm) | Status | Run |
|---|---|---|---|
| Ours (MAE) | 97.63 | ✅ | `vit1d_full_body_pose_probe_nymeria_mae_multidata` |
| Ours (MAE) finetune | 92.39 | ✅ | `vit1d_full_body_pose_finetune_nymeria_mae_multidata` |
| Ours (MAE, no alt-attn) | 99.81 | ✅* | `vit1d_full_body_pose_probe_nymeria_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 103.37 | ✅† (job 7461721, best_mpjpe.pt @ step 56,000, cancelled after plateau) | `vit1d_full_body_pose_probe_nymeria_mae_acconly_multidata` |
| FOCAL | 108.29 | ✅ | `vit1d_full_body_pose_probe_nymeria_focal_multidata` |
| SimMTM | 103.48 | ✅ | `vit1d_full_body_pose_probe_nymeria_simmtm_multidata` |
| RelCon | **102.47** | ✅ (job 7431101, `TIMEOUT` at 1-day walltime step 639,000 — best stayed at step 32,000 across all 639 evals, resolved itself without manual cancellation) | `vit1d_full_body_pose_probe_nymeria_relcon_multidata` |
| Scratch | 92.56 | ✅ | `vit1d_full_body_pose_supervised_nymeria_altattn_scratch` |
| MobilePoser** | **91.88** | ✅ | `baselines/full_body_pose/SimplePoser` |
| Ours (MAE, Nymeria-only, no norm) probe | 99.86 | ✅ | `vit1d_full_body_pose_probe_nymeria_only_nonorm` |
| Ours (MAE, Nymeria-only, no norm) finetune | 93.19 | ✅ | `vit1d_full_body_pose_finetune_nymeria_only_nonorm` |
| Inertia-1 PatchTST-small (NHANES repro)† | 100.48 | ✅ (job 9567108, fixed 100k budget, best @100k) | `vit1d_full_body_pose_probe_nymeria_inertia1_patchtst` |
| Inertia-1 PatchTST-small (NHANES repro)† finetune | 97.89 | ✅ (job 9567109) | `vit1d_full_body_pose_finetune_nymeria_inertia1_patchtst` |
| Inertia-1 ART-small (NHANES repro)† | 101.45 | ✅ (job 9567135, best @49k of 100k) | `vit1d_full_body_pose_probe_nymeria_inertia1_art` |
| Inertia-1 ART-small (NHANES repro)† finetune | 96.04 | ✅ (job 9567136) | `vit1d_full_body_pose_finetune_nymeria_inertia1_art` |

### 2a. LR sweep on Ours (MAE) finetune (✅ complete, 11 of 11 done)

Baseline finetune row above uses `trainer.optim.lr: 0.0001` (untuned). Same
recipe, LR-only sweep — extended upward since 3e-4 hadn't peaked, and it
never clearly turned over even at the top of the range:

| LR | best mpjpe (mm) | Job |
|---|---|---|
| 3e-5 | 96.27 | 7491703 |
| 5e-5 | 94.16 | 7491704 |
| 7e-5 | 92.86 | 7491706 |
| 1e-4 (baseline) | 92.39 | see main table |
| 1.5e-4 | 90.81 | 7491710 |
| 2e-4 | 90.44 | 7491716 |
| 3e-4 | 89.07 | 7491718 |
| 4e-4 | 88.71 | 7522400 |
| 5e-4 | 87.80 | 7522401 |
| 7e-4 | 87.79 | 7522402 |
| 1e-3 | **87.77** | 7522403 |

`1e-3` is the nominal best, but `5e-4`/`7e-4`/`1e-3` (87.80/87.79/87.77mm)
are functionally tied — within 0.03mm, noise-level. Using `1e-3`:
**4.11mm/4.5% better than MobilePoser** (91.88mm), the best result in this
entire benchmark. Not swept above 1e-3 since gains had already flattened to
<0.03mm/step. See EXPERIMENTS.md §2 for full discussion.

### 2c. LR sweep on MobilePoser baseline (✅ complete, 5 of 5 done)

MobilePoser's own baseline (9.19cm / 91.88mm) also uses an untuned LR
(`learning_rate: 0.001`). 5 more from-scratch training runs at
`{3e-4, 5e-4, 7e-4, 1.5e-3, 2e-3}` to check whether the baseline itself
could be improved before crediting Ours(MAE) with beating it:

| LR | full-body pos_err (cm) | Job |
|---|---|---|
| 3e-4 | 9.42 | 7522405 |
| 5e-4 | 9.27 | 7522406 |
| 7e-4 | 9.22 | 7522407 |
| 1e-3 (baseline) | **9.19** | `mobileposer_nymeria` |
| 1.5e-3 | 9.29 | 7522411 |
| 2e-3 | 9.25 | 7522412 |

**Firm conclusion: no improvement found.** Every swept LR is worse than
the untuned baseline on full-body pos_err — MobilePoser's default was
already at or near its optimum, unlike Ours(MAE) finetune's default (which
had ~5mm of headroom). The Ours(MAE)-vs-MobilePoser gap isn't an artifact
of an undertuned baseline. See EXPERIMENTS.md §7 for full methodology.

## 2b. Full-body pose estimation (TotalCapture DIP, 24 joints) — see EXPERIMENTS.md

| Model | mpjpe (mm) | Status | Run |
|---|---|---|---|
| Ours (MAE) | 193.51 | ✅ | `vit1d_full_body_pose_probe_totalcapture_mae_multidata` |
| Ours (MAE) finetune | **174.19** | ✅ | `vit1d_full_body_pose_finetune_totalcapture_mae_multidata` |
| Ours (MAE, no alt-attn) | 182.27 | ✅ | `vit1d_full_body_pose_probe_totalcapture_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 182.23 | ✅ | `vit1d_full_body_pose_probe_totalcapture_mae_acconly_multidata` |
| FOCAL | 184.45 | ✅ | `vit1d_full_body_pose_probe_totalcapture_focal_multidata` |
| SimMTM | 183.75 | ✅ | `vit1d_full_body_pose_probe_totalcapture_simmtm_multidata` |
| RelCon | 197.32 | ✅ | `vit1d_full_body_pose_probe_totalcapture_relcon_multidata` |
| Scratch | 175.89 | ✅ | `vit1d_full_body_pose_supervised_totalcapture_altattn_scratch` |

## 3. Human Activity Recognition (WISDM, UTD-MHAD, PAMAP2)

### WISDM
| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 65.50% | 65.52% | 0.639 | ✅ | `vit1d_har_probe_wisdm_mae_multidata` |
| Ours (MAE) finetune | **71.27%** | **71.22%** | **0.704** | ✅ | `vit1d_har_finetune_wisdm_mae_multidata` |
| Ours (MAE, no alt-attn) | 64.84% | 64.85% | 0.629 | ✅ | `vit1d_har_probe_wisdm_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 55.04% | 55.05% | 0.534 | ✅ | `vit1d_har_probe_wisdm_mae_acconly_multidata` |
| FOCAL | 41.95% | 41.98% | 0.390 | ✅ | `vit1d_har_probe_wisdm_focal_multidata` |
| SimMTM | 53.45% | 53.57% | 0.509 | ✅ | `vit1d_har_probe_wisdm_simmtm_multidata` |
| RelCon | 37.83% | 37.88% | 0.331 | ✅ | `vit1d_har_probe_wisdm_relcon_multidata` |
| Scratch | 60.89% | 60.89% | 0.606 | ✅ | `vit1d_har_supervised_wisdm_altattn_scratch` |
| DeepConvLSTM | 70.47% | 70.47% | 0.700 | ✅ | `deepconvlstm_har_wisdm` |

### UTD-MHAD
| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 63.77% | 63.83% | 0.633 | ✅ | `vit1d_har_probe_utd_mhad_mae_multidata` |
| Ours (MAE) finetune | **73.91%** | **74.98%** | **0.727** | ✅ | `vit1d_har_finetune_utd_mhad_mae_multidata` |
| Ours (MAE, no alt-attn) | 70.05% | 70.12% | 0.704 | ✅ | `vit1d_har_probe_utd_mhad_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 60.39% | 59.23% | 0.579 | ✅ | `vit1d_har_probe_utd_mhad_mae_acconly_multidata` |
| FOCAL | 53.62% | 52.10% | 0.504 | ✅ | `vit1d_har_probe_utd_mhad_focal_multidata` |
| SimMTM | 64.73% | 64.77% | 0.626 | ✅ | `vit1d_har_probe_utd_mhad_simmtm_multidata` |
| RelCon | 63.77% | 65.79% | 0.642 | ✅ | `vit1d_har_probe_utd_mhad_relcon_multidata` |
| Scratch | 71.01% | 71.37% | 0.701 | ✅ | `vit1d_har_supervised_utd_mhad_altattn_scratch` |
| DeepConvLSTM | 56.04% | 55.01% | 0.518 | ✅ | `deepconvlstm_har_utd_mhad` |

### PAMAP2
| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | **56.12%** | **52.86%** | **0.542** | ✅ | `vit1d_har_probe_pamap2_mae_multidata` |
| Ours (MAE) finetune | 51.77% | 45.09% | 0.412 | ✅ | `vit1d_har_finetune_pamap2_mae_multidata` |
| Ours (MAE, no alt-attn) | 54.00% | 48.07% | 0.482 | ✅ | `vit1d_har_probe_pamap2_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 47.51% | 46.00% | 0.462 | ✅ | `vit1d_har_probe_pamap2_mae_acconly_multidata` |
| FOCAL | 34.67% | 32.37% | 0.269 | ✅ | `vit1d_har_probe_pamap2_focal_multidata` |
| SimMTM | 47.42% | 42.75% | 0.409 | ✅ | `vit1d_har_probe_pamap2_simmtm_multidata` |
| RelCon | 22.50% | 16.56% | 0.129 | ✅ | `vit1d_har_probe_pamap2_relcon_multidata` |
| Scratch | 28.50% | 26.04% | 0.219 | ✅ | `vit1d_har_supervised_pamap2_altattn_scratch` |
| DeepConvLSTM | 37.45% | 35.02% | 0.292 | ✅ | `deepconvlstm_har_pamap2` |

### RecoFit
| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 78.21% | 61.67% | 0.627 | ✅ | `vit1d_har_probe_recofit_mae_multidata` |
| Ours (MAE) finetune | **87.84%** | **77.52%** | **0.768** | ✅ | `vit1d_har_finetune_recofit_mae_multidata` |
| Ours (MAE, no alt-attn) | 79.27% | 63.64% | 0.648 | ✅ | `vit1d_har_probe_recofit_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 69.54% | 57.14% | 0.568 | ✅ | `vit1d_har_probe_recofit_mae_acconly_multidata` |
| FOCAL | 55.23% | 44.91% | 0.437 | ✅ | `vit1d_har_probe_recofit_focal_multidata` |
| SimMTM | 68.91% | 54.04% | 0.541 | ✅ | `vit1d_har_probe_recofit_simmtm_multidata` |
| RelCon | 48.87% | 32.47% | 0.342 | ✅ | `vit1d_har_probe_recofit_relcon_multidata` |
| Scratch | 80.26% | 66.60% | 0.663 | ✅ | `vit1d_har_supervised_recofit_altattn_scratch` |
| DeepConvLSTM | 85.18% | 72.15% | 0.721 | ✅ | `deepconvlstm_har_recofit` |

### WHARF (2026-07-23, ✅ complete)
Wearable Human Activity Recognition Folder (Bruno et al., RO-MAN 2014):
single wrist-worn tri-axial **accelerometer only** (no gyro), native 32 Hz,
892 trials / 17 subjects / 14 activities — downloaded directly from
[fulviomas/WHARF](https://github.com/fulviomas/WHARF) (no access gate) into
`/projects/b53017/vbc1591/WHARF`, converted to the master-parquet HAR schema
via `scripts/prepare_wharf_dataset.py` (resampled 32→50 Hz canonical,
6-bit-coded raw ints decoded to m/s²) into
`/projects/b53017/vbc1591/wharf_dataset.parquet`. Since WHARF has no gyro
channel, the ViT1D columns use the **acc-only pretrained checkpoint**
(`vit1d_ssl_mae_varlen_acconly_multidata`, 3-channel) rather than the usual
full acc+gyro MAE — same architecture choice as the acc-only ablation used
elsewhere. Some classes are extremely small (eat_soup: 3 trials, eat_meat: 5
trials), so per-class metrics on those are noisy and val is a single
subject-disjoint 20% split (no separate held-out test split configured).
Metrics below are at each run's best-val-loss checkpoint.
| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE, acc-only) | 46.12% | 36.41% | 0.345 | ✅ | `vit1d_har_probe_wharf_mae_acconly_multidata` (job 7715641) |
| Ours (MAE, acc-only) finetune | **48.59%** | **43.54%** | 0.347 | ✅ | `vit1d_har_finetune_wharf_mae_acconly_multidata` (job 7715642) |
| Scratch | 44.00% | 37.67% | 0.349 | ✅ | `vit1d_har_supervised_wharf_altattn_scratch` (job 7715536) |
| DeepConvLSTM | 45.93% | 41.46% | **0.348** | ✅ | `deepconvlstm_har_wharf` (job 7715639) |

### HHAR (2026-07-23, ✅ complete)
Heterogeneity Human Activity Recognition (Stisen et al., SenSys 2015):
smartwatch accelerometer + gyroscope, 9 users, 4 watch devices (2 LG, 2
Samsung Gear), 6 activities (bike/sit/stand/walk/stairsup/stairsdown) —
downloaded directly from [UCI 344](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition)
(no access gate) into `/projects/b53017/vbc1591/HHAR`. Uses the **watch**
streams only (not phone), matching the wrist-worn convention used by most
other HAR sources in this repo. Accelerometer and gyroscope are independent
asynchronous streams per (user, device); `scripts/prepare_hhar_dataset.py`
aligns them via nearest-timestamp asof-merge on the device's own sensor
clock (`Creation_Time`; `Arrival_Time` is not monotonic under sort and isn't
used), segments into contiguous same-label runs, and resamples each to the
canonical 50 Hz → `/projects/b53017/vbc1591/hhar_dataset.parquet` (320
sessions, 857K rows). Units verified empirically (accel ~9.8 m/s² at rest,
gyro rad/s), no unit conversion needed. Standard full acc+gyro MAE checkpoint
(`vit1d_ssl_mae_varlen_altattn_multidata`) is used for probe/finetune, same
as WISDM/PAMAP2/UTD-MHAD/RecoFit.
| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | **66.79%** | **68.24%** | **0.679** | ✅ | `vit1d_har_probe_hhar_mae_multidata` (job 7745122, resubmitted on `gengpu` after `b1232` outage) |
| Ours (MAE) finetune | **61.85%** | **63.46%** | **0.628** | ✅ | `vit1d_har_finetune_hhar_mae_multidata` (job 7723045) |
| Ours (MAE, no alt-attn) | 65.11% | 66.13% | 0.661 | ✅ | `vit1d_har_probe_hhar_mae_noaltattn_multidata` (job 9562586) |
| Ours (MAE, acc-only) | 61.22% | 60.71% | 0.600 | ✅ | `vit1d_har_probe_hhar_mae_acconly_multidata` (job 9562588) |
| FOCAL | 51.18% | 51.50% | 0.500 | ✅ | `vit1d_har_probe_hhar_focal_multidata` (job 9519689) |
| SimMTM | 56.73% | 57.78% | 0.573 | ✅ | `vit1d_har_probe_hhar_simmtm_multidata` (job 9519690) |
| RelCon | 36.63% | 33.10% | 0.283 | ✅ | `vit1d_har_probe_hhar_relcon_multidata` (job 9519691) |
| Scratch | 50.63% | 50.91% | 0.504 | ✅ | `vit1d_har_supervised_hhar_altattn_scratch` (job 7745119, resubmitted on `gengpu` after `b1232` outage) |
| DeepConvLSTM | 44.80% | 46.60% | 0.448 | ✅ | `deepconvlstm_har_hhar` (job 7723042) |

### MHEALTH (2026-07-23, ✅ complete)
Mobile Health dataset (Banos et al., UGR): three Shimmer2 sensors
(chest/left-ankle/right-lower-arm) + ECG, 10 subjects, 12 activities, native
50 Hz — downloaded directly from [UCI 319](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
(no access gate) into `/projects/b53017/vbc1591/MHEALTH`. Uses only the
**right-lower-arm accelerometer + gyroscope** columns (dropping chest ECG,
all three magnetometers, and the ankle sensor) as the project's single
wrist/forearm-analog 6-channel stream, matching the wrist-worn convention
used elsewhere. `scripts/prepare_mhealth_dataset.py` segments each subject's
continuous log into contiguous same-label runs (dropping the null class) →
`/projects/b53017/vbc1591/mhealth_dataset.parquet` (126 sessions, 343K rows).
No resampling needed (already 50 Hz). **Caveat**: raw gyro magnitude is
nearly identical across standing vs. running/jumping (~1.0-1.2 in raw units
regardless of activity), i.e. bias-dominated/uncalibrated — a known quality
issue in the raw MHEALTH release, not a preprocessing bug (accel channel and
column offsets both check out against the README). Converted deg/s→rad/s per
the README's stated units. Standard full acc+gyro MAE checkpoint used for
probe/finetune, same as WISDM/PAMAP2/UTD-MHAD/RecoFit.
| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | **72.70%** | **69.92%** | **0.689** | ✅ | `vit1d_har_probe_mhealth_mae_multidata` (job 7745124, resubmitted on `gengpu` after `b1232` outage) |
| Ours (MAE) finetune | **64.58%** | **66.65%** | **0.627** | ✅ | `vit1d_har_finetune_mhealth_mae_multidata` (job 7723049) |
| Ours (MAE, no alt-attn) | 73.81% | 70.55% | 0.699 | ✅ | `vit1d_har_probe_mhealth_mae_noaltattn_multidata` (job 9562592) |
| Ours (MAE, acc-only) | 70.24% | 68.51% | 0.661 | ✅ | `vit1d_har_probe_mhealth_mae_acconly_multidata` (job 9562593) |
| FOCAL | 65.63% | 65.52% | 0.644 | ✅ | `vit1d_har_probe_mhealth_focal_multidata` (job 9562589) |
| SimMTM | 59.27% | 62.35% | 0.607 | ✅ | `vit1d_har_probe_mhealth_simmtm_multidata` (job 9562590) |
| RelCon | 43.62% | 42.04% | 0.401 | ✅ | `vit1d_har_probe_mhealth_relcon_multidata` (job 9562591) |
| Scratch | 69.83% | 68.49% | 0.659 | ✅ | `vit1d_har_supervised_mhealth_altattn_scratch` (job 7745123, resubmitted on `gengpu` after `b1232` outage) |
| DeepConvLSTM | **60.25%** | **59.48%** | **0.556** | ✅ | `deepconvlstm_har_mhealth` (job 7723047) |

### MMAct — ⛔ blocked, waiting on Hitachi
Large-scale multimodal action dataset (Kong et al., ICCV 2019; RGB + keypoints
+ acc/gyro/orientation + WiFi + pressure, 37 classes, 40 subjects). Not
downloadable programmatically: the [official site](https://mmact19.github.io/2019/)
only provides a [Google Form request](https://docs.google.com/forms/d/e/1FAIpQLScM3OTHIil2b39MZHkORTSHcRpWC6qXcvFnfGivNgb4ZpoRzA/viewform)
(needs the requester's own name/institution/email and agreement to an
academic-only, no-redistribution license); Hitachi emails the download link
after manual review. User has submitted the request (2026-07-23) and is
waiting to hear back — once the link arrives, download/convert/config/submit
follows the same pattern as WHARF/HHAR/MHEALTH above.

## 4. Odometry (Nymeria real-wrist local velocity)

| Model | vel_mse | ATE (m) | drift | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 0.0245 | **0.0726** | 49.42% | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata` |
| Ours (MAE) finetune | **0.01675** | 0.1521 | **43.70%** | ✅ | `vit1d_odom_finetune_nymeria_mae_multidata` |
| Ours (MAE, no alt-attn) | 0.02563 | 0.0754 | 49.56% | ✅ | `vit1d_ssl_mae_varlen_noaltattn_multidata` |
| Ours (MAE, acc-only) | — | — | — | ⏳ pretrain done, val MSE loss 0.7509 (job 7461706), odometry scoring pending | `vit1d_ssl_mae_varlen_acconly_multidata` |
| FOCAL | 0.0332 | 0.0967 | 63.46% | ✅ | `vit1d_ssl_focal_multidata` |
| SimMTM | 0.0300 | 0.0858 | 55.95% | ✅ | `vit1d_ssl_simmtm_multidata` |
| RelCon | 0.02819 | 0.0867 | 55.29% | ✅ | `vit1d_ssl_relcon_multidata` |
| Scratch | 0.0200 | 0.1670 | 49.37% | ✅ | `vit1d_odom_supervised_nymeria_altattn_scratch` |
| MobilePoser* | — | 0.150 | — | ✅ | `baselines/full_body_pose/SimplePoser` |
| Inertia-1 PatchTST-small (NHANES repro)† | 0.02748 | 0.0830 | 60.84% | ✅ (job 9567111) | `inertia1_patchtst_small_nhanes` |
| Inertia-1 ART-small (NHANES repro)† | 0.02767 | 0.0864 | 60.86% | ✅ (job 9567138) | `inertia1_art_small_nhanes` |

## 4b. Odometry — canonical world-frame variant (Nymeria pelvis motion)

| Model | vel_mse | ATE (m) | drift | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 0.02086 | **0.0787** | 48.28% | ✅ done (job 7262762) | `vit1d_ssl_mae_varlen_altattn_multidata` |
| Ours (MAE) finetune | **0.01656** | 0.1293 | **34.08%** | ✅ done (job 7321032 train, 7325714 score) | `vit1d_odom_worldframe_finetune_nymeria_mae_multidata` |
| Scratch | 0.01835 | 0.1419 | 37.31% | ✅ done (job 7262761 train, 7318026 score) | `vit1d_odom_worldframe_supervised_nymeria_altattn_scratch` |
| MobilePoser (bench protocol, 2026-08-17)‡ | 0.02963 | 0.2355 | 59.42% | ✅ (job 9719402) | `baselines/full_body_pose/SimplePoser/eval_nymeria_worldframe_odom.py` → `bench/results/odometry_nymeria_worldframe/mobileposer__external__s0.json` |
| MobilePoser* (its own 1-s translation-error metric, NOT an ATE) | — | 0.150 | — | ✅ | `baselines/full_body_pose/SimplePoser` |
| TLIO† | 0.02419 | 0.1510 | 38.79% | ✅ | `baselines/odometry/TLIO` |
| Air-IO† | 0.02461 | 0.1444 | 38.56% | ✅ | `baselines/odometry/Air-IO` |
| Inertia-1 PatchTST-small (NHANES repro)† | 0.02272 | 0.0862 | 64.74% | ✅ (job 9567110) | `inertia1_patchtst_small_nhanes` |
| Inertia-1 PatchTST-small (NHANES repro)† finetune | 0.02099 | 0.1572 | 41.68% | ✅ (jobs 9567112 train, 9567113 score) | `vit1d_odom_worldframe_finetune_nymeria_inertia1_patchtst` |
| Inertia-1 ART-small (NHANES repro)† | 0.02314 | 0.0911 | 64.69% | ✅ (job 9567137) | `inertia1_art_small_nhanes` |
| Inertia-1 ART-small (NHANES repro)† finetune | 0.02062 | 0.1539 | 39.90% | ✅ (jobs 9567139 train, 9567140 score) | `vit1d_odom_worldframe_finetune_nymeria_inertia1_art` |

‡ **MobilePoser scored on the actual §4b protocol (2026-08-17):** same 2.56 s windows on the 101 val sessions (73,515 windows), same per-window heading-canonical frame, its own predicted root trajectory (65 pts @ 25 Hz, `forward_offline` per window with only that window's IMU) interpolated onto the 33-pt grid exactly as TLIO/Air-IO (`baselines/odometry/common_worldframe_eval.score_window`). Session alignment to its shuffled `test.pt` verified by wrist-acc content. Result: ATE 0.2355 m, drift 59.4 %, vel MSE 0.0296 — worse than TLIO/Air-IO and every ViT1D row; the old 0.150 was a different quantity (root translation error over 1 s).

## 5. Microgesture recognition (OpenWatch, 6-class)

| Model | acc | Status | Run |
|---|---|---|---|
| Ours (MAE) | **67.35%** | ✅ | `vit1d_gesture_probe_openwatch_6class_mae_multidata` |
| Ours (MAE) finetune | 66.33% | ✅ | `vit1d_gesture_finetune_openwatch_6class_mae_multidata` |
| Ours (MAE, no alt-attn) | 63.27% | ✅ | `vit1d_gesture_probe_openwatch_6class_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 60.20% | ✅ | `vit1d_gesture_probe_openwatch_6class_mae_acconly_multidata` |
| FOCAL | 45.92% | ✅ | `vit1d_gesture_probe_openwatch_6class_focal_multidata` |
| SimMTM | 58.16% | ✅ | `vit1d_gesture_probe_openwatch_6class_simmtm_multidata` |
| RelCon | 54.08% | ✅ | `vit1d_gesture_probe_openwatch_6class_relcon_multidata` |
| Scratch | 44.90% | ✅ | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch` |
| NormWear-Base* (Bonazzi et al., external)‡ | 69.0% (clip-level) | ✅ | Bonazzi et al. Table 2 |

‡ OpenWatch paper's own foundation-model baseline on the *identical* 5-gesture
+ background taxonomy, matched to our exact condition (no augmentation, IMU-only —
verified via their Fig. 7 ablation, not their headline augmented/possibly-PPG
Table 2 row). 1.65pp ahead of our 67.35%, essentially tied given our ~763-sample
split. See EXPERIMENTS.md §5 for full methodology and caveats.

**⚠️ Superseded by §5e**: this table's ranking depends on an unstratified
train/val subject split with a real posture confound (val is 40%
walking-covered vs. 7.5% in train). Under a posture-balanced split, every
frozen-probe row above drops 6-18pp while both full-finetune rows improve
8-17pp — Ours(MAE) finetune becomes this task's best number (74.70%), not
the frozen Ours(MAE) probe. See §5e below before citing this ranking.

### 5a. OpenWatch's own "augmented" raw-data release — matching NormWear's headline (not just no-aug) condition

| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 66.67% | 56.91% | 0.590 | ✅ (job 7620347) | `vit1d_gesture_probe_openwatch_6class_mae_multidata_aug` |
| Ours (MAE) finetune | **73.02%** | **63.30%** | **0.672** | ✅ (job 7620349) | `vit1d_gesture_finetune_openwatch_6class_mae_multidata_aug` |
| Ours (MAE, no alt-attn) | — | — | — | ⚠️ cancelled (job 7620352, not by this session) | `vit1d_gesture_probe_openwatch_6class_mae_noaltattn_multidata_aug` |
| Ours (MAE, acc-only) | — | — | — | ⚠️ cancelled (job 7620353, not by this session) | `vit1d_gesture_probe_openwatch_6class_mae_acconly_multidata_aug` |
| FOCAL | 55.56% | 40.65% | 0.412 | ✅ (job 7620354) | `vit1d_gesture_probe_openwatch_6class_focal_multidata_aug` |
| SimMTM | 62.43% | 53.59% | 0.540 | ✅ (job 7620355) | `vit1d_gesture_probe_openwatch_6class_simmtm_multidata_aug` |
| RelCon | 67.20% | 58.27% | 0.604 | ✅ (job 7620356) | `vit1d_gesture_probe_openwatch_6class_relcon_multidata_aug` |
| Scratch | 58.73% | 53.83% | 0.517 | ✅ (job 7620357) | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch_aug` |
| **NormWear-Base (this repo, first-party)** | 53.97% | 59.62% | 0.517 | ✅ (job 9382934, n_val=189) | `eval_openwatch_normwear.py --root .../open_watch_augmentations` |

**⚠️ Whole-table caveat (2026-08-15, see EXPERIMENTS.md §5f)**: the OpenWatch
release itself is contaminated — real ids 39/41/42, 40/43, 46/48(/49) are
byte-identical duplicate subjects, and aug ids 51-85 are wrist-mirrored
copies of specific real subjects. Every row above (ours and NormWear alike)
uses the pre-discovery split convention, which can place a twin/mirror of a
held-out subject in train; rows remain comparable to *each other* but not to
a leakage-free protocol. Paper-protocol contamination-safe reruns of
Ours(MAE): jobs 9388527-9388847 (ViT1D + Mix-Token + NormWear, one shared protocol).

‡ **The paper's cited 78.8% is no longer a row in this table (2026-08-15).**
It was never in the same units as the column it sat in: every "acc" above is
*whole-recording* accuracy (one resampled 512-sample window per CSV = one
prediction), while 78.8% is **clip-level (k=3)** — 1s/100-sample windows at
stride 20, clip label by 3-consecutive-run else majority vote — on the
paper's *own* posture-stratified split, and not confirmed IMU-only. Their
own window-level number for that same row is 68.2%, so the aggregation alone
moves it ~10pp. A reader comparing 73.02% to 78.8% was reading a difference
that doesn't exist. Replaced by a first-party NormWear-Base run on this same
augmented release, same subject split, same `compute_metrics` (job 9382934);
the paper's cited numbers moved to the citation block below. Note §5c's
existing first-party 52.04% is the **no-aug** release and belongs to §5's
table, not this one — hence the new run rather than reusing it.

**Cited numbers from Bonazzi et al. Table 2 (arXiv:2605.04791) — their
protocol, their split, NOT comparable to the table above**:

| Method (cited, not run here) | Window-level Acc | Clip-level (k=3) Acc |
|---|---|---|
| Mix-Token (the paper's own proposed method) | 90.8% | 95.5% |
| Apple-CNN | not reported | 87.9% |
| NormWear-Base (headline; aug, possibly +PPG) | 68.2% | 78.8% |
| NormWear-Base* (no-aug, IMU-only) | 37.0% | 69.0% |

Mix-Token/Apple-CNN have no asterisked no-aug/IMU-only row in Table 2 (only
NormWear does), and no public checkpoint was reproduced here — so unlike
NormWear-Base they cannot be given a first-party row at all, and any gap to
them is uninterpretable in both data condition and aggregation level.

Ranking flip vs. §5 (unchanged by this edit): finetune *beats* probe here
(73.02% vs 66.67%) instead of losing to it, and every no-pretraining/
scratch-capacity model (FOCAL/SimMTM/RelCon/Scratch) improves by 4-14pp —
consistent with augmentation relieving the small-dataset overfitting that
handicapped those rows in §5. 6/8 of our own jobs done; no-alt-attn/acc-only
still **cancelled, not pending** (jobs 7620352/7620353) — they need
resubmitting before those two rows can be filled.
See EXPERIMENTS.md §5a for full analysis.

### 5b. Window-level vs. clip-level (k=3) — testing whether aggregation explains the Mix-Token/NormWear gap

Every §5/§5a number is single-window (one whole recording = one prediction);
the paper's own Table 2 reports window-level *and* clip-level(k=3), the
latter ~10pp higher per model there. Added a same-checkpoint clip-level(k=3)
number (job 7700685) to check if that aggregation gap alone explains why our
numbers trail the paper's headline Mix-Token/NormWear rows.

| Model | Window Acc | Clip(k=3) Acc | vs. §5 whole-recording Acc |
|---|---|---|---|
| Ours (MAE) | 0.493 | **0.510** | 67.35% → -16.35pp |
| Ours (MAE) finetune | 0.490 | 0.490 | 66.33% → -17.33pp |
| Ours (MAE, no alt-attn) | 0.401 | 0.480 | 63.27% → -15.27pp |
| Ours (MAE, acc-only) | 0.425 | 0.469 | 60.20% → -13.30pp |
| FOCAL | 0.401 | 0.439 | 45.92% → -2.02pp |
| SimMTM | 0.435 | 0.459 | 58.16% → -12.26pp |
| RelCon | 0.480 | **0.510** | 54.08% → -3.08pp |
| Scratch | 0.395 | 0.480 | 44.90% → **+3.10pp** |

**Result: aggregation is not the explanation** — every row's clip-level(k=3)
accuracy lands *below* its own §5 whole-recording accuracy, the opposite of
the paper's own rows. Likely cause: our loader already resamples a whole
(short) recording up to the model's fixed 512-sample/10.24s window; slicing
into thirds first stretches each third ~3x further, plausibly cropping a
short gesture's real motion into one third while the other two look like
noise. Drop is uneven though — the MAE/SimMTM family loses 12-17pp while
FOCAL/RelCon/Scratch barely move or improve, flipping the ranking (RelCon
ties Ours(MAE) for best here despite trailing by 13pp in §5). See
EXPERIMENTS.md §5b for full analysis. (§5's "tied" framing vs. NormWear-Base*
is itself superseded by §5c below.)

### 5c. First-party NormWear-Base reproduction — actually running NormWear ourselves

§5 compared our 67.35% against the *paper's own cited* NormWear-Base* number
(69.0%) — a citation, not something we ran. This experiment removes that
assumption: vendored the real NormWear repo (Luo et al., ACM Trans. Comput.
Healthcare 2026, arXiv:2412.09758; Apache-2.0), downloaded its official
136M-param pretrained checkpoint, and trained a linear head on frozen
NormWear embeddings under our own exact val split (98 clips) and metrics,
following the OpenWatch paper's own stated NormWear-Base recipe (frozen
backbone + head, class-weighted CE, AdamW, early-stop on val macro-F1).

| Model | acc | macro_f1 | Status | Run |
|---|---|---|---|---|
| Ours (MAE) probe | **67.35%** | — | ✅ (§5) | `vit1d_gesture_probe_openwatch_6class_mae_multidata` |
| NormWear-Base* (Bonazzi et al., cited) | 69.0% | — | citation only | Bonazzi et al. Table 2 |
| **NormWear-Base (this repo, first-party)** | **52.04%** | 0.498 | ✅ (job 7705059) | `scripts/eval_openwatch_normwear.py` |

**Result: the "tied" conclusion doesn't survive actually running NormWear.**
Our own reproduction lands 15.3pp behind Ours(MAE) and 17pp behind the
paper's cited number — either the paper used a materially more-tuned recipe
than its one-paragraph description, our pooling/hyperparameter choices
differ from Table 2's specifically, or the 98-clip val split is just noisy.
Read as "our faithful-effort NormWear-Base baseline underperforms
Ours(MAE) here," not as a refutation of the paper's own numbers. See
EXPERIMENTS.md §5c for full methodology and caveats.

### 5d. Corrected window/clip protocol — real 1s/20-stride windowing + 3-consecutive-run aggregation

§5b/§5c both predated finding the paper's actual protocol (arXiv:2605.04791):
1s windows (100 samples @ 100Hz), 20-sample stride, clip label = first run of
3 consecutive same-class windows else majority vote. Redid both evals with
this real rule (`imu_lm/utils/clip_voting.py`). Our raw recordings are a
fixed 300 samples (3.0s), giving 11 windows/clip at stride=20 vs. the paper's
implied ~6 (at their implied 2.0s clip length) — tested stride=40 (6
windows/clip on our real data) as a direct test of this discrepancy.

| Model | Stride=20 Clip Acc | Stride=40 Clip Acc |
|---|---|---|
| Ours (MAE) | 0.480 | 0.480 |
| Ours (MAE) finetune | 0.490 | 0.490 |
| Ours (MAE, no alt-attn) | 0.439 | 0.449 |
| Ours (MAE, acc-only) | 0.459 | **0.500** |
| FOCAL | 0.469 | 0.490 |
| SimMTM | 0.469 | 0.429 |
| RelCon | **0.500** | **0.510** |
| Scratch | 0.449 | 0.388 |
| **NormWear-Base (first-party)** | **0.459** | 0.296 |

(External citation: NormWear-Base* window acc 0.370, clip acc 0.690.)

**Window-level validates the pipeline**: at stride=20, our NormWear-Base
window accuracy (37.8%) nearly exactly matches the paper's cited window
accuracy (37.0%, 0.8pp gap) — the embedding/classification pipeline itself
looks sound. **But the stride=40 test refutes the windows-per-clip
hypothesis**: matching the paper's implied 6-windows/clip count made things
*worse* (window 37.8%→31.1%, clip 45.9%→29.6%), not better — likely because
fewer/sparser windows means less training data and noisier voting, not a
protocol fix. The remaining ~23pp clip-level gap (45.9% vs. 69.0%) is most
plausibly explained by the paper's split being "balanced per posture/activity
condition" (stratified) vs. ours (`_split_subjects`, plain unstratified
random shuffle, ratio never confirmed against the paper) or undisclosed
training-recipe details — neither resolvable without the paper's code. Our
own 8-model ranking is stable across both strides (RelCon best ~50-51%,
Scratch/no-alt-attn weakest). See EXPERIMENTS.md §5d for full analysis.

### 5e. Posture-stratified split — a bigger finding than expected

The default random split (`_split_subjects`) puts 2 of the dataset's only 3
walking-posture subjects into the 5-subject val set (40% walking-covered
val vs. 7.5% train) — a real train/val mismatch the paper's own "split
balanced per posture and activity condition" is designed to avoid. Fixed
via a new opt-in `splits.strategy: posture_stratified` split
(`_stratified_split_subjects_by_position`); retrained fresh linear heads on
frozen features for the 7 probe-style rows (cheap) and fully retrained the
2 end-to-end rows via real SLURM jobs (their old-split-adapted encoders
can't be reused with a new-split head).

| Model | Unstratified Acc | Stratified Acc | Δ |
|---|---|---|---|
| Ours (MAE) | 67.35% | 49.40% | -17.95pp |
| Ours (MAE) finetune | 66.33% | **74.70%** | +8.37pp |
| Ours (MAE, no alt-attn) | 63.27% | 49.40% | -13.87pp |
| Ours (MAE, acc-only) | 60.20% | 45.78% | -14.42pp |
| FOCAL | 45.92% | 39.76% | -6.16pp |
| SimMTM | 58.16% | 49.40% | -8.76pp |
| RelCon | 54.08% | 44.58% | -9.50pp |
| Scratch | 44.90% | **61.45%** | +16.55pp |
| NormWear-Base (first-party) | 52.04% | 51.81% | -0.23pp (flat) |

**The split choice flips which architecture pattern wins.** Every
frozen-probe row (6 ours + NormWear-Base) gets *worse* under stratification;
every full end-to-end trainable row gets *better* (Scratch +16.55pp,
finetune +8.37pp to a new task-best 74.70%). NormWear-Base is essentially
unaffected (-0.23pp) while our own frozen probes swing 6-18pp — and under
the corrected split, **NormWear-Base (51.81%) now beats every one of our own
frozen-probe rows**, reversing §5's "Ours(MAE) probe clearly best" reading.
The original single-split numbers were not a stable ranking; split choice
alone moves results by double-digit points on this small a task. See
EXPERIMENTS.md §5e for full methodology and caveats.

## 6. Language alignment (IMU↔text CLIP, HumanML3D)

| Model | acc (imu→text) | acc (text→imu) | Status | Run |
|---|---|---|---|---|
| Ours (MAE) | 19.69% | 21.11% | ✅ | `vit1d_clip_humanml3d_mae_probe` |
| Ours (MAE) finetune | 33.00% | **34.25%** | ✅ | `vit1d_clip_humanml3d_finetune_mae_multidata` |
| Ours (MAE, no alt-attn) | 20.02% | 22.20% | ✅ | `vit1d_clip_humanml3d_mae_noaltattn_probe` |
| Ours (MAE, acc-only) | 12.77% | 13.22% | ✅ | `vit1d_clip_humanml3d_mae_acconly_probe` |
| FOCAL | 11.72% | 12.00% | ✅ | `vit1d_clip_humanml3d_focal_probe` |
| SimMTM | 16.06% | 17.95% | ✅ | `vit1d_clip_humanml3d_simmtm_probe` |
| RelCon | 11.59% | 13.59% | ✅ | `vit1d_clip_humanml3d_relcon_probe` |
| Scratch | 28.81% | 30.44% | ✅ | `vit1d_clip_humanml3d_scratch` |
| Ours (MAE) finetune, fixed-window | 24.73% | 24.92% | ✅ (job 7543850) — worse than the random-crop finetune row above (33.00/34.25%) by ~8-9pp both directions; see EXPERIMENTS.md §6 | `vit1d_clip_humanml3d_finetune_mae_fixedwindow_multidata` |
| Ours (MAE) finetune, aug+adapter | 30.06% | 30.47% | ✅ (job 7567181) — *worse* than the plain finetune baseline (33.00/34.25%); rotation augmentation is the leading suspect (a contrastive task's frozen text targets don't reward rotation invariance). See EXPERIMENTS.md §6 | `vit1d_clip_humanml3d_finetune_mae_augadapter_multidata` |
| UniMTS (zero-shot, external)† | 8.79% | 1.76% | ✅ | `baselines/language_alignment/UniMTS/eval_unimts_humanml3d.py` (job 7390552) |
| UniMTS (finetuned on HumanML3D, external)†‡ | 53.99% | 5.15% | ✅ | `baselines/language_alignment/UniMTS/train_unimts_humanml3d.py` (job 7390743) |
| UniMTS (finetuned, frozen text tower, external)†‡ | **61.11%** | 4.11% | ✅ | `baselines/language_alignment/UniMTS/train_unimts_humanml3d_frozen_text.sh` (job 7481114) |

‡ **Caveat**: UniMTS's officially released checkpoint was pretrained directly
on HumanML3D itself (synthetic IMU + LLM-enriched captions derived from
HumanML3D's own motions, per the vendored repo's own README) — these two
finetuned rows are lightly recalibrating an already-aligned space, not
learning alignment from scratch, so they aren't a clean external ceiling.
See EXPERIMENTS.md §6 for the full writeup.

† UniMTS (Zhang et al., NeurIPS 2024) is a literature baseline, accel-only
(released checkpoint has no gyro channels), with the single wrist IMU placed
at joint 21 of UniMTS's 22-joint skeleton graph (other joints zeroed),
native 20Hz/200-sample window. First row is **zero-shot** (official
pretrained checkpoint, no finetuning). Second row is **finetuned** on this
repo's own `HumanML3D/train.txt` (8191 valid clips), full end-to-end
(ST-GCN + CLIP text tower, nothing frozen), using UniMTS's own official
`pretrain.py` recipe (Adam lr=1e-4, temperature=0.1, rotation aug), early-
stopped at step 10500 (best ckpt step 500). Both rows evaluated on the same
518-clip val split with the identical retrieval-accuracy formula as the
other rows. The 53.99% imu→text value is the single highest in that column
across the whole table and is bolded accordingly, but text→imu (5.15%)
stays far below every ViT1D row on the very same run -- a consistent
imu→text >> text→imu asymmetry present even zero-shot (see EXPERIMENTS.md
§6 for the full discussion, incl. a plausible explanation via HumanML3D's
caption repetitiveness) -- so this is not a strict apples-to-apples
comparison; see EXPERIMENTS.md §6 for full methodology.

## 6b. Train/eval-overlap leakage control — NOT a valid held-out comparison

Tests the UniMTS-overlap theory above directly: train our own recipe on
`train_val.txt` (HumanML3D's official train+val union) instead of
`train.txt`, then score retrieval on `val.txt` — i.e. every eval clip was
also a training clip. Not a genuine generalization number; measures how
much train/eval overlap alone can inflate this exact metric.

| Model | acc (imu→text) | acc (text→imu) | Status | Run |
|---|---|---|---|---|
| Ours (MAE) finetune (held-out, reference) | 33.00% | 34.25% | ✅ | `vit1d_clip_humanml3d_finetune_mae_multidata` |
| Ours (MAE) finetune, train-on-val (leakage control) | **92.52%** | **92.77%** | ✅ (job 7568586, best step 109,000) | `vit1d_clip_humanml3d_finetune_mae_trainonval_multidata` |

**Result: a ~59pp jump** (33.00→92.52% / 34.25→92.77%) from the identical
recipe, changing nothing but whether the eval clips were trained on —
strong evidence the leakage mechanism alone can produce inflation of
roughly the magnitude seen in UniMTS's finetuned HumanML3D numbers (§6),
though not proof of that exact mechanism for UniMTS specifically (this
experiment's own jump is larger than UniMTS's zero-shot→finetuned jump,
~45pp, consistent with leakage being *a* major driver, not the only one).
See EXPERIMENTS.md §6b for full methodology.

## 6c. Language alignment (IMU↔text CLIP, Nymeria motion-narration captions)

Same CLIP machinery as §6, but real recorded wrist IMU + real human-written
captions (Nymeria's own `motion_narration.csv` annotations, 236/1100
sequences, downloaded via `scripts/download_nymeria_captions.py`) instead of
HumanML3D's synthetic IMU + templated captions. New loader
`imu_lm/data/nymeria_caption_dataset_FLU.py`, new objective
`imu_text_clip_nymeria_flu`. Same 867/101-session train/val split as every
other Nymeria task in this benchmark (already leakage-verified). 193 train
sequences / 32,642 caption rows, 20 val sequences / 3,488 caption rows.

| Model | acc (imu→text) | acc (text→imu) | Status | Run |
|---|---|---|---|---|
| Ours (MAE) probe | 6.73% | 6.86% | ✅ (job 7578437) | `vit1d_clip_nymeria_mae_probe` |
| Ours (MAE) probe, lr1e-3 | 6.44% | 6.44% | ✅ (job 7587278) — LR-convention fix, ~flat vs default, didn't help | `vit1d_clip_nymeria_mae_probe_lr1e-3` |
| Ours (MAE) finetune | 5.81% | 5.69% | ✅ (job 7578438) | `vit1d_clip_nymeria_finetune_mae_multidata` |
| Ours (MAE) finetune, lr3e-5 | **6.72%** | **6.69%** | ✅ (job 7587279) — lower LR fixes the overfitting-vs-probe gap | `vit1d_clip_nymeria_finetune_mae_lr3e-5` |
| Ours (MAE) finetune, lr3e-4 | 6.00% | 6.27% | ✅ (job 7593255) — better than lr1e-4 default, worse than lr3e-5, confirming lower-LR direction | `vit1d_clip_nymeria_finetune_mae_lr3e-4` |
| UniMTS (zero-shot, external)† | 1.94% | 1.77% | ✅ (job 7579739) | `baselines/language_alignment/UniMTS/eval_unimts_nymeria.py` |
| UniMTS (finetuned on Nymeria, external)† | 6.12% | 6.36% | ✅ (job 7581974, early-stopped step 10,500, best step 500) | `baselines/language_alignment/UniMTS/train_unimts_nymeria.py` |
| Ours (MAE) probe, caption-merged | 5.88% | 6.23% | ✅ (job 7596783) — slightly worse than raw (6.73/6.86%) | `vit1d_clip_nymeria_mae_probe_capmerged` |
| Ours (MAE) finetune, caption-merged | 5.95% | 6.59% | ✅ (job 7596797) — ~flat vs raw (5.81/5.69%) | `vit1d_clip_nymeria_finetune_mae_capmerged` |
| UniMTS (zero-shot, caption-merged, external)† | 2.01% | 1.60% | ✅ (job 7596798) — ~flat vs raw (1.94/1.77%) | `baselines/language_alignment/UniMTS/eval_unimts_nymeria.py` |
| Ours (MAE) probe, atomic_action captions | 7.97% | 8.28% | ✅ (job 7602000) — beats motion_narration probe (6.73/6.86%) | `vit1d_clip_nymeria_atomicaction_mae_probe` |
| Ours (MAE) finetune, atomic_action captions | **10.09%** | **10.50%** | ✅ (job 7602001) — best Nymeria CLIP result overall | `vit1d_clip_nymeria_atomicaction_finetune_mae_multidata` |
| Ours (MAE) probe, atomic_action_augmented captions | 6.95% | 7.72% | ✅ (job 7679669) — worse than plain atomic_action (7.97/8.28%) | `vit1d_clip_nymeria_atomicaction_augmented_probe_mae_multidata` |
| Ours (MAE) finetune, atomic_action_augmented captions | 8.80% | 9.06% | ✅ (job 7679670) — worse than plain atomic_action (10.09/10.50%) | `vit1d_clip_nymeria_atomicaction_augmented_finetune_mae_multidata` |
| UniMTS (finetuned, atomic_action_augmented, external) | 2.11% | 1.80% | ✅ (job 7679671) — sharp regression vs. plain atomic_action (6.12/6.36%) | `baselines/language_alignment/UniMTS/train_unimts_nymeria_atomicaction_augmented.sh` |

†Same UniMTS adaptation as §6 (accel-only, right-wrist joint 21, 20Hz/200-sample
windows) — **no contamination caveat here**, since UniMTS's released
checkpoint was pretrained on HumanML3D, not Nymeria, so this comparison is
clean. Every number on this task is far weaker than the equivalent
HumanML3D row (including UniMTS, ruling out leakage as the explanation) —
plausibly because Nymeria's captions are far more fine-grained (~5s
segments) and repetitive *within* a session than HumanML3D's one-per-clip
captions, a harder discrimination problem given only 20 val sequences.
Notably, **probe beat finetune at the untuned default LR** (opposite of
HumanML3D) — but a lower finetune LR (3e-5) closes almost the entire gap
(5.81→6.72%), supporting an overfitting explanation rather than a
fundamental probe > finetune result. Raising the probe's LR to match this
repo's own frozen-probe convention (1e-3) did *not* help, unlike the
finetune LR fix. All three finetune LR points now complete: 1e-4
(5.81/5.69) < 3e-4 (6.00/6.27) < 3e-5 (6.72/6.69) — lower LR is
unambiguously better, `lr:3e-5` is the best finetune setting found.
UniMTS finetune final: 6.12/6.36% (early-stopped step 10,500) — essentially
tied with Ours(MAE) probe and Ours(MAE) finetune@lr3e-5; all three "serious"
models land within ~1pp once each is given a fair shot, suggesting this
task's ceiling is set more by the data than by model/recipe choice.

**Caption cleaning**: measured why this task is so much harder than
HumanML3D — only 64.1% of val captions are unique (top one repeats 51x),
9-15% of any batch has a near-duplicate (cos 0.90-0.95) the loss's tight
0.999 threshold scores as a false negative, and ~9.5% of segments are
near-stationary with near-identical captions (genuine noise floor, not a
labeling bug). Implemented a conservative fix: `_parse_caption_csv` now
merges consecutive rows with identical captions (gap ≤1s) by default — real
but modest effect (val rows 3,488→3,187, unique-caption fraction
64.1%→70.2%), since most repetition recurs at scattered non-contiguous
points, not just oversegmented runs. **Result: a wash** — probe 5.88/6.23%
(slightly worse than raw 6.73/6.86%), finetune 5.95/6.59% (~flat vs raw
5.81/5.69%), UniMTS zero-shot 2.01/1.60% (~flat vs raw 1.94/1.77%). Kept as
the new default (harmless), but doesn't solve the repetitiveness problem.

**Fuller manifest follow-up**: checked Nymeria's complete download manifest
(`scripts/download/Nymeria_download_urls_all.json`, all file types × 1,100 sequences) for a
more motion-focused caption source. Two more narration types exist:
`atomic_action.csv` (845/1,100 seqs, 3.6x more coverage than
`motion_narration.csv`'s 236) and `activity_summarization.csv` (864/1,100,
coarser ~30s summaries). **Neither is actually motion-only** — verified by
reading real files, both reference rooms/objects just as much as
`motion_narration.csv`. Doesn't solve the original hypothesis, but
`atomic_action.csv`'s much larger coverage is independently valuable
(more data, richer text) — both files downloaded cleanly (845/845,
864/864, zero failures). Pipeline built: `caption_source` registry added to
the shared loader/objective (`vit1d_clip_nymeria_atomicaction_{probe,
finetune}_mae_multidata.yaml`, probe at lr1e-4, finetune at lr3e-5 — the
better LR already found for each). Text-embedding cache built (143,055
unique captions, vs. 83,402 for motion_narration).

**Result: a genuine win.** Probe 7.97/8.28% (beats motion_narration probe
6.73/6.86%), finetune **10.09/10.50%** — the best result anywhere on this
task, well clear of motion_narration finetune@lr3e-5 (6.72/6.69%, the
previous best) and UniMTS's finetuned 6.12/6.36%. Unlike the caption-merge
cleaning (a wash), more coverage + richer text moved the needle
substantially — this task's low numbers were significantly a
data-scarcity/caption-quality problem, not purely an inherent domain
ceiling. Finetune also beats probe here (unlike motion_narration's
untuned-LR reversal), consistent with more data reducing overfitting risk.
See EXPERIMENTS.md §6c for full methodology.

**LLM-simplified caption augmentation (2026-07-23): a regression across all
three models.** Used `Qwen/Qwen2.5-7B-Instruct` to rewrite each of
atomic_action's 143,054 unique captions into a short, HumanML3D-style
motion-only sentence (drop object nouns / other people), added as a second
description column (`atomic_action_augmented.csv`, 845/845 sequences, 0
missing) rather than replacing the originals. Both probe (6.95/7.72%) and
finetune (8.80/9.06%) came out *worse* than plain atomic_action
(7.97/8.28% / 10.09/10.50%) by ~1-1.4pp — plausibly because the shorter,
more generic simplified captions increase near-duplicate collisions in the
text embedding space (the same false-negative mechanism behind this task's
difficulty generally), losing more discriminative signal than the
motion-only style gains. **UniMTS regresses far more sharply**: 2.11/1.80%
vs. its plain-atomic_action 6.12/6.36% (job 7581974), a -4.01/-4.56pp drop
— 3-4x the size of Ours(MAE)'s regression, plausibly because its finetune
only moves 500 steps before its best checkpoint (same as the non-augmented
run), leaving it far less able to adapt to the doubled, more homogeneous
caption pool than Ours(MAE)'s longer training run. All three models agree
in direction (augmentation hurts), though the magnitude differs sharply.
See EXPERIMENTS.md §6c for full discussion.

**Synthetic (sensor-noise-free) IMU test (2026-07-23): rules out sensor
noise.** Checked why HumanML3D's finetune (33.00/34.25%) is so much higher
than Nymeria atomic_action's (10.09/10.50%) despite similar caption style.
First ruled out caption repetitiveness (HumanML3D val is actually *more*
repetitive, 68.3% unique, vs. atomic_action val's 85.0% unique). Then
tested the sensor-noise hypothesis directly: derived a clean, synthetic
wrist IMU from Nymeria's own ground-truth XSens mocap
(`gt_data['xsens_pose']`, same math as the existing
`nymeria_syn_wrist_loader_FLU.py`, new `imu_source: synthetic` flag) instead
of the real recorded hardware sensor, keeping identical captions/split/LRs.

| Model | acc (imu→text) | acc (text→imu) | Status | Run |
|---|---|---|---|---|
| Ours (MAE) probe, atomic_action + synthetic IMU | 7.58% | 7.94% | ✅ (job 7692003) — flat vs. real IMU (7.97/8.28%) | `vit1d_clip_nymeria_atomicaction_synthetic_mae_probe` |
| Ours (MAE) finetune, atomic_action + synthetic IMU | 9.39% | 10.64% | ✅ (job 7692004) — flat vs. real IMU (10.09/10.50%) | `vit1d_clip_nymeria_atomicaction_synthetic_finetune_mae_multidata` |

**Result: essentially no change** — both land within ~0.3-0.7pp of their
real-IMU counterparts, nowhere near HumanML3D's ceiling. Swapping a noisy
hardware sensor for a perfectly clean, ground-truth-mocap-derived signal
bought nothing, decisively ruling out sensor noise as the bottleneck. The
likely remaining explanation is the caption-*generation* process itself:
HumanML3D's captions were written watching the exact mocap clip the IMU
came from (guaranteed correspondence); Nymeria's were written watching
separate egocentric video with full scene/object context a wrist sensor
can't recover, against only approximately-aligned segment timestamps — a
structural dataset-curation gap, not a fixable sensor or caption-wording
issue. See EXPERIMENTS.md §6c for full discussion.

## 8. Cross-task transfer (does finetuning on task A help/hurt task B?)

### 8a. Frozen-probe transfer

### Target: WISDM HAR
| Source encoder | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| *(none — SSL pretrain only)* | 65.50% | 65.52% | 0.639 | ✅ | `vit1d_har_probe_wisdm_mae_multidata` |
| *(scratch, no pretraining at all)* | 60.89% | 60.89% | 0.606 | ✅ | `vit1d_har_supervised_wisdm_altattn_scratch` |
| Full-body pose (finetuned) | 64.09% | 64.06% | 0.629 | ✅ | `vit1d_transfer_wisdm_from_fullbodypose` |
| Odometry, world-frame (finetuned) | 62.31% | 62.28% | 0.606 | ✅ | `vit1d_transfer_wisdm_from_odomworldframe` |
| Gesture (finetuned) | 64.78% | 64.79% | 0.632 | ✅ | `vit1d_transfer_wisdm_from_gesture` |
| Parkinson's, PADS (finetuned) | **65.96%** | **65.96%** | **0.645** | ✅ | `vit1d_transfer_wisdm_from_pd` |
| *(in-domain upper bound)* | **71.27%** | **71.22%** | **0.704** | ✅ | `vit1d_har_finetune_wisdm_mae_multidata` |

### Target: Full-body pose, Nymeria (mpjpe, mm)
| Source encoder | mpjpe (mm) | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 97.63 | ✅ | `vit1d_full_body_pose_probe_nymeria_mae_multidata` |
| *(scratch, no pretraining at all)* | 92.56 | ✅ | `vit1d_full_body_pose_supervised_nymeria_altattn_scratch` |
| WISDM HAR (finetuned) | 100.08 | ✅ | `vit1d_transfer_fullbodypose_from_wisdm` |
| Odometry, world-frame (finetuned) | **95.93** | ✅ | `vit1d_transfer_fullbodypose_from_odomworldframe` |
| Gesture (finetuned) | 99.70 | ✅ | `vit1d_transfer_fullbodypose_from_gesture` |
| Parkinson's, PADS (finetuned) | 100.42 | ✅ | `vit1d_transfer_fullbodypose_from_pd` |
| *(in-domain upper bound)* | 92.39 | ✅ | `vit1d_full_body_pose_finetune_nymeria_mae_multidata` |

### Target: Gesture, OpenWatch 6-class
| Source encoder | acc | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 67.35% | ✅ | `vit1d_gesture_probe_openwatch_6class_mae_multidata` |
| *(scratch, no pretraining at all)* | 44.90% | ✅ | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch` |
| WISDM HAR (finetuned) | 64.29% | ✅ | `vit1d_transfer_gesture_from_wisdm` |
| Full-body pose (finetuned) | 62.24% | ✅ | `vit1d_transfer_gesture_from_fullbodypose` |
| Odometry, world-frame (finetuned) | 56.12% | ✅ | `vit1d_transfer_gesture_from_odomworldframe` |
| Parkinson's, PADS (finetuned) | 65.31% | ✅ | `vit1d_transfer_gesture_from_pd` |
| *(in-domain upper bound)* | 66.33% | ✅ | `vit1d_gesture_finetune_openwatch_6class_mae_multidata` |

### Target: Odometry, world-frame, Nymeria (vel_mse / ATE m / drift %)
| Source encoder | vel_mse | ATE (m) | drift | Status | Run |
|---|---|---|---|---|---|
| *(none — SSL pretrain only)* | 0.02086 | 0.0787 | 48.28% | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata` |
| *(scratch, no pretraining at all)* | 0.01835 | 0.1419 | 37.31% | ✅ | `vit1d_odom_worldframe_supervised_nymeria_altattn_scratch` |
| WISDM HAR (finetuned) | 0.02091 | 0.0783 | 49.15% | ✅ | `vit1d_transfer_odomworldframe_from_wisdm` |
| Full-body pose (finetuned) | **0.01978** | **0.0753** | **43.99%** | ✅ | `vit1d_transfer_odomworldframe_from_fullbodypose` |
| Gesture (finetuned) | 0.02115 | 0.0790 | 49.21% | ✅ | `vit1d_transfer_odomworldframe_from_gesture` |
| Parkinson's, PADS (finetuned) | 0.02107 | **0.0773** | 48.86% | ✅ | `vit1d_transfer_odomworldframe_from_pd` |
| *(in-domain upper bound)* | **0.01656** | 0.1293 | **34.08%** | ✅ | `vit1d_odom_worldframe_finetune_nymeria_mae_multidata` |

### Target: Parkinson's differential diagnosis, PADS 3-class (bal_acc)
Read `bal_acc`, not `acc` — the PD-heavy majority puts every column within
1.5pp of ~62% accuracy, chance `bal_acc` is 33.33% (§14).

| Source encoder | bal_acc | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 38.94% | ✅ | `vit1d_pd_probe_pads_mae` |
| *(scratch, no pretraining at all)* | 35.99% | ✅ | `vit1d_pd_supervised_pads_scratch` |
| WISDM HAR (finetuned) | 37.67% | ✅ | `vit1d_transfer_pd_from_wisdm` |
| Full-body pose (finetuned) | 37.62% | ✅ | `vit1d_transfer_pd_from_fullbodypose` |
| Gesture (finetuned) | **40.23%** | ✅ | `vit1d_transfer_pd_from_gesture` |
| Odometry, world-frame (finetuned) | 36.11% | ✅ | `vit1d_transfer_pd_from_odomworldframe` |
| *(in-domain upper bound)* | 39.19% | ✅ | `vit1d_pd_finetune_pads_mae` |

### 8b. Full end-to-end finetune transfer (encoder warm-started AND kept trainable)

Sharply different from 8a: with the encoder kept trainable, cross-task
transfer routinely matches or **beats** the in-domain finetune upper bound.
A full-body-pose-finetuned source encoder is the standout: transferred into
*every other target it was tried on* (WISDM, gesture, world-frame
odometry), it beats that target's own in-domain upper bound. Full-body-pose
is also the easiest *target* to beat -- all three cross-task sources land
below its own in-domain upper bound, with world-frame-odometry-as-source
(91.49mm) the single best cell in the whole table -- see EXPERIMENTS.md for
the full writeup. 5 of the original 12 job attempts crashed from a
disk-quota checkpoint-write issue (same root cause as RelCon's Stage-2
pretrain) and were resubmitted under new job IDs, noted below.

#### Target: WISDM HAR
| Source encoder | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| *(none — SSL pretrain only)* | 65.50% | 65.52% | 0.639 | ✅ | `vit1d_har_probe_wisdm_mae_multidata` |
| *(scratch, no pretraining at all)* | 60.89% | 60.89% | 0.606 | ✅ | `vit1d_har_supervised_wisdm_altattn_scratch` |
| Full-body pose (finetuned, full finetune) | **71.80%** | **71.75%** | **0.710** | ✅ | `vit1d_transfer_finetune_wisdm_from_fullbodypose` |
| Odometry, world-frame (finetuned, full finetune) | 71.29% | 71.24% | 0.705 | ✅ | `vit1d_transfer_finetune_wisdm_from_odomworldframe` |
| Gesture (finetuned, full finetune) | 71.18% | 71.12% | 0.703 | ✅ | `vit1d_transfer_finetune_wisdm_from_gesture` |
| Parkinson's, PADS (finetuned, full finetune) | **72.25%** | **72.19%** | **0.717** | ✅ | `vit1d_transfer_finetune_wisdm_from_pd` |
| *(in-domain upper bound)* | 71.27% | 71.22% | 0.704 | ✅ | `vit1d_har_finetune_wisdm_mae_multidata` |

#### Target: Full-body pose, Nymeria (mpjpe, mm)
| Source encoder | mpjpe (mm) | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 97.63 | ✅ | `vit1d_full_body_pose_probe_nymeria_mae_multidata` |
| *(scratch, no pretraining at all)* | 92.56 | ✅ | `vit1d_full_body_pose_supervised_nymeria_altattn_scratch` |
| WISDM HAR (finetuned, full finetune) | 92.16 | ✅ | `vit1d_transfer_finetune_fullbodypose_from_wisdm` |
| Odometry, world-frame (finetuned, full finetune) | **91.49** | ✅ | `vit1d_transfer_finetune_fullbodypose_from_odomworldframe` |
| Gesture (finetuned, full finetune) | 92.24 | ✅ | `vit1d_transfer_finetune_fullbodypose_from_gesture` |
| Parkinson's, PADS (finetuned, full finetune) | 92.02 | ✅ | `vit1d_transfer_finetune_fullbodypose_from_pd` |
| *(in-domain upper bound)* | 92.39 | ✅ | `vit1d_full_body_pose_finetune_nymeria_mae_multidata` |

#### Target: Gesture, OpenWatch 6-class
| Source encoder | acc | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 67.35% | ✅ | `vit1d_gesture_probe_openwatch_6class_mae_multidata` |
| *(scratch, no pretraining at all)* | 44.90% | ✅ | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch` |
| WISDM HAR (finetuned, full finetune) | 63.27% | ✅ | `vit1d_transfer_finetune_gesture_from_wisdm` |
| Full-body pose (finetuned, full finetune) | **68.37%** | ✅ | `vit1d_transfer_finetune_gesture_from_fullbodypose` |
| Odometry, world-frame (finetuned, full finetune) | 67.35% | ✅ | `vit1d_transfer_finetune_gesture_from_odomworldframe` |
| Parkinson's, PADS (finetuned, full finetune) | 65.31% | ✅ | `vit1d_transfer_finetune_gesture_from_pd` |
| *(in-domain upper bound)* | 66.33% | ✅ | `vit1d_gesture_finetune_openwatch_6class_mae_multidata` |

#### Target: Odometry, world-frame, Nymeria (vel_mse / ATE m / drift %)
| Source encoder | vel_mse | ATE (m) | drift | Status | Run |
|---|---|---|---|---|---|
| *(none — SSL pretrain only)* | 0.02086 | 0.0787 | 48.28% | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata` |
| *(scratch, no pretraining at all)* | 0.01835 | 0.1419 | 37.31% | ✅ | `vit1d_odom_worldframe_supervised_nymeria_altattn_scratch` |
| WISDM HAR (finetuned, full finetune) | 0.01684 | 0.1278 | 34.09% | ✅ | `vit1d_transfer_finetune_odomworldframe_from_wisdm` |
| Full-body pose (finetuned, full finetune) | **0.01622** | **0.1259** | **33.13%** | ✅ | `vit1d_transfer_finetune_odomworldframe_from_fullbodypose` |
| Gesture (finetuned, full finetune) | 0.01666 | 0.1323 | 36.42% | ✅ | `vit1d_transfer_finetune_odomworldframe_from_gesture` |
| Parkinson's, PADS (finetuned, full finetune) | 0.01686 | 0.1306 | 35.06% | ✅ | `vit1d_transfer_finetune_odomworldframe_from_pd` |
| *(in-domain upper bound)* | 0.01656 | 0.1293 | 34.08% | ✅ | `vit1d_odom_worldframe_finetune_nymeria_mae_multidata` |

#### Target: Parkinson's differential diagnosis, PADS 3-class (bal_acc)
| Source encoder | bal_acc | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 38.94% | ✅ | `vit1d_pd_probe_pads_mae` |
| *(scratch, no pretraining at all)* | 35.99% | ✅ | `vit1d_pd_supervised_pads_scratch` |
| WISDM HAR (finetuned, full finetune) | **42.01%** | ✅ | `vit1d_transfer_finetune_pd_from_wisdm` |
| Full-body pose (finetuned, full finetune) | 39.82% | ✅ | `vit1d_transfer_finetune_pd_from_fullbodypose` |
| Gesture (finetuned, full finetune) | 39.16% | ✅ | `vit1d_transfer_finetune_pd_from_gesture` |
| Odometry, world-frame (finetuned, full finetune) | 39.71% | ✅ | `vit1d_transfer_finetune_pd_from_odomworldframe` |
| *(in-domain upper bound)* | 39.19% | ✅ | `vit1d_pd_finetune_pads_mae` |

**Reads for the PD row and column (added 2026-08-18).** Every one of the eight
PD-as-target cells beats the from-scratch baseline (35.99% bal_acc), so the row
behaves like the others: cross-task initialisation helps. Two cells stand out.
Frozen, the **gesture**-finetuned encoder is the best PD source at **40.23%** —
above both the MAE probe (38.94%) and the in-domain MAE finetune upper bound
(39.19%). Trainable, **WISDM** is the best PD source at **42.01%**, +2.8pp over
that same upper bound and the best PD number anywhere in this repo at the
best-val-loss checkpoint. World-frame odometry is the weakest source in both
protocols (36.11% frozen, barely above scratch), which fits: PD diagnosis keys
on 4-6 Hz wrist tremor, and an encoder tuned for pelvis translation has the
least reason to preserve it.

As a *source*, the PD encoder is unexpectedly strong on HAR. It is the best
frozen source for WISDM (**65.96%**, above the MAE probe's 65.50% and every
other cross-task source) and the best trainable one too (**72.25%**, beating
WISDM's own in-domain upper bound of 71.27%). Some of that is likely window
alignment rather than task content — PD and WISDM are the only two 2.56 s tasks,
so the PD encoder's finetuned position embeddings line up exactly with WISDM's
and only partially with the 10.24 s tasks. Elsewhere the PD column is mid-pack:
second-best frozen odometry source (0.0773 ATE, behind pose's 0.0753), mid-pack
on gesture, and negative transfer into full-body pose (100.42 mm) exactly like
every other frozen source in that row.

⚠️ The two OpenWatch-gesture-from-PD cells are both 65.31% — genuinely
different runs (best checkpoints at step 800 vs 200, val losses 0.875 vs 1.652)
that collide because the gesture val split is only 49 samples, so accuracy is
quantised to 1/49 ≈ 2.04pp. Do not read the equality as a duplicated run.

The matrix figures (`analysis/transferability_scratch.png`,
`analysis/transferability_finetune.png`) are regenerated with
`python analysis/harvest_pd_transfer.py && python analysis/plot_transferability.py`;
the PD row/column are read from the runs, the other 20 cells stay hard-coded
from the tables above. Caveat for the PD **column**: the source encoder
(`vit1d_pd_finetune_pads_mae`) was finetuned on 2.56 s windows, so only its
first 8 position embeddings per channel carry PD-specific adaptation — the
same situation as the WISDM source column.

## 9. Pretraining data scaling (MAE reconstruction loss vs. % of pretraining data)

| % of pretraining data | `overall_fraction` | best val MSE loss | step @ best | epoch @ best | Status | Run |
|---|---|---|---|---|---|---|
| 0.1% | 0.001 | 0.8111 | 29,000 | 9,667 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data0.1pct` |
| 1% | 0.01 | 0.7643 | 135,000 | 3,649 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data1pct` |
| 5% | 0.05 | 0.7430 | 221,000 | 1,170 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data5pct` |
| 25% | 0.25 | 0.7438 | 221,000 | 234 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data25pct` |
| 50% | 0.5 | 0.7419 | 221,000 | 117 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data50pct` |
| 100% (main run) | 1.0 | **0.7351** | 291,000 | 77 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata` |

Monotonic improvement with more data, diminishing returns past ~5% (25%/50%
barely beat 5%). See EXPERIMENTS.md §9 for full methodology and discussion.

**🐛 2026-07-22: found the pooled `overall_fraction` draw can zero out small
sources at 0.1%/1%** (totalcapture/imuposer, each <0.1% of the pool) —
fixed with an opt-in `stratify_overall_fraction: true` flag that samples
each source independently. Stratified reruns (jobs 7624656 @ 0.1%, 7624657
@ 1%) completed: val loss 0.8100 (vs. pooled 0.8111) and 0.7657 (vs. pooled
0.7643) respectively — both deltas (~0.001-0.0014) are noise-level, **a
wash**, not the driver of the steep low-end curve. Pooled rows above kept
as canonical (all downstream §9b-9e evals already ran against those
checkpoints). See EXPERIMENTS.md §9 for full writeup.

### 9b. Downstream HAR scaling (linear-probe accuracy vs. % of pretraining data)

Frozen linear probes on top of each scaling checkpoint above (same probe
recipe/config as the "Ours (MAE)" rows in Section 3). Metric: best
val_loss checkpoint's accuracy, same convention as Section 3. 100% row
reuses Section 3's "Ours (MAE)" probe numbers (same checkpoint).

#### WISDM
| % of pretraining data | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| 0.1% | 57.29% | 57.35% | 0.550 | ✅ | `vit1d_har_probe_wisdm_mae_data0.1pct` |
| 1% | **65.75%** | **65.76%** | **0.644** | ✅ | `vit1d_har_probe_wisdm_mae_data1pct` |
| 5% | 65.68% | 65.70% | 0.641 | ✅ | `vit1d_har_probe_wisdm_mae_data5pct` |
| 25% | 65.29% | 65.29% | 0.637 | ✅ | `vit1d_har_probe_wisdm_mae_data25pct` |
| 50% | 65.61% | 65.61% | 0.639 | ✅ | `vit1d_har_probe_wisdm_mae_data50pct` |
| 100% (main run) | 65.50% | 65.52% | 0.639 | ✅ | `vit1d_har_probe_wisdm_mae_multidata` |

#### UTD-MHAD
| % of pretraining data | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| 0.1% | 54.59% | 55.56% | 0.542 | ✅ | `vit1d_har_probe_utd_mhad_mae_data0.1pct` |
| 1% | 58.45% | 58.85% | 0.555 | ✅ | `vit1d_har_probe_utd_mhad_mae_data1pct` |
| 5% | 62.32% | 61.60% | 0.610 | ✅ | `vit1d_har_probe_utd_mhad_mae_data5pct` |
| 25% | **69.08%** | **69.24%** | **0.680** | ✅ | `vit1d_har_probe_utd_mhad_mae_data25pct` |
| 50% | 64.73% | 64.61% | 0.639 | ✅ | `vit1d_har_probe_utd_mhad_mae_data50pct` |
| 100% (main run) | 63.77% | 63.83% | 0.633 | ✅ | `vit1d_har_probe_utd_mhad_mae_multidata` |

#### PAMAP2
| % of pretraining data | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| 0.1% | 51.14% | 47.53% | 0.438 | ✅ | `vit1d_har_probe_pamap2_mae_data0.1pct` |
| 1% | 57.60% | 55.33% | 0.560 | ✅ | `vit1d_har_probe_pamap2_mae_data1pct` |
| 5% | 57.73% | 55.32% | 0.562 | ✅ | `vit1d_har_probe_pamap2_mae_data5pct` |
| 25% | 57.27% | 54.84% | 0.560 | ✅ | `vit1d_har_probe_pamap2_mae_data25pct` |
| 50% | **58.02%** | **56.75%** | **0.567** | ✅ | `vit1d_har_probe_pamap2_mae_data50pct` |
| 100% (main run) | 56.12% | 52.86% | 0.542 | ✅ | `vit1d_har_probe_pamap2_mae_multidata` |

#### RecoFit
| % of pretraining data | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| 0.1% | 71.65% | 56.62% | 0.565 | ✅ | `vit1d_har_probe_recofit_mae_data0.1pct` |
| 1% | 78.67% | 64.33% | 0.654 | ✅ | `vit1d_har_probe_recofit_mae_data1pct` |
| 5% | 79.37% | **65.87%** | **0.665** | ✅ | `vit1d_har_probe_recofit_mae_data5pct` |
| 25% | **80.23%** | 65.17% | 0.656 | ✅ | `vit1d_har_probe_recofit_mae_data25pct` |
| 50% | 78.11% | 64.20% | 0.643 | ✅ | `vit1d_har_probe_recofit_mae_data50pct` |
| 100% (main run) | 78.21% | 61.67% | 0.627 | ✅ | `vit1d_har_probe_recofit_mae_multidata` |

Unlike the pretraining-loss curve (monotonic through 100%), HAR probe
accuracy saturates or peaks well before 100% of pretraining data on every
one of the four datasets — WISDM/PAMAP2/RecoFit all peak around 1-50% and
are flat-to-slightly-*down* at 100%; UTD-MHAD peaks hardest at 25% (69.08%,
+5.3pp over the 100% checkpoint). Consistent with Section 9's reconstruction
loss already showing diminishing returns past ~5% data: once the encoder's
reconstruction loss stops improving much, extra pretraining data isn't
buying extra downstream-task signal either, and on the smaller HAR datasets
(UTD-MHAD, PAMAP2 val splits are small) the differences across 5-100% are
plausibly within eval noise rather than a real regression. Gesture scaling
is not yet run; arm pose/full-body pose/odometry scaling are below.

### 9c. Downstream arm pose scaling (Nymeria probe val_loss vs. % of pretraining data)

| % of pretraining data | best val_loss | Status | Run |
|---|---|---|---|
| 0.1% | 0.017561 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data0.1pct_multidata` |
| 1% | 0.016630 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data1pct_multidata` |
| 5% | 0.016286 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data5pct_multidata` |
| 25% | **0.015866** | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data25pct_multidata` |
| 50% | 0.016193 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data50pct_multidata` |
| 100% (main run) | 0.01638 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_multidata` (Section 1) |

pos_err/ang_err not scored for these (Section 1's scoring script isn't
extended to these checkpoints yet). 25% is already the best val_loss in
the column, beating the 100%-data run — same early-saturation pattern as
9b.

### 9d. Downstream full-body pose scaling (Nymeria probe mpjpe vs. % of pretraining data) — ✅ complete

All 5 fractions hit the 12h walltime before early-stopping fired on its
own (same `early_stopping.enabled: false` gap as other runaway jobs in
this benchmark); `best_mpjpe.pt` pulled for each, nothing lost.

| % of pretraining data | best mpjpe (mm) | Job |
|---|---|---|
| 0.1% | 101.18 | 7491653 |
| 1% | 98.20 | 7491654 |
| 5% | 97.25 | 7491746 |
| 25% | **96.75** | 7491747 |
| 50% | 97.00 | 7491750 |
| 100% (main run) | 97.63 | `vit1d_full_body_pose_probe_nymeria_mae_multidata` (Section 2) |

25% and 50% both beat the 100%-data probe's own mpjpe outright — same
early-saturation pattern as 9b/9c. Slightly non-monotonic at the top (50%
a hair worse than 25%), plausibly noise this close to saturation.

### 9e. Downstream odometry scaling (Nymeria probe vel_mse/ATE/drift vs. % of pretraining data)

| % of pretraining data | vel_mse | ATE (m) | drift | Status | Job |
|---|---|---|---|---|---|
| 0.1% | — | — | — | ❌ failed silently (empty log), not retried | 7491659 |
| 1% | 0.02533 | 0.0729 | 50.49% | ✅ | 7491661 |
| 5% | 0.02375 | 0.0708 | 49.51% | ✅ | 7491665 |
| 25% | 0.02376 | 0.0707 | 48.54% | ✅ | 7491666 |
| 50% | 0.02375 | 0.0717 | 49.94% | ✅ | 7491668 |
| 100% (main run) | 0.0245 | 0.0726 | 49.42% | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata` (Section 4) |

4 of 5 fractions already match the 100%-data probe's own ATE almost
exactly — odometry saturates its pretraining-data signal by 1% already,
with no further gain out to 50%. Only 0.1% remains a genuine gap (job
failed silently). See EXPERIMENTS.md §9e for full discussion.

## 10. Zero-shot & finetuned HAR classification via language alignment (WISDM, UTD-MHAD, PAMAP2)

Different question from §3: can the CLIP-aligned encoder
(`vit1d_clip_humanml3d_finetune_mae_multidata`, the 33.00%/34.25% HumanML3D
checkpoint) classify real HAR activities *zero-shot* via text-embedding
cosine similarity, and how much does finetuning on the target dataset close
the gap — benchmarked directly against UniMTS (external zero-shot HAR
foundation model) on the *same* data/split (new
`baselines/language_alignment/UniMTS/prepare_har_data.py` exports this
repo's own windows into UniMTS's expected format).

| Dataset | Model | Mode | Result | Job |
|---|---|---|---|---|
| UTD-MHAD (21cls) | Ours (MAE) | Zero-shot | Top-1 6.21% / Top-3 16.67% / Top-5 27.01% | 7703806 |
| UTD-MHAD | Ours (MAE) | Finetune | acc 5.88% / F1 1.01% | 7783289 |
| UTD-MHAD | UniMTS | Zero-shot | acc 11.31% / F1 7.21% | 7702128 |
| UTD-MHAD | UniMTS | Finetune | acc **79.19%** / F1 **77.60%** | 7702129 |
| WISDM (18cls) | Ours (MAE) | Zero-shot | Top-1 17.74% / Top-3 33.86% / Top-5 46.73% | 7703974 |
| WISDM | Ours (MAE) | Finetune | acc 5.76% / F1 0.67% | 7783289 |
| WISDM | UniMTS | Zero-shot | acc 28.73% / F1 24.83% | 7703982 |
| WISDM | UniMTS | Finetune | acc **67.92%** / F1 **68.36%** | 7704431 |
| PAMAP2 (18cls) | Ours (MAE) | Zero-shot | Top-1 11.52% / Top-3 40.78% / Top-5 60.42% | 7703975 |
| PAMAP2 | Ours (MAE) | Finetune | acc 0.13% / F1 0.14% | 7783289 |
| PAMAP2 | UniMTS | Zero-shot | acc 16.14% / F1 12.71% | 7703984 |
| PAMAP2 | UniMTS | Finetune | acc **57.06%** / F1 **44.78%** | 7703987 |

**Zero-shot: UniMTS wins clearly on all 3 datasets** (~2x Ours(MAE)'s top-1
across the board) -- expected, UniMTS was purpose-built for zero-shot
text-prompted HAR, while Ours(MAE)'s CLIP alignment only ever saw
HumanML3D's synthetic captions. **Finetune: UniMTS wins all 3, decisively.**
The originally-reported Ours(MAE) finetune numbers (88-94% "acc_imu2text/
acc_text2imu") were a metric artifact: that's an in-batch retrieval
accuracy computed on a *non-shuffled* val loader, and since WISDM/UTD-MHAD/
PAMAP2 sessions are each one continuous single-activity recording, a
64-window non-shuffled batch is almost entirely one class -- "top-1 lands
on any same-activity item in this near-single-class batch" is nearly
guaranteed regardless of real discriminative learning. Scored the *correct*
way (accuracy/macro-F1 against the full fixed class set, same metric as
UniMTS, on the same held-out val split), Ours(MAE) finetune collapses to
near-chance or worse (5.88%/5.76%/0.13% vs. UniMTS's 79.19%/67.92%/57.06%) --
e.g. on WISDM it predicts "eating soup" for virtually every window
regardless of true activity. **Correction (2026-07-24): these 3 jobs also
had a compounding bug** -- `imu_text_clip_realdata` never checked
`alt_attention` when building its encoder, so the warm-start from the
HumanML3D-CLIP checkpoint silently failed (near-random init, not fixed
until later this session). The near-chance result had two causes, not one;
doesn't change the practical conclusion (the matched-recipe finetune below
independently confirms a *properly* warm-started encoder finetunes well).
Conclusion: the CLIP-alignment recipe
(in-batch multi-positive contrastive loss) is a poor fit for classification
finetuning on few-class, session-blocked HAR data -- a standard supervised
classification finetune (§3) remains the right recipe there. Building this
pipeline also surfaced and fixed 5 real bugs (stale `base.yaml` paths,
`run_zeroshot_har.py` using the wrong encoder class for alt-attention
checkpoints, missing `HF_HOME`, PAMAP2's `max_gaps_per_session` gap
silently zeroing its val split, UniMTS OOM on WISDM at 32G). See
EXPERIMENTS.md §10 for full methodology, the metric-collapse root-cause
writeup, and an unrelated note on a batch of `launch_*.sh` scripts deleted
externally mid-session.

**Follow-up (2026-07-24): matched-recipe comparison.** UniMTS's own finetune
doesn't actually use its text branch -- it trains a standard linear
classifier + cross-entropy, i.e. Section 3's own recipe. Built a matched
Ours(MAE) version (`vit1d_har_finetune_{wisdm,utd_mhad,pamap2}_humanml3dclip_multidata_flu.yaml`
-- same `utd_mhad_finetune` objective as Section 3, warm-started from the
HumanML3D-CLIP checkpoint instead of plain MAE) and reran Ours(MAE)'s
zero-shot on the matched `--val-only` split too.

| Dataset | Model | Mode | acc / F1 | Job |
|---|---|---|---|---|
| UTD-MHAD | Ours (MAE+HumanML3D-CLIP init) | Zero-shot | 5.43% / 2.55% | 7788952 |
| UTD-MHAD | Ours (MAE+HumanML3D-CLIP init) | Finetune | 74.40% / 74.47% | 7788956 |
| UTD-MHAD | UniMTS | Zero-shot | 11.31% / 7.21% | 7702128 |
| UTD-MHAD | UniMTS | Finetune | **79.19%** / **77.60%** | 7702129 |
| WISDM | Ours (MAE+HumanML3D-CLIP init) | Zero-shot | 17.41% / 14.76% | 7788952 |
| WISDM | Ours (MAE+HumanML3D-CLIP init) | Finetune | **74.89%** / **74.26%** | 7788957 |
| WISDM | UniMTS | Zero-shot | 28.73% / 24.83% | 7703982 |
| WISDM | UniMTS | Finetune | 67.92% / 68.36% | 7704431 |
| PAMAP2 | Ours (MAE+HumanML3D-CLIP init) | Zero-shot | 14.20% / 11.74% | 7788952 |
| PAMAP2 | Ours (MAE+HumanML3D-CLIP init) | Finetune | 55.73% / 44.43% | 7788960 |
| PAMAP2 | UniMTS | Zero-shot | 16.14% / 12.71% | 7703984 |
| PAMAP2 | UniMTS | Finetune | 57.06% / 44.78% | 7703987 |

Zero-shot: UniMTS still wins clearly on all 3 (unchanged). **Finetune with
a fair recipe flips the story again**: vs. Section 3's own plain-MAE
finetune baseline (WISDM 71.27%/70.4%, UTD-MHAD 73.91%/72.7%, PAMAP2
51.77%/41.2% acc/F1), HumanML3D-CLIP pretraining as an *init* wins on all
3 datasets (+3.6/+0.5/+4.0pp acc) -- language-alignment pretraining does
help downstream HAR classification, it was specifically *continuing* the
CLIP objective during finetuning that hurt, not the pretraining itself.
Against UniMTS: UniMTS still wins UTD-MHAD (79.19% vs. 74.40%, smallest
dataset), **Ours(MAE+CLIP init) wins WISDM** (74.89% vs. 67.92%), and
PAMAP2 is essentially tied (55.73% vs. 57.06%). A genuinely competitive,
mixed result once the recipe is fair. See EXPERIMENTS.md §10 for the full
writeup.

### 10 (continued). Closing the zero-shot gap: rotation, midtraining, IMU-synthesis fidelity (2026-07-25)

Rotation augmentation (SO(3), matching UniMTS's own `augment_data()`) tied
UniMTS on UTD-MHAD zero-shot (11.31%/8.50% vs 11.31%/7.21%). Six rounds of
"midtraining" (continuing CLIP training on real HAR datasets) found and
fixed a real bug (`clip_head` never warm-started, only the encoder was --
`run.py` ~line 2921) and landed on v5 as the best midtrain checkpoint
(PAMAP2 28.01%/21.94% zero-shot, beats UniMTS by a wide margin) but never
fully fixed a UTD-MHAD zero-shot collapse. A z-score-normalization mismatch
hypothesis was tested and **disproven** (`--zscore-input` flag test gave
byte-identical results; real datasets were already normalized correctly by
`base.yaml`'s own default). Adding gravity to HumanML3D's synthetic
acceleration (missing entirely from the original kinematic-derivative
synthesis) was **negative in isolation** (worse than baseline on all 3
datasets, job 7946093/7946317).

**Biggest lever of the whole investigation**: `_resample_joints`
(`imu_lm/data/humanml3d_dataset.py`) upsamples HumanML3D's native 20Hz
joints to 50Hz via **linear** interpolation before double-differentiating
into acceleration -- confirmed this creates a 10-20x "roughness" artifact
(spurious spikes at each 20Hz breakpoint, a numerical artifact of linear
interpolation, not real motion) vs. real accelerometer data. Cubic-spline
resampling (opt-in `resample_method: cubic`, default unchanged) drops this
to ~0.2-0.7, matching real sensor characteristics for the first time.

| Dataset | Smoothresample alone | Smoothresample + gravity | UniMTS |
|---|---|---|---|
| UTD-MHAD | **16.74% / 11.66%** | 12.67% / 8.91% | 11.31% / 7.21% |
| WISDM | 19.17% / 17.51% | 25.52% / 23.50% | **28.73% / 24.83%** |
| PAMAP2 | 16.87% / 14.53% | 16.41% / 15.81% | 16.14% / 12.71% |

Smoothresample alone beats UniMTS outright on UTD-MHAD and PAMAP2 -- first
clean Stage-2-only checkpoint to do so. Combining with gravity trades UTD-MHAD
for a big WISDM gain; **averaged across all 3 datasets this essentially
matches UniMTS on accuracy (18.20% vs 18.73%) and exceeds it on macro-F1
(16.07% vs 14.92%)** -- from ~2x behind at session start. See EXPERIMENTS.md
§10 (continued) for full job IDs and the complete midtrain v1-v6 saga.

**LR tuning on smoothresample+gravity -- final best result.** The `lr:1e-4`
used throughout every Stage-2 config this session was never itself tuned.
Sweep `{3e-5, 5e-5, 1e-4, 2e-4}` (jobs 7948162-64, zero-shot 7948504-06):

| LR | UTD-MHAD | WISDM | PAMAP2 | Average |
|---|---|---|---|---|
| 3e-5 | 15.38/13.58 | 24.36/21.81 | 14.20/12.99 | 17.98/16.13 |
| **5e-5** | **17.65/12.48** | **25.69/24.32** | **19.01/17.25** | **20.78/18.02** |
| 1e-4 | 12.67/8.91 | 25.52/23.50 | 16.41/15.81 | 18.20/16.07 |
| 2e-4 | 12.67/8.73 | 26.58/24.89 | 13.81/12.72 | 17.69/15.45 |
| UniMTS | 11.31/7.21 | 28.73/24.83 | 16.14/12.71 | 18.73/14.92 |

`lr=5e-5` wins clearly: beats UniMTS on UTD-MHAD and PAMAP2, nearly ties
WISDM F1. **First single "Ours" checkpoint to beat UniMTS on BOTH average
acc (20.78% vs 18.73%) and macro-F1 (18.02% vs 14.92%)** -- best zero-shot
result of the session. Config:
`vit1d_clip_humanml3d_flu_finetune_mae_multidata_anyrot_smoothresample_gravity_lr5e-5.yaml`.
See EXPERIMENTS.md §10 (continued) for full detail.

**Extending to WHARF/HHAR/MHEALTH: earlier "beats UniMTS on average" doesn't
generalize.** WHARF has no gyro at all (zero-filled to fit our 6ch model --
a handicap UniMTS's accel-only checkpoint doesn't share). Jobs: Ours 7948675,
UniMTS 7948639/7948640/7948641.

| Dataset | Ours (lr5e-5) | UniMTS |
|---|---|---|
| WHARF | 6.97% / 3.48% | **19.09% / 12.67%** |
| HHAR | 32.35% / 25.82% | **50.02% / 44.41%** |
| MHEALTH | 22.35% / 18.74% | **59.31% / 51.06%** |

UniMTS wins clearly on all 3, likely partly because HHAR/MHEALTH are plausibly
part of UniMTS's own "18 benchmark datasets" zero-shot suite. **6-dataset
average: Ours 20.67%/17.02% vs UniMTS 30.77%/25.48%** -- the earlier win was
specific to the original 3-dataset scope, not a general result. See
EXPERIMENTS.md §10 (continued) for full detail and honest framing.

**Matching train/eval window at 10.24s -- small net regression, not a win.**
Found and fixed two bugs along the way: (1) a documented HumanML3D
all-clips-dropped hang risk at min==max==10.24s (fixed via `pad_mode: wrap`),
(2) `compute_T_and_hop` reads a different config key than Stage-2 training
sets, silently defaulting T_model to 2.56s instead of the true 10.24s
(patched the saved checkpoint directly). Jobs: train 7948887, eval 7949096.

| Dataset | Fixed-10.24s | Current best (2.56s eval) | UniMTS |
|---|---|---|---|
| UTD-MHAD (full-session, not like-for-like) | 14.88/10.32 | **17.65/12.48** | 11.31/7.21 |
| WISDM | **26.31/22.37** | 25.69/24.32 | 28.73/24.83 |
| PAMAP2 | **19.51/15.11** | 19.01/17.25 | 16.14/12.71 |
| Average | 20.23/15.93 | **20.78/18.02** | 18.73/14.92 |

Still beats UniMTS on average, but slightly worse than the existing best --
matching window size didn't help. See EXPERIMENTS.md §10 (continued) for
full detail on the two bugs found.

**Matching UniMTS's exact clip crop/pad strategy -- best "Ours" result of
the session, but NOT equivalence with UniMTS.** Verified UniMTS's own
`CLIPDataset.__getitem__` (`baselines/language_alignment/UniMTS/data.py`):
wrap-pad short clips (already matched here) + **deterministic first-N-
samples truncation** for long clips (this repo previously used a random
crop). Added `humanml3d.crop_mode: 'first'` to match exactly. Jobs: train
7949239, eval 7949377/7949378.

| Dataset | UniMTS-crop-matched | Current best (2.56s) | UniMTS |
|---|---|---|---|
| UTD-MHAD | 14.88/10.65 | **17.65/12.48** | 11.31/7.21 |
| WISDM | **29.00/25.91** | 25.69/24.32 | 28.73/24.83 |
| PAMAP2 | 17.49/14.19 | **19.01/17.25** | 16.14/12.71 |
| WHARF | **16.18/6.38** | 6.97/3.48 | 19.09/12.67 |
| HHAR | **43.76/35.35** | 32.35/25.82 | 50.02/44.41 |
| MHEALTH | **28.63/27.15** | 22.35/18.74 | 59.31/51.06 |
| **Average (6)** | **24.99/19.94** | 20.67/17.02 | 30.77/25.48 |

Best "Ours" 6-dataset average of the whole investigation, closing real
ground on WISDM/WHARF/HHAR/MHEALTH -- but still trailing UniMTS overall
(MHEALTH gap especially large). **Not the "same pipeline except window
size"**: architecture (graph net vs ViT1D), IMU synthesis fidelity (IMUSim
vs finite-difference), captions (GPT-paraphrased vs original only), joint-
masking augmentation (22-joint vs single-joint), sampling rate (20Hz vs
50Hz), and the extra MAE pretraining stage here all remain real
differences. See EXPERIMENTS.md §10 (continued) for full detail.

**Resample round-trip to 20Hz for zero-shot eval (`--roundtrip-20hz`).**
Hypothesis: HumanML3D-derived training data is trained-on at 50Hz but its
real content is capped at ~20Hz (AMASS mocap native rate, upsampled via
cubic spline) -- test datasets have genuine higher-frequency content the
model never saw in training. Fix: downsample target windows to 20Hz then
upsample back to 50Hz before encoding (verified on synthetic tones: <10Hz
content preserved, >10Hz removed to ~1e-14). Same `unimtscrop` checkpoint,
jobs 7953922/7953923:

| Dataset | Baseline | +roundtrip-20hz | UniMTS |
|---|---|---|---|
| UTD-MHAD | 14.88/10.65 | 15.48/11.22 | 11.31/7.21 |
| WISDM | **29.00/25.91** | 27.08/24.55 | 28.73/24.83 |
| PAMAP2 | 17.49/14.19 | 18.84/16.18 | 16.14/12.71 |
| WHARF | **16.18/6.38** | 14.19/6.07 | 19.09/12.67 |
| HHAR | 43.76/35.35 | **45.76/40.12** | 50.02/44.41 |
| MHEALTH | 28.63/27.15 | **29.06/29.21** | 59.31/51.06 |
| **Average (6)** | 24.99/19.94 | **25.07/21.23** | 30.77/25.48 |

Acc flat, F1 +1.29pp on average; biggest gains on HHAR/MHEALTH (the two
datasets UniMTS previously won most decisively), small regressions on
WISDM/WHARF. Doesn't close the gap but is the first hypothesis this session
to move HHAR/MHEALTH in the right direction. New recommended default for
zero-shot eval going forward.

**Trainable CLIP text encoder (`freeze_text_encoder: false`).** UniMTS trains
its full text tower jointly (no freezing); "Ours" always froze it. Tested
unfreezing on top of unimtscrop (job 7954257). Required fixing a real gap:
run.py wasn't checkpointing the fine-tuned text tower at all (silently
discarded), and run_zeroshot_har.py wasn't loading it back -- fixed both
sides, verified the checkpoint's `text_enc` key (196 tensors) round-trips.
Also fixed an unrelated but real bug found along the way: augmentation
(random_rotation etc.) was being applied to VAL windows too, not just train,
in HumanML3DIMUVarLenDatasetFLU -- corrupting the val_loss early-stopping
relies on. Result:

| Dataset | unimtscrop | +trainable text |
|---|---|---|
| UTD-MHAD | 14.88/10.65 | 13.69/8.46 |
| WISDM | 29.00/25.91 | 26.69/23.34 |
| PAMAP2 | 17.49/14.19 | 16.06/12.92 |
| WHARF | 16.18/6.38 | **1.99/0.47** |
| HHAR | 43.76/35.35 | 42.76/36.09 |
| MHEALTH | 28.63/27.15 | 25.64/20.74 |
| **Average (6)** | **24.99/19.94** | 21.14/17.00 |

Clear negative result (-3.85pp acc avg) despite a slightly better internal
HumanML3D val_loss (2.5425 vs 2.564) -- text tower overfits to HumanML3D's
caption phrasing, hurting transfer. WHARF nearly collapses. UniMTS's
zero-shot advantage isn't explained by trainable-text alone.

**Gaussian-noise augmentation (`gaussian_noise.enabled: true, sigma: 0.6`).**
HumanML3D synthetic IMU is noise-free; real sensors and UniMTS's own
IMUSim synthesis aren't. Tested on top of unimtscrop (job 7954301):

| Dataset | unimtscrop | +gaussian noise |
|---|---|---|
| UTD-MHAD | 14.88/10.65 | **20.83/13.23** |
| WISDM | **29.00/25.91** | 19.70/17.78 |
| PAMAP2 | **17.49/14.19** | 16.32/14.00 |
| WHARF | **16.18/6.38** | 10.67/6.17 |
| HHAR | **43.76/35.35** | 31.11/28.56 |
| MHEALTH | **28.63/27.15** | 17.09/13.61 |
| **Average (6)** | **24.99/19.94** | 19.29/15.56 |

Net negative (-5.70pp acc avg) despite a clear UTD-MHAD win (+5.95pp acc) --
plausibly larger-amplitude/more-distinct gestures benefit from noise
regularization where finer postural discrimination (HHAR/MHEALTH) doesn't.
Joins z-score removal, bare-word captions, and trainable-text as UniMTS
recipe elements that do NOT transfer cleanly in isolation.

**Combined: trainable text + Gaussian noise together (job 7954562).**
Both tested alone came back negative; tested together to see if they
compound or offset:

| Dataset | unimtscrop | +text only | +noise only | +both |
|---|---|---|---|---|
| UTD-MHAD | 14.88/10.65 | 13.69/8.46 | **20.83/13.23** | 20.83/14.57 |
| WISDM | **29.00/25.91** | 26.69/23.34 | 19.70/17.78 | 21.33/19.01 |
| PAMAP2 | 17.49/14.19 | 16.06/12.92 | 16.32/14.00 | **18.25/14.26** |
| WHARF | **16.18/6.38** | 1.99/0.47 | 10.67/6.17 | 8.32/5.44 |
| HHAR | **43.76/35.35** | 42.76/36.09 | 31.11/28.56 | 33.61/24.42 |
| MHEALTH | 28.63/27.15 | 25.64/20.74 | 17.09/13.61 | **4.27/3.73** |
| **Average (6)** | **24.99/19.94** | 21.14/17.00 | 19.29/15.56 | 17.77/13.57 |

They compound negatively rather than cancel out -- combined is worse than
either alone, MHEALTH nearly collapses entirely (below random-guess floor).
UTD-MHAD's noise-driven gain is the one consistent positive across all
three variants. **Conclusion: none of z-score removal, bare-word captions,
trainable text, or Gaussian noise -- tested alone or combined -- improves
"Ours" zero-shot transfer.** `unimtscrop` (24.99/19.94, this session's best)
plus `--roundtrip-20hz` at eval time (25.07/21.23) remain the best results
found this investigation.

**UniMTS architecture, from scratch, on our own all-joint HumanML3D data.**
Isolates architecture+all-joint-coverage from IMUSim-specific synthesis.
Correction: UniMTS's own README confirms their pretraining ALSO derives
from HumanML3D (not a separate corpus) via IMUSim synthesis -- the real
differences are synthesis fidelity (IMUSim's noise/dynamics vs. our clean
finite-difference) and clip yield (we only got 8,246/23,384 train clips,
35%, due to this cluster's incomplete local AMASS archive; UniMTS's own
pipeline likely processes closer to the full set). Built
`prepare_humanml3d_alljoint.py` (all-22-joint FK + per-joint local-frame
IMU + UniMTS's own chain-compressed joint numbering, decoded from their
`joint_assignment.png` and cross-verified against the existing
`RIGHT_WRIST_JOINT=21` convention). Trained via UniMTS's own unmodified
`pretrain.py`, job 7953827, evaluated at epoch 90 (its last-ever checkpoint
per their own save-interval logic), jobs 7954835-7954840:

| Dataset | UniMTS-arch/Ours-data (scratch) | Ours (unimtscrop) | Official UniMTS |
|---|---|---|---|
| UTD-MHAD | 7.24/2.21 | 14.88/10.65 | 11.31/7.21 |
| WISDM | **5.91/1.16** | **29.00/25.91** | 28.73/24.83 |
| PAMAP2 | 12.17/4.14 | **17.49/14.19** | 16.14/12.71 |
| WHARF | **2.64/0.75** | **16.18/6.38** | 19.09/12.67 |
| HHAR | 17.40/4.94 | 43.76/35.35 | **50.02/44.41** |
| MHEALTH | 16.18/7.02 | 28.63/27.15 | **59.31/51.06** |
| **Average (6)** | **10.26/3.37** | 24.99/19.94 | **30.77/25.48** |

Far worse than BOTH other rows -- worse than random chance on WHARF.
Severe mode collapse on HHAR/PAMAP2 (e.g. PAMAP2: 2 classes get all
predictions, 14 others get 0%) despite training loss converging smoothly
to 0.58. **Conclusion: UniMTS's advantage is not explained by architecture
or all-joint coverage alone** -- it depends on their actual pretraining
data (IMUSim synthesis fidelity and/or genuinely more clips than our
AMASS-archive-limited export could supply). This closes the
architecture-vs-data question this session's investigation set out to
answer; the remaining open lever (corpus scale) would need either a more
complete local AMASS mirror or access to IMUSim itself to test further.

## 12. UniMTS_data arc: user-provided synthesis, four bug fixes, and the final aligned comparison (2026-07-28)

Full narrative in EXPERIMENTS.md Section 12 (four bugs fixed: 34%-coverage
clip filter; text_cache_path silently no-op'ing trainable text -- which also
invalidates the 2026-07-26 "trainable text is net-negative" rows; empty-
caption fallback; rotation-after-zscore). All rows below on the SAME
seed-42 held-out subjects. "Ours proj" = ViT1D CLIP-align trained on
/projects/b53017/vbc1591/HumanML3D/UniMTS_data/data (right-wrist channel of
the pre-synthesized [T,22,6] sim files + aug_texts captions), original MAE
Stage-1 warm start, trainable text tower WITH CLIP text_projection, 10.24s
window, SO(3) rotation aug in raw space, cosine to 8k steps, lr 5e-5 (job
8147232). "noproj" = same minus text_projection + minus real cosine horizon
(job 8135882).

| Dataset | Mode | Ours proj | Ours noproj | UniMTS official | UniMTS from-scratch |
|---|---|---|---|---|---|
| UTD-MHAD | Zero-shot | **16.7 / 7.4** | 13.1 / 6.4 | 11.3 / 7.2 | 19.0 / 11.8 |
| WISDM | Zero-shot | 28.6 / **26.5** | 23.9 / 21.3 | 28.7 / 24.8 | 25.0 / 21.1 |
| PAMAP2 | Zero-shot | 11.1 / 14.2 | 15.8 / 14.8 | 16.1 / 12.7 | 21.4 / 18.8 |
| HHAR | Zero-shot | 39.4 / 30.0 | 33.6 / 22.1 | **50.0 / 44.4** | 35.0 / 26.0 |
| MHEALTH | Zero-shot | 28.6 / 19.1 | 35.5 / 22.9 | **59.3 / 51.1** | 47.7 / 37.0 |
| **Avg (5)** | Zero-shot | 24.9 / 19.4 | 24.4 / 17.5 | **33.1 / 28.0** | 29.6 / 22.9 |
| UTD-MHAD | Finetune | 77.4 / 74.6 | 73.3 / 70.8 | **78.3 / 76.6** | 76.5 / 75.4 |
| WISDM | Finetune | **70.9 / 70.1** | 70.7 / 69.7 | 67.9 / 68.0 | 68.5 / 68.9 |
| PAMAP2 | Finetune | 53.7 / 42.2 | **58.6 / 45.1** | 54.3 / 42.5 | 55.6 / 43.5 |
| HHAR | Finetune | 66.6 / 64.9 | 70.8 / 70.5 | **79.0 / 78.2** | 78.3 / 77.0 |
| MHEALTH | Finetune | 74.0 / 72.3 | 79.1 / 76.3 | 87.7 / 85.4 | **88.5 / 86.4** |

Key reads: (1) fine-tune is near-parity on UTD-MHAD/WISDM/PAMAP2 (Ours wins
WISDM outright, best PAMAP2 is Ours-noproj); (2) both zero-shot and finetune
gaps concentrate in HHAR/MHEALTH, and per-class analysis attributes them to
static-posture/leg-cyclic classes (sitting, stairs, cycling) whose gravity-
orientation cue our per-window mean-subtracting z-score erases -- while Ours
wins wrist-articulated classes (drinking/clapping/typing) by wide margins;
(3) UTD-MHAD zero-shot is near-floor for BOTH models (15-16/21 zero-accuracy
classes each) -- protocol-intrinsic, not a model failure.

Addendum (2026-07-28, late): "Ours scratch" ablation (same proj recipe, NO
MAE warm start, job 8155972) -- zero-shot avg 22.2/19.1 (MHEALTH 42.3/35.5,
the best "Ours" MHEALTH number of the session); finetune utd_mhad 65.2/63.1,
wisdm 67.4/66.9, pamap2 58.9/53.5 (best PAMAP2 macro-F1 of all five
methods), hhar 69.6/68.9, mhealth 73.3/71.6. Full 5-column tables +
MAE-ablation read in EXPERIMENTS.md Section 12e.

## 13. Alignment-retrieval comparisons on identical data (2026-07-28/29)

Full writeup in EXPERIMENTS.md Section 13. Batch-64 top-1 retrieval on
held-out val splits; best-val-loss checkpoint rows.

**13a. Nymeria (synthesized wrist IMU + atomic_action captions, same 867/101 session split):**

| Model | imu→text | text→imu | Job |
|---|---|---|---|
| Ours, MAE finetune | **9.39%** | **10.64%** | 7692004 |
| Ours, MAE probe | 7.58% | 7.94% | 7692003 |
| UniMTS from scratch (synthetic IMU) | 1.80% | 1.80% (random floor) | 8158173 |
| (ref) official UniMTS finetuned, real IMU, motion_narration | 6.12% | 6.36% | 7581974 |

**13b. HumanML3D UniMTS_data (same 19,751 train / 1,197 val clips for every row):**

| Model | loss | input | imu→text | text→imu | Job |
|---|---|---|---|---|---|
| Ours ViT1D, MAE-pretrained (proj) | symmetric | wrist | 44.0% | 43.2% | 8147232 |
| Ours ViT1D, MAE-pretrained | i2t-only | wrist | **44.9%** | **43.9%** | 8173825 |
| Ours ViT1D, from scratch | symmetric | wrist | 27.3% | 28.2% | 8155972 |
| Ours ViT1D, from scratch | i2t-only | wrist | 26.2% | 25.8% | 8173826 |
| UniMTS, from scratch | i2t-only (native) | 22 joints | **47.1%** | 2.5% | 8163821 |

Key reads: loss direction explains none of the architecture gap (Ours i2t-only
≈ Ours symmetric, both directions intact); UniMTS's text→imu floor is specific
to its model/training combo; all-22-joint from-scratch UniMTS (47.1%) edges
wrist-only MAE-pretrained Ours (44.9%) on the trained direction -- with a ~20x
sensor-channel advantage; pretraining is decisive on small/noisy Nymeria,
worth +17-18pp on wrist-only HumanML3D, and catchable-up on big clean data
with rich input.

## 13. Alignment-retrieval comparisons on identical data (2026-07-28/29)

Full writeup: EXPERIMENTS.md Section 13. Batch-64 top-1 retrieval on
held-out val splits; best-val-loss-checkpoint rows.

**13a. Nymeria (synthesized wrist IMU + atomic_action captions), same
train/val sessions:**

| Model | imu→text | text→imu |
|---|---|---|
| Ours, MAE probe (frozen) | 7.58% | 7.94% |
| Ours, MAE finetune | **9.39%** | **10.64%** |
| UniMTS from scratch (8158173) | 1.80% | 1.80% (never left random floor) |
| (ref) official UniMTS finetuned, real IMU, motion_narration | 6.12% | 6.36% |

**13b. HumanML3D UniMTS_data (same 19,751 train / 1,197 val clips as Ours):**

| Model | Train input | Eval input | Loss | i2t | t2i |
|---|---|---|---|---|---|
| Ours MAE-pretrained (proj) | wrist | wrist | symmetric | 44.0% | 43.2% |
| Ours MAE-pretrained | wrist | wrist | i2t-only | **44.9%** | **43.9%** |
| Ours from scratch | wrist | wrist | symmetric | 27.3% | 28.2% |
| Ours from scratch | wrist | wrist | i2t-only | 26.3% | 25.8% |
| UniMTS scratch, full-grid trained (8163821) | 22 joints | 22 joints | i2t (native) | 49.8% | 2.8% |
| UniMTS scratch, full-grid trained | 22 joints | wrist only | i2t (native) | 3.1% | 1.7% |
| UniMTS scratch, native joint-mask aug, exact original duration/selection (8192640) | 1-5 rand joints | wrist only | i2t (native) | 29.9% | 2.7% |

Reads: input-matched at the wrist, MAE-pretrained Ours beats the
native-recipe from-scratch UniMTS by ~+18pp i2t (and ~+41pp t2i -- UniMTS's
one-way loss + architecture never learns text→imu); from-scratch-vs-
from-scratch at the wrist is a tie on i2t, so the Ours edge IS the MAE
pretraining; the full 22-joint grid is worth ~+20pp but doesn't exist in a
wrist-only deployment; on Nymeria (small/noisy) from-scratch UniMTS cannot
learn the task at all.

## 13c. Section-6 ladder re-evaluated under the new setting (2026-07-29)

Full writeup EXPERIMENTS.md 13c. New setting = unimts_sim data + trainable
projected text + fixed 10.24s window + raw-space SO(3) aug, lr 5e-5/8k;
held-out val, best-val-loss rows.

| Row | OLD (derived data, frozen text) | NEW setting |
|---|---|---|
| MAE finetune (headline) | 33.00 / 34.25 | **44.94 / 43.91** |
| MAE finetune + text adapter | 30.06 / 30.47 | 44.45 / 43.69 |
| MAE finetune, window ablation | fixed: 24.73 / 24.92 | variable: 37.39 / 37.02 |
| MAE probe | 19.69 / 21.11 | 31.34 / 29.97 |
| MAE conventional-attn probe | 20.02 / 22.20 | 33.73 / 32.17 |
| MAE acc-only probe | 12.77 / 13.22 | 25.84 / 24.34 |
| Scratch | 28.81 / 30.44 | 27.25 / 28.17 |
| SimMTM probe | 16.06 / 17.95 | 20.66 / 19.03 |
| RelCon probe | 11.59 / 13.59 | 18.22 / 15.86 |
| FOCAL probe | 11.72 / 12.00 | 10.67 / 10.72 |
| (Nymeria) Ours from scratch | -- | 8.55 / 8.81 (vs MAE finetune 9.39/10.64) |

## 13d. Complete bidirectional retrieval metrics — R@1/5/10 + MRR + MedR, fixed vs. full gallery (2026-08-15, ✅ complete)

Replaces the single in-batch top-1 number (Sections 6/12/13) as the **main
language-alignment evaluation**: Recall@{1,5,10} and mean reciprocal rank
in BOTH directions, plus median rank, on the same held-out items for every
row, reported for a **fixed 100-item gallery** (AnyMo convention) and the
**full gallery** (TMR convention) — retrieval difficulty changes drastically
with gallery size, so both are needed. Every checkpoint is the same
best-val-loss checkpoint as in Sections 13a/13b/13c; nothing was retrained.
Protocol details, per-gallery tables incl. G=64 (the legacy batch size —
its R@1 reproduces the old acc_imu2text/acc_text2imu within ~1-2 pp on
every row, validating the pipeline), the secondary all-captions protocol,
and the reads are in EXPERIMENTS.md §13d. Evaluator:
`scripts/eval_retrieval_alignment.py`; results `analysis/retrieval_eval/*.json`;
tables `python scripts/summarize_retrieval_alignment.py analysis/retrieval_eval`.

Setting: HumanML3D = UniMTS_data val split (1,197 held-out clips, aug_texts
captions, one caption per clip → 900 unique caption strings, duplicates are
multi-positives), wrist-only input for every row except the 22-joint
reference. Nymeria = synthetic wrist IMU + atomic_action captions, held-out
val sessions (12,065 segments ≥0.32 s, 11,517 unique captions), same
867/101 session split as everything else. Chance R@1 = 1/G (1.0% at G=100;
0.11% HumanML3D full; 0.008% Nymeria full). All numbers in %.

**Fixed gallery, G = 100** (200 random 100-item subsets, ranks pooled; single-subset R@1 std ≈ 4 pp HumanML3D / 2.4 pp Nymeria, so the pooled means carry ≈ ±0.3 / ±0.2 pp):

| Model | HumanML3D IMU→Text | HumanML3D Text→IMU | Nymeria IMU→Text | Nymeria Text→IMU |
|---|---|---|---|---|
| | R@1 / R@5 / R@10 / MRR | R@1 / R@5 / R@10 / MRR | R@1 / R@5 / R@10 / MRR | R@1 / R@5 / R@10 / MRR |
| **Ours pretrained** (MAE init, finetune) | **39.4 / 73.9 / 84.3 / 54.9** | **37.5 / 72.0 / 82.4 / 53.0** | **7.5 / 25.5 / 37.5 / 17.6** | **8.6 / 26.5 / 38.8 / 18.7** |
| Ours pretrained (MAE init, frozen probe) | 26.0 / 57.7 / 71.3 / 40.8 | 24.3 / 55.8 / 70.6 / 39.2 | 6.0 / 20.2 / 30.8 / 14.6 | 6.4 / 20.9 / 31.3 / 15.0 |
| Ours scratch (random init) | 21.9 / 53.1 / 67.8 / 36.6 | 21.6 / 52.1 / 66.9 / 36.3 | 6.1 / 21.9 / 33.2 / 15.3 | 7.3 / 22.6 / 33.8 / 16.4 |
| UniMTS from scratch (wrist-only)† | 24.4 / 55.3 / 68.9 / 39.1 | 1.7 / 7.4 / 13.5 / 6.7 | 2.4 / 9.8 / 17.0 / 8.2 | 1.0 / 5.1 / 10.1 / 5.3 |
| UniMTS official checkpoint, zero-shot (wrist-only)‡ | 33.6 / 68.6 / 81.2 / 49.3 | 1.6 / 7.2 / 12.6 / 6.4 | 1.2 / 6.5 / 12.7 / 6.1 | 1.0 / 5.1 / 10.2 / 5.2 |
| UniMTS from scratch, 22-joint input (reference)† | 40.3 / 79.7 / 89.4 / 57.2 | 2.1 / 8.0 / 15.1 / 7.3 | — | — |
| Inertia-1 PatchTST-small (NHANES repro)†, finetune i2t | 25.8 / 60.1 / 74.1 / 41.4 | 24.5 / 57.0 / 71.3 / 39.7 | — | — |
| Inertia-1 PatchTST-small (NHANES repro)†, frozen probe | 18.4 / 48.5 / 64.7 / 32.9 | 17.2 / 47.1 / 63.1 / 31.6 | — | — |
| Inertia-1 ART-small (NHANES repro)†, finetune i2t | 28.3 / 60.6 / 74.7 / 43.5 | 26.1 / 57.8 / 73.3 / 41.1 | — | — |
| Inertia-1 ART-small (NHANES repro)†, frozen probe | 20.6 / 51.0 / 67.1 / 34.9 | 20.2 / 49.1 / 65.5 / 34.2 | — | — |

**Full gallery** (HumanML3D: 1,197 IMU windows ↔ 900 captions; Nymeria: 12,065 IMU windows ↔ 11,517 captions; MedR = median rank, chance ≈ 450 / 600 for HumanML3D and ≈ 5,750 / 6,000 for Nymeria):

| Model | HumanML3D IMU→Text | HumanML3D Text→IMU | Nymeria IMU→Text | Nymeria Text→IMU |
|---|---|---|---|---|
| | R@1 / R@5 / R@10 / MRR (MedR) | R@1 / R@5 / R@10 / MRR (MedR) | R@1 / R@5 / R@10 / MRR (MedR) | R@1 / R@5 / R@10 / MRR (MedR) |
| **Ours pretrained** (MAE init, finetune) | **11.1 / 35.3 / 47.8 / 22.7** (11) | **10.4 / 29.7 / 44.6 / 20.5** (13) | **0.10 / 0.52 / 1.11 / 0.63** (1,920) | **0.13 / 0.68 / 1.27 / 0.71** (1,987) |
| Ours pretrained (MAE init, frozen probe) | 6.6 / 20.5 / 31.0 / 14.5 (29) | 5.4 / 18.1 / 25.9 / 12.7 (36) | 0.08 / 0.44 / 0.80 / 0.49 (2,665) | 0.12 / 0.43 / 0.94 / 0.53 (2,762) |
| Ours scratch (random init) | 4.8 / 16.3 / 26.0 / 11.9 (38) | 3.2 / 14.1 / 25.3 / 10.1 (40) | 0.07 / 0.46 / 0.90 / 0.50 (2,376) | 0.10 / 0.54 / 0.99 / 0.57 (2,421) |
| UniMTS from scratch (wrist-only)† | 5.3 / 19.8 / 30.5 / 13.2 (31) | 0.2 / 0.8 / 1.8 / 1.1 (507) | 0.02 / 0.06 / 0.20 / 0.17 (4,203) | 0.01 / 0.04 / 0.09 / 0.09 (5,813) |
| UniMTS official checkpoint, zero-shot (wrist-only)‡ | 9.7 / 28.6 / 41.8 / 19.9 (16) | 0.1 / 0.7 / 1.3 / 0.9 (534) | 0.01 / 0.05 / 0.08 / 0.10 (4,931) | 0.01 / 0.04 / 0.10 / 0.09 (5,937) |
| UniMTS from scratch, 22-joint input (reference)† | 9.4 / 34.9 / 50.0 / 22.4 (11) | 0.6 / 1.2 / 1.8 / 1.4 (504) | — | — |
| Inertia-1 PatchTST-small (NHANES repro)†, finetune i2t | 5.9 / 20.7 / 31.6 / 14.2 | 5.8 / 17.6 / 28.1 / 13.0 | — | — |
| Inertia-1 PatchTST-small (NHANES repro)†, frozen probe | 4.3 / 14.2 / 22.2 / 10.4 | 3.7 / 11.3 / 20.2 / 8.9 | — | — |
| Inertia-1 ART-small (NHANES repro)†, finetune i2t | 7.2 / 23.6 / 34.5 / 16.1 | 6.4 / 19.1 / 30.7 / 14.0 | — | — |
| Inertia-1 ART-small (NHANES repro)†, frozen probe | 4.8 / 16.5 / 25.1 / 11.5 | 5.6 / 14.9 / 21.6 / 11.1 | — | — |

† HumanML3D UniMTS-from-scratch = the native joint-mask recipe, 100 ep,
`last.pth` (job 8192640) for the wrist-only row; the 22-joint reference is
the full-grid-trained model (job 8163821) evaluated with all 22 joints (a
~20× sensor-channel advantage no wrist deployment has); its Nymeria cells
are "—" because Nymeria only has a wrist IMU (no 22-joint synthetic grid
exists), so that row is a HumanML3D-only reference, not an unfinished
run. Nymeria
UniMTS-from-scratch = job 8158173. ‡ The official UniMTS checkpoint was
pretrained on HumanML3D itself (Section 6 caveat), so its HumanML3D row is
contaminated; its Nymeria row is clean. Ours rows: HumanML3D headline =
`vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t` (job
8173825; the symmetric-loss twin 8147232 is within noise: 39.0/38.9 at
G=100, 11.1/12.1 full), probe = `vit1d_clip_humanml3d_probe_mae_unimtssim`,
scratch = `..._fromscratch_unimtssim_texttrainable_lr5e-5` (8155972);
Nymeria = `vit1d_clip_nymeria_atomicaction_synthetic_{finetune_mae_multidata,
mae_probe,fromscratch_multidata}` (7692004 / 7692003 / 8282334).

Key reads: (1) **model ordering is stable across gallery sizes and metrics**
— Ours MAE finetune > Ours MAE probe ≳ Ours scratch on both datasets, in both
directions, at G=64/100/full; MAE pretraining is worth ≈ +17 pp R@1 (G=100)
/ ≈ +6 pp R@1 and ≈ 2× MRR (full) over the identically-trained scratch model
on HumanML3D, and on Nymeria moves the full-gallery median rank from 2,376
(scratch) to 1,920 (of 11,517). (2) **Gallery size dominates absolute
numbers**: the same headline model goes 46.7 → 39.4 → 11.1 % R@1 as the
gallery grows 64 → 100 → 900, and 10.6 → 7.5 → 0.10 % on Nymeria (64 → 100
→ 11.5k); at full gallery MRR/R@10/MedR are the informative columns (Nymeria
MedR ≈ 1,900-2,000 vs. chance ≈ 6,000: the true caption lands in roughly the
top sixth on median — real signal, but nowhere near usable at scale).
(3) **UniMTS is one-directional at every gallery size**: wrist-only i2t
sits between our probe and our scratch model (24.4 % at G=100; 5.3 % full),
but t2i never leaves chance (1.7 % at G=100 = 1.7× chance; MedR 507 of 1,197
full), and both directions are at chance on Nymeria — the batch-64 asymmetry
was not a metric artefact. (4) With a 22-joint input UniMTS reaches parity
with wrist-only Ours on i2t (40.3 vs 39.4 at G=100; 9.4 vs 11.1 full,
MRR 22.4 vs 22.7) but its t2i is still 2 %. (5) Loss direction (i2t-only vs
symmetric) is a wash for Ours in both directions (Δ ≤ 1.7 pp anywhere).

**13d-tuned (2026-08-15): recipe tuning of the HumanML3D "Ours pretrained
(MAE init, finetune)" row against the UniMTS 22-joint reference.** 47 tuning
runs (`configs/retrieval_tune/`, run names `rtune_*`, full log in
EXPERIMENTS.md §13e). Only two changes to the parent recipe survive:
`trainer.optim.lr 5e-5 → 1e-4` and a **fixed** (non-learnable) contrastive
temperature τ = 0.1 (UniMTS's own value) instead of CLIP's learnable τ
initialised at 0.07. Everything else (regularisation, text-tower LR /
depth / size, batch size, schedule length, augmentation) is neutral or
negative. Selection was done on the val split (the set every §13 row reports
on); the **never-before-used HumanML3D test split (test.txt ∩ sim ∩ aug_texts
= 3,701 clips)** is reported alongside as an untouched confirmation. Means ±
sd over 4 seeds for the two Ours recipes; the UniMTS rows are single
checkpoints (their own `best.pth`). Same first-caption protocol as above.

| HumanML3D, G=100 (200 subsets) | split | IMU→Text R@1 / R@5 / R@10 / MRR | Text→IMU R@1 / R@5 / R@10 / MRR |
|---|---|---|---|
| Ours pretrained, parent recipe (lr 5e-5, learnable τ) — 4 seeds | val | 38.9 / 72.6 / 83.3 / 54.1 ± 0.7 | 36.9 / 70.4 / 81.2 / 52.0 |
| **Ours pretrained, tuned (lr 1e-4, τ = 0.1 fixed) — 4 seeds** | val | **40.6 / 75.0 / 84.8 / 55.9 ± 0.5** | **38.9 / 73.4 / 83.4 / 54.4** |
| Ours pretrained, tuned — best seed (`rtune_lr1e-4_tau0.1fixed`) | val | 41.4 / 75.6 / 84.9 / 56.5 | 39.7 / 74.4 / 84.0 / 55.1 |
| **Ours pretrained, tuned + 16k steps + soup[6000,16000] — 3 seeds**§ | val | **41.8 / 75.6 / 84.8 / 56.8 ± 0.6** | **40.7 / 74.6 / 83.5 / 55.7** |
| UniMTS from scratch, wrist-only (input-matched)† | val | 24.4 / 55.3 / 68.9 / 39.1 | 1.7 / 7.4 / 13.5 / 6.7 |
| UniMTS from scratch, 22-joint input (reference)† | val | 40.3 / 79.7 / 89.4 / 57.2 | 2.1 / 8.0 / 15.1 / 7.3 |
| Ours pretrained, parent recipe — 4 seeds | test | 37.1 / 71.8 / 82.8 / 52.6 ± 0.4 | 35.6 / 69.6 / 80.8 / 50.9 |
| **Ours pretrained, tuned — 4 seeds** | test | **38.9 / 73.8 / 83.6 / 54.3 ± 0.4** | **37.9 / 72.7 / 82.6 / 53.4** |
| **Ours pretrained, tuned + 16k steps + soup[6000,16000] — 3 seeds**§ | test | **41.5 / 74.8 / 84.0 / 56.3 ± 0.2** | **40.3 / 74.0 / 83.2 / 55.3** |
| UniMTS from scratch, wrist-only (input-matched)† | test | 24.9 / 55.8 / 69.2 / 39.3 | 1.5 / 7.3 / 13.9 / 6.5 |
| UniMTS from scratch, 22-joint input (reference)† | test | 40.7 / 76.6 / 87.6 / 56.6 | 2.1 / 8.6 / 15.3 / 7.4 |

| HumanML3D, full gallery | split | IMU→Text R@1 / R@5 / R@10 / MRR | Text→IMU R@1 / R@5 / R@10 / MRR |
|---|---|---|---|
| Ours parent — 4 seeds | val (1,197 ↔ 900) | 11.9 / 34.5 / 46.6 / 23.0 | 10.7 / 29.5 / 42.4 / 20.6 |
| **Ours tuned — 4 seeds** | val | **12.2 / 35.6 / 49.2 / 23.8** | **10.3 / 31.7 / 45.8 / 21.3** |
| **Ours tuned + 16k + soup[6000,16000] — 3 seeds**§ | val | **13.3 / 37.9 / 50.0 / 25.1** | **11.4 / 34.4 / 47.3 / 22.7** |
| UniMTS 22-joint (reference) | val | 9.4 / 34.9 / 50.0 / 22.4 | 0.6 / 1.2 / 1.8 / 1.4 |
| Ours parent — 4 seeds | test (3,701 ↔ 2,730) | 5.0 / 16.9 / 26.6 / 12.0 | 4.0 / 14.8 / 23.3 / 10.4 |
| **Ours tuned — 4 seeds** | test | **5.1 / 18.3 / 28.2 / 12.6** | **4.4 / 15.7 / 25.3 / 11.2** |
| **Ours tuned + 16k + soup[6000,16000] — 3 seeds**§ | test | **6.1 / 20.4 / 31.2 / 14.1** | **5.5 / 18.3 / 28.1 / 12.7** |
| UniMTS 22-joint (reference) | test | 5.6 / 18.5 / 28.8 / 13.3 | 0.1 / 0.4 / 0.8 / 0.5 |

§ `rtune_lr1e-4_tau0.1fixed_steps16k_soupsrc{,_s43,_s44}` (seeds 42/43/44):
same tuned recipe trained for 16k instead of 8k cosine steps, uniform
weight average of the 11 checkpoints at steps 6000-16000 (the best of the
soup ranges on val; the runs' own val-loss `best.pt` average 55.1 val /
54.8 test, so the soup is worth +1.7 / +1.5 here). Per seed, val i2t MRR
56.1 / 56.7 / 57.5, test 56.6 / 56.2 / 56.1. Checkpoints:
`runs/rtune_lr1e-4_tau0.1fixed_steps16k_soupsrc*/checkpoints/soup_6000_16000.pt`.

Read: the tuned recipe is a real, seed-robust **+1.8 MRR / +1.7 R@1 / +2.4
R@5** on val and **+1.7 MRR / +1.8 R@1 / +2.0 R@5** on the untouched test
split. Against the input-matched (wrist-only) UniMTS it was already +15 pp
R@1 and is now +16; against the **22-joint** UniMTS reference (≈ 20× the
sensor channels) wrist-only Ours now wins IMU→Text R@1 on val (40.6 mean /
41.4 best vs 40.3), wins the full-gallery IMU→Text R@1/R@5/MRR on val, and
wins Text→IMU everywhere by > 30 pp — but still trails it on G=100 R@5 /
R@10 / MRR (75.0 / 84.8 / 55.9 vs 79.7 / 89.4 / 57.2 val; 73.8 / 83.6 / 54.3
vs 76.6 / 87.6 / 56.6 test). The 16k-step + soup variant (§, 3 seeds) goes
further and is the final recipe: **it beats the 22-joint reference on
IMU→Text R@1 on both splits (41.8 vs 40.3 val, 41.5 vs 40.7 test), on the
full gallery on both splits (val 13.3 / 37.9 / 50.0 / 25.1 vs 9.4 / 34.9 /
50.0 / 22.4; test 6.1 / 20.4 / 31.2 / 14.1 vs 5.6 / 18.5 / 28.8 / 13.3), and
on every Text→IMU number; G=100 MRR is a statistical tie (56.8 ± 0.6 vs
57.2 val, 56.3 ± 0.2 vs 56.6 test); only R@5 / R@10 at G=100 remain behind
(75.6 / 84.8 vs 79.7 / 89.4 val; 74.8 / 84.0 vs 76.6 / 87.6 test).** That residual ~1-4 pp gap on the recall@k tail
is not closable by hyper-parameters — 47 arms across every axis of the recipe
moved nothing else outside seed noise (sd ≈ 0.5-0.8 MRR) — it is the
22-joint-vs-1-wrist input gap. Two protocol findings from the sweep also
matter for how the §13 rows are read: (a) **val-loss checkpoint selection
under-selects for retrieval** — per-step val MRR rises monotonically to the
end of the cosine schedule while the i2t CE loss bottoms out 2-4k steps
earlier, so `latest.pt` ≥ `best.pt` by 0-1 MRR on every arm and a uniform
weight soup over steps 3000-8000 recovers +1.5 MRR on the parent recipe
(55.5 val / 53.8 test, 2 seeds) — the tuned recipe's `best.pt` is already
its final checkpoint, so soups add only +0.2 there; (b) the §13a-d
"Ours pretrained" headline row (single seed 42, job 8173825: 39.4/54.9 val)
sits at the top of the 4-seed parent band (54.1 ± 0.7), i.e. it is a
slightly lucky seed, not a biased one. Runs: `runs/rtune_*`; JSONs:
`analysis/retrieval_tune/{val,test,steps}/`; tables:
`scripts/summarize_retrieval_tune.py`.

## 14. Parkinson's disease differential diagnosis (PADS smartwatch, 3-class) (2026-07-30, ✅ complete)

PADS — Parkinson's Disease Smartwatch dataset (Varghese et al., PhysioNet
1.0.0): Apple Watch Series 4 accelerometer + gyroscope, native 100 Hz,
**469 patients**, 11 scripted clinical movement tasks (Relaxed, RelaxedTask,
StretchHold, HoldWeight, DrinkGlas, CrossArms, LiftHold, PointFinger,
TouchIndex, TouchNose, Entrainment) recorded on **both wrists** — downloaded
directly from [PhysioNet](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)
(no access gate) into `/projects/b53017/vbc1591/physionet.org` (10,318
recordings, 1.5 GB, all SHA256 verified). Converted to the master-parquet HAR
schema via `scripts/prepare_pads_dataset.py` →
`/projects/b53017/vbc1591/pads_dataset.parquet` (10,318 sessions, 6.48M rows,
**36.0 h** @ canonical 50 Hz). Each (subject, task, wrist) recording is one
session (`session_id = <Task>_<LeftWrist|RightWrist>`), keeping the
single-IMU wrist convention used elsewhere and leaving the task name
recoverable. Resampling interpolates on the recorded `Time` column rather
than the nominal rate (the manifest declares 100 Hz but actual spacing is
~102 Hz with jitter). Standard full acc+gyro checkpoints are used for
probe/finetune, same as WISDM/PAMAP2/UTD-MHAD/RecoFit/HHAR/MHEALTH.

Unlike the HAR sections, **the label here is subject-level** (clinical
condition), not a per-sample activity — every window of every recording of a
patient carries that patient's diagnosis. Splits are therefore grouped by
`subject_id` (`splits.group_key`, already the base.yaml default); grouping by
session would put the same patient's other 21 recordings in train and leak the
label outright. Split is a single subject-disjoint 80/20: **375 train / 94 val
subjects**, 8,250 / 2,068 sessions, 67,500 / 16,920 windows.

Labels reproduce the 3-class differential-diagnosis grouping the release
itself ships in `preprocessed/file_list.csv`: **0=Healthy, 1=Parkinson's,
2=Differential Diagnosis** (Other Movement Disorders / Essential Tremor /
Atypical Parkinsonism / Multiple Sclerosis). Subject counts are imbalanced —
276 PD / 114 DD / 79 HC — so `acc` is nearly uninformative here (the PD-heavy
majority sits at ~62% of val windows, and every column lands within 1.5pp of
that); **read `bal_acc` and `macro_f1`, where chance is 33.33% / ~0.25.**
`--label-scheme {pd_binary,condition6}` in the prep script builds the two
alternative labelings.

Two data properties worth recording. (1) The PADS accelerometer is
**gravity-free** — measured per-axis means are ~0 and |a| averages 0.05–0.33 g,
i.e. CoreMotion `userAcceleration`, not raw readings (the release's own
`run_preprocessing.py` only strips residual drift on top of that). All four
pretrained encoders saw gravity-bearing accelerometer, so there is a genuine
distribution shift and no gravity-direction cue. (2) The first ~0.5 s of every
recording is the watch's start-of-task notification vibration; the release
drops it (`data[:, 48:]` @100 Hz) and so does the prep script
(`--trim-start-seconds 0.5`). Sanity check that the signal survives
conversion: on `Relaxed_RightWrist`, median 3–7 Hz gyro band power is **206
(PD) vs 69 (HC)** — the tremor signature is intact.

Metrics at each run's best-val-loss checkpoint (repo convention). The
best-`bal_acc` checkpoint is given alongside because the gap is large here —
val loss bottoms out early while `bal_acc` keeps climbing, so best-val-loss
systematically understates every column by 3–7pp:

| Model | acc | bal_acc | macro_f1 | bal_acc @best-bal_acc ckpt | Status | Run |
|---|---|---|---|---|---|---|
| Ours (MAE) | 63.07% | **38.94%** | **0.375** | **43.37%** | ✅ | `vit1d_pd_probe_pads_mae` (job 8321785) |
| Ours (MAE) finetune | 63.71% | 39.19% | 0.378 | 45.38% | ✅ | `vit1d_pd_finetune_pads_mae` (job 8321809) |
| FOCAL | 62.22% | 33.34% | 0.255 | 33.96% | ✅ | `vit1d_pd_probe_pads_focal` (job 8321808) |
| FOCAL finetune | 62.42% | 33.70% | 0.263 | 40.37% | ✅ | `vit1d_pd_finetune_pads_focal` (job 8321812) |
| SimMTM | 63.22% | 35.58% | 0.304 | 41.82% | ✅ | `vit1d_pd_probe_pads_simmtm` (job 8321806) |
| SimMTM finetune | 63.42% | **40.17%** | **0.394** | **47.06%** | ✅ | `vit1d_pd_finetune_pads_simmtm` (job 8321810) |
| RelCon | 62.29% | 33.44% | 0.257 | 35.42% | ✅ | `vit1d_pd_probe_pads_relcon` (job 8321807) |
| RelCon finetune | 63.55% | 37.08% | 0.337 | 45.34% | ✅ | `vit1d_pd_finetune_pads_relcon` (job 8321811) |
| Scratch | 63.02% | 35.99% | 0.316 | 41.34% | ✅ | `vit1d_pd_supervised_pads_scratch` (job 8321813) |
| Inertia-1 PatchTST-small (NHANES repro)† | 62.82% | 34.34% | 0.276 | 34.74% | ✅ | `vit1d_pd_probe_pads_inertia1_patchtst` (job 9567126) |
| Inertia-1 PatchTST-small (NHANES repro)† finetune | 62.81% | 34.75% | 0.287 | 39.53% | ✅ | `vit1d_pd_finetune_pads_inertia1_patchtst` (job 9567127) |
| Inertia-1 ART-small (NHANES repro)† | 62.27% | 33.44% | 0.257 | 35.19% | ✅ | `vit1d_pd_probe_pads_inertia1_art` (job 9567149) |
| Inertia-1 ART-small (NHANES repro)† finetune | 63.26% | 37.28% | 0.341 | 38.70% | ✅ | `vit1d_pd_finetune_pads_inertia1_art` (job 9567150) |

Reads: **Ours (MAE) is the best frozen probe by a wide margin** — 38.94% /
0.375 vs SimMTM 35.58% / 0.304, and RelCon (33.44%) and FOCAL (33.34%) sit
*at* the 33.33% chance floor with macro-precision ≈ 0.206 ≈ (majority
freq)/3, the signature of collapsing onto a single class. Their frozen
features carry essentially no PD-relevant information. Every finetune beats
Scratch on `bal_acc` except FOCAL, so pretraining does help end-to-end, but
only by ~3–4pp, and SimMTM overtakes MAE once the encoder is trainable
(40.17% vs 39.19%). All finetunes overfit hard — final-step val losses reach
2.3–3.2 against ~0.85–0.90 at the best checkpoint, and all early-stopped by
epoch 17–20; with 375 training subjects and a subject-level label the encoder
memorizes patients.

⚠️ **These are window-level metrics** — each 2.56 s window scored
independently — whereas PADS' own benchmark is *subject-level* diagnosis. Each
val subject contributes 22 recordings ≈ 180 windows, so per-subject
aggregation (majority vote or mean logit) should raise these numbers
substantially and is the number that is actually comparable to the
literature. That aggregation does not exist in this repo's eval path yet;
until it does, treat the table above as a relative ranking of encoders, not
as PADS diagnostic performance. DeepConvLSTM baseline not run for this
dataset.

## 16. Inertia-1 external baseline (Xu et al. 2026) on all six tasks (2026-08-16/17, ✅ complete)

**No official weights exist** (repo ships code only; issue #1 unanswered) → both rows are our
NHANES reproductions of their two robust objectives at their small preset (≈8.5M):
`inertia1_patchtst_small_nhanes` (masked patch reconstruction) and `inertia1_art_small_nhanes`
(next-patch prediction), 30 s @ 20 Hz triaxial acc in g, no normalisation, 170k×1024 steps
(pretrain jobs 9563086/9563087). Downstream = the same configs as the sibling rows with only
the encoder swapped (acc-only, raw g input, adapter resamples 50→20 Hz; see EXPERIMENTS.md §16
for the six adaptation decisions). Fair reference column = "Ours (MAE, acc-only)".

Rows added to §1 (arm pose), §2 (full-body), §4/§4b (odometry), 5f/Table 6
(microgesture, `analysis/openwatch_fair_comparison.md`), §13d (retrieval) and §14 (PADS) as
`Inertia-1 PatchTST-small (NHANES repro)†` / `Inertia-1 ART-small (NHANES repro)†` × {probe, finetune}
(† = our reproduction, acc-only, raw-g input; fair reference column = "Ours (MAE, acc-only)").
Full per-task collation next to reference rows: `analysis/inertia1/inertia1_results.md`
(`python baselines/inertia1/scripts/harvest_inertia1.py`). Bench/leaderboard cells:
`bench_<task>__inertia1_{patchtst,art}__{probe,finetune}__s42`.
