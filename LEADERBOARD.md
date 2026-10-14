# IMU-Bench Leaderboard

_Generated 2026-09-25T08:09:44-05:00 from `bench/results/` (2314 scored results, git unknown). Regenerate with `python -m bench.leaderboard`. Every number links to a result JSON produced by a `bench/score/*` scorer — nothing is hand-typed._

**Controlled protocol (v1):** one encoder architecture (ViT1D, 21M), one recipe per task shared by every encoder (`bench/registry/tasks.yaml`), seed 42 (+43/44 where shown), checkpoint selected on val by the task's rule, scored by one scorer per task. `†` = harvested from a pre-harness run whose config is recipe-equivalent (same template/LR/selection); all other cells were trained by the harness. Bold = best per column. Lower-is-better metrics are marked ↓.

## Frozen probe — primary metric per task

| Encoder | Arm pose<br><sub>pos err (cm) ↓</sub> | Full-body pose<br><sub>MPJPE (mm) ↓</sub> | Odometry<br><sub>ATE (m) ↓</sub> | Microgesture<br><sub>clip F1 (test) ↑</sub> | Language alignment<br><sub>R@1 G=100 (mean i2t/t2i) ↑</sub> | Parkinson's disease<br><sub>subject macro-F1 ↑</sub> | HAR (4 datasets)<br><sub>mean macro-F1 (4 sets) ↑</sub> | Fall detection · SmartFallMM<br><sub>macro-F1 ↑</sub> | Mean rank | Tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WristMAE (5-source) ⚠ eval-overlap pool | **8.14** | **97.85** | **0.1999** | 0.647 | 27.9 | 0.552 | **0.578** | **0.759** | 1.38 | 8/8 |
| NormWear (Luo et al. 2024, released checkpoint) | 8.39 | 107.74 | 0.2955 | 0.438 | 15.5 | **0.566** | 0.382 | · | 5.57 | 7/8 |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | 9.46 | 106.52 | 0.2352 | 0.215 | 22.2 | 0.254 | 0.519 | · | 7.14 | 7/8 |
| Ours (MAE, no alt-attn) | · | · | · | **0.647**<sup>†</sup> | **28.6** | 0.386 | · | · | 3.33 | 3/8 |
| MAE 4-source (no HAR pool), per-window z-score | · | · | · | 0.613 | 24.1 | 0.463 | · | · | 4.33 | 3/8 |
| MAE 4-source, no CLS token (mean-pooled) | · | · | · | 0.594 | 24.7 | 0.474 | · | · | 4.33 | 3/8 |
| Ours (MAE, 11-source) ⚠ eval-overlap pool | · | · | · | 0.606<sup>†</sup> | 27.1 | 0.388 | · | · | 4.67 | 3/8 |
| Ours (MAE, acc-only) | · | · | · | 0.586<sup>†</sup> | 19.3 | 0.397 | · | · | 6.33 | 3/8 |
| MAE 4-source (no HAR pool), physical rescale | · | · | · | 0.556 | 17.0 | 0.334 | · | · | 9.00 | 3/8 |
| SimMTM | · | · | · | 0.532<sup>†</sup> | 16.7 | 0.368 | · | · | 9.33 | 3/8 |
| SimMTM (4-source pool) | · | · | · | 0.522 | 16.6 | 0.386 | · | · | 9.67 | 3/8 |
| MAE 4-source, physical input + patch-normalised loss | · | · | · | 0.492 | 16.5 | 0.357 | · | · | 11.33 | 3/8 |
| RelCon (4-source pool) | · | · | · | 0.553 | 15.5 | 0.254 | · | · | 12.00 | 3/8 |
| MAE 4-source (no HAR pool), raw/DC-preserving | · | · | · | 0.472 | 8.3 | 0.317 | · | · | 14.00 | 3/8 |
| RelCon | · | · | · | 0.419<sup>†</sup> | 14.5 | 0.285 | · | · | 14.33 | 3/8 |
| FOCAL | · | · | · | 0.376<sup>†</sup> | 8.3 | 0.254 | · | · | 15.33 | 3/8 |
| MAE acc-only (4-source pool) | · | · | · | 0.521 | · | 0.487 | · | · | 7.00 | 2/8 |
| FOCAL (4-source pool) | · | · | · | 0.369 | · | · | · | · | 17.00 | 1/8 |
| MAE 4-source, plain attention (no alt-attn) | · | · | · | · | · | · | · | · | — | 0/8 |
| IMUPoser (Mollyn et al.) | · | · | · | · | · | · | · | · | — | 0/8 |
| PADS pipeline, BOSS + CatBoost (Varghese et al.) | · | · | · | · | · | · | · | · | — | 0/8 |
| PADS pipeline, BOSS + SVM (Varghese et al.) | · | · | · | · | · | · | · | · | — | 0/8 |
| PADS pipeline, spectral features + CatBoost (Varghese et al.) | · | · | · | · | · | · | · | · | — | 0/8 |
| _Scratch (no pretraining) — reference_ | 8.21 | 92.72 | 0.1618 | 0.608 | 23.2 | 0.299 | 0.533 | 0.727 | ref | 8/8 |

## Full finetune — primary metric per task

| Encoder | Arm pose<br><sub>pos err (cm) ↓</sub> | Full-body pose<br><sub>MPJPE (mm) ↓</sub> | Odometry<br><sub>ATE (m) ↓</sub> | Microgesture<br><sub>clip F1 (test) ↑</sub> | Language alignment<br><sub>R@1 G=100 (mean i2t/t2i) ↑</sub> | Parkinson's disease<br><sub>subject macro-F1 ↑</sub> | HAR (4 datasets)<br><sub>mean macro-F1 (4 sets) ↑</sub> | Fall detection · SmartFallMM<br><sub>macro-F1 ↑</sub> | Mean rank | Tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WristMAE (5-source) ⚠ eval-overlap pool | 8.01 | 88.11 | **0.1541** | 0.542 | 38.8 | 0.519 | 0.639 | **0.784** | 3.50 | 8/8 |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | 8.63 | 94.43 | 0.1736 | 0.528 | 37.5 | 0.463 | **0.735** | · | 8.43 | 7/8 |
| NormWear (Luo et al. 2024, released checkpoint) | 8.54 | 102.24 | 0.2966 | 0.210 | 15.7 | 0.431 | 0.278 | · | 11.71 | 7/8 |
| Ours (MAE, 11-source) ⚠ eval-overlap pool | **8.00**<sup>†</sup> | **87.46**<sup>†</sup> | · | 0.580 | 38.8<sup>†</sup> | 0.542 | · | · | 2.80 | 5/8 |
| MAE 4-source (no HAR pool), per-window z-score | 8.04 | 89.22 | · | 0.587 | 37.5 | 0.590 | · | · | 3.80 | 5/8 |
| MAE 4-source, no CLS token (mean-pooled) | 8.02 | 89.27 | · | 0.607 | 36.5 | 0.505 | · | · | 4.80 | 5/8 |
| Ours (MAE, no alt-attn) | 8.16 | 88.28 | · | 0.544 | **39.4** | 0.458 | · | · | 6.00 | 5/8 |
| RelCon | 8.56 | 91.25 | · | **0.618** | 30.7 | **0.619** | · | · | 6.80 | 5/8 |
| SimMTM | 8.45 | 93.27 | · | 0.579 | 31.3 | 0.515 | · | · | 8.20 | 5/8 |
| RelCon (4-source pool) | 8.47 | 93.76 | · | 0.548 | 30.1 | 0.505 | · | · | 10.00 | 5/8 |
| MAE 4-source (no HAR pool), physical rescale | 8.29 | 92.00 | · | 0.543 | 30.1 | 0.313 | · | · | 10.60 | 5/8 |
| Ours (MAE, acc-only) | 8.81 | 96.20 | · | 0.548 | 29.7 | 0.606 | · | · | 10.80 | 5/8 |
| MAE 4-source, physical input + patch-normalised loss | 8.16 | 91.19 | · | 0.543 | 29.2 | 0.254 | · | · | 11.00 | 5/8 |
| MAE 4-source (no HAR pool), raw/DC-preserving | 8.42 | 90.94 | · | 0.504 | 22.9 | 0.364 | · | · | 11.80 | 5/8 |
| FOCAL | 8.76 | 97.80 | · | 0.531 | 23.5 | 0.331 | · | · | 14.80 | 5/8 |
| SimMTM (4-source pool) | 8.12 | · | · | 0.551 | 33.7 | 0.563 | · | · | 5.50 | 4/8 |
| FOCAL (4-source pool) | 8.38 | · | · | 0.482 | · | 0.364 | · | · | 13.33 | 3/8 |
| MAE acc-only (4-source pool) | 8.50 | 96.01 | · | 0.496 | · | · | · | · | 14.00 | 3/8 |
| MAE 4-source, plain attention (no alt-attn) | · | · | · | · | · | · | · | · | — | 0/8 |
| IMUPoser (Mollyn et al.) | · | · | · | · | · | · | · | · | — | 0/8 |
| PADS pipeline, BOSS + CatBoost (Varghese et al.) | · | · | · | · | · | · | · | · | — | 0/8 |
| PADS pipeline, BOSS + SVM (Varghese et al.) | · | · | · | · | · | · | · | · | — | 0/8 |
| PADS pipeline, spectral features + CatBoost (Varghese et al.) | · | · | · | · | · | · | · | · | — | 0/8 |
| _Scratch (no pretraining) — reference_ | 8.21 | 92.72 | 0.1618 | 0.608 | 23.2 | 0.299 | 0.533 | 0.727 | ref | 8/8 |

## Probe-decoder capacity — `mae_4src_z`

Same frozen encoder, same data and recipe; only the task head changes, from the board's 1-layer default to ~12-layer decoders. **Not comparable to the frozen-probe table above** — an ablation, excluded from the ranking until one decoder is promoted into the task cards for every encoder. Bold = best across the `mae_4src_z` rows only.

| Probe decoder | Arm pose (pos err (cm) ↓) | Full-body pose (MPJPE (mm) ↓) | Odometry (ATE (m) ↓) |
|---|---|---|---|
| small decoder (board default, 1 layer) | 8.14 | 98.09 | — |
| 12L balanced | 8.05 | 91.93 | — |
| 12L temporal-heavy | 8.13 | 92.33 | — |
| 12L wide | **8.05** | **91.64** | — |
| _Scratch (no pretraining)_ (reference) | 8.21 | 92.72 | 0.1618 |
| _MobilePoser (Xu et al.)_ (reference) | 9.74 | 91.88 | — |
| _TLIO_ (reference) | — | — | 0.1874 |
| _Air-IO_ (reference) | — | — | 0.1803 |

> **Contamination caveat.** Every cell in the 3×3 grid above ran with a concurrent duplicate job writing into the same run directory for part of training (a dual-partition submission bug, fixed 2026-09-04). Single-job re-runs of the promoted arms reproduce full-body (91.64 → 91.45 mm) and odometry (0.1224 → 0.1228 ATE) but NOT arm pose: the promoted d12w decoder scores 8.14–8.16 cm when run alone, i.e. no better than the 1-layer default. **Treat the arm-pose row as noise, not a win.** The `mae_4src_z` numbers in the frozen-probe table are from the clean single-job re-runs.

## Task cards — unified input / output

| Task | Dataset / split | Input | Output | Selection | Reported split | Primary metric |
|---|---|---|---|---|---|---|
| **Arm pose**<br><sub>Nymeria real right-wrist IMU → 5-joint right-arm chain rotations</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=5, 6]`<br><sub>6D-continuous local rotations of SMPL joints [14,17,19,21,23] (collar, shoulder, elbow, wrist, hand), one per 0.32 s patch</sub> | val_loss (min) on val | val | pos err (cm) ↓ |
| **Full-body pose**<br><sub>Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=23, 6]`<br><sub>6D local rotations for SMPL joints 1..23, one per 0.32 s patch (sub_patches=1)</sub> | mpjpe_mm (min) on val | val | MPJPE (mm) ↓ |
| **Odometry**<br><sub>Nymeria real right-wrist IMU → pelvis velocity in the wrist (FLU) frame</sub> | Nymeria (real wrist IMU + Project-Aria pelvis trajectory)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, FLU | `[P_t=8, K=4, 3]`<br><sub>3-D pelvis velocity (m/s) expressed in the wrist FLU frame at each sub-patch midpoint (imu_lm/data/nymeria_real_wrist_odom_loader_FLU.py), 4 sub-patches per 0.32 s patch (0.08 s resolution)</sub> | val_loss (min) on val | val | ATE (m) ↓ |
| **Microgesture**<br><sub>OpenWatch wrist IMU → 5 gestures + background (paper protocol, real recordings)</sub> | OpenWatch (Bonazzi et al.), open_watch_without_augmentations<br><sub>contamination-safe posture-stratified subject split over canonical identities, seed 42, 80/10/10 (33/5/5 subjects; val 102 clips / test 87 clips)</sub> | `[6, 128]` @ 100 Hz, 1.0 s, FLU | `[6]`<br><sub>class logits per 1 s window; clip label = first run of k=3 identical window predictions, else majority (imu_lm/utils/clip_voting.py)</sub> | macro_f1 (max) on val | test | clip F1 (test) ↑ |
| **Language alignment**<br><sub>HumanML3D synthetic wrist IMU (UniMTS_data) ↔ text retrieval</sub> | HumanML3D / UniMTS_data sim (right-wrist channel), aug_texts captions<br><sub>HumanML3D official train/val/test lists (val 1,197 clips; test 3,701 clips)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[512]`<br><sub>L2-normalised IMU embedding in CLIP ViT-B/32 text space (ContrastiveCLIPHead); text tower trainable, i2t InfoNCE, τ=0.1 fixed</sub> | fixed_budget (latest) on none | test | R@1 G=100 (mean i2t/t2i) ↑ |
| **Parkinson's disease**<br><sub>PADS smartwatch IMU → {Healthy, PD, Differential diagnosis} per subject</sub> | PADS (PhysioNet 1.0.0), 469 patients, 11 tasks × 2 wrists<br><sub>subject-disjoint 80/20 (375 / 94 subjects), splits.group_key=subject_id, seed 0</sub> | `[6, 128]` @ 50 Hz, 2.56 s, FLU | `[3]`<br><sub>class logits per 2.56 s window; subject prediction = argmax of mean softmax over all of the subject's windows</sub> | bal_acc (max) on val | val | subject macro-F1 ↑ |
| **HAR · HHAR**<br><sub>HHAR activity recognition from one wrist/arm IMU (6 classes)</sub> | HHAR: smartwatch acc+gyro (Stisen et al. 2015), 6 activities, 9 users x 4 watches; asof-aligned, 50 Hz canonical<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=6]`<br><sub>class logits over the 6 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · MHEALTH**<br><sub>MHEALTH activity recognition from one wrist/arm IMU (12 classes)</sub> | MHEALTH: right-lower-arm Shimmer2 acc+gyro (Banos et al. 2014), 12 activities, 10 subjects, 50 Hz; raw gyro is bias-dominated<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=12]`<br><sub>class logits over the 12 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · User-study watch**<br><sub>User-study smartwatch, 60 fine-grained everyday activities (10 participants)</sub> | User-study watch (ours): 10 participants x 7 activities each, 70 recordings, 2.0 h, 60 activity classes after merging spelling variants; wrist smartwatch acc (user accel + gravity, m/s^2) + gyro (rad/s), 50 Hz<br><sub>TIME split within each recording: first 80 % train, last 20 % val, one 2.56 s window dropped at the boundary (splits.split_by_session_suffix). Participants appear in both splits -- NOT subject-disjoint; most activities were performed by a single participant</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=60]`<br><sub>class logits over the 60 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · PRISM**<br><sub>PrISM-Tracker (Arakawa et al.) — PRISM activity recognition from one wrist IMU (5 global ADL classes, 31 subjects)</sub> | PRISM: held-out probe dataset (base.yaml splits.probe_dataset), 31 subjects, never in any pretraining pool; 16.3 h, 20 sessions, 96 dataset_activity_id labels (head size = the trainer's own label map)<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K]`<br><sub>class logits over the dataset's activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **Fall detection · SmartFallMM**<br><sub>SmartFallMM right-wrist IMU → fall vs activity of daily living (31 participants)</sub> | SmartFallMM (Texas State SmartFall group): right-wrist MbientLab Meta sensor, acc + gyro, 50 Hz; 31 young participants, 1,352 trials (538 simulated falls onto an air mattress in 5 directions, 814 ADLs of 9 kinds), 3.6 h. Sensor ranges +-2 g / +-125 deg/s: the gyroscope saturates for a median 14 % of a trial's samples, the accelerometer at fall impacts<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=2]`<br><sub>fall / ADL logits, one per 2.56 s window (every window of a trial carries the trial's label)</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **Arm pose · rotated frame**<br><sub>Nymeria real right-wrist IMU → 5-joint right-arm chain rotations — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=5, 6]`<br><sub>6D-continuous local rotations of SMPL joints [14,17,19,21,23] (collar, shoulder, elbow, wrist, hand), one per 0.32 s patch</sub> | val_loss (min) on val | val | pos err (cm) ↓ |
| **Microgesture · rotated frame**<br><sub>OpenWatch wrist IMU → 5 gestures + background (paper protocol, real recordings) — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split</sub> | OpenWatch (Bonazzi et al.), open_watch_without_augmentations<br><sub>contamination-safe posture-stratified subject split over canonical identities, seed 42, 80/10/10 (33/5/5 subjects; val 102 clips / test 87 clips)</sub> | `[6, 128]` @ 100 Hz, 1.0 s, FLU | `[6]`<br><sub>class logits per 1 s window; clip label = first run of k=3 identical window predictions, else majority (imu_lm/utils/clip_voting.py)</sub> | macro_f1 (max) on val | test | clip F1 (test) ↑ |
| **Full-body pose (probe decoder 0x0)**<br><sub>Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=23, 6]`<br><sub>6D local rotations for SMPL joints 1..23, one per 0.32 s patch (sub_patches=1)</sub> | mpjpe_mm (min) on val | val | MPJPE (mm) ↓ |
| **Full-body pose (probe decoder 1x0)**<br><sub>Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=23, 6]`<br><sub>6D local rotations for SMPL joints 1..23, one per 0.32 s patch (sub_patches=1)</sub> | mpjpe_mm (min) on val | val | MPJPE (mm) ↓ |
| **Full-body pose (probe decoder 1x4)**<br><sub>Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=23, 6]`<br><sub>6D local rotations for SMPL joints 1..23, one per 0.32 s patch (sub_patches=1)</sub> | mpjpe_mm (min) on val | val | MPJPE (mm) ↓ |
| **Microgesture (release raw-ID split, leaky)**<br><sub>OpenWatch wrist IMU → 5 gestures + background (paper protocol, real recordings)</sub> | OpenWatch (Bonazzi et al.), open_watch_without_augmentations<br><sub>contamination-safe posture-stratified subject split over canonical identities, seed 42, 80/10/10 (33/5/5 subjects; val 102 clips / test 87 clips)</sub> | `[6, 128]` @ 100 Hz, 1.0 s, FLU | `[6]`<br><sub>class logits per 1 s window; clip label = first run of k=3 identical window predictions, else majority (imu_lm/utils/clip_voting.py)</sub> | macro_f1 (max) on val | test | clip F1 (test) ↑ |
| **Arm pose (deep probe decoder)**<br><sub>Nymeria real right-wrist IMU → 5-joint right-arm chain rotations</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=5, 6]`<br><sub>6D-continuous local rotations of SMPL joints [14,17,19,21,23] (collar, shoulder, elbow, wrist, hand), one per 0.32 s patch</sub> | val_loss (min) on val | val | pos err (cm) ↓ |
| **Full-body pose (deep probe decoder)**<br><sub>Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=23, 6]`<br><sub>6D local rotations for SMPL joints 1..23, one per 0.32 s patch (sub_patches=1)</sub> | mpjpe_mm (min) on val | val | MPJPE (mm) ↓ |
| **Odometry (deep probe decoder)**<br><sub>Nymeria real right-wrist IMU → pelvis velocity in the wrist (FLU) frame</sub> | Nymeria (real wrist IMU + Project-Aria pelvis trajectory)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, FLU | `[P_t=8, K=4, 3]`<br><sub>3-D pelvis velocity (m/s) expressed in the wrist FLU frame at each sub-patch midpoint (imu_lm/data/nymeria_real_wrist_odom_loader_FLU.py), 4 sub-patches per 0.32 s patch (0.08 s resolution)</sub> | val_loss (min) on val | val | ATE (m) ↓ |
| **Arm pose (2-layer probe decoder)**<br><sub>Nymeria real right-wrist IMU → 5-joint right-arm chain rotations</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=5, 6]`<br><sub>6D-continuous local rotations of SMPL joints [14,17,19,21,23] (collar, shoulder, elbow, wrist, hand), one per 0.32 s patch</sub> | val_loss (min) on val | val | pos err (cm) ↓ |
| **Arm pose (4-layer probe decoder)**<br><sub>Nymeria real right-wrist IMU → 5-joint right-arm chain rotations</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=5, 6]`<br><sub>6D-continuous local rotations of SMPL joints [14,17,19,21,23] (collar, shoulder, elbow, wrist, hand), one per 0.32 s patch</sub> | val_loss (min) on val | val | pos err (cm) ↓ |
| **Full-body pose (2-layer probe decoder)**<br><sub>Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=23, 6]`<br><sub>6D local rotations for SMPL joints 1..23, one per 0.32 s patch (sub_patches=1)</sub> | mpjpe_mm (min) on val | val | MPJPE (mm) ↓ |
| **Full-body pose (4-layer probe decoder)**<br><sub>Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=23, 6]`<br><sub>6D local rotations for SMPL joints 1..23, one per 0.32 s patch (sub_patches=1)</sub> | mpjpe_mm (min) on val | val | MPJPE (mm) ↓ |
| **Odometry (2-layer probe decoder)**<br><sub>Nymeria real right-wrist IMU → pelvis velocity in the wrist (FLU) frame</sub> | Nymeria (real wrist IMU + Project-Aria pelvis trajectory)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, FLU | `[P_t=8, K=4, 3]`<br><sub>3-D pelvis velocity (m/s) expressed in the wrist FLU frame at each sub-patch midpoint (imu_lm/data/nymeria_real_wrist_odom_loader_FLU.py), 4 sub-patches per 0.32 s patch (0.08 s resolution)</sub> | val_loss (min) on val | val | ATE (m) ↓ |
| **Odometry (4-layer probe decoder)**<br><sub>Nymeria real right-wrist IMU → pelvis velocity in the wrist (FLU) frame</sub> | Nymeria (real wrist IMU + Project-Aria pelvis trajectory)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, FLU | `[P_t=8, K=4, 3]`<br><sub>3-D pelvis velocity (m/s) expressed in the wrist FLU frame at each sub-patch midpoint (imu_lm/data/nymeria_real_wrist_odom_loader_FLU.py), 4 sub-patches per 0.32 s patch (0.08 s resolution)</sub> | val_loss (min) on val | val | ATE (m) ↓ |

## Coverage of the controlled grid

✅ scored · 🔄 running · ⏳ queued · 🟡 trained but not yet scored · ⬜ not launched

| Encoder | Protocol | Arm pose | Full-body pose | Odometry | Microgesture | Language alignment | Parkinson's disease | HAR · HHAR | HAR · MHEALTH | HAR · User-study watch | HAR · PRISM | Fall detection · SmartFallMM | Arm pose · rotated frame | Microgesture · rotated frame | Full-body pose (probe decoder 0x0) | Full-body pose (probe decoder 1x0) | Full-body pose (probe decoder 1x4) | Microgesture (release raw-ID split, leaky) | Arm pose (deep probe decoder) | Full-body pose (deep probe decoder) | Odometry (deep probe decoder) | Arm pose (2-layer probe decoder) | Arm pose (4-layer probe decoder) | Full-body pose (2-layer probe decoder) | Full-body pose (4-layer probe decoder) | Odometry (2-layer probe decoder) | Odometry (4-layer probe decoder) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAE | probe | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE | finetune | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| WristMAE-5src | probe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| WristMAE-5src | finetune | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-noAA | probe | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-noAA | finetune | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-acc | probe | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-acc | finetune | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| FOCAL | probe | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| FOCAL | finetune | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| SimMTM | probe | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| SimMTM | finetune | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| RelCon | probe | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| RelCon | finetune | ✅ | ✅ | ⬜ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| Scratch | scratch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⬜ | ⬜ | ⬜ | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |

## HAR (4 datasets) — `har_mean`

_mean window macro-F1 over the 4 IMU-Bench HAR datasets (HHAR, MHEALTH, PRISM, user-study watch); WISDM, UTD-MHAD, PAMAP2, RecoFit, SAMoSA and UT-Watch were dropped on 2026-09-17._ Scorer: `bench.leaderboard.inject_aggregates`. Selection: {'split': 'val', 'monitor': 'per member task', 'mode': '-'}. Reported split: **val**.
Primary metric **mean macro-F1 (4 sets)**: mean of the members' primary metric over 4 datasets: har_hhar, har_mhealth, har_prism, har_userwatch.

| Encoder | Protocol | **mean macro-F1 (4 sets)** | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.578 | 42 |  | — | aggregate |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.639 | 42 |  | — | aggregate |
| WristMAE (5-source, no alt-attn) | probe | 0.569 | 42 |  | — | aggregate |
| WristMAE (5-source, no alt-attn) | finetune | 0.590 | 42 |  | — | aggregate |
| WristMAE (5-source, acc-only) | probe | 0.559 | 42 |  | — | aggregate |
| WristMAE (5-source, acc-only) | finetune | 0.580 | 42 |  | — | aggregate |
| FOCAL (5-source) | probe | 0.442 | 42 |  | — | aggregate |
| FOCAL (5-source) | finetune | 0.561 | 42 |  | — | aggregate |
| FOCAL (5-source, native frequency-domain input) | probe | 0.629 | 42 |  | — | aggregate |
| SimMTM (5-source) | probe | 0.531 | 42 |  | — | aggregate |
| SimMTM (5-source) | finetune | 0.521 | 42 |  | — | aggregate |
| SimMTM (5-source, native normalisation) | probe | 0.461 | 42 |  | — | aggregate |
| RelCon (5-source) | probe | 0.354 | 42 |  | — | aggregate |
| RelCon (5-source) | finetune | 0.528 | 42 |  | — | aggregate |
| RelCon (5-source, native normalisation) | probe | 0.560 | 42 |  | — | aggregate |
| LIMU-BERT (5-source) | probe | 0.562 | 42 |  | — | aggregate |
| LIMU-BERT (5-source) | finetune | 0.561 | 42 |  | — | aggregate |
| SSL-Wearables (5-source) | probe | 0.551 | 42 |  | — | aggregate |
| SSL-Wearables (5-source) | finetune | 0.567 | 42 |  | — | aggregate |
| Scratch (no pretraining) | scratch | 0.533 | 42 |  | — | aggregate |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.519 | — |  | — | legacy† |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 0.519 | 42 |  | — | aggregate |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 0.735 | 42 |  | — | aggregate |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 0.382 | 42 |  | — | aggregate |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 0.278 | 42 |  | — | aggregate |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 0.543 | 42 |  | — | aggregate |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 0.547 | 42 |  | — | aggregate |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 0.539 | 42 |  | — | aggregate |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 0.569 | 42 |  | — | aggregate |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 0.585 | 42 |  | — | aggregate |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 0.564 | 42 |  | — | aggregate |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 0.538 | 42 |  | — | aggregate |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 0.525 | 42 |  | — | aggregate |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.574 | 42 |  | — | aggregate |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.557 | 42 |  | — | aggregate |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | probe | 0.680 | 42 |  | — | aggregate |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | finetune | 0.656 | 42 |  | — | aggregate |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.262 | 42 |  | — | aggregate |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.056 | 42 |  | — | aggregate |
| MAE-5src fine-tuned on full-body pose | probe | 0.594 | 42 |  | — | aggregate |
| MAE-5src fine-tuned on microgesture | probe | 0.556 | 42 |  | — | aggregate |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 0.586 | 42 |  | — | aggregate |
| Multi-task (full-body pose + odometry, 1:1) | probe | 0.571 | 42 |  | — | aggregate |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 0.582 | 42 |  | — | aggregate |

## Arm pose — `arm_pose_nymeria`

_Nymeria real right-wrist IMU → 5-joint right-arm chain rotations._ Scorer: `bench.score.arm_pose`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **pos err (cm)**: FK joint-position error at the 5 arm joints with GT ancestors (scripts/eval_arm_pose_position_error.py).

| Encoder | Protocol | **pos err (cm)** | ang err (°) | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) ⚠ eval-overlap pool | finetune | 8.00 | 9.89 | 0.01599 | 42 | best.pt (3000) | `vit1d_arm_pose_finetune_nymeria_mae_multidata` | legacy† |
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 8.14 | 9.89 | 0.01650 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 8.01 | 9.86 | 0.01586 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | finetune | 8.16 | 9.98 | 0.01630 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__mae_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 8.36 | 10.17 | 0.01737 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 8.19 | 10.06 | 0.01662 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__mae_5src_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | finetune | 8.81 | 10.90 | 0.01892 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__mae_acconly__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 8.76 | 10.61 | 0.01862 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 8.88 | 10.80 | 0.01874 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__mae_5src_acconly__finetune__s42` | bench |
| FOCAL | finetune | 8.76 | 10.85 | 0.01915 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__focal__finetune__s42` | bench |
| FOCAL (5-source) | probe | 8.71 | 10.70 | 0.01900 | 42 | best.pt (11000) | `bench_arm_pose_nymeria__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 8.66 | 10.69 | 0.01866 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 8.38 | 10.29 | 0.01778 | 42 | best.pt (10000) | `bench_arm_pose_nymeria__focal_5src_native__probe__s42` | bench |
| SimMTM | finetune | 8.45 | 10.38 | 0.01787 | 42 | best.pt (4000) | `bench_arm_pose_nymeria__simmtm__finetune__s42` | bench |
| SimMTM (5-source) | probe | 8.47 | 10.27 | 0.01770 | 42 | best.pt (10000) | `bench_arm_pose_nymeria__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 8.44 | 10.29 | 0.01788 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 8.50 | 10.37 | 0.01799 | 42 | best.pt (10000) | `bench_arm_pose_nymeria__simmtm_5src_native__probe__s42` | bench |
| RelCon | finetune | 8.56 | 10.49 | 0.01771 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__relcon__finetune__s42` | bench |
| RelCon (5-source) | probe | 8.50 | 10.41 | 0.01816 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 8.55 | 10.53 | 0.01810 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 8.22 | 10.09 | 0.01730 | 42 | best.pt (17000) | `bench_arm_pose_nymeria__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 8.14 | 9.86 | 0.01649 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 8.18 | 9.97 | 0.01642 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 8.42 | 10.14 | 0.01719 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 8.57 | 10.58 | 0.01834 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__sslwear_5src__finetune__s42` | bench |
| DINO (5-source) | probe | 8.57 | 10.39 | 0.01811 | 42 | best.pt (10000) | `bench_arm_pose_nymeria__dino_5src__probe__s42` | bench |
| DINO (5-source) | finetune | 8.76 | 10.83 | 0.01918 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__dino_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 8.21 | 9.89 | 0.01674 | 42 | best.pt (50000) | `bench_arm_pose_nymeria__scratch__scratch__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 8.04 | 9.83 | 0.01581 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 8.42 | 10.26 | 0.01752 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | finetune | 8.02 | 9.92 | 0.01601 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__mae_4src_nocls__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 8.29 | 10.21 | 0.01695 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 8.16 | 10.08 | 0.01700 | 42 | best.pt (5000) | `bench_arm_pose_nymeria__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | finetune | 8.12 | 9.86 | 0.01624 | 42 | best.pt (4000) | `bench_arm_pose_nymeria__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | finetune | 8.38 | 10.50 | 0.01800 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__focal_4src__finetune__s42` | bench |
| RelCon (4-source pool) | finetune | 8.47 | 10.41 | 0.01743 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__relcon_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 8.50 | 10.38 | 0.01739 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__acconly_4src__finetune__s42` | bench |
| *MobilePoser (Xu et al.)* | external | 9.74 | 24.28 | — | — |  | — | harvest:Experiments_table.md §1 / baseli |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 8.51 | 10.38 | 0.01755 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 8.35 | 10.31 | 0.01744 | 42 | best.pt (5000) | `bench_arm_pose_nymeria__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 8.27 | 10.17 | 0.01696 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 8.55 | 10.51 | 0.01755 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__inertia1_art_large__finetune__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 9.46 | 11.65 | 0.02246 | 42 | best.pt (24000) | `bench_arm_pose_nymeria__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 8.63 | 10.50 | 0.01770 | 42 | best.pt (4000) | `bench_arm_pose_nymeria__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 8.39 | 10.27 | 0.01763 | 42 | best.pt (11000) | `bench_arm_pose_nymeria__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 8.54 | 10.42 | 0.01803 | 42 | best.pt (21000) | `bench_arm_pose_nymeria__normwear_base__finetune__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 8.37 | 10.17 | 0.01731 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 8.36 | 10.18 | 0.01747 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 8.34 | 10.11 | 0.01729 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 8.22 | 10.01 | 0.01680 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 8.28 | 10.02 | 0.01684 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 8.36 | 10.19 | 0.01741 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 8.41 | 10.21 | 0.01758 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 8.36 | 10.14 | 0.01738 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 8.25 | 10.05 | 0.01695 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 8.42 | 10.22 | 0.01744 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__abl_so3__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | probe | 8.03 | 9.74 | 0.01619 | 42 | best.pt (11000) | `bench_arm_pose_nymeria__mae_5src_physin__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 8.46 | 10.35 | 0.01786 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 8.51 | 10.30 | 0.01784 | 42 | best.pt (10000) | `bench_arm_pose_nymeria__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 7.73 | 9.37 | 0.01493 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 8.20 | 9.96 | 0.01684 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 8.07 | 9.76 | 0.01617 | 42 | best.pt (13000) | `bench_arm_pose_nymeria__tx5_odom__probe__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | probe | 8.55 | 10.48 | 0.01813 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__txs_prism__probe__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | finetune | 8.56 | 10.50 | 0.01807 | 42 | best.pt (5000) | `bench_arm_pose_nymeria__txs_prism__finetune__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | probe | 8.13 | 9.93 | 0.01674 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__txs_fullbody__probe__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | finetune | 8.27 | 10.12 | 0.01677 | 42 | best.pt (1000) | `bench_arm_pose_nymeria__txs_fullbody__finetune__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | probe | 8.52 | 10.43 | 0.01820 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__txs_gesture__probe__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | finetune | 8.64 | 10.47 | 0.01822 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__txs_gesture__finetune__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | probe | 8.26 | 10.14 | 0.01733 | 42 | best.pt (11000) | `bench_arm_pose_nymeria__txs_odom__probe__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | finetune | 8.12 | 9.97 | 0.01678 | 42 | best.pt (5000) | `bench_arm_pose_nymeria__txs_odom__finetune__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | probe | 8.53 | 10.34 | 0.01803 | 42 | best.pt (14000) | `bench_arm_pose_nymeria__txs_pd__probe__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | finetune | 8.41 | 10.31 | 0.01774 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__txs_pd__finetune__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | probe | 8.54 | 10.36 | 0.01813 | 42 | best.pt (12000) | `bench_arm_pose_nymeria__txs_language__probe__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | finetune | 8.50 | 10.37 | 0.01805 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__txs_language__finetune__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 7.93 | 9.65 | 0.01594 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 8.19 | 9.96 | 0.01653 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__mt_body_gesture__probe__s42` | bench |
| IMUPoser (Mollyn et al.) | external | 10.15 | 23.82 | — | 0 |  | — | bench |

## Full-body pose — `full_body_pose_nymeria`

_Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)._ Scorer: `bench.score.full_body_pose`. Selection: {'split': 'val', 'monitor': 'mpjpe_mm', 'mode': 'min'}. Reported split: **val**.
Primary metric **MPJPE (mm)**: root-relative FK joint position error, joints 1-23 (imu_lm/objectives/pose_1d.py::_compute_val_mpjpe), at the selected checkpoint.

| Encoder | Protocol | **MPJPE (mm)** | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) ⚠ eval-overlap pool | finetune | 87.46 | 0.01800 | 42 | best_mpjpe.pt (21000) | `vit1d_full_body_pose_finetune_nymeria_mae_multidata_lr1e-3` | legacy† |
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 97.85 | 0.01935 | 42 | best_mpjpe.pt (34000) | `bench_full_body_pose_nymeria__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 88.11 | 0.01817 | 42 | best_mpjpe.pt (21000) | `bench_full_body_pose_nymeria__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | finetune | 88.28 | 0.01873 | 42 | best_mpjpe.pt (42000) | `bench_full_body_pose_nymeria__mae_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 99.96 | 0.01970 | 42 | best_mpjpe.pt (28000) | `bench_full_body_pose_nymeria__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 88.38 | 0.01827 | 42 | best_mpjpe.pt (17000) | `bench_full_body_pose_nymeria__mae_5src_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | finetune | 96.20 | 0.02017 | 42 | best_mpjpe.pt (11000) | `bench_full_body_pose_nymeria__mae_acconly__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 102.79 | 0.02037 | 42 | best_mpjpe.pt (18000) | `bench_full_body_pose_nymeria__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 94.42 | 0.01993 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__mae_5src_acconly__finetune__s42` | bench |
| FOCAL | finetune | 97.80 | 0.02083 | 42 | best_mpjpe.pt (24000) | `bench_full_body_pose_nymeria__focal__finetune__s42` | bench |
| FOCAL (5-source) | probe | 108.41 | 0.02205 | 42 | best_mpjpe.pt (71000) | `bench_full_body_pose_nymeria__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 95.92 | 0.02014 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 103.94 | 0.02074 | 42 | best_mpjpe.pt (25000) | `bench_full_body_pose_nymeria__focal_5src_native__probe__s42` | bench |
| SimMTM | finetune | 93.27 | 0.01947 | 42 | best_mpjpe.pt (16000) | `bench_full_body_pose_nymeria__simmtm__finetune__s42` | bench |
| SimMTM (5-source) | probe | 103.65 | 0.02055 | 42 | best_mpjpe.pt (44000) | `bench_full_body_pose_nymeria__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 92.91 | 0.01958 | 42 | best_mpjpe.pt (20000) | `bench_full_body_pose_nymeria__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 104.37 | 0.02100 | 42 | best_mpjpe.pt (68000) | `bench_full_body_pose_nymeria__simmtm_5src_native__probe__s42` | bench |
| RelCon | finetune | 91.25 | 0.01892 | 42 | best_mpjpe.pt (17000) | `bench_full_body_pose_nymeria__relcon__finetune__s42` | bench |
| RelCon (5-source) | probe | 104.29 | 0.02073 | 42 | best_mpjpe.pt (30000) | `bench_full_body_pose_nymeria__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 92.15 | 0.01889 | 42 | best_mpjpe.pt (17000) | `bench_full_body_pose_nymeria__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 104.75 | 0.02108 | 42 | best_mpjpe.pt (68000) | `bench_full_body_pose_nymeria__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 98.55 | 0.01952 | 42 | best_mpjpe.pt (56000) | `bench_full_body_pose_nymeria__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 89.74 | 0.01849 | 42 | best_mpjpe.pt (14000) | `bench_full_body_pose_nymeria__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 101.67 | 0.02003 | 42 | best_mpjpe.pt (29000) | `bench_full_body_pose_nymeria__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 90.61 | 0.01889 | 42 | best_mpjpe.pt (22000) | `bench_full_body_pose_nymeria__sslwear_5src__finetune__s42` | bench |
| DINO (5-source) | probe | 106.71 | 0.02127 | 42 | best_mpjpe.pt (27000) | `bench_full_body_pose_nymeria__dino_5src__probe__s42` | bench |
| DINO (5-source) | finetune | 95.71 | 0.02010 | 42 | best_mpjpe.pt (32000) | `bench_full_body_pose_nymeria__dino_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 92.72 | 0.01967 | 42 | best_mpjpe.pt (17000) | `bench_full_body_pose_nymeria__scratch__scratch__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | finetune | 90.54 | 0.01919 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__mae_div60_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 89.22 | 0.01881 | 42 | best_mpjpe.pt (16000) | `bench_full_body_pose_nymeria__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 90.94 | 0.01910 | 42 | best_mpjpe.pt (19000) | `bench_full_body_pose_nymeria__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | finetune | 89.27 | 0.01857 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__mae_4src_nocls__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 92.00 | 0.01934 | 42 | best_mpjpe.pt (8000) | `bench_full_body_pose_nymeria__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 91.19 | 0.01933 | 42 | best_mpjpe.pt (13000) | `bench_full_body_pose_nymeria__mae_4src_physnp__finetune__s42` | bench |
| RelCon (4-source pool) | finetune | 93.76 | 0.01973 | 42 | best_mpjpe.pt (24000) | `bench_full_body_pose_nymeria__relcon_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 96.01 | 0.02073 | 42 | best_mpjpe.pt (10000) | `bench_full_body_pose_nymeria__acconly_4src__finetune__s42` | bench |
| *MobilePoser (Xu et al.)* | external | 91.88 | — | — |  | — | harvest:Experiments_table.md §2 / baseli |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 93.61 | 0.01970 | 42 | best_mpjpe.pt (10000) | `bench_full_body_pose_nymeria__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 94.12 | 0.01991 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 92.74 | 0.01972 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 94.77 | 0.02035 | 42 | best_mpjpe.pt (14000) | `bench_full_body_pose_nymeria__inertia1_art_large__finetune__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 106.52 | 0.02129 | 42 | best_mpjpe.pt (81000) | `bench_full_body_pose_nymeria__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 94.43 | 0.02004 | 42 | best_mpjpe.pt (29000) | `bench_full_body_pose_nymeria__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 107.74 | 0.02094 | 42 | best_mpjpe.pt (13000) | `bench_full_body_pose_nymeria__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 102.24 | 0.02163 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__normwear_base__finetune__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 100.69 | 0.01986 | 42 | best_mpjpe.pt (34000) | `bench_full_body_pose_nymeria__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 100.64 | 0.01980 | 42 | best_mpjpe.pt (29000) | `bench_full_body_pose_nymeria__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 101.01 | 0.01977 | 42 | best_mpjpe.pt (28000) | `bench_full_body_pose_nymeria__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 98.70 | 0.01932 | 42 | best_mpjpe.pt (29000) | `bench_full_body_pose_nymeria__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 99.08 | 0.01953 | 42 | best_mpjpe.pt (29000) | `bench_full_body_pose_nymeria__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 100.70 | 0.01994 | 42 | best_mpjpe.pt (45000) | `bench_full_body_pose_nymeria__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 100.69 | 0.02012 | 42 | best_mpjpe.pt (64000) | `bench_full_body_pose_nymeria__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 100.35 | 0.01982 | 42 | best_mpjpe.pt (62000) | `bench_full_body_pose_nymeria__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 99.85 | 0.01946 | 42 | best_mpjpe.pt (28000) | `bench_full_body_pose_nymeria__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 100.26 | 0.01992 | 42 | best_mpjpe.pt (64000) | `bench_full_body_pose_nymeria__abl_so3__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 104.99 | 0.02072 | 42 | best_mpjpe.pt (87000) | `bench_full_body_pose_nymeria__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 106.36 | 0.02134 | 42 | best_mpjpe.pt (103000) | `bench_full_body_pose_nymeria__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | probe | 98.77 | 0.01928 | 42 | best_mpjpe.pt (27000) | `bench_full_body_pose_nymeria__tx5_prism__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | finetune | 88.20 | 0.01806 | 42 | best_mpjpe.pt (17000) | `bench_full_body_pose_nymeria__tx5_prism__finetune__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 97.98 | 0.01955 | 42 | best_mpjpe.pt (34000) | `bench_full_body_pose_nymeria__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | finetune | 88.14 | 0.01862 | 42 | best_mpjpe.pt (37000) | `bench_full_body_pose_nymeria__tx5_gesture__finetune__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 95.05 | 0.01833 | 42 | best_mpjpe.pt (22000) | `bench_full_body_pose_nymeria__tx5_odom__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | finetune | 87.28 | 0.01780 | 42 | best_mpjpe.pt (16000) | `bench_full_body_pose_nymeria__tx5_odom__finetune__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | probe | 99.55 | 0.01963 | 42 | best_mpjpe.pt (26000) | `bench_full_body_pose_nymeria__tx5_pd__probe__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | finetune | 87.49 | 0.01807 | 42 | best_mpjpe.pt (20000) | `bench_full_body_pose_nymeria__tx5_pd__finetune__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | probe | 98.99 | 0.01963 | 42 | best_mpjpe.pt (34000) | `bench_full_body_pose_nymeria__tx5_language__probe__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | finetune | 87.94 | 0.01797 | 42 | best_mpjpe.pt (14000) | `bench_full_body_pose_nymeria__tx5_language__finetune__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | probe | 103.49 | 0.02072 | 42 | best_mpjpe.pt (85000) | `bench_full_body_pose_nymeria__txs_prism__probe__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | finetune | 92.39 | 0.01905 | 42 | best_mpjpe.pt (18000) | `bench_full_body_pose_nymeria__txs_prism__finetune__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | probe | 103.91 | 0.02049 | 42 | best_mpjpe.pt (62000) | `bench_full_body_pose_nymeria__txs_gesture__probe__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | finetune | 92.78 | 0.01912 | 42 | best_mpjpe.pt (26000) | `bench_full_body_pose_nymeria__txs_gesture__finetune__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | probe | 99.92 | 0.01953 | 42 | best_mpjpe.pt (30000) | `bench_full_body_pose_nymeria__txs_odom__probe__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | finetune | 87.89 | 0.01799 | 42 | best_mpjpe.pt (17000) | `bench_full_body_pose_nymeria__txs_odom__finetune__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | probe | 104.33 | 0.02068 | 42 | best_mpjpe.pt (62000) | `bench_full_body_pose_nymeria__txs_pd__probe__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | finetune | 92.89 | 0.01923 | 42 | best_mpjpe.pt (27000) | `bench_full_body_pose_nymeria__txs_pd__finetune__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | probe | 103.31 | 0.02065 | 42 | best_mpjpe.pt (77000) | `bench_full_body_pose_nymeria__txs_language__probe__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | finetune | 92.11 | 0.01929 | 42 | best_mpjpe.pt (30000) | `bench_full_body_pose_nymeria__txs_language__finetune__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | probe | 95.99 | 0.01952 | 42 | best_mpjpe.pt (49000) | `bench_full_body_pose_nymeria__txs_armpose__probe__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | finetune | 89.82 | 0.01860 | 42 | best_mpjpe.pt (15000) | `bench_full_body_pose_nymeria__txs_armpose__finetune__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 91.13 | 0.01904 | 42 | best_mpjpe.pt (19000) | `bench_full_body_pose_nymeria__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | finetune | 91.51 | — | 42 | best.pt (40000) | `mt_body_odom__full_body_pose_nymeria` | bench.rescore_full_body_extra.evaluate ( |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 94.34 | 0.01952 | 42 | best_mpjpe.pt (17000) | `bench_full_body_pose_nymeria__mt_body_gesture__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | finetune | 94.43 | — | 42 | best.pt (22000) | `mt_body_gesture__full_body_pose_nymeria` | bench.rescore_full_body_extra.evaluate ( |
| IMUPoser (Mollyn et al.) | external | 96.87 | — | 0 |  | — | bench |

## Odometry — `odometry_nymeria_local`

_Nymeria real right-wrist IMU → pelvis velocity in the wrist (FLU) frame._ Scorer: `bench.score.odometry_local`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **ATE (m)**: mean absolute trajectory error over all val windows; predicted FLU velocities are rotated to the world frame with the ground-truth start orientation propagated by raw-gyro integration, then summed (bench/score/odometry_local.py, = scripts/eval_odom_nymeria_altattn.py::reconstruct_and_score).

| Encoder | Protocol | **ATE (m)** | drift (%) | vel MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.1999 | 53.93 | 0.02732 | 42 | best.pt (144000) | `bench_odometry_nymeria_local__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.1541 | 45.13 | 0.01824 | 42 | best.pt (32000) | `bench_odometry_nymeria_local__mae_5src__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 0.2104 | 58.07 | 0.02892 | 42 | best.pt (176000) | `bench_odometry_nymeria_local__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 0.1569 | 45.76 | 0.01880 | 42 | best.pt (38000) | `bench_odometry_nymeria_local__mae_5src_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 0.2296 | 60.46 | 0.03232 | 42 | best.pt (81000) | `bench_odometry_nymeria_local__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 0.1898 | 51.45 | 0.02558 | 42 | best.pt (22000) | `bench_odometry_nymeria_local__mae_5src_acconly__finetune__s42` | bench |
| FOCAL (5-source) | probe | 0.2404 | 66.41 | 0.03334 | 42 | best.pt (199000) | `bench_odometry_nymeria_local__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 0.1767 | 51.49 | 0.02218 | 42 | best.pt (58000) | `bench_odometry_nymeria_local__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 0.2185 | 62.01 | 0.02953 | 42 | best.pt (231000) | `bench_odometry_nymeria_local__focal_5src_native__probe__s42` | bench |
| SimMTM (5-source) | probe | 0.2240 | 62.51 | 0.03064 | 42 | best.pt (136000) | `bench_odometry_nymeria_local__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 0.1629 | 48.24 | 0.01946 | 42 | best.pt (51000) | `bench_odometry_nymeria_local__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 0.2271 | 61.38 | 0.03130 | 42 | best.pt (149000) | `bench_odometry_nymeria_local__simmtm_5src_native__probe__s42` | bench |
| RelCon (5-source) | probe | 0.2153 | 59.46 | 0.02893 | 42 | best.pt (187000) | `bench_odometry_nymeria_local__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 0.1672 | 49.70 | 0.02037 | 42 | best.pt (40000) | `bench_odometry_nymeria_local__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 0.2320 | 66.84 | 0.03150 | 42 | best.pt (160000) | `bench_odometry_nymeria_local__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 0.2078 | 56.40 | 0.02856 | 42 | best.pt (161000) | `bench_odometry_nymeria_local__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 0.1576 | 46.13 | 0.01885 | 42 | best.pt (34000) | `bench_odometry_nymeria_local__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 0.2205 | 59.14 | 0.03078 | 42 | best.pt (110000) | `bench_odometry_nymeria_local__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 0.1656 | 47.04 | 0.02030 | 42 | best.pt (38000) | `bench_odometry_nymeria_local__sslwear_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.1618 | 47.67 | 0.01940 | 42 | best.pt (43000) | `bench_odometry_nymeria_local__scratch__scratch__s42` | bench |
| *TLIO* | external | 0.1874 | 48.86 | 0.02978 | — |  | — | bench |
| *Air-IO* | external | 0.1803 | 48.99 | 0.03148 | — |  | — | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 0.2352 | 59.64 | 0.03655 | 42 | best.pt (81000) | `bench_odometry_nymeria_local__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 0.1736 | 50.44 | 0.02263 | 42 | best.pt (42000) | `bench_odometry_nymeria_local__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 0.2955 | 78.28 | 0.04398 | 42 | best.pt (14000) | `bench_odometry_nymeria_local__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 0.2966 | 78.66 | 0.04415 | 42 | best.pt (14000) | `bench_odometry_nymeria_local__normwear_base__finetune__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 0.2090 | 57.55 | 0.02868 | 42 | best.pt (159000) | `bench_odometry_nymeria_local__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 0.2161 | 58.65 | 0.02962 | 42 | best.pt (94000) | `bench_odometry_nymeria_local__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 0.2108 | 56.61 | 0.02913 | 42 | best.pt (137000) | `bench_odometry_nymeria_local__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 0.2006 | 54.45 | 0.02726 | 42 | best.pt (187000) | `bench_odometry_nymeria_local__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 0.2007 | 54.75 | 0.02722 | 42 | best.pt (176000) | `bench_odometry_nymeria_local__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 0.2095 | 58.45 | 0.02843 | 42 | best.pt (161000) | `bench_odometry_nymeria_local__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 0.2161 | 59.40 | 0.02946 | 42 | best.pt (117000) | `bench_odometry_nymeria_local__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 0.2142 | 58.50 | 0.02953 | 42 | best.pt (119000) | `bench_odometry_nymeria_local__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.2043 | 57.05 | 0.02756 | 42 | best.pt (188000) | `bench_odometry_nymeria_local__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.2123 | 57.67 | 0.02934 | 42 | best.pt (144000) | `bench_odometry_nymeria_local__abl_so3__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.2334 | 64.06 | 0.03217 | 42 | best.pt (134000) | `bench_odometry_nymeria_local__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.2558 | 68.76 | 0.03720 | 42 | best.pt (134000) | `bench_odometry_nymeria_local__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | probe | 0.1998 | 54.88 | 0.02679 | 42 | best.pt (140000) | `bench_odometry_nymeria_local__tx5_prism__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | finetune | 0.1567 | 45.97 | 0.01812 | 42 | best.pt (27000) | `bench_odometry_nymeria_local__tx5_prism__finetune__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 0.1667 | 48.05 | 0.02054 | 42 | best.pt (137000) | `bench_odometry_nymeria_local__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | finetune | 0.1528 | 44.79 | 0.01792 | 42 | best.pt (14000) | `bench_odometry_nymeria_local__tx5_fullbody__finetune__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 0.1988 | 54.34 | 0.02700 | 42 | best.pt (199000) | `bench_odometry_nymeria_local__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | finetune | 0.1560 | 45.58 | 0.01818 | 42 | best.pt (27000) | `bench_odometry_nymeria_local__tx5_gesture__finetune__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | probe | 0.2025 | 55.26 | 0.02776 | 42 | best.pt (137000) | `bench_odometry_nymeria_local__tx5_pd__probe__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | finetune | 0.1551 | 45.20 | 0.01826 | 42 | best.pt (22000) | `bench_odometry_nymeria_local__tx5_pd__finetune__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | probe | 0.2052 | 55.80 | 0.02825 | 42 | best.pt (154000) | `bench_odometry_nymeria_local__tx5_language__probe__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | finetune | 0.1554 | 45.30 | 0.01828 | 42 | best.pt (32000) | `bench_odometry_nymeria_local__tx5_language__finetune__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | probe | 0.2252 | 64.73 | 0.03012 | 42 | best.pt (230000) | `bench_odometry_nymeria_local__txs_prism__probe__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | finetune | 0.1672 | 49.17 | 0.02019 | 42 | best.pt (36000) | `bench_odometry_nymeria_local__txs_prism__finetune__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | probe | 0.1865 | 53.03 | 0.02407 | 42 | best.pt (113000) | `bench_odometry_nymeria_local__txs_fullbody__probe__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | finetune | 0.1626 | 47.73 | 0.01966 | 42 | best.pt (25000) | `bench_odometry_nymeria_local__txs_fullbody__finetune__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | probe | 0.2308 | 65.84 | 0.03132 | 42 | best.pt (140000) | `bench_odometry_nymeria_local__txs_gesture__probe__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | finetune | 0.1648 | 48.17 | 0.01993 | 42 | best.pt (41000) | `bench_odometry_nymeria_local__txs_gesture__finetune__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | probe | 0.2255 | 63.29 | 0.03057 | 42 | best.pt (170000) | `bench_odometry_nymeria_local__txs_pd__probe__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | finetune | 0.1638 | 48.13 | 0.01975 | 42 | best.pt (66000) | `bench_odometry_nymeria_local__txs_pd__finetune__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | probe | 0.2271 | 62.97 | 0.03137 | 42 | best.pt (199000) | `bench_odometry_nymeria_local__txs_language__probe__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | finetune | 0.1647 | 48.34 | 0.01992 | 42 | best.pt (36000) | `bench_odometry_nymeria_local__txs_language__finetune__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | probe | 0.1925 | 54.54 | 0.02528 | 42 | best.pt (99000) | `bench_odometry_nymeria_local__txs_armpose__probe__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | finetune | 0.1715 | 49.84 | 0.02096 | 42 | best.pt (20000) | `bench_odometry_nymeria_local__txs_armpose__finetune__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 0.1497 | 44.21 | 0.01728 | 42 | best.pt (34000) | `bench_odometry_nymeria_local__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | finetune | 0.1488 | 44.27 | 0.01732 | 42 | best.pt (40000) | `mt_body_odom__odometry_nymeria_local` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 0.1774 | 50.65 | 0.02265 | 42 | best.pt (155000) | `bench_odometry_nymeria_local__mt_body_gesture__probe__s42` | bench |

## Microgesture — `microgesture_openwatch`

_OpenWatch wrist IMU → 5 gestures + background (paper protocol, real recordings)._ Scorer: `bench.score.microgesture`. Selection: {'split': 'val', 'monitor': 'macro_f1', 'mode': 'max', 'note': 'window-level val macro-F1 (paper §3.3.3), inverse-frequency class-weighted CE'}. Reported split: **test**.

| Encoder | Protocol | **clip F1 (test)** | clip acc (test) | win F1 (test) | win acc (test) | clip F1 (val) | win F1 (val) | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) ⚠ eval-overlap pool | probe | 0.606 | 0.598 | 0.490 | 0.508 | 0.674 | 0.561 | 42 | best.pt (1200) | `vit1d_gesture_probe_openwatch_6class_mae_windowed_strat` | legacy† |
| Ours (MAE, 11-source) ⚠ eval-overlap pool | finetune | 0.580 | 0.598 | 0.519 | 0.563 | 0.564 | 0.548 | 42 | best.pt (200) | `bench_microgesture_openwatch__mae_11src__finetune__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.647 | 0.678 | 0.494 | 0.536 | 0.729 | 0.544 | 42 | best.pt (8400) | `bench_microgesture_openwatch__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.542 | 0.655 | 0.480 | 0.594 | 0.789 | 0.671 | 42 | best.pt (1400) | `bench_microgesture_openwatch__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 0.647 | 0.655 | 0.524 | 0.552 | 0.647 | 0.552 | 42 | best.pt (9000) | `vit1d_gesture_probe_openwatch_6class_mae_noaltattn_windowed_strat` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.544 | 0.644 | 0.481 | 0.586 | 0.620 | 0.579 | 42 | best.pt (3400) | `bench_microgesture_openwatch__mae_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 0.639 | 0.667 | 0.494 | 0.533 | 0.626 | 0.572 | 42 | best.pt (9600) | `bench_microgesture_openwatch__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 0.703 | 0.747 | 0.562 | 0.636 | 0.780 | 0.654 | 42 | best.pt (1200) | `bench_microgesture_openwatch__mae_5src_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.586 | 0.586 | 0.478 | 0.496 | 0.625 | 0.508 | 42 | best.pt (3600) | `vit1d_gesture_probe_openwatch_6class_mae_acconly_windowed_strat` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.548 | 0.621 | 0.502 | 0.582 | 0.695 | 0.611 | 42 | best.pt (2200) | `bench_microgesture_openwatch__mae_acconly__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 0.622 | 0.644 | 0.491 | 0.515 | 0.617 | 0.515 | 42 | best.pt (10400) | `bench_microgesture_openwatch__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 0.606 | 0.678 | 0.497 | 0.592 | 0.766 | 0.583 | 42 | best.pt (1600) | `bench_microgesture_openwatch__mae_5src_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.376 | 0.414 | 0.337 | 0.379 | 0.478 | 0.393 | 42 | best.pt (4800) | `vit1d_gesture_probe_openwatch_6class_focal_windowed_strat` | legacy† |
| FOCAL | finetune | 0.531 | 0.598 | 0.486 | 0.559 | 0.530 | 0.489 | 42 | best.pt (3200) | `bench_microgesture_openwatch__focal__finetune__s42` | bench |
| FOCAL (5-source) | probe | 0.481 | 0.471 | 0.369 | 0.383 | 0.486 | 0.408 | 42 | best.pt (13600) | `bench_microgesture_openwatch__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 0.393 | 0.494 | 0.395 | 0.475 | 0.645 | 0.507 | 42 | best.pt (2000) | `bench_microgesture_openwatch__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 0.432 | 0.494 | 0.416 | 0.467 | 0.548 | 0.515 | 42 | best.pt (10800) | `bench_microgesture_openwatch__focal_5src_native__probe__s42` | bench |
| SimMTM | probe | 0.532 | 0.552 | 0.447 | 0.471 | 0.607 | 0.502 | 42 | best.pt (9600) | `vit1d_gesture_probe_openwatch_6class_simmtm_windowed_strat` | legacy† |
| SimMTM | finetune | 0.579 | 0.598 | 0.502 | 0.536 | 0.589 | 0.534 | 42 | best.pt (600) | `bench_microgesture_openwatch__simmtm__finetune__s42` | bench |
| SimMTM (5-source) | probe | 0.559 | 0.575 | 0.483 | 0.506 | 0.628 | 0.512 | 42 | best.pt (7400) | `bench_microgesture_openwatch__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 0.619 | 0.690 | 0.505 | 0.598 | 0.798 | 0.635 | 42 | best.pt (3800) | `bench_microgesture_openwatch__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 0.548 | 0.575 | 0.438 | 0.475 | 0.581 | 0.492 | 42 | best.pt (7400) | `bench_microgesture_openwatch__simmtm_5src_native__probe__s42` | bench |
| SimMTM (5-source, native normalisation) | finetune | 0.573 | 0.678 | 0.478 | 0.598 | 0.747 | 0.607 | 42 | best.pt (3000) | `bench_microgesture_openwatch__simmtm_5src_native__finetune__s42` | bench |
| RelCon | probe | 0.419 | 0.425 | 0.407 | 0.421 | 0.411 | 0.386 | 42 | best.pt (16800) | `vit1d_gesture_probe_openwatch_6class_relcon_windowed_strat` | legacy† |
| RelCon | finetune | 0.618 | 0.609 | 0.542 | 0.550 | 0.580 | 0.543 | 42 | best.pt (600) | `bench_microgesture_openwatch__relcon__finetune__s42` | bench |
| RelCon (5-source) | probe | 0.385 | 0.391 | 0.399 | 0.425 | 0.509 | 0.409 | 42 | best.pt (5400) | `bench_microgesture_openwatch__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 0.427 | 0.575 | 0.383 | 0.510 | 0.628 | 0.482 | 42 | best.pt (1800) | `bench_microgesture_openwatch__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 0.315 | 0.425 | 0.282 | 0.377 | 0.457 | 0.389 | 42 | best.pt (3600) | `bench_microgesture_openwatch__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 0.617 | 0.609 | 0.498 | 0.519 | 0.590 | 0.548 | 42 | best.pt (7400) | `bench_microgesture_openwatch__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 0.573 | 0.678 | 0.491 | 0.594 | 0.802 | 0.612 | 42 | best.pt (4000) | `bench_microgesture_openwatch__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 0.547 | 0.563 | 0.456 | 0.489 | 0.586 | 0.502 | 42 | best.pt (7400) | `bench_microgesture_openwatch__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 0.583 | 0.667 | 0.531 | 0.607 | 0.688 | 0.582 | 42 | best.pt (1800) | `bench_microgesture_openwatch__sslwear_5src__finetune__s42` | bench |
| DINO (5-source) | probe | 0.485 | 0.506 | 0.405 | 0.427 | 0.533 | 0.440 | 42 | best.pt (7400) | `bench_microgesture_openwatch__dino_5src__probe__s42` | bench |
| DINO (5-source) | finetune | 0.555 | 0.690 | 0.439 | 0.559 | 0.698 | 0.505 | 42 | best.pt (2800) | `bench_microgesture_openwatch__dino_5src__finetune__s42` | bench |
| I-JEPA (5-source) | probe | 0.217 | 0.264 | 0.244 | 0.259 | 0.289 | 0.280 | 42 | best.pt (6000) | `bench_microgesture_openwatch__jepa_5src__probe__s42` | bench |
| I-JEPA (5-source) | finetune | 0.449 | 0.586 | 0.363 | 0.496 | 0.502 | 0.421 | 42 | best.pt (9800) | `bench_microgesture_openwatch__jepa_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.608 | 0.690 | 0.451 | 0.565 | 0.636 | 0.500 | 42 | best.pt (5000) | `bench_microgesture_openwatch__scratch__scratch__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.501 | 0.563 | 0.440 | 0.487 | 0.655 | 0.544 | 42 | best.pt (7400) | `bench_microgesture_openwatch__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.568 | 0.598 | 0.480 | 0.525 | 0.625 | 0.524 | 42 | best.pt (3400) | `bench_microgesture_openwatch__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.351 | 0.345 | 0.305 | 0.320 | 0.411 | 0.337 | 42 | best.pt (2000) | `bench_microgesture_openwatch__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.355 | 0.345 | 0.325 | 0.331 | 0.363 | 0.295 | 42 | best.pt (3200) | `bench_microgesture_openwatch__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.503 | 0.563 | 0.458 | 0.517 | 0.662 | 0.510 | 42 | best.pt (5400) | `bench_microgesture_openwatch__mae_recipe_r0__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 0.536 | 0.586 | 0.473 | 0.523 | 0.641 | 0.534 | 42 | best.pt (7400) | `bench_microgesture_openwatch__mae_mr40__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.394 | 0.425 | 0.385 | 0.410 | 0.521 | 0.473 | 42 | best.pt (8800) | `bench_microgesture_openwatch__mae_div60_dc__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | finetune | 0.495 | 0.517 | 0.438 | 0.466 | 0.560 | 0.477 | 42 | best.pt (600) | `bench_microgesture_openwatch__mae_div60_dc__finetune__s42` | bench |
| MAE mask-ratio 0.50 | probe | 0.501 | 0.540 | 0.472 | 0.515 | 0.621 | 0.503 | 42 | best.pt (9400) | `bench_microgesture_openwatch__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.613 | 0.632 | 0.543 | 0.567 | 0.614 | 0.558 | 42 | best.pt (6200) | `bench_microgesture_openwatch__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.587 | 0.667 | 0.557 | 0.634 | 0.708 | 0.618 | 42 | best.pt (8800) | `bench_microgesture_openwatch__mae_4src_z__finetune__s42` | bench |
| MAE mask-ratio 0.60 | probe | 0.550 | 0.575 | 0.441 | 0.498 | 0.575 | 0.524 | 42 | best.pt (7400) | `bench_microgesture_openwatch__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.472 | 0.494 | 0.414 | 0.443 | 0.521 | 0.483 | 42 | best.pt (4200) | `bench_microgesture_openwatch__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.504 | 0.517 | 0.444 | 0.473 | 0.533 | 0.487 | 42 | best.pt (1000) | `bench_microgesture_openwatch__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | probe | 0.594 | 0.632 | 0.522 | 0.548 | 0.566 | 0.532 | 42 | best.pt (9400) | `bench_microgesture_openwatch__mae_4src_nocls__probe__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | finetune | 0.607 | 0.667 | 0.538 | 0.613 | 0.666 | 0.575 | 42 | best.pt (5000) | `bench_microgesture_openwatch__mae_4src_nocls__finetune__s42` | bench |
| MAE mask-ratio 0.85 | probe | 0.541 | 0.563 | 0.459 | 0.500 | 0.620 | 0.501 | 42 | best.pt (3400) | `bench_microgesture_openwatch__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.556 | 0.575 | 0.458 | 0.490 | 0.543 | 0.486 | 42 | best.pt (7400) | `bench_microgesture_openwatch__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.543 | 0.598 | 0.492 | 0.557 | 0.638 | 0.586 | 42 | best.pt (4200) | `bench_microgesture_openwatch__mae_4src_phys__finetune__s42` | bench |
| MAE mask-ratio 0.90 | probe | 0.490 | 0.529 | 0.437 | 0.471 | 0.638 | 0.505 | 42 | best.pt (5800) | `bench_microgesture_openwatch__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.492 | 0.517 | 0.456 | 0.489 | 0.523 | 0.477 | 42 | best.pt (12200) | `bench_microgesture_openwatch__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.543 | 0.563 | 0.479 | 0.506 | 0.611 | 0.528 | 42 | best.pt (400) | `bench_microgesture_openwatch__mae_4src_physnp__finetune__s42` | bench |
| MAE frame-canonical (no rotation) | probe | 0.459 | 0.506 | 0.442 | 0.498 | 0.636 | 0.542 | 42 | best.pt (13600) | `bench_microgesture_openwatch__mae_frame_canon__probe__s42` | bench |
| SimMTM (4-source pool) | probe | 0.522 | 0.540 | 0.455 | 0.473 | 0.548 | 0.503 | 42 | best.pt (3800) | `bench_microgesture_openwatch__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.551 | 0.632 | 0.467 | 0.554 | 0.653 | 0.591 | 42 | best.pt (1200) | `bench_microgesture_openwatch__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.369 | 0.437 | 0.339 | 0.406 | 0.483 | 0.426 | 42 | best.pt (6000) | `bench_microgesture_openwatch__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.482 | 0.540 | 0.398 | 0.450 | 0.573 | 0.515 | 42 | best.pt (1000) | `bench_microgesture_openwatch__focal_4src__finetune__s42` | bench |
| RelCon (4-source pool) | probe | 0.553 | 0.586 | 0.506 | 0.540 | 0.541 | 0.525 | 42 | best.pt (4800) | `bench_microgesture_openwatch__relcon_4src__probe__s42` | bench |
| RelCon (4-source pool) | finetune | 0.548 | 0.644 | 0.481 | 0.575 | 0.677 | 0.570 | 42 | best.pt (5400) | `bench_microgesture_openwatch__relcon_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.521 | 0.529 | 0.512 | 0.510 | 0.572 | 0.554 | 42 | best.pt (8800) | `bench_microgesture_openwatch__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.496 | 0.621 | 0.506 | 0.609 | 0.639 | 0.553 | 42 | best.pt (3000) | `bench_microgesture_openwatch__acconly_4src__finetune__s42` | bench |
| *Mix-Token (reimpl.)* | external | 0.500 | 0.621 | 0.490 | 0.592 | — | — | — |  | — | harvest:results_openwatch_mixtoken_noaug |
| *NormWear-Base (first-party)* | external | 0.453 | 0.517 | 0.425 | 0.504 | — | — | — |  | — | harvest:results_openwatch_normwear_proto |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.517 | 0.529 | 0.485 | 0.496 | 0.542 | 0.486 | 42 | best.pt (12000) | `bench_microgesture_openwatch__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.270 | 0.299 | 0.262 | 0.287 | 0.239 | 0.254 | 42 | best.pt (15600) | `bench_microgesture_openwatch__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.488 | 0.517 | 0.472 | 0.506 | 0.599 | 0.517 | 42 | best.pt (3000) | `bench_microgesture_openwatch__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.154 | 0.356 | 0.141 | 0.299 | 0.133 | 0.152 | 42 | best.pt (8400) | `bench_microgesture_openwatch__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 0.557 | 0.575 | 0.482 | 0.506 | 0.568 | 0.522 | 42 | best.pt (1600) | `bench_microgesture_openwatch__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 0.272 | 0.310 | 0.261 | 0.284 | 0.313 | 0.298 | 42 | best.pt (8000) | `bench_microgesture_openwatch__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 0.508 | 0.575 | 0.474 | 0.517 | 0.604 | 0.535 | 42 | best.pt (7000) | `bench_microgesture_openwatch__inertia1_art_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 0.126 | 0.172 | 0.141 | 0.174 | 0.176 | 0.192 | 42 | best.pt (8600) | `bench_microgesture_openwatch__inertia1_art_large__finetune__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 0.215 | 0.310 | 0.203 | 0.291 | 0.322 | 0.258 | 42 | best.pt (6600) | `bench_microgesture_openwatch__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 0.528 | 0.632 | 0.450 | 0.565 | 0.609 | 0.545 | 42 | best.pt (5800) | `bench_microgesture_openwatch__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 0.438 | 0.425 | 0.388 | 0.418 | 0.453 | 0.409 | 42 | best.pt (5400) | `bench_microgesture_openwatch__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 0.210 | 0.195 | 0.222 | 0.215 | 0.234 | 0.229 | 42 | best.pt (14600) | `bench_microgesture_openwatch__normwear_base__finetune__s42` | bench |
| VolDiv single-source, narrow motion band, N=12,500 | probe | 0.584 | 0.609 | 0.534 | 0.556 | 0.600 | 0.545 | 42 | best.pt (8800) | `bench_microgesture_openwatch__vd_nar_n12k__probe__s42` | bench |
| VolDiv single-source, narrow motion band, N=50,000 | probe | 0.550 | 0.575 | 0.496 | 0.525 | 0.579 | 0.543 | 42 | best.pt (6200) | `bench_microgesture_openwatch__vd_nar_n50k__probe__s42` | bench |
| VolDiv single-source, narrow motion band, N=200,000 | probe | 0.577 | 0.598 | 0.500 | 0.529 | 0.576 | 0.536 | 42 | best.pt (13600) | `bench_microgesture_openwatch__vd_nar_n200k__probe__s42` | bench |
| VolDiv single-source, narrow motion band, N=800,000 | probe | 0.555 | 0.586 | 0.523 | 0.561 | 0.588 | 0.555 | 42 | best.pt (12200) | `bench_microgesture_openwatch__vd_nar_n800k__probe__s42` | bench |
| VolDiv single-source, uniform random, N=12,500 | probe | 0.588 | 0.586 | 0.531 | 0.550 | 0.636 | 0.595 | 42 | best.pt (8800) | `bench_microgesture_openwatch__vd_one_n12k__probe__s42` | bench |
| VolDiv single-source, uniform random, N=50,000 | probe | 0.569 | 0.575 | 0.515 | 0.538 | 0.575 | 0.546 | 42 | best.pt (12000) | `bench_microgesture_openwatch__vd_one_n50k__probe__s42` | bench |
| VolDiv single-source, uniform random, N=200,000 | probe | 0.542 | 0.563 | 0.514 | 0.540 | 0.579 | 0.551 | 42 | best.pt (11600) | `bench_microgesture_openwatch__vd_one_n200k__probe__s42` | bench |
| VolDiv single-source, uniform random, N=800,000 | probe | 0.514 | 0.540 | 0.491 | 0.511 | 0.588 | 0.541 | 42 | best.pt (6200) | `bench_microgesture_openwatch__vd_one_n800k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=12,500 | probe | 0.568 | 0.586 | 0.490 | 0.527 | 0.584 | 0.527 | 42 | best.pt (5400) | `bench_microgesture_openwatch__vd_nat_n12k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=50,000 | probe | 0.522 | 0.552 | 0.489 | 0.523 | 0.561 | 0.530 | 42 | best.pt (4600) | `bench_microgesture_openwatch__vd_nat_n50k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=200,000 | probe | 0.549 | 0.563 | 0.479 | 0.506 | 0.635 | 0.553 | 42 | best.pt (13800) | `bench_microgesture_openwatch__vd_nat_n200k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=800,000 | probe | 0.531 | 0.552 | 0.486 | 0.513 | 0.545 | 0.521 | 42 | best.pt (6400) | `bench_microgesture_openwatch__vd_nat_n800k__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=12,500 | probe | 0.516 | 0.540 | 0.468 | 0.498 | 0.507 | 0.503 | 42 | best.pt (9200) | `bench_microgesture_openwatch__vd_bal_n12k__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=50,000 | probe | 0.497 | 0.517 | 0.480 | 0.498 | 0.527 | 0.508 | 42 | best.pt (8800) | `bench_microgesture_openwatch__vd_bal_n50k__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=200,000 | probe | 0.510 | 0.529 | 0.457 | 0.479 | 0.582 | 0.515 | 42 | best.pt (12200) | `bench_microgesture_openwatch__vd_bal_n200k__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=800,000 | probe | 0.558 | 0.575 | 0.480 | 0.506 | 0.568 | 0.555 | 42 | best.pt (5200) | `bench_microgesture_openwatch__vd_bal_n800k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=1,738,094 | probe | 0.490 | 0.540 | 0.490 | 0.534 | 0.608 | 0.556 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_nat_nfull__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=1,738,094 | probe | 0.564 | 0.575 | 0.489 | 0.511 | 0.568 | 0.534 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_bal_nfull__probe__s42` | bench |
| VolDiv src1, N=50,000 | probe | 0.454 | 0.471 | 0.469 | 0.492 | 0.567 | 0.499 | 42 | best.pt (6800) | `bench_microgesture_openwatch__vd_src1_n50k__probe__s42` | bench |
| VolDiv src2, N=50,000 | probe | 0.503 | 0.540 | 0.499 | 0.527 | 0.568 | 0.541 | 42 | best.pt (13600) | `bench_microgesture_openwatch__vd_src2_n50k__probe__s42` | bench |
| VolDiv src3, N=50,000 | probe | 0.590 | 0.586 | 0.504 | 0.521 | 0.578 | 0.549 | 42 | best.pt (10200) | `bench_microgesture_openwatch__vd_src3_n50k__probe__s42` | bench |
| VolDiv src4, N=50,000 | probe | 0.473 | 0.517 | 0.455 | 0.500 | 0.581 | 0.534 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_src4_n50k__probe__s42` | bench |
| VolDiv ocdetect only, 3 T-phys type(s), water-filled, N=12,500 | probe | 0.542 | 0.563 | 0.506 | 0.534 | 0.593 | 0.516 | 42 | best.pt (4800) | `bench_microgesture_openwatch__vd_typ3_n12k__probe__s42` | bench |
| VolDiv ocdetect only, 30 T-phys type(s), water-filled, N=12,500 | probe | 0.533 | 0.575 | 0.489 | 0.519 | 0.577 | 0.550 | 42 | best.pt (5400) | `bench_microgesture_openwatch__vd_typ30_n12k__probe__s42` | bench |
| VolDiv ocdetect only, 300 T-phys type(s), water-filled, N=12,500 | probe | 0.557 | 0.575 | 0.497 | 0.525 | 0.556 | 0.550 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_typ300_n12k__probe__s42` | bench |
| VolDiv ocdetect only, every T-phys type(s), water-filled, N=12,500 | probe | 0.523 | 0.552 | 0.494 | 0.527 | 0.555 | 0.547 | 42 | best.pt (13600) | `bench_microgesture_openwatch__vd_typall_n12k__probe__s42` | bench |
| VolDiv ocdetect only, 1 subject(s), type-matched, N=50,000 | probe | 0.528 | 0.552 | 0.482 | 0.511 | 0.593 | 0.527 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_sub1_n50k__probe__s42` | bench |
| VolDiv ocdetect only, 3 subject(s), type-matched, N=50,000 | probe | 0.538 | 0.563 | 0.493 | 0.517 | 0.553 | 0.515 | 42 | best.pt (12200) | `bench_microgesture_openwatch__vd_sub3_n50k__probe__s42` | bench |
| VolDiv ocdetect only, 10 subject(s), type-matched, N=50,000 | probe | 0.498 | 0.529 | 0.474 | 0.500 | 0.610 | 0.540 | 42 | best.pt (4800) | `bench_microgesture_openwatch__vd_sub10_n50k__probe__s42` | bench |
| VolDiv ocdetect only, every subject(s), type-matched, N=50,000 | probe | 0.516 | 0.540 | 0.495 | 0.533 | 0.585 | 0.538 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_suball_n50k__probe__s42` | bench |
| VolDiv MAE-space types, 3 T-enc type(s), ocdetect only, N=12,500 | probe | 0.585 | 0.598 | 0.535 | 0.554 | 0.615 | 0.553 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_etyp3_n12k__probe__s42` | bench |
| VolDiv MAE-space types, 30 T-enc type(s), ocdetect only, N=12,500 | probe | 0.531 | 0.552 | 0.479 | 0.498 | 0.589 | 0.532 | 42 | best.pt (6600) | `bench_microgesture_openwatch__vd_etyp30_n12k__probe__s42` | bench |
| VolDiv MAE-space types, 300 T-enc type(s), ocdetect only, N=12,500 | probe | 0.564 | 0.586 | 0.519 | 0.536 | 0.600 | 0.566 | 42 | best.pt (9600) | `bench_microgesture_openwatch__vd_etyp300_n12k__probe__s42` | bench |
| VolDiv MAE-space types, every T-enc type(s), ocdetect only, N=12,500 | probe | 0.526 | 0.552 | 0.488 | 0.515 | 0.548 | 0.522 | 42 | best.pt (10000) | `bench_microgesture_openwatch__vd_etypall_n12k__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 | probe | 0.428 | 0.437 | 0.404 | 0.429 | 0.512 | 0.471 | 42 | best.pt (8800) | `bench_microgesture_openwatch__vd_ek1_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 | probe | 0.561 | 0.598 | 0.520 | 0.567 | 0.593 | 0.511 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_ek1_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 | probe | 0.467 | 0.483 | 0.438 | 0.452 | 0.546 | 0.485 | 42 | best.pt (8800) | `bench_microgesture_openwatch__vd_ek5_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 | probe | 0.530 | 0.540 | 0.485 | 0.498 | 0.635 | 0.565 | 42 | best.pt (12400) | `bench_microgesture_openwatch__vd_ek5_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 | probe | 0.505 | 0.540 | 0.477 | 0.504 | 0.587 | 0.509 | 42 | best.pt (13400) | `bench_microgesture_openwatch__vd_ek5_f1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 | probe | 0.437 | 0.460 | 0.416 | 0.439 | 0.501 | 0.451 | 42 | best.pt (9400) | `bench_microgesture_openwatch__vd_ek10_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 | probe | 0.545 | 0.598 | 0.483 | 0.529 | 0.600 | 0.542 | 42 | best.pt (6000) | `bench_microgesture_openwatch__vd_ek10_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 | probe | 0.489 | 0.494 | 0.443 | 0.464 | 0.608 | 0.540 | 42 | best.pt (1800) | `bench_microgesture_openwatch__vd_ek10_f1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 | probe | 0.491 | 0.517 | 0.452 | 0.477 | 0.501 | 0.466 | 42 | best.pt (13400) | `bench_microgesture_openwatch__vd_ek25_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 | probe | 0.622 | 0.632 | 0.532 | 0.552 | 0.597 | 0.554 | 42 | best.pt (9000) | `bench_microgesture_openwatch__vd_ek25_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 | probe | 0.486 | 0.529 | 0.481 | 0.515 | 0.594 | 0.540 | 42 | best.pt (4600) | `bench_microgesture_openwatch__vd_ek25_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 | probe | 0.494 | 0.506 | 0.439 | 0.458 | 0.555 | 0.497 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_ek50_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 | probe | 0.578 | 0.586 | 0.531 | 0.548 | 0.592 | 0.533 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_ek50_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 | probe | 0.544 | 0.563 | 0.530 | 0.550 | 0.559 | 0.537 | 42 | best.pt (9400) | `bench_microgesture_openwatch__vd_ek50_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 | probe | 0.570 | 0.598 | 0.506 | 0.540 | 0.598 | 0.531 | 42 | best.pt (12200) | `bench_microgesture_openwatch__vd_ek50_f10__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 | probe | 0.473 | 0.506 | 0.406 | 0.443 | 0.507 | 0.477 | 42 | best.pt (5400) | `bench_microgesture_openwatch__vd_ek100_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 | probe | 0.521 | 0.540 | 0.475 | 0.498 | 0.559 | 0.515 | 42 | best.pt (1400) | `bench_microgesture_openwatch__vd_ek100_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 | probe | 0.501 | 0.529 | 0.477 | 0.508 | 0.531 | 0.507 | 42 | best.pt (10400) | `bench_microgesture_openwatch__vd_ek100_f1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 | probe | 0.508 | 0.552 | 0.486 | 0.529 | 0.575 | 0.532 | 42 | best.pt (5400) | `bench_microgesture_openwatch__vd_ek100_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 | probe | 0.474 | 0.494 | 0.408 | 0.437 | 0.554 | 0.487 | 42 | best.pt (2600) | `bench_microgesture_openwatch__vd_ek1000_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 | probe | 0.551 | 0.552 | 0.501 | 0.513 | 0.568 | 0.547 | 42 | best.pt (9200) | `bench_microgesture_openwatch__vd_ek1000_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 | probe | 0.541 | 0.563 | 0.479 | 0.502 | 0.595 | 0.535 | 42 | best.pt (6800) | `bench_microgesture_openwatch__vd_ek1000_f1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 | probe | 0.542 | 0.575 | 0.476 | 0.515 | 0.585 | 0.552 | 42 | best.pt (12200) | `bench_microgesture_openwatch__vd_ek1000_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 | probe | 0.497 | 0.540 | 0.460 | 0.502 | 0.573 | 0.528 | 42 | best.pt (12000) | `bench_microgesture_openwatch__vd_ek1000_f100__probe__s42` | bench |
| VolDiv 4-source pool, 1% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 0.529 | 0.552 | 0.479 | 0.504 | 0.564 | 0.514 | 42 | best.pt (3800) | `bench_microgesture_openwatch__vd_pk1_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 1% of T-phys types (random, source-stratified, gyro-unit fixed), N=16,113 | probe | 0.565 | 0.575 | 0.496 | 0.510 | 0.464 | 0.480 | 42 | best.pt (9400) | `bench_microgesture_openwatch__vd_pk1_n1__probe__s42` | bench |
| VolDiv 4-source pool, 10% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 0.527 | 0.552 | 0.500 | 0.534 | 0.602 | 0.532 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_pk10_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 10% of T-phys types (random, source-stratified, gyro-unit fixed), N=17,381 | probe | 0.500 | 0.540 | 0.478 | 0.511 | 0.604 | 0.541 | 42 | best.pt (4600) | `bench_microgesture_openwatch__vd_pk10_n1__probe__s42` | bench |
| VolDiv 4-source pool, 10% of T-phys types (random, source-stratified, gyro-unit fixed), N=173,809 | probe | 0.547 | 0.552 | 0.504 | 0.527 | 0.602 | 0.534 | 42 | best.pt (16200) | `bench_microgesture_openwatch__vd_pk10_n10__probe__s42` | bench |
| VolDiv 4-source pool, 50% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 0.604 | 0.621 | 0.540 | 0.571 | 0.564 | 0.537 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_pk50_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 50% of T-phys types (random, source-stratified, gyro-unit fixed), N=17,381 | probe | 0.566 | 0.586 | 0.488 | 0.521 | 0.559 | 0.532 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_pk50_n1__probe__s42` | bench |
| VolDiv 4-source pool, 50% of T-phys types (random, source-stratified, gyro-unit fixed), N=173,809 | probe | 0.507 | 0.540 | 0.489 | 0.534 | 0.575 | 0.543 | 42 | best.pt (10000) | `bench_microgesture_openwatch__vd_pk50_n10__probe__s42` | bench |
| VolDiv 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 0.537 | 0.552 | 0.495 | 0.508 | 0.581 | 0.535 | 42 | best.pt (4400) | `bench_microgesture_openwatch__vd_pk100_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=17,381 | probe | 0.541 | 0.563 | 0.503 | 0.527 | 0.600 | 0.534 | 42 | best.pt (10000) | `bench_microgesture_openwatch__vd_pk100_n1__probe__s42` | bench |
| VolDiv 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=173,809 | probe | 0.549 | 0.575 | 0.507 | 0.533 | 0.552 | 0.513 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_pk100_n10__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.574 | 0.609 | 0.451 | 0.490 | 0.519 | 0.455 | 42 | best.pt (3400) | `bench_microgesture_openwatch__vd_ek1_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.614 | 0.644 | 0.462 | 0.513 | 0.556 | 0.491 | 42 | best.pt (8000) | `bench_microgesture_openwatch__vd_ek1_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.583 | 0.609 | 0.445 | 0.492 | 0.613 | 0.493 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_ek5_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.568 | 0.586 | 0.471 | 0.508 | 0.640 | 0.506 | 42 | best.pt (3400) | `bench_microgesture_openwatch__vd_ek5_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.601 | 0.632 | 0.489 | 0.542 | 0.597 | 0.497 | 42 | best.pt (3400) | `bench_microgesture_openwatch__vd_ek5_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.583 | 0.632 | 0.496 | 0.550 | 0.534 | 0.453 | 42 | best.pt (5400) | `bench_microgesture_openwatch__vd_ek10_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.579 | 0.644 | 0.466 | 0.525 | 0.639 | 0.527 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_ek10_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.548 | 0.563 | 0.494 | 0.534 | 0.642 | 0.508 | 42 | best.pt (8400) | `bench_microgesture_openwatch__vd_ek10_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.522 | 0.540 | 0.457 | 0.496 | 0.568 | 0.454 | 42 | best.pt (1400) | `bench_microgesture_openwatch__vd_ek25_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.547 | 0.586 | 0.461 | 0.511 | 0.603 | 0.526 | 42 | best.pt (2000) | `bench_microgesture_openwatch__vd_ek25_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.588 | 0.598 | 0.484 | 0.511 | 0.632 | 0.535 | 42 | best.pt (6400) | `bench_microgesture_openwatch__vd_ek25_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.600 | 0.632 | 0.453 | 0.494 | 0.641 | 0.499 | 42 | best.pt (8400) | `bench_microgesture_openwatch__vd_ek50_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.596 | 0.621 | 0.472 | 0.517 | 0.654 | 0.536 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_ek50_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.620 | 0.644 | 0.476 | 0.523 | 0.575 | 0.521 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_ek50_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.563 | 0.621 | 0.487 | 0.542 | 0.623 | 0.507 | 42 | best.pt (5800) | `bench_microgesture_openwatch__vd_ek50_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.547 | 0.563 | 0.458 | 0.489 | 0.645 | 0.491 | 42 | best.pt (8400) | `bench_microgesture_openwatch__vd_ek100_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.621 | 0.655 | 0.499 | 0.550 | 0.612 | 0.535 | 42 | best.pt (11800) | `bench_microgesture_openwatch__vd_ek100_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.628 | 0.655 | 0.494 | 0.540 | 0.642 | 0.526 | 42 | best.pt (6000) | `bench_microgesture_openwatch__vd_ek100_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.568 | 0.632 | 0.469 | 0.529 | 0.611 | 0.534 | 42 | best.pt (13600) | `bench_microgesture_openwatch__vd_ek100_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.552 | 0.586 | 0.440 | 0.487 | 0.558 | 0.468 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_ek1000_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.615 | 0.655 | 0.558 | 0.594 | 0.694 | 0.543 | 42 | best.pt (7400) | `bench_microgesture_openwatch__vd_ek1000_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.512 | 0.563 | 0.467 | 0.515 | 0.619 | 0.523 | 42 | best.pt (6000) | `bench_microgesture_openwatch__vd_ek1000_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.591 | 0.609 | 0.496 | 0.534 | 0.662 | 0.521 | 42 | best.pt (6000) | `bench_microgesture_openwatch__vd_ek1000_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 (z-score probe input) | probe | 0.482 | 0.529 | 0.439 | 0.479 | 0.682 | 0.557 | 42 | best.pt (8400) | `bench_microgesture_openwatch__vd_ek1000_f100_zs__probe__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.01% of pool, N=196 | probe | 0.298 | 0.333 | 0.282 | 0.322 | 0.425 | 0.348 | 42 | best.pt (5800) | `bench_microgesture_openwatch__g5_k1_f0p01__probe__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.303 | 0.333 | 0.283 | 0.310 | 0.375 | 0.353 | 42 | best.pt (5800) | `bench_microgesture_openwatch__g5_k1_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.01% of pool, N=196 | probe | 0.564 | 0.563 | 0.445 | 0.462 | 0.573 | 0.441 | 42 | best.pt (8400) | `bench_microgesture_openwatch__g5_k5_f0p01__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.539 | 0.552 | 0.415 | 0.450 | 0.585 | 0.464 | 42 | best.pt (7400) | `bench_microgesture_openwatch__g5_k5_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 1% of pool, N=19,593 | probe | 0.541 | 0.563 | 0.447 | 0.475 | 0.542 | 0.464 | 42 | best.pt (12000) | `bench_microgesture_openwatch__g5_k5_f1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.01% of pool, N=196 | probe | 0.587 | 0.598 | 0.432 | 0.464 | 0.560 | 0.497 | 42 | best.pt (5400) | `bench_microgesture_openwatch__g5_k10_f0p01__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.599 | 0.609 | 0.504 | 0.529 | 0.702 | 0.568 | 42 | best.pt (6000) | `bench_microgesture_openwatch__g5_k10_f0p1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 1% of pool, N=19,593 | probe | 0.623 | 0.655 | 0.495 | 0.531 | 0.627 | 0.504 | 42 | best.pt (13600) | `bench_microgesture_openwatch__g5_k10_f1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.01% of pool, N=196 | probe | 0.542 | 0.563 | 0.443 | 0.487 | 0.566 | 0.478 | 42 | best.pt (8200) | `bench_microgesture_openwatch__g5_k25_f0p01__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.484 | 0.529 | 0.422 | 0.458 | 0.589 | 0.497 | 42 | best.pt (9600) | `bench_microgesture_openwatch__g5_k25_f0p1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 1% of pool, N=19,593 | probe | 0.586 | 0.609 | 0.473 | 0.519 | 0.576 | 0.501 | 42 | best.pt (9600) | `bench_microgesture_openwatch__g5_k25_f1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.01% of pool, N=196 | probe | 0.548 | 0.586 | 0.431 | 0.481 | 0.589 | 0.465 | 42 | best.pt (5800) | `bench_microgesture_openwatch__g5_k50_f0p01__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.534 | 0.586 | 0.500 | 0.548 | 0.603 | 0.490 | 42 | best.pt (7400) | `bench_microgesture_openwatch__g5_k50_f0p1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 1% of pool, N=19,593 | probe | 0.535 | 0.575 | 0.460 | 0.517 | 0.622 | 0.531 | 42 | best.pt (9600) | `bench_microgesture_openwatch__g5_k50_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.01% of pool, N=196 | probe | 0.600 | 0.609 | 0.466 | 0.498 | 0.535 | 0.456 | 42 | best.pt (2600) | `bench_microgesture_openwatch__g5_k100_f0p01__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.614 | 0.632 | 0.507 | 0.550 | 0.600 | 0.520 | 42 | best.pt (7400) | `bench_microgesture_openwatch__g5_k100_f0p1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 1% of pool, N=19,593 | probe | 0.507 | 0.552 | 0.465 | 0.527 | 0.588 | 0.517 | 42 | best.pt (8400) | `bench_microgesture_openwatch__g5_k100_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 10% of pool, N=195,929 | probe | 0.536 | 0.586 | 0.467 | 0.515 | 0.560 | 0.504 | 42 | best.pt (9600) | `bench_microgesture_openwatch__g5_k100_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.595 | 0.598 | 0.482 | 0.517 | 0.682 | 0.540 | 42 | best.pt (7400) | `bench_microgesture_openwatch__g5_k1000_f0p1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 0.558 | 0.575 | 0.480 | 0.510 | 0.564 | 0.494 | 42 | best.pt (5400) | `bench_microgesture_openwatch__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 0.573 | 0.609 | 0.477 | 0.511 | 0.597 | 0.528 | 42 | best.pt (5800) | `bench_microgesture_openwatch__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 0.586 | 0.609 | 0.467 | 0.513 | 0.636 | 0.533 | 42 | best.pt (8400) | `bench_microgesture_openwatch__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 0.626 | 0.667 | 0.503 | 0.556 | 0.649 | 0.565 | 42 | best.pt (6000) | `bench_microgesture_openwatch__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 0.551 | 0.575 | 0.475 | 0.519 | 0.660 | 0.546 | 42 | best.pt (6000) | `bench_microgesture_openwatch__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 0.585 | 0.609 | 0.457 | 0.500 | 0.702 | 0.543 | 42 | best.pt (8400) | `bench_microgesture_openwatch__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 0.526 | 0.540 | 0.463 | 0.490 | 0.648 | 0.539 | 42 | best.pt (8400) | `bench_microgesture_openwatch__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 0.524 | 0.552 | 0.447 | 0.498 | 0.665 | 0.561 | 42 | best.pt (7400) | `bench_microgesture_openwatch__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.539 | 0.552 | 0.485 | 0.511 | 0.629 | 0.528 | 42 | best.pt (1400) | `bench_microgesture_openwatch__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.526 | 0.563 | 0.486 | 0.525 | 0.654 | 0.558 | 42 | best.pt (9600) | `bench_microgesture_openwatch__abl_so3__probe__s42` | bench |
| Design ablation 'fixedlen' (fixed 10.24 s windows, no variable-length crop; otherwise the corpus 'ref' arm) | probe | 0.567 | 0.586 | 0.471 | 0.504 | 0.564 | 0.496 | 42 | best.pt (1400) | `bench_microgesture_openwatch__abl_fixedlen__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | probe | 0.614 | 0.644 | 0.545 | 0.575 | 0.511 | 0.529 | 42 | best.pt (4600) | `bench_microgesture_openwatch__mae_5src_physin__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.364 | 0.402 | 0.319 | 0.354 | 0.398 | 0.354 | 42 | best.pt (5400) | `bench_microgesture_openwatch__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.114 | 0.517 | 0.114 | 0.517 | 0.121 | 0.121 | 42 | best.pt (1400) | `bench_microgesture_openwatch__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | probe | 0.595 | 0.632 | 0.510 | 0.559 | 0.671 | 0.538 | 42 | best.pt (4800) | `bench_microgesture_openwatch__tx5_prism__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | finetune | 0.499 | 0.655 | 0.472 | 0.605 | 0.695 | 0.601 | 42 | best.pt (1200) | `bench_microgesture_openwatch__tx5_prism__finetune__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 0.499 | 0.529 | 0.426 | 0.462 | 0.565 | 0.478 | 42 | best.pt (2600) | `bench_microgesture_openwatch__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | finetune | 0.598 | 0.690 | 0.556 | 0.655 | 0.694 | 0.612 | 42 | best.pt (2600) | `bench_microgesture_openwatch__tx5_fullbody__finetune__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 0.595 | 0.609 | 0.476 | 0.506 | 0.560 | 0.486 | 42 | best.pt (12000) | `bench_microgesture_openwatch__tx5_odom__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | finetune | 0.564 | 0.667 | 0.510 | 0.623 | 0.720 | 0.598 | 42 | best.pt (1200) | `bench_microgesture_openwatch__tx5_odom__finetune__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | probe | 0.605 | 0.598 | 0.459 | 0.479 | 0.657 | 0.481 | 42 | best.pt (1200) | `bench_microgesture_openwatch__tx5_pd__probe__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | finetune | 0.498 | 0.609 | 0.459 | 0.580 | 0.715 | 0.580 | 42 | best.pt (1800) | `bench_microgesture_openwatch__tx5_pd__finetune__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | probe | 0.589 | 0.609 | 0.479 | 0.504 | 0.677 | 0.541 | 42 | best.pt (600) | `bench_microgesture_openwatch__tx5_language__probe__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | finetune | 0.595 | 0.701 | 0.535 | 0.638 | 0.728 | 0.608 | 42 | best.pt (1200) | `bench_microgesture_openwatch__tx5_language__finetune__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | probe | 0.329 | 0.379 | 0.318 | 0.354 | 0.432 | 0.384 | 42 | best.pt (1400) | `bench_microgesture_openwatch__txs_prism__probe__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | finetune | 0.335 | 0.540 | 0.326 | 0.485 | 0.589 | 0.454 | 42 | best.pt (3000) | `bench_microgesture_openwatch__txs_prism__finetune__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | probe | 0.416 | 0.448 | 0.370 | 0.408 | 0.478 | 0.383 | 42 | best.pt (1200) | `bench_microgesture_openwatch__txs_fullbody__probe__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | finetune | 0.360 | 0.563 | 0.369 | 0.506 | 0.605 | 0.448 | 42 | best.pt (600) | `bench_microgesture_openwatch__txs_fullbody__finetune__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | probe | 0.434 | 0.506 | 0.392 | 0.444 | 0.585 | 0.468 | 42 | best.pt (7400) | `bench_microgesture_openwatch__txs_odom__probe__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | finetune | 0.475 | 0.609 | 0.427 | 0.552 | 0.620 | 0.502 | 42 | best.pt (600) | `bench_microgesture_openwatch__txs_odom__finetune__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | probe | 0.392 | 0.471 | 0.348 | 0.398 | 0.522 | 0.394 | 42 | best.pt (7400) | `bench_microgesture_openwatch__txs_pd__probe__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | finetune | 0.401 | 0.586 | 0.357 | 0.494 | 0.568 | 0.432 | 42 | best.pt (1600) | `bench_microgesture_openwatch__txs_pd__finetune__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | probe | 0.460 | 0.494 | 0.393 | 0.421 | 0.496 | 0.410 | 42 | best.pt (2400) | `bench_microgesture_openwatch__txs_language__probe__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | finetune | 0.486 | 0.609 | 0.397 | 0.546 | 0.627 | 0.511 | 42 | best.pt (4400) | `bench_microgesture_openwatch__txs_language__finetune__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | probe | 0.446 | 0.483 | 0.348 | 0.397 | 0.442 | 0.399 | 42 | best.pt (800) | `bench_microgesture_openwatch__txs_armpose__probe__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | finetune | 0.352 | 0.552 | 0.347 | 0.525 | 0.638 | 0.504 | 42 | best.pt (800) | `bench_microgesture_openwatch__txs_armpose__finetune__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 0.650 | 0.690 | 0.523 | 0.563 | 0.629 | 0.533 | 42 | best.pt (7400) | `bench_microgesture_openwatch__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 0.659 | 0.736 | 0.611 | 0.688 | 0.815 | 0.694 | 42 | best.pt (3400) | `bench_microgesture_openwatch__mt_body_gesture__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | finetune | 0.650 | 0.724 | 0.607 | 0.682 | 0.815 | 0.693 | 42 | best.pt (22000) | `mt_body_gesture__microgesture_openwatch` | bench |

## Language alignment — `language_humanml3d`

_HumanML3D synthetic wrist IMU (UniMTS_data) ↔ text retrieval._ Scorer: `bench.score.language`. Selection: {'split': 'none', 'monitor': 'fixed_budget', 'mode': 'latest', 'note': 'fixed 8k-step budget, final checkpoint (latest.pt); §13e: val-loss best.pt under-selects for retrieval, latest ≥ best on every arm'}. Reported split: **test**.
Primary metric **R@1 G=100 (mean i2t/t2i)**: mean of IMU→Text and Text→IMU Recall@1 over 200 random 100-item galleries, first-caption protocol.

| Encoder | Protocol | **R@1 G=100 (mean i2t/t2i)** | i2t R@1 | t2i R@1 | i2t MRR | t2i MRR | i2t R@1 full | t2i R@1 full | i2t MRR full | t2i MRR full | R@1 G=100 (val) | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) ⚠ eval-overlap pool | probe | 27.1 | 27.6 | 26.6 | 42.1 | 41.1 | 3.3 | 2.9 | 8.5 | 7.5 | 27.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_11src__probe__s42` | bench |
| Ours (MAE, 11-source) ⚠ eval-overlap pool | finetune | 38.8 | 39.3 | 38.3 | 54.9 | 53.9 | 5.3 | 4.1 | 12.9 | 10.8 | 40.7 | 42 | latest.pt (8000) | `rtune_lr1e-4_tau0.1fixed` | legacy† |
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 27.9 | 28.3 | 27.5 | 42.5 | 41.8 | 3.6 | 3.4 | 9.0 | 8.2 | 27.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 38.8 | 39.5 | 38.1 | 54.9 | 53.9 | 5.1 | 4.4 | 12.6 | 11.0 | 39.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 28.6 | 29.5 | 27.7 | 43.7 | 42.2 | 3.4 | 2.5 | 8.9 | 7.4 | 27.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_noaltattn__probe__s42` | bench |
| Ours (MAE, no alt-attn) | finetune | 39.4 | 39.8 | 38.9 | 55.2 | 54.2 | 5.3 | 4.4 | 13.2 | 11.6 | 39.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 26.3 | 26.7 | 26.0 | 40.4 | 39.7 | 3.4 | 2.4 | 8.3 | 7.0 | 24.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 37.0 | 37.7 | 36.3 | 53.0 | 51.8 | 4.6 | 4.1 | 11.7 | 10.6 | 38.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_5src_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 19.3 | 19.8 | 18.7 | 32.8 | 31.9 | 2.0 | 2.0 | 5.6 | 5.3 | 18.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_acconly__probe__s42` | bench |
| Ours (MAE, acc-only) | finetune | 29.7 | 30.0 | 29.4 | 45.2 | 44.6 | 3.1 | 3.3 | 8.7 | 8.2 | 30.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_acconly__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 20.7 | 21.0 | 20.3 | 34.4 | 33.7 | 2.1 | 1.7 | 6.1 | 5.4 | 20.1 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 31.5 | 31.7 | 31.3 | 46.9 | 46.3 | 3.6 | 3.3 | 9.6 | 8.8 | 32.1 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_5src_acconly__finetune__s42` | bench |
| FOCAL | probe | 8.3 | 8.6 | 8.0 | 19.2 | 18.2 | 0.5 | 0.3 | 2.2 | 1.8 | 7.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__focal__probe__s42` | bench |
| FOCAL | finetune | 23.5 | 23.6 | 23.3 | 38.2 | 37.8 | 1.8 | 1.9 | 6.1 | 6.1 | 22.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__focal__finetune__s42` | bench |
| FOCAL (5-source) | probe | 8.6 | 9.1 | 8.0 | 19.9 | 18.6 | 0.7 | 0.7 | 2.3 | 2.2 | 8.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 22.4 | 22.7 | 22.0 | 37.2 | 36.5 | 2.1 | 1.9 | 6.2 | 5.7 | 21.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 10.0 | 10.6 | 9.5 | 22.1 | 20.8 | 0.8 | 0.5 | 2.8 | 2.2 | 10.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__focal_5src_native__probe__s42` | bench |
| SimMTM | probe | 16.7 | 17.3 | 16.1 | 29.8 | 28.6 | 1.7 | 1.2 | 4.9 | 4.0 | 17.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm__probe__s42` | bench |
| SimMTM | finetune | 31.3 | 32.0 | 30.6 | 47.4 | 46.0 | 3.9 | 3.1 | 9.9 | 8.6 | 32.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm__finetune__s42` | bench |
| SimMTM (5-source) | probe | 17.2 | 17.4 | 17.0 | 30.3 | 29.8 | 1.7 | 1.2 | 5.0 | 4.2 | 17.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 32.4 | 32.7 | 32.1 | 48.1 | 47.4 | 4.0 | 3.4 | 10.3 | 9.0 | 34.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 14.7 | 15.5 | 14.0 | 27.6 | 26.2 | 1.2 | 0.7 | 4.1 | 3.1 | 13.7 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm_5src_native__probe__s42` | bench |
| SimMTM (5-source, native normalisation) | finetune | 31.4 | 32.0 | 30.9 | 47.4 | 46.3 | 3.8 | 3.4 | 9.9 | 8.9 | 32.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm_5src_native__finetune__s42` | bench |
| RelCon | probe | 14.5 | 15.3 | 13.6 | 28.2 | 26.1 | 1.1 | 1.1 | 3.8 | 3.4 | 13.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__relcon__probe__s42` | bench |
| RelCon | finetune | 30.7 | 31.2 | 30.3 | 46.3 | 45.5 | 3.5 | 3.2 | 9.5 | 8.5 | 31.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__relcon__finetune__s42` | bench |
| RelCon (5-source) | probe | 13.2 | 13.9 | 12.4 | 26.9 | 25.0 | 1.0 | 1.0 | 3.6 | 3.1 | 12.7 | 42 | latest.pt (8000) | `bench_language_humanml3d__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 30.0 | 30.5 | 29.5 | 45.5 | 44.5 | 3.2 | 2.9 | 8.9 | 8.1 | 30.7 | 42 | latest.pt (8000) | `bench_language_humanml3d__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 9.1 | 9.7 | 8.4 | 20.8 | 19.0 | 0.4 | 0.7 | 2.3 | 2.1 | 8.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 24.9 | 25.3 | 24.5 | 39.2 | 38.4 | 3.1 | 2.5 | 7.8 | 6.6 | 24.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 36.6 | 37.2 | 36.0 | 52.5 | 51.4 | 4.4 | 3.9 | 11.5 | 10.3 | 37.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 24.8 | 25.6 | 24.0 | 39.2 | 37.9 | 2.5 | 2.5 | 7.4 | 6.6 | 24.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 34.5 | 34.9 | 34.1 | 50.4 | 49.5 | 4.2 | 4.5 | 11.1 | 10.5 | 35.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__sslwear_5src__finetune__s42` | bench |
| DINO (5-source) | probe | 14.7 | 15.0 | 14.4 | 27.6 | 26.8 | 1.1 | 1.1 | 4.0 | 3.6 | 14.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__dino_5src__probe__s42` | bench |
| DINO (5-source) | finetune | 25.3 | 25.7 | 25.0 | 40.7 | 40.0 | 2.4 | 2.3 | 7.4 | 6.6 | 27.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__dino_5src__finetune__s42` | bench |
| I-JEPA (5-source) | probe | 0.9 | 0.9 | 1.0 | 5.2 | 5.2 | 0.0 | 0.0 | 0.3 | 0.3 | 1.1 | 42 | latest.pt (8000) | `bench_language_humanml3d__jepa_5src__probe__s42` | bench |
| I-JEPA (5-source) | finetune | 16.7 | 17.0 | 16.4 | 30.9 | 30.1 | 1.1 | 1.0 | 4.2 | 3.9 | 17.1 | 42 | latest.pt (8000) | `bench_language_humanml3d__jepa_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 23.2 | 23.6 | 22.8 | 38.5 | 37.6 | 2.1 | 2.1 | 6.5 | 6.0 | 23.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__scratch__scratch__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 24.1 | 24.5 | 23.7 | 38.6 | 37.7 | 2.6 | 2.5 | 7.2 | 6.7 | 24.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 37.5 | 38.3 | 36.7 | 53.5 | 52.2 | 4.4 | 4.7 | 11.8 | 10.8 | 37.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 8.3 | 8.9 | 7.7 | 19.7 | 18.2 | 0.5 | 0.5 | 2.2 | 1.8 | 8.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 22.9 | 23.1 | 22.7 | 38.0 | 37.5 | 2.1 | 2.2 | 6.2 | 6.1 | 22.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | probe | 24.7 | 25.1 | 24.2 | 39.1 | 38.3 | 2.9 | 2.2 | 7.4 | 6.5 | 26.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_nocls__probe__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | finetune | 36.5 | 36.9 | 36.0 | 52.5 | 51.7 | 4.3 | 4.2 | 11.6 | 10.3 | 36.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_nocls__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 17.0 | 17.3 | 16.6 | 30.4 | 29.6 | 1.8 | 1.5 | 4.9 | 4.6 | 16.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 30.1 | 30.7 | 29.5 | 45.8 | 44.6 | 3.5 | 3.4 | 9.2 | 8.6 | 30.1 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 16.5 | 17.0 | 16.0 | 30.0 | 29.0 | 1.3 | 1.5 | 4.4 | 4.1 | 15.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 29.2 | 29.5 | 28.9 | 44.8 | 44.1 | 3.0 | 3.2 | 8.6 | 8.2 | 29.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 16.6 | 16.8 | 16.5 | 29.8 | 29.2 | 1.6 | 1.4 | 4.7 | 4.2 | 18.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 33.7 | 34.0 | 33.5 | 49.3 | 48.5 | 4.6 | 4.5 | 10.9 | 10.1 | 34.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm_4src__finetune__s42` | bench |
| RelCon (4-source pool) | probe | 15.5 | 16.1 | 14.9 | 29.4 | 28.0 | 0.9 | 1.2 | 3.8 | 3.8 | 14.7 | 42 | latest.pt (8000) | `bench_language_humanml3d__relcon_4src__probe__s42` | bench |
| RelCon (4-source pool) | finetune | 30.1 | 30.7 | 29.4 | 45.8 | 44.6 | 3.7 | 3.5 | 9.4 | 8.8 | 31.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__relcon_4src__finetune__s42` | bench |
| *UniMTS (official ckpt, zero-shot)* ⚠ invalid | external | 17.6 | 33.6 | 1.6 | 49.3 | 6.4 | — | — | — | — | — | — |  | — | harvest:analysis/retrieval_eval/humanml3 |
| *UniMTS (from scratch, wrist-only)* | external | 13.2 | 24.9 | 1.5 | 39.3 | 6.5 | 2.3 | 0.0 | 6.9 | 0.3 | 13.1 | — |  | — | harvest:analysis/retrieval_eval/humanml3 |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 19.8 | 20.2 | 19.5 | 34.3 | 33.5 | 1.3 | 1.9 | 5.3 | 5.4 | 21.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 27.7 | 28.2 | 27.2 | 42.8 | 41.8 | 3.6 | 3.1 | 8.9 | 7.9 | 27.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 19.8 | 20.5 | 19.1 | 34.3 | 33.1 | 2.0 | 1.6 | 5.8 | 5.1 | 20.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 31.3 | 31.7 | 30.9 | 46.7 | 45.8 | 4.0 | 3.6 | 10.0 | 9.1 | 32.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 24.2 | 24.4 | 23.9 | 38.6 | 38.0 | 2.9 | 3.0 | 7.6 | 7.3 | 25.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 35.3 | 35.7 | 34.9 | 50.8 | 50.0 | 4.3 | 4.0 | 11.1 | 10.4 | 36.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 21.3 | 22.0 | 20.6 | 35.8 | 34.4 | 2.1 | 2.2 | 6.2 | 5.9 | 20.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_art_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 37.0 | 37.5 | 36.6 | 52.7 | 51.9 | 4.5 | 4.4 | 11.7 | 11.1 | 37.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_art_large__finetune__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 22.2 | 22.8 | 21.7 | 36.7 | 35.7 | 2.1 | 2.1 | 6.4 | 6.0 | 22.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 37.5 | 37.7 | 37.3 | 53.1 | 52.5 | 5.1 | 4.0 | 12.3 | 10.6 | 38.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 15.5 | 15.9 | 15.2 | 28.9 | 28.0 | 1.4 | 1.2 | 4.3 | 4.0 | 15.7 | 42 | latest.pt (8000) | `bench_language_humanml3d__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 15.7 | 15.8 | 15.6 | 29.0 | 28.4 | 1.0 | 1.3 | 4.0 | 3.9 | 15.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__normwear_base__finetune__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 21.8 | 22.1 | 21.4 | 36.0 | 35.2 | 2.2 | 2.0 | 6.4 | 5.8 | 23.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 22.6 | 22.9 | 22.3 | 36.7 | 35.9 | 2.8 | 2.3 | 7.1 | 6.2 | 22.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 21.8 | 22.4 | 21.2 | 35.8 | 34.8 | 2.4 | 1.9 | 6.4 | 5.9 | 20.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 25.0 | 25.4 | 24.6 | 39.5 | 38.8 | 2.8 | 2.6 | 7.6 | 6.8 | 25.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 25.6 | 26.1 | 25.0 | 40.1 | 39.0 | 2.9 | 2.3 | 7.7 | 6.7 | 24.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 23.2 | 24.0 | 22.5 | 37.9 | 36.7 | 3.1 | 2.5 | 7.5 | 6.3 | 23.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 22.1 | 22.8 | 21.4 | 36.5 | 35.5 | 2.2 | 2.2 | 6.4 | 5.7 | 21.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 23.4 | 23.7 | 23.0 | 37.4 | 36.8 | 2.7 | 2.2 | 7.0 | 6.2 | 22.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 23.5 | 23.9 | 23.1 | 38.1 | 37.1 | 2.6 | 2.2 | 7.0 | 5.9 | 21.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 23.2 | 23.7 | 22.7 | 37.7 | 36.9 | 2.3 | 2.5 | 6.7 | 6.3 | 23.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__abl_so3__probe__s42` | bench |
| Design ablation 'fixedlen' (fixed 10.24 s windows, no variable-length crop; otherwise the corpus 'ref' arm) | probe | 23.7 | 24.0 | 23.4 | 38.1 | 37.5 | 2.8 | 3.1 | 7.2 | 7.1 | 23.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__abl_fixedlen__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 9.5 | 10.0 | 9.1 | 21.3 | 19.7 | 0.6 | 0.4 | 2.6 | 2.1 | 9.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.5 | 1.0 | 0.0 | 5.2 | 2.0 | 0.0 | 0.0 | 0.3 | 0.1 | 0.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__control_raw__probe__s42` | legacy† |
| MAE-5src fine-tuned on PRISM activity recognition | probe | 23.7 | 24.3 | 23.2 | 38.5 | 37.6 | 2.8 | 1.9 | 7.2 | 6.2 | 23.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_prism__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | finetune | 39.4 | 39.9 | 38.9 | 55.2 | 54.2 | 5.2 | 3.9 | 13.0 | 11.2 | 39.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_prism__finetune__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 19.6 | 20.2 | 18.9 | 34.0 | 32.7 | 1.6 | 1.5 | 5.4 | 4.9 | 19.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | finetune | 35.0 | 35.5 | 34.4 | 51.0 | 50.0 | 4.6 | 4.2 | 11.3 | 10.1 | 35.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_fullbody__finetune__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 25.2 | 25.5 | 24.9 | 39.8 | 39.1 | 2.8 | 2.5 | 7.6 | 7.0 | 25.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | finetune | 39.0 | 39.3 | 38.6 | 54.8 | 54.1 | 5.1 | 4.4 | 12.6 | 11.3 | 40.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_gesture__finetune__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 19.9 | 20.3 | 19.6 | 34.3 | 33.4 | 2.0 | 1.5 | 5.6 | 5.0 | 19.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_odom__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | finetune | 38.7 | 39.2 | 38.1 | 54.9 | 53.7 | 4.2 | 4.2 | 11.9 | 10.8 | 39.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_odom__finetune__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | probe | 22.1 | 22.7 | 21.6 | 36.7 | 35.7 | 1.9 | 1.9 | 6.2 | 5.7 | 22.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_pd__probe__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | finetune | 38.8 | 39.4 | 38.1 | 54.3 | 53.2 | 5.2 | 4.4 | 12.4 | 11.3 | 39.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__tx5_pd__finetune__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | probe | 8.3 | 8.9 | 7.7 | 20.0 | 18.3 | 0.4 | 0.4 | 2.1 | 1.9 | 7.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_prism__probe__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | finetune | 24.9 | 25.0 | 24.8 | 40.4 | 39.9 | 2.1 | 2.1 | 6.9 | 6.6 | 26.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_prism__finetune__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | probe | 11.8 | 12.5 | 11.0 | 24.7 | 23.1 | 0.6 | 0.7 | 2.9 | 2.7 | 11.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_fullbody__probe__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | finetune | 30.1 | 30.8 | 29.4 | 45.8 | 44.6 | 3.3 | 3.4 | 9.2 | 8.5 | 32.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_fullbody__finetune__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | probe | 9.3 | 9.9 | 8.8 | 21.5 | 19.9 | 0.7 | 0.5 | 2.5 | 2.0 | 9.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_gesture__probe__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | finetune | 25.7 | 26.1 | 25.2 | 41.0 | 40.1 | 2.4 | 2.5 | 7.2 | 6.8 | 25.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_gesture__finetune__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | probe | 11.7 | 12.3 | 11.0 | 24.6 | 22.8 | 0.6 | 0.6 | 2.9 | 2.6 | 11.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_odom__probe__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | finetune | 30.6 | 31.0 | 30.2 | 46.6 | 45.8 | 3.1 | 3.0 | 8.8 | 8.3 | 33.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_odom__finetune__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | probe | 9.4 | 10.0 | 8.8 | 21.4 | 19.7 | 0.9 | 0.4 | 2.8 | 2.1 | 8.7 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_pd__probe__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | finetune | 25.8 | 25.9 | 25.6 | 41.1 | 40.6 | 2.8 | 2.6 | 7.6 | 7.1 | 26.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_pd__finetune__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | probe | 11.1 | 11.8 | 10.4 | 24.1 | 22.3 | 0.9 | 0.7 | 3.1 | 2.7 | 11.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_armpose__probe__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | finetune | 29.4 | 29.7 | 29.0 | 45.2 | 44.4 | 3.4 | 3.1 | 8.9 | 8.1 | 31.1 | 42 | latest.pt (8000) | `bench_language_humanml3d__txs_armpose__finetune__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 21.0 | 21.4 | 20.7 | 35.5 | 34.6 | 2.0 | 1.4 | 5.9 | 5.3 | 21.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 17.2 | 17.8 | 16.6 | 31.5 | 30.1 | 1.2 | 1.8 | 4.5 | 4.6 | 17.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__mt_body_gesture__probe__s42` | bench |

## Parkinson's disease — `pd_pads`

_PADS smartwatch IMU → {Healthy, PD, Differential diagnosis} per subject._ Scorer: `bench.score.pd`. Selection: {'split': 'val', 'monitor': 'bal_acc', 'mode': 'max', 'note': 'window-level val balanced accuracy'}. Reported split: **val**.
Primary metric **subject macro-F1**: mean-prob aggregation over ~180 windows/subject (scripts/eval_pads_subject_level.py), macro-F1 over HC/PD/DD; a class never predicted contributes F1=0. Chance ≈ 0.25-0.33.

| Encoder | Protocol | **subject macro-F1** | subject bal-acc | #classes predicted (of 3) | window bal-acc | window macro-F1 | window acc | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) ⚠ eval-overlap pool | probe | 0.388 | 40.9% | 3 | 43.4% | 0.441 | 59.8% | 42 | best.pt (5200) | `bench_pd_pads__mae_11src__probe__s42` | bench |
| Ours (MAE, 11-source) ⚠ eval-overlap pool | finetune | 0.542 | 54.8% | 3 | 44.9% | 0.441 | 53.4% | 42 | best.pt (1600) | `bench_pd_pads__mae_11src__finetune__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.552 | 60.0% | 3 | 48.3% | 0.441 | 47.9% | 42 | best.pt (1800) | `bench_pd_pads__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.519 | 63.9% | 3 | 51.0% | 0.436 | 45.3% | 42 | best.pt (800) | `bench_pd_pads__mae_5src__finetune__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool ⚠ partial | legacy_valloss | 0.299 | 35.6% | 2 | 38.3% | 0.361 | 63.1% | 42,42 | best.pt (400) | `vit1d_pd_finetune_pads_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.386 | 40.9% | 2 | 43.5% | 0.443 | 61.1% | 42 | best.pt (5200) | `bench_pd_pads__mae_noaltattn__probe__s42` | bench |
| Ours (MAE, no alt-attn) | finetune | 0.458 | 45.0% | 3 | 45.5% | 0.456 | 59.3% | 42 | best.pt (1400) | `bench_pd_pads__mae_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 0.481 | 59.7% | 3 | 48.8% | 0.414 | 42.6% | 42 | best.pt (4000) | `bench_pd_pads__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 0.610 | 68.3% | 3 | 50.1% | 0.449 | 49.7% | 42 | best.pt (1000) | `bench_pd_pads__mae_5src_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.397 | 41.8% | 2 | 41.1% | 0.410 | 57.2% | 42 | best.pt (600) | `bench_pd_pads__mae_acconly__probe__s42` | bench |
| Ours (MAE, acc-only) | finetune | 0.606 | 55.8% | 3 | 45.8% | 0.453 | 54.7% | 42 | best.pt (1600) | `bench_pd_pads__mae_acconly__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 0.482 | 61.6% | 3 | 47.6% | 0.401 | 41.2% | 42 | best.pt (4000) | `bench_pd_pads__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 0.599 | 67.7% | 3 | 48.7% | 0.426 | 44.7% | 42 | best.pt (800) | `bench_pd_pads__mae_5src_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.254 | 33.3% | 1 | 33.9% | 0.285 | 59.8% | 42 | best.pt (3400) | `bench_pd_pads__focal__probe__s42` | bench |
| FOCAL | finetune | 0.331 | 37.2% | 3 | 40.1% | 0.398 | 51.5% | 42 | best.pt (5400) | `bench_pd_pads__focal__finetune__s42` | bench |
| FOCAL ⚠ partial | legacy_valloss | 0.254 | 33.3% | 1 | 33.5% | 0.259 | 61.8% | 42,42 | best.pt (800) | `vit1d_pd_finetune_pads_focal` | legacy† |
| FOCAL (5-source) | probe | 0.407 | 53.9% | 3 | 39.2% | 0.313 | 31.5% | 42 | best.pt (1200) | `bench_pd_pads__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 0.463 | 57.7% | 3 | 45.1% | 0.370 | 37.8% | 42 | best.pt (2800) | `bench_pd_pads__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 0.308 | 35.8% | 2 | 35.4% | 0.330 | 52.8% | 42 | best.pt (600) | `bench_pd_pads__focal_5src_native__probe__s42` | bench |
| SimMTM | probe | 0.368 | 39.6% | 2 | 39.1% | 0.359 | 54.6% | 42 | best.pt (600) | `bench_pd_pads__simmtm__probe__s42` | bench |
| SimMTM | finetune | 0.515 | 48.8% | 3 | 46.8% | 0.465 | 56.7% | 42 | best.pt (2200) | `bench_pd_pads__simmtm__finetune__s42` | bench |
| SimMTM ⚠ partial | legacy_valloss | 0.254 | 33.3% | 1 | 35.6% | 0.304 | 62.7% | 42 | best.pt (10800) | `vit1d_pd_probe_pads_simmtm` | legacy† |
| SimMTM (5-source) | probe | 0.487 | 59.3% | 3 | 48.0% | 0.403 | 41.6% | 42 | best.pt (13800) | `bench_pd_pads__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 0.468 | 59.3% | 3 | 49.1% | 0.406 | 41.5% | 42 | best.pt (800) | `bench_pd_pads__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 0.424 | 44.3% | 2 | 39.1% | 0.356 | 54.3% | 42 | best.pt (600) | `bench_pd_pads__simmtm_5src_native__probe__s42` | bench |
| RelCon | probe | 0.285 | 34.8% | 2 | 35.4% | 0.317 | 59.8% | 42 | best.pt (600) | `bench_pd_pads__relcon__probe__s42` | bench |
| RelCon | finetune | 0.619 | 58.7% | 3 | 45.1% | 0.440 | 51.5% | 42 | best.pt (1600) | `bench_pd_pads__relcon__finetune__s42` | bench |
| RelCon ⚠ partial | legacy_valloss | 0.254 | 33.3% | 1 | 35.3% | 0.297 | 62.4% | 42,42 | best.pt (1000) | `vit1d_pd_finetune_pads_relcon` | legacy† |
| RelCon (5-source) | probe | 0.405 | 51.5% | 2 | 40.7% | 0.348 | 42.1% | 42 | best.pt (8000) | `bench_pd_pads__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 0.491 | 58.1% | 3 | 46.9% | 0.399 | 41.4% | 42 | best.pt (1800) | `bench_pd_pads__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 0.254 | 33.3% | 1 | 33.5% | 0.260 | 61.6% | 42 | best.pt (3400) | `bench_pd_pads__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 0.505 | 58.4% | 3 | 49.0% | 0.438 | 46.7% | 42 | best.pt (12000) | `bench_pd_pads__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 0.460 | 55.0% | 3 | 49.3% | 0.430 | 48.9% | 42 | best.pt (1000) | `bench_pd_pads__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 0.574 | 61.7% | 3 | 48.5% | 0.439 | 47.3% | 42 | best.pt (1000) | `bench_pd_pads__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 0.487 | 59.0% | 3 | 50.0% | 0.419 | 43.2% | 42 | best.pt (800) | `bench_pd_pads__sslwear_5src__finetune__s42` | bench |
| DINO (5-source) | probe | 0.514 | 59.2% | 3 | 46.4% | 0.401 | 42.2% | 42 | best.pt (6800) | `bench_pd_pads__dino_5src__probe__s42` | bench |
| DINO (5-source) | finetune | 0.567 | 59.3% | 3 | 47.0% | 0.434 | 49.9% | 42 | best.pt (1000) | `bench_pd_pads__dino_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.299 | 45.5% | 3 | 43.9% | 0.318 | 31.9% | 42 | best.pt (3400) | `bench_pd_pads__scratch__scratch__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.463 | 47.3% | 2 | 45.7% | 0.461 | 57.6% | 42 | best.pt (1600) | `bench_pd_pads__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.590 | 55.2% | 3 | 45.0% | 0.443 | 54.5% | 42 | best.pt (5600) | `bench_pd_pads__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.317 | 39.2% | 2 | 38.2% | 0.341 | 51.4% | 42 | best.pt (2400) | `bench_pd_pads__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.364 | 39.4% | 2 | 38.3% | 0.375 | 59.2% | 42 | best.pt (5800) | `bench_pd_pads__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | probe | 0.474 | 46.3% | 3 | 46.5% | 0.462 | 56.5% | 42 | best.pt (1600) | `bench_pd_pads__mae_4src_nocls__probe__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | finetune | 0.505 | 48.1% | 3 | 45.9% | 0.451 | 55.1% | 42 | best.pt (5600) | `bench_pd_pads__mae_4src_nocls__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.334 | 37.3% | 2 | 38.8% | 0.384 | 56.1% | 42 | best.pt (1600) | `bench_pd_pads__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.313 | 36.4% | 2 | 41.1% | 0.414 | 54.7% | 42 | best.pt (6200) | `bench_pd_pads__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.357 | 38.8% | 2 | 39.3% | 0.374 | 57.2% | 42 | best.pt (800) | `bench_pd_pads__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.254 | 33.3% | 1 | 41.2% | 0.403 | 52.8% | 42 | best.pt (5600) | `bench_pd_pads__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.386 | 40.9% | 2 | 36.8% | 0.344 | 55.8% | 42 | best.pt (600) | `bench_pd_pads__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.563 | 51.9% | 3 | 44.9% | 0.453 | 57.0% | 42 | best.pt (5200) | `bench_pd_pads__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.364 | 39.4% | 2 | 40.4% | 0.407 | 56.3% | 42 | best.pt (12200) | `bench_pd_pads__focal_4src__finetune__s42` | bench |
| RelCon (4-source pool) | probe | 0.254 | 33.3% | 1 | 36.9% | 0.341 | 59.5% | 42 | best.pt (6200) | `bench_pd_pads__relcon_4src__probe__s42` | bench |
| RelCon (4-source pool) | finetune | 0.505 | 48.1% | 3 | 44.2% | 0.439 | 56.4% | 42 | best.pt (7000) | `bench_pd_pads__relcon_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.487 | 46.5% | 3 | 43.3% | 0.408 | 50.8% | 42 | best.pt (5600) | `bench_pd_pads__acconly_4src__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.254 | 33.3% | 1 | 34.7% | 0.284 | 62.5% | 42 | best.pt (9800) | `bench_pd_pads__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.386 | 40.9% | 2 | 40.2% | 0.403 | 59.8% | 42 | best.pt (9000) | `bench_pd_pads__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.254 | 33.3% | 1 | 35.8% | 0.312 | 62.5% | 42 | best.pt (7600) | `bench_pd_pads__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.386 | 40.9% | 2 | 38.9% | 0.374 | 62.5% | 42 | best.pt (2600) | `bench_pd_pads__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 0.315 | 36.4% | 3 | 39.5% | 0.388 | 61.2% | 42 | best.pt (5600) | `bench_pd_pads__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 0.364 | 39.4% | 2 | 38.7% | 0.378 | 60.7% | 42 | best.pt (4800) | `bench_pd_pads__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 0.298 | 54.3% | 2 | 42.5% | 0.248 | 25.0% | 42 | best.pt (600) | `bench_pd_pads__inertia1_art_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 0.254 | 33.3% | 1 | 37.4% | 0.359 | 59.6% | 42 | best.pt (600) | `bench_pd_pads__inertia1_art_large__finetune__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 0.254 | 33.3% | 1 | 33.4% | 0.257 | 61.7% | 42 | best.pt (9800) | `bench_pd_pads__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 0.463 | 45.1% | 3 | 45.7% | 0.463 | 60.4% | 42 | best.pt (2200) | `bench_pd_pads__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 0.566 | 59.7% | 3 | 42.9% | 0.364 | 38.0% | 42 | best.pt (1000) | `bench_pd_pads__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 0.431 | 53.3% | 3 | 41.7% | 0.336 | 34.6% | 42 | best.pt (4400) | `bench_pd_pads__normwear_base__finetune__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.01% of pool, N=196 | probe | 0.565 | 59.6% | 3 | 43.0% | 0.386 | 41.7% | 42 | best.pt (5400) | `bench_pd_pads__g5_k1_f0p01__probe__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.445 | 54.8% | 3 | 40.2% | 0.337 | 36.3% | 42 | best.pt (4000) | `bench_pd_pads__g5_k1_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.01% of pool, N=196 | probe | 0.530 | 62.2% | 3 | 45.4% | 0.390 | 41.0% | 42 | best.pt (10000) | `bench_pd_pads__g5_k5_f0p01__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.521 | 60.8% | 3 | 46.6% | 0.412 | 43.7% | 42 | best.pt (13800) | `bench_pd_pads__g5_k5_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 1% of pool, N=19,593 | probe | 0.524 | 63.9% | 3 | 45.0% | 0.381 | 39.3% | 42 | best.pt (400) | `bench_pd_pads__g5_k5_f1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.01% of pool, N=196 | probe | 0.452 | 58.9% | 3 | 47.3% | 0.394 | 40.2% | 42 | best.pt (4000) | `bench_pd_pads__g5_k10_f0p01__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.553 | 60.2% | 3 | 47.4% | 0.424 | 45.3% | 42 | best.pt (4400) | `bench_pd_pads__g5_k10_f0p1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 1% of pool, N=19,593 | probe | 0.584 | 63.9% | 3 | 47.7% | 0.423 | 44.8% | 42 | best.pt (11400) | `bench_pd_pads__g5_k10_f1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.01% of pool, N=196 | probe | 0.464 | 56.5% | 3 | 45.7% | 0.389 | 40.2% | 42 | best.pt (4000) | `bench_pd_pads__g5_k25_f0p01__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.357 | 51.1% | 3 | 46.0% | 0.365 | 36.7% | 42 | best.pt (4000) | `bench_pd_pads__g5_k25_f0p1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 1% of pool, N=19,593 | probe | 0.407 | 53.9% | 3 | 46.7% | 0.388 | 39.7% | 42 | best.pt (4000) | `bench_pd_pads__g5_k25_f1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.01% of pool, N=196 | probe | 0.411 | 56.7% | 3 | 46.0% | 0.381 | 38.9% | 42 | best.pt (4000) | `bench_pd_pads__g5_k50_f0p01__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.541 | 57.8% | 3 | 47.5% | 0.444 | 51.0% | 42 | best.pt (1000) | `bench_pd_pads__g5_k50_f0p1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 1% of pool, N=19,593 | probe | 0.425 | 57.2% | 3 | 47.5% | 0.399 | 41.0% | 42 | best.pt (4000) | `bench_pd_pads__g5_k50_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.01% of pool, N=196 | probe | 0.548 | 63.7% | 3 | 47.6% | 0.407 | 42.3% | 42 | best.pt (13600) | `bench_pd_pads__g5_k100_f0p01__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.467 | 56.4% | 3 | 46.9% | 0.404 | 41.9% | 42 | best.pt (4000) | `bench_pd_pads__g5_k100_f0p1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 1% of pool, N=19,593 | probe | 0.555 | 65.2% | 3 | 47.8% | 0.417 | 43.8% | 42 | best.pt (4000) | `bench_pd_pads__g5_k100_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 10% of pool, N=195,929 | probe | 0.485 | 59.7% | 3 | 46.7% | 0.397 | 40.9% | 42 | best.pt (4000) | `bench_pd_pads__g5_k100_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.498 | 59.0% | 3 | 47.8% | 0.414 | 43.0% | 42 | best.pt (400) | `bench_pd_pads__g5_k1000_f0p1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 0.500 | 60.0% | 3 | 46.8% | 0.409 | 42.6% | 42 | best.pt (4000) | `bench_pd_pads__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 0.481 | 60.6% | 3 | 46.7% | 0.398 | 41.2% | 42 | best.pt (4000) | `bench_pd_pads__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 0.454 | 59.8% | 3 | 47.6% | 0.395 | 40.2% | 42 | best.pt (4000) | `bench_pd_pads__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 0.533 | 62.3% | 3 | 50.3% | 0.438 | 45.9% | 42 | best.pt (1800) | `bench_pd_pads__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 0.600 | 64.7% | 3 | 48.3% | 0.438 | 47.7% | 42 | best.pt (1800) | `bench_pd_pads__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 0.481 | 59.8% | 3 | 48.6% | 0.410 | 42.1% | 42 | best.pt (1800) | `bench_pd_pads__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 0.541 | 61.9% | 3 | 47.7% | 0.418 | 44.1% | 42 | best.pt (4000) | `bench_pd_pads__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 0.467 | 57.3% | 3 | 48.1% | 0.407 | 41.8% | 42 | best.pt (4000) | `bench_pd_pads__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.500 | 61.9% | 3 | 48.9% | 0.413 | 42.4% | 42 | best.pt (4000) | `bench_pd_pads__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.582 | 63.6% | 3 | 48.5% | 0.438 | 47.0% | 42 | best.pt (4400) | `bench_pd_pads__abl_so3__probe__s42` | bench |
| Design ablation 'fixedlen' (fixed 10.24 s windows, no variable-length crop; otherwise the corpus 'ref' arm) | probe | 0.488 | 58.5% | 3 | 48.8% | 0.422 | 43.9% | 42 | best.pt (1800) | `bench_pd_pads__abl_fixedlen__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.517 | 55.9% | 3 | 42.7% | 0.384 | 42.3% | 42 | best.pt (4800) | `bench_pd_pads__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.126 | 33.3% | 1 | 33.3% | 0.126 | 23.4% | 42 | best.pt (200) | `bench_pd_pads__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | probe | 0.568 | 62.0% | 3 | 48.3% | 0.440 | 48.6% | 42 | best.pt (1000) | `bench_pd_pads__tx5_prism__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | finetune | 0.552 | 65.3% | 3 | 50.0% | 0.434 | 45.5% | 42 | best.pt (800) | `bench_pd_pads__tx5_prism__finetune__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 0.435 | 53.5% | 3 | 47.4% | 0.404 | 43.2% | 42 | best.pt (3000) | `bench_pd_pads__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | finetune | 0.634 | 63.2% | 3 | 48.7% | 0.453 | 49.8% | 42 | best.pt (800) | `bench_pd_pads__tx5_fullbody__finetune__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 0.556 | 60.0% | 3 | 48.5% | 0.439 | 47.6% | 42 | best.pt (1800) | `bench_pd_pads__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | finetune | 0.652 | 64.7% | 3 | 50.2% | 0.472 | 53.4% | 42 | best.pt (200) | `bench_pd_pads__tx5_gesture__finetune__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 0.502 | 56.6% | 3 | 47.3% | 0.415 | 43.6% | 42 | best.pt (11000) | `bench_pd_pads__tx5_odom__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | finetune | 0.521 | 64.9% | 3 | 50.9% | 0.432 | 44.7% | 42 | best.pt (800) | `bench_pd_pads__tx5_odom__finetune__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | probe | 0.580 | 64.8% | 3 | 49.5% | 0.441 | 47.6% | 42 | best.pt (1800) | `bench_pd_pads__tx5_language__probe__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | finetune | 0.629 | 68.2% | 3 | 51.8% | 0.470 | 51.1% | 42 | best.pt (400) | `bench_pd_pads__tx5_language__finetune__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | probe | 0.475 | 55.9% | 3 | 41.3% | 0.359 | 39.9% | 42 | best.pt (4000) | `bench_pd_pads__txs_prism__probe__s42` | bench |
| Trained from scratch on PRISM activity recognition (no pretraining) | finetune | 0.585 | 61.2% | 3 | 43.2% | 0.397 | 44.3% | 42 | best.pt (2200) | `bench_pd_pads__txs_prism__finetune__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | probe | 0.380 | 54.0% | 3 | 42.3% | 0.349 | 35.6% | 42 | best.pt (2000) | `bench_pd_pads__txs_fullbody__probe__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | finetune | 0.490 | 61.3% | 3 | 44.2% | 0.382 | 40.1% | 42 | best.pt (400) | `bench_pd_pads__txs_fullbody__finetune__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | probe | 0.491 | 56.9% | 3 | 41.6% | 0.355 | 36.8% | 42 | best.pt (5400) | `bench_pd_pads__txs_gesture__probe__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | finetune | 0.424 | 55.4% | 3 | 43.1% | 0.342 | 34.6% | 42 | best.pt (1200) | `bench_pd_pads__txs_gesture__finetune__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | probe | 0.343 | 47.0% | 3 | 43.0% | 0.363 | 37.3% | 42 | best.pt (4000) | `bench_pd_pads__txs_odom__probe__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | finetune | 0.387 | 54.7% | 3 | 46.3% | 0.383 | 39.1% | 42 | best.pt (800) | `bench_pd_pads__txs_odom__finetune__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | probe | 0.473 | 57.9% | 3 | 44.9% | 0.385 | 40.0% | 42 | best.pt (4000) | `bench_pd_pads__txs_language__probe__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | finetune | 0.435 | 56.8% | 3 | 45.0% | 0.376 | 38.6% | 42 | best.pt (1800) | `bench_pd_pads__txs_language__finetune__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | probe | 0.308 | 45.8% | 3 | 43.1% | 0.355 | 36.3% | 42 | best.pt (4000) | `bench_pd_pads__txs_armpose__probe__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | finetune | 0.483 | 58.5% | 3 | 44.2% | 0.388 | 41.0% | 42 | best.pt (400) | `bench_pd_pads__txs_armpose__finetune__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 0.379 | 40.3% | 2 | 40.8% | 0.406 | 61.8% | 42 | best.pt (5200) | `bench_pd_pads__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 0.332 | 37.3% | 2 | 38.8% | 0.379 | 53.9% | 42 | best.pt (600) | `bench_pd_pads__mt_body_gesture__probe__s42` | bench |
| PADS pipeline, BOSS + CatBoost (Varghese et al.) | external | 0.481 | 58.5% | — | 42.1% | 0.358 | 37.2% | 0 |  | — | bench |
| PADS pipeline, BOSS + SVM (Varghese et al.) ⚠ invalid | external | 0.254 | 33.3% | — | 33.4% | 0.256 | 61.7% | 0 |  | — | bench |
| PADS pipeline, spectral features + CatBoost (Varghese et al.) | external | 0.618 | 66.8% | — | 49.1% | 0.446 | 48.5% | 0 |  | — | bench |

## HAR · HHAR — `har_hhar`

_HHAR activity recognition from one wrist/arm IMU (6 classes)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) ⚠ eval-overlap pool | probe | 0.679 | 68.2% | 67.0% | 0.9045 | 42 | best.pt (3200) | `vit1d_har_probe_hhar_mae_multidata` | legacy† |
| Ours (MAE, 11-source) ⚠ eval-overlap pool | finetune | 0.628 | 63.5% | 61.7% | 1.1546 | 42 | best.pt (200) | `vit1d_har_finetune_hhar_mae_multidata` | legacy† |
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.621 | 62.7% | 61.9% | 0.9777 | 42 | best.pt (800) | `bench_har_hhar__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.603 | 61.0% | 60.2% | 1.1782 | 42 | best.pt (200) | `bench_har_hhar__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 0.661 | 66.1% | 65.5% | 0.9308 | 42 | best.pt (7400) | `vit1d_har_probe_hhar_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.599 | 60.6% | 59.6% | 1.1973 | 42 | best.pt (200) | `bench_har_hhar__mae_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 0.602 | 60.4% | 60.3% | 1.0184 | 42 | best.pt (7600) | `bench_har_hhar__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 0.526 | 54.9% | 55.2% | 1.1972 | 42 | best.pt (200) | `bench_har_hhar__mae_5src_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.600 | 60.7% | 61.4% | 0.9892 | 42 | best.pt (1400) | `vit1d_har_probe_hhar_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.569 | 58.0% | 58.0% | 1.1880 | 42 | best.pt (200) | `bench_har_hhar__mae_acconly__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 0.578 | 58.3% | 58.8% | 1.0451 | 42 | best.pt (400) | `bench_har_hhar__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 0.568 | 57.9% | 58.0% | 1.2661 | 42 | best.pt (200) | `bench_har_hhar__mae_5src_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.500 | 51.5% | 51.1% | 1.2454 | 42 | best.pt (11000) | `vit1d_har_probe_hhar_focal_multidata` | legacy† |
| FOCAL | finetune | 0.588 | 58.3% | 56.8% | 1.1406 | 42 | best.pt (600) | `bench_har_hhar__focal__finetune__s42` | bench |
| FOCAL (5-source) | probe | 0.449 | 46.7% | 45.8% | 1.3790 | 42 | best.pt (18200) | `bench_har_hhar__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 0.613 | 61.7% | 61.4% | 1.1066 | 42 | best.pt (800) | `bench_har_hhar__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 0.628 | 61.7% | 58.5% | 1.0702 | 42 | best.pt (4400) | `bench_har_hhar__focal_5src_native__probe__s42` | bench |
| SimMTM | probe | 0.573 | 57.7% | 56.5% | 1.1478 | 42 | best.pt (7800) | `vit1d_har_probe_hhar_simmtm_multidata` | legacy† |
| SimMTM | finetune | 0.479 | 52.5% | 51.0% | 1.1589 | 42 | best.pt (200) | `bench_har_hhar__simmtm__finetune__s42` | bench |
| SimMTM (5-source) | probe | 0.575 | 57.8% | 56.8% | 1.1589 | 42 | best.pt (7800) | `bench_har_hhar__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 0.470 | 48.1% | 48.4% | 1.1418 | 42 | best.pt (200) | `bench_har_hhar__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 0.496 | 50.1% | 48.9% | 1.2452 | 42 | best.pt (7800) | `bench_har_hhar__simmtm_5src_native__probe__s42` | bench |
| SimMTM (5-source, native normalisation) | finetune | 0.451 | 48.8% | 49.0% | 1.1939 | 42 | best.pt (200) | `bench_har_hhar__simmtm_5src_native__finetune__s42` | bench |
| RelCon | probe | 0.283 | 33.2% | 35.4% | 1.6737 | 42 | best.pt (13600) | `vit1d_har_probe_hhar_relcon_multidata` | legacy† |
| RelCon | finetune | 0.454 | 47.2% | 46.7% | 1.2264 | 42 | best.pt (200) | `bench_har_hhar__relcon__finetune__s42` | bench |
| RelCon (5-source) | probe | 0.333 | 35.5% | 35.5% | 1.6279 | 42 | best.pt (15600) | `bench_har_hhar__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 0.429 | 46.3% | 46.7% | 1.1537 | 42 | best.pt (200) | `bench_har_hhar__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 0.628 | 66.0% | 61.3% | 0.9430 | 42 | best.pt (4600) | `bench_har_hhar__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 0.612 | 60.8% | 59.9% | 1.0177 | 42 | best.pt (7600) | `bench_har_hhar__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 0.563 | 58.5% | 57.2% | 1.0172 | 42 | best.pt (200) | `bench_har_hhar__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 0.592 | 59.7% | 59.5% | 1.0186 | 42 | best.pt (15600) | `bench_har_hhar__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 0.535 | 55.7% | 56.0% | 1.0824 | 42 | best.pt (200) | `bench_har_hhar__sslwear_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.488 | 50.3% | 49.6% | 1.2332 | 42 | best.pt (800) | `bench_har_hhar__scratch__scratch__s42` | bench |
| Inertia-1 PatchTST-small arch, from scratch (no pretraining) | scratch | 0.697 | 70.4% | 67.4% | 1.0853 | 42 | best.pt (3200) | `bench_har_hhar__inertia1_patchtst_scratch__scratch__s42` | bench |
| Inertia-1 PatchTST-small arch, random init, frozen (probe control) | probe | 0.455 | 48.2% | 46.5% | 1.3517 | 42 | best.pt (2600) | `bench_har_hhar__inertia1_patchtst_randinit__probe__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.513 | 50.8% | 50.4% | 1.1179 | 42 | best.pt (400) | `bench_har_hhar__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.565 | 56.9% | 57.2% | 1.0935 | 42 | best.pt (400) | `bench_har_hhar__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.575 | 58.7% | 58.5% | 1.1792 | 42 | best.pt (7600) | `bench_har_hhar__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.677 | 68.0% | 64.1% | 0.9864 | 42 | best.pt (4800) | `bench_har_hhar__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.545 | 55.0% | 54.5% | 1.1052 | 42 | best.pt (400) | `bench_har_hhar__mae_recipe_r0__probe__s42` | bench |
| MAE recipe R1 (physical scaling + yaw rotation + lr 3e-4) | probe | 0.532 | 56.1% | 51.7% | 1.1901 | 42 | best.pt (12800) | `bench_har_hhar__mae_recipe_r1__probe__s42` | bench |
| MAE recipe R2 (R1 + band-weighted loss (4/2/1/0.5)) | probe | 0.586 | 60.3% | 56.5% | 1.0768 | 42 | best.pt (4200) | `bench_har_hhar__mae_recipe_r2__probe__s42` | bench |
| MAE recipe R3 (R2 + modality masking p=0.3 (replace-mode loss)) | probe | 0.545 | 55.7% | 52.3% | 1.2226 | 42 | best.pt (9200) | `bench_har_hhar__mae_recipe_r3__probe__s42` | bench |
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.323 | 35.4% | 36.2% | 1.5012 | 42 | best.pt (18400) | `bench_har_hhar__mae_recipe_rraw__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 0.489 | 49.9% | 50.1% | 1.1596 | 42 | best.pt (400) | `bench_har_hhar__mae_mr40__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.531 | 54.4% | 54.3% | 1.0841 | 42 | best.pt (7000) | `bench_har_hhar__mae_div60_dc__probe__s42` | bench |
| MAE mask-ratio 0.50 | probe | 0.553 | 55.7% | 55.2% | 1.1024 | 42 | best.pt (400) | `bench_har_hhar__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.663 | 66.4% | 64.0% | 1.0254 | 42 | best.pt (6400) | `bench_har_hhar__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.655 | 68.8% | 66.4% | 1.5983 | 42 | best.pt (800) | `bench_har_hhar__mae_4src_z__finetune__s42` | bench |
| MAE mask-ratio 0.60 | probe | 0.525 | 53.2% | 53.4% | 1.0811 | 42 | best.pt (400) | `bench_har_hhar__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.662 | 67.2% | 63.9% | 1.0839 | 42 | best.pt (3600) | `bench_har_hhar__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.593 | 59.8% | 59.3% | 1.4928 | 42 | best.pt (800) | `bench_har_hhar__mae_4src_dc__finetune__s42` | bench |
| MAE mask-ratio 0.85 | probe | 0.483 | 50.7% | 49.5% | 1.1674 | 42 | best.pt (400) | `bench_har_hhar__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.608 | 61.9% | 58.2% | 1.1049 | 42 | best.pt (7000) | `bench_har_hhar__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.707 | 70.7% | 67.5% | 1.4558 | 42 | best.pt (800) | `bench_har_hhar__mae_4src_phys__finetune__s42` | bench |
| MAE mask-ratio 0.90 | probe | 0.544 | 54.3% | 53.4% | 1.1358 | 42 | best.pt (7800) | `bench_har_hhar__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.637 | 63.9% | 60.8% | 1.0304 | 42 | best.pt (11000) | `bench_har_hhar__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.670 | 67.4% | 65.6% | 2.1834 | 42 | best.pt (2600) | `bench_har_hhar__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.675 | 67.4% | 63.6% | 1.0035 | 42 | best.pt (3000) | `bench_har_hhar__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.683 | 67.4% | 65.0% | 1.7191 | 42 | best.pt (1200) | `bench_har_hhar__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.591 | 58.0% | 54.8% | 1.2151 | 42 | best.pt (3600) | `bench_har_hhar__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.651 | 64.7% | 62.2% | 1.4404 | 42 | best.pt (800) | `bench_har_hhar__focal_4src__finetune__s42` | bench |
| RelCon (4-source pool) | probe | 0.619 | 62.2% | 58.2% | 1.3008 | 42 | best.pt (6400) | `bench_har_hhar__relcon_4src__probe__s42` | bench |
| RelCon (4-source pool) | finetune | 0.661 | 68.3% | 65.0% | 1.6311 | 42 | best.pt (800) | `bench_har_hhar__relcon_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.704 | 70.6% | 68.3% | 0.9448 | 42 | best.pt (6000) | `bench_har_hhar__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.700 | 70.6% | 69.9% | 1.2029 | 42 | best.pt (1000) | `bench_har_hhar__acconly_4src__finetune__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.448 | 46.6% | 44.8% | 1.2853 | — | best.pt | `deepconvlstm_har_hhar` | legacy† |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.713 | 73.1% | 69.7% | 0.8740 | 42 | best.pt (15400) | `bench_har_hhar__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.785 | 78.5% | 75.8% | 0.6651 | 42 | best.pt (800) | `bench_har_hhar__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.717 | 71.9% | 68.8% | 0.8406 | 42 | best.pt (7000) | `bench_har_hhar__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.670 | 67.6% | 64.9% | 0.8168 | 42 | best.pt (200) | `bench_har_hhar__inertia1_art__finetune__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 0.622 | 62.8% | 64.8% | 0.8476 | 42 | best.pt (19800) | `bench_har_hhar__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 0.762 | 76.4% | 77.0% | 0.6355 | 42 | best.pt (2000) | `bench_har_hhar__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 0.437 | 44.6% | 45.8% | 1.3432 | 42 | best.pt (7800) | `bench_har_hhar__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 0.371 | 39.0% | 40.6% | 1.4824 | 42 | best.pt (18200) | `bench_har_hhar__normwear_base__finetune__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 | probe | 0.607 | 61.7% | 57.9% | 1.1205 | 42 | best.pt (8600) | `bench_har_hhar__vd_ek1_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 | probe | 0.664 | 66.8% | 62.7% | 1.0327 | 42 | best.pt (7400) | `bench_har_hhar__vd_ek1_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 | probe | 0.568 | 57.1% | 53.6% | 1.1812 | 42 | best.pt (13200) | `bench_har_hhar__vd_ek5_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 | probe | 0.626 | 63.2% | 60.0% | 1.0565 | 42 | best.pt (7400) | `bench_har_hhar__vd_ek5_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 | probe | 0.634 | 63.9% | 60.4% | 1.1021 | 42 | best.pt (14400) | `bench_har_hhar__vd_ek5_f1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 | probe | 0.640 | 64.6% | 60.2% | 1.0542 | 42 | best.pt (9400) | `bench_har_hhar__vd_ek10_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 | probe | 0.632 | 62.8% | 61.1% | 1.0324 | 42 | best.pt (6400) | `bench_har_hhar__vd_ek10_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 | probe | 0.700 | 70.3% | 66.9% | 0.9180 | 42 | best.pt (7600) | `bench_har_hhar__vd_ek10_f1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 | probe | 0.661 | 66.6% | 62.5% | 1.0399 | 42 | best.pt (9400) | `bench_har_hhar__vd_ek25_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 | probe | 0.689 | 69.5% | 65.8% | 0.9803 | 42 | best.pt (16600) | `bench_har_hhar__vd_ek25_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 | probe | 0.666 | 67.2% | 63.2% | 0.9802 | 42 | best.pt (16600) | `bench_har_hhar__vd_ek25_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 | probe | 0.524 | 53.1% | 51.6% | 1.1786 | 42 | best.pt (3200) | `bench_har_hhar__vd_ek50_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 | probe | 0.649 | 65.5% | 62.2% | 0.9738 | 42 | best.pt (16000) | `bench_har_hhar__vd_ek50_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 | probe | 0.637 | 64.1% | 61.0% | 0.9988 | 42 | best.pt (13400) | `bench_har_hhar__vd_ek50_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 | probe | 0.671 | 67.4% | 63.8% | 0.9606 | 42 | best.pt (11600) | `bench_har_hhar__vd_ek50_f10__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 | probe | 0.652 | 66.4% | 61.9% | 0.9517 | 42 | best.pt (13200) | `bench_har_hhar__vd_ek100_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 | probe | 0.535 | 55.2% | 54.0% | 1.0694 | 42 | best.pt (5400) | `bench_har_hhar__vd_ek100_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 | probe | 0.584 | 59.1% | 56.8% | 1.0965 | 42 | best.pt (14400) | `bench_har_hhar__vd_ek100_f1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 | probe | 0.674 | 68.4% | 63.9% | 0.9134 | 42 | best.pt (4600) | `bench_har_hhar__vd_ek100_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 | probe | 0.575 | 57.7% | 55.4% | 1.0533 | 42 | best.pt (4800) | `bench_har_hhar__vd_ek1000_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 | probe | 0.579 | 58.1% | 56.7% | 1.0668 | 42 | best.pt (7000) | `bench_har_hhar__vd_ek1000_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 | probe | 0.617 | 61.7% | 58.2% | 1.0581 | 42 | best.pt (16000) | `bench_har_hhar__vd_ek1000_f1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 | probe | 0.575 | 59.1% | 58.5% | 1.0695 | 42 | best.pt (7000) | `bench_har_hhar__vd_ek1000_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 | probe | 0.644 | 64.5% | 61.0% | 1.0577 | 42 | best.pt (7400) | `bench_har_hhar__vd_ek1000_f100__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.495 | 50.3% | 49.7% | 1.2290 | 42 | best.pt (9600) | `bench_har_hhar__vd_ek1_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.575 | 57.2% | 56.6% | 1.0606 | 42 | best.pt (9600) | `bench_har_hhar__vd_ek1_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.498 | 50.0% | 49.8% | 1.1940 | 42 | best.pt (13400) | `bench_har_hhar__vd_ek5_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.558 | 55.5% | 55.5% | 1.1591 | 42 | best.pt (9600) | `bench_har_hhar__vd_ek5_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.576 | 57.0% | 56.4% | 1.1012 | 42 | best.pt (7800) | `bench_har_hhar__vd_ek5_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.486 | 48.7% | 48.0% | 1.2225 | 42 | best.pt (13400) | `bench_har_hhar__vd_ek10_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.579 | 57.3% | 56.5% | 1.0951 | 42 | best.pt (5600) | `bench_har_hhar__vd_ek10_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.608 | 60.4% | 60.3% | 1.0839 | 42 | best.pt (4800) | `bench_har_hhar__vd_ek10_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.532 | 53.3% | 53.1% | 1.1617 | 42 | best.pt (7800) | `bench_har_hhar__vd_ek25_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.571 | 57.0% | 57.1% | 1.0818 | 42 | best.pt (400) | `bench_har_hhar__vd_ek25_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.534 | 53.0% | 52.5% | 1.1614 | 42 | best.pt (400) | `bench_har_hhar__vd_ek25_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.538 | 53.9% | 53.4% | 1.1596 | 42 | best.pt (9600) | `bench_har_hhar__vd_ek50_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.549 | 54.8% | 55.4% | 1.0957 | 42 | best.pt (400) | `bench_har_hhar__vd_ek50_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.559 | 55.5% | 55.0% | 1.1041 | 42 | best.pt (2000) | `bench_har_hhar__vd_ek50_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.524 | 51.8% | 53.0% | 1.1014 | 42 | best.pt (400) | `bench_har_hhar__vd_ek50_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.490 | 51.0% | 50.4% | 1.2261 | 42 | best.pt (400) | `bench_har_hhar__vd_ek100_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.583 | 58.0% | 57.6% | 1.0900 | 42 | best.pt (9600) | `bench_har_hhar__vd_ek100_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.548 | 54.2% | 55.4% | 1.1067 | 42 | best.pt (400) | `bench_har_hhar__vd_ek100_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.556 | 55.0% | 55.1% | 1.0782 | 42 | best.pt (400) | `bench_har_hhar__vd_ek100_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.498 | 51.1% | 51.1% | 1.2274 | 42 | best.pt (9600) | `bench_har_hhar__vd_ek1000_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.589 | 59.3% | 60.1% | 1.0041 | 42 | best.pt (400) | `bench_har_hhar__vd_ek1000_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.556 | 56.1% | 56.8% | 1.0684 | 42 | best.pt (400) | `bench_har_hhar__vd_ek1000_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.573 | 57.4% | 58.1% | 1.0107 | 42 | best.pt (400) | `bench_har_hhar__vd_ek1000_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 (z-score probe input) | probe | 0.540 | 55.8% | 56.3% | 1.0596 | 42 | best.pt (200) | `bench_har_hhar__vd_ek1000_f100_zs__probe__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.01% of pool, N=196 | probe | 0.407 | 41.5% | 43.0% | 1.4416 | 42 | best.pt (16000) | `bench_har_hhar__g5_k1_f0p01__probe__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.341 | 35.1% | 36.9% | 1.5850 | 42 | best.pt (16000) | `bench_har_hhar__g5_k1_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.01% of pool, N=196 | probe | 0.525 | 53.1% | 54.2% | 1.1129 | 42 | best.pt (7600) | `bench_har_hhar__g5_k5_f0p01__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.571 | 57.6% | 58.1% | 1.0772 | 42 | best.pt (7600) | `bench_har_hhar__g5_k5_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 1% of pool, N=19,593 | probe | 0.561 | 56.6% | 57.0% | 1.0840 | 42 | best.pt (7600) | `bench_har_hhar__g5_k5_f1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.01% of pool, N=196 | probe | 0.498 | 49.6% | 49.8% | 1.1916 | 42 | best.pt (400) | `bench_har_hhar__g5_k10_f0p01__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.564 | 56.9% | 56.8% | 1.1480 | 42 | best.pt (7400) | `bench_har_hhar__g5_k10_f0p1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 1% of pool, N=19,593 | probe | 0.572 | 56.5% | 56.8% | 1.0808 | 42 | best.pt (4800) | `bench_har_hhar__g5_k10_f1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.01% of pool, N=196 | probe | 0.542 | 54.4% | 54.0% | 1.1889 | 42 | best.pt (13400) | `bench_har_hhar__g5_k25_f0p01__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.536 | 53.6% | 53.4% | 1.1568 | 42 | best.pt (13400) | `bench_har_hhar__g5_k25_f0p1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 1% of pool, N=19,593 | probe | 0.570 | 56.9% | 56.7% | 1.1091 | 42 | best.pt (13400) | `bench_har_hhar__g5_k25_f1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.01% of pool, N=196 | probe | 0.524 | 52.5% | 52.5% | 1.1877 | 42 | best.pt (7400) | `bench_har_hhar__g5_k50_f0p01__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.516 | 51.6% | 51.4% | 1.1787 | 42 | best.pt (400) | `bench_har_hhar__g5_k50_f0p1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 1% of pool, N=19,593 | probe | 0.495 | 50.1% | 50.7% | 1.1797 | 42 | best.pt (200) | `bench_har_hhar__g5_k50_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.01% of pool, N=196 | probe | 0.521 | 52.0% | 51.0% | 1.1885 | 42 | best.pt (9600) | `bench_har_hhar__g5_k100_f0p01__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.572 | 56.6% | 56.2% | 1.1002 | 42 | best.pt (5600) | `bench_har_hhar__g5_k100_f0p1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 1% of pool, N=19,593 | probe | 0.579 | 57.2% | 57.6% | 1.0582 | 42 | best.pt (4800) | `bench_har_hhar__g5_k100_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 10% of pool, N=195,929 | probe | 0.499 | 50.3% | 48.7% | 1.1569 | 42 | best.pt (400) | `bench_har_hhar__g5_k100_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.552 | 55.3% | 55.7% | 1.0938 | 42 | best.pt (400) | `bench_har_hhar__g5_k1000_f0p1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 0.606 | 60.6% | 60.9% | 1.0139 | 42 | best.pt (9600) | `bench_har_hhar__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 0.584 | 58.1% | 57.8% | 1.0665 | 42 | best.pt (14200) | `bench_har_hhar__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 0.560 | 57.2% | 58.3% | 1.0606 | 42 | best.pt (200) | `bench_har_hhar__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 0.595 | 61.1% | 61.4% | 1.0183 | 42 | best.pt (800) | `bench_har_hhar__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 0.639 | 63.7% | 63.6% | 1.0095 | 42 | best.pt (3200) | `bench_har_hhar__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 0.545 | 54.9% | 55.3% | 1.1184 | 42 | best.pt (400) | `bench_har_hhar__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 0.595 | 59.1% | 59.3% | 1.0855 | 42 | best.pt (15600) | `bench_har_hhar__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 0.553 | 56.9% | 56.9% | 1.0890 | 42 | best.pt (400) | `bench_har_hhar__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.559 | 56.3% | 56.7% | 1.0677 | 42 | best.pt (400) | `bench_har_hhar__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.565 | 56.3% | 56.3% | 1.0621 | 42 | best.pt (400) | `bench_har_hhar__abl_so3__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | probe | 0.653 | 65.5% | 62.6% | 0.9562 | 42 | best.pt (7000) | `bench_har_hhar__mae_5src_physin__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | finetune | 0.545 | 57.8% | 57.2% | 1.2469 | 42 | best.pt (200) | `bench_har_hhar__mae_5src_physin__finetune__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.324 | 36.2% | 37.7% | 1.4808 | 42 | best.pt (18400) | `bench_har_hhar__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.053 | 16.7% | 19.0% | 1.7903 | 42 | best.pt (2400) | `bench_har_hhar__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 0.592 | 58.9% | 58.2% | 1.1308 | 42 | best.pt (3000) | `bench_har_hhar__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 0.585 | 60.0% | 60.1% | 1.0266 | 42 | best.pt (800) | `bench_har_hhar__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 0.616 | 62.0% | 60.6% | 1.0698 | 42 | best.pt (7400) | `bench_har_hhar__tx5_odom__probe__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 0.561 | 57.1% | 56.1% | 1.1220 | 42 | best.pt (7400) | `bench_har_hhar__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 0.582 | 58.6% | 57.3% | 1.0864 | 42 | best.pt (2000) | `bench_har_hhar__mt_body_gesture__probe__s42` | bench |

## HAR · MHEALTH — `har_mhealth`

_MHEALTH activity recognition from one wrist/arm IMU (12 classes)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) ⚠ eval-overlap pool | probe | 0.689 | 69.9% | 68.4% | 0.5826 | 42 | best.pt (1400) | `vit1d_har_probe_mhealth_mae_multidata` | legacy† |
| Ours (MAE, 11-source) ⚠ eval-overlap pool | finetune | 0.627 | 66.7% | 65.6% | 1.0757 | 42 | best.pt (200) | `vit1d_har_finetune_mhealth_mae_multidata` | legacy† |
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.686 | 70.0% | 68.3% | 0.6036 | 42 | best.pt (600) | `bench_har_mhealth__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.724 | 74.9% | 73.7% | 0.7071 | 42 | best.pt (200) | `bench_har_mhealth__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 0.699 | 70.5% | 68.9% | 0.5657 | 42 | best.pt (2200) | `vit1d_har_probe_mhealth_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.726 | 73.3% | 72.3% | 0.9625 | 42 | best.pt (400) | `bench_har_mhealth__mae_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 0.681 | 69.6% | 68.5% | 0.6110 | 42 | best.pt (2800) | `bench_har_mhealth__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 0.686 | 72.7% | 71.4% | 1.0480 | 42 | best.pt (400) | `bench_har_mhealth__mae_5src_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.661 | 68.5% | 67.2% | 0.7463 | 42 | best.pt (800) | `vit1d_har_probe_mhealth_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.628 | 66.0% | 64.4% | 1.1452 | 42 | best.pt (200) | `bench_har_mhealth__mae_acconly__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 0.685 | 70.8% | 69.3% | 0.6453 | 42 | best.pt (800) | `bench_har_mhealth__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 0.683 | 70.7% | 68.8% | 1.0656 | 42 | best.pt (200) | `bench_har_mhealth__mae_5src_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.644 | 65.5% | 65.1% | 0.8744 | 42 | best.pt (12800) | `vit1d_har_probe_mhealth_focal_multidata` | legacy† |
| FOCAL | finetune | 0.705 | 71.2% | 70.4% | 0.8267 | 42 | best.pt (600) | `bench_har_mhealth__focal__finetune__s42` | bench |
| FOCAL (5-source) | probe | 0.630 | 63.0% | 63.4% | 0.9642 | 42 | best.pt (5800) | `bench_har_mhealth__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 0.659 | 68.2% | 67.8% | 0.8352 | 42 | best.pt (200) | `bench_har_mhealth__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 0.750 | 75.0% | 76.2% | 0.4802 | 42 | best.pt (1600) | `bench_har_mhealth__focal_5src_native__probe__s42` | bench |
| SimMTM | probe | 0.605 | 62.2% | 62.5% | 0.9021 | 42 | best.pt (1400) | `vit1d_har_probe_mhealth_simmtm_multidata` | legacy† |
| SimMTM | finetune | 0.574 | 63.3% | 64.6% | 1.0800 | 42 | best.pt (200) | `bench_har_mhealth__simmtm__finetune__s42` | bench |
| SimMTM (5-source) | probe | 0.614 | 63.1% | 62.5% | 0.8316 | 42 | best.pt (1400) | `bench_har_mhealth__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 0.568 | 62.8% | 62.3% | 0.9701 | 42 | best.pt (200) | `bench_har_mhealth__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 0.584 | 59.4% | 59.5% | 0.8846 | 42 | best.pt (15800) | `bench_har_mhealth__simmtm_5src_native__probe__s42` | bench |
| RelCon | probe | 0.401 | 42.0% | 44.4% | 1.7410 | 42 | best.pt (12000) | `vit1d_har_probe_mhealth_relcon_multidata` | legacy† |
| RelCon | finetune | 0.602 | 65.1% | 65.1% | 0.8966 | 42 | best.pt (200) | `bench_har_mhealth__relcon__finetune__s42` | bench |
| RelCon (5-source) | probe | 0.384 | 40.4% | 42.9% | 1.8627 | 42 | best.pt (12000) | `bench_har_mhealth__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 0.603 | 63.4% | 62.2% | 0.9014 | 42 | best.pt (200) | `bench_har_mhealth__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 0.668 | 67.4% | 68.5% | 0.6034 | 42 | best.pt (5200) | `bench_har_mhealth__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 0.676 | 68.1% | 66.3% | 0.6853 | 42 | best.pt (9200) | `bench_har_mhealth__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 0.629 | 69.2% | 68.1% | 0.8574 | 42 | best.pt (200) | `bench_har_mhealth__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 0.663 | 67.0% | 66.3% | 0.6681 | 42 | best.pt (3600) | `bench_har_mhealth__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 0.648 | 68.5% | 68.9% | 1.0729 | 42 | best.pt (200) | `bench_har_mhealth__sslwear_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.632 | 66.2% | 65.2% | 1.0080 | 42 | best.pt (600) | `bench_har_mhealth__scratch__scratch__s42` | bench |
| Inertia-1 PatchTST-small arch, from scratch (no pretraining) | scratch | 0.779 | 79.9% | 79.6% | 0.6890 | 42 | best.pt (1000) | `bench_har_mhealth__inertia1_patchtst_scratch__scratch__s42` | bench |
| Inertia-1 PatchTST-small arch, random init, frozen (probe control) | probe | 0.670 | 67.3% | 67.8% | 0.7984 | 42 | best.pt (5200) | `bench_har_mhealth__inertia1_patchtst_randinit__probe__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.613 | 61.6% | 61.4% | 0.8551 | 42 | best.pt (3600) | `bench_har_mhealth__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.632 | 64.4% | 63.7% | 0.7860 | 42 | best.pt (3600) | `bench_har_mhealth__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.805 | 79.9% | 80.6% | 0.4108 | 42 | best.pt (6800) | `bench_har_mhealth__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.820 | 81.5% | 82.7% | 0.3985 | 42 | best.pt (13800) | `bench_har_mhealth__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.622 | 63.0% | 62.3% | 0.8260 | 42 | best.pt (11000) | `bench_har_mhealth__mae_recipe_r0__probe__s42` | bench |
| MAE recipe R1 (physical scaling + yaw rotation + lr 3e-4) | probe | 0.708 | 71.4% | 72.3% | 0.5796 | 42 | best.pt (16600) | `bench_har_mhealth__mae_recipe_r1__probe__s42` | bench |
| MAE recipe R2 (R1 + band-weighted loss (4/2/1/0.5)) | probe | 0.642 | 64.9% | 66.0% | 0.6903 | 42 | best.pt (7400) | `bench_har_mhealth__mae_recipe_r2__probe__s42` | bench |
| MAE recipe R3 (R2 + modality masking p=0.3 (replace-mode loss)) | probe | 0.611 | 62.3% | 63.5% | 0.7885 | 42 | best.pt (12600) | `bench_har_mhealth__mae_recipe_r3__probe__s42` | bench |
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.314 | 33.4% | 33.8% | 1.4301 | 42 | best.pt (14400) | `bench_har_mhealth__mae_recipe_rraw__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 0.644 | 65.0% | 64.2% | 0.7403 | 42 | best.pt (11000) | `bench_har_mhealth__mae_mr40__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.674 | 67.5% | 68.2% | 0.6773 | 42 | best.pt (12000) | `bench_har_mhealth__mae_div60_dc__probe__s42` | bench |
| MAE mask-ratio 0.50 | probe | 0.613 | 64.1% | 63.3% | 0.8019 | 42 | best.pt (1400) | `bench_har_mhealth__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.875 | 87.4% | 87.1% | 0.3407 | 42 | best.pt (4000) | `bench_har_mhealth__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.900 | 90.2% | 90.3% | 0.3170 | 42 | best.pt (800) | `bench_har_mhealth__mae_4src_z__finetune__s42` | bench |
| MAE mask-ratio 0.60 | probe | 0.668 | 67.0% | 66.5% | 0.7454 | 42 | best.pt (9200) | `bench_har_mhealth__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.769 | 78.0% | 78.2% | 0.4619 | 42 | best.pt (14600) | `bench_har_mhealth__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.895 | 89.4% | 89.7% | 0.8998 | 42 | best.pt (5200) | `bench_har_mhealth__mae_4src_dc__finetune__s42` | bench |
| MAE mask-ratio 0.85 | probe | 0.623 | 63.7% | 63.5% | 0.7853 | 42 | best.pt (10000) | `bench_har_mhealth__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.857 | 85.3% | 87.0% | 0.3784 | 42 | best.pt (10600) | `bench_har_mhealth__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.833 | 82.7% | 84.2% | 0.5160 | 42 | best.pt (600) | `bench_har_mhealth__mae_4src_phys__finetune__s42` | bench |
| MAE mask-ratio 0.90 | probe | 0.643 | 65.2% | 64.3% | 0.7551 | 42 | best.pt (11000) | `bench_har_mhealth__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.849 | 85.4% | 85.9% | 0.3299 | 42 | best.pt (6600) | `bench_har_mhealth__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.832 | 84.6% | 85.8% | 1.5500 | 42 | best.pt (3800) | `bench_har_mhealth__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.661 | 67.6% | 68.5% | 0.6935 | 42 | best.pt (3000) | `bench_har_mhealth__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.831 | 85.2% | 85.6% | 0.7526 | 42 | best.pt (2600) | `bench_har_mhealth__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.665 | 68.9% | 68.0% | 1.4501 | 42 | best.pt (6800) | `bench_har_mhealth__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.868 | 87.6% | 87.7% | 0.6349 | 42 | best.pt (1000) | `bench_har_mhealth__focal_4src__finetune__s42` | bench |
| RelCon (4-source pool) | probe | 0.817 | 82.2% | 83.5% | 0.4884 | 42 | best.pt (2000) | `bench_har_mhealth__relcon_4src__probe__s42` | bench |
| RelCon (4-source pool) | finetune | 0.830 | 84.2% | 85.7% | 0.5657 | 42 | best.pt (600) | `bench_har_mhealth__relcon_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.848 | 84.9% | 85.1% | 0.3949 | 42 | best.pt (4000) | `bench_har_mhealth__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.892 | 89.6% | 89.3% | 0.5978 | 42 | best.pt (1600) | `bench_har_mhealth__acconly_4src__finetune__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.556 | 59.5% | 60.2% | 0.8779 | — | best.pt | `deepconvlstm_har_mhealth` | legacy† |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.893 | 89.5% | 89.3% | 0.3359 | 42 | best.pt (19200) | `bench_har_mhealth__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.896 | 89.7% | 89.6% | 0.3246 | 42 | best.pt (800) | `bench_har_mhealth__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.919 | 91.9% | 91.9% | 0.2088 | 42 | best.pt (2600) | `bench_har_mhealth__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.851 | 85.6% | 85.7% | 0.3519 | 42 | best.pt (200) | `bench_har_mhealth__inertia1_art__finetune__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 0.750 | 74.8% | 74.3% | 0.5756 | 42 | best.pt (19600) | `bench_har_mhealth__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 0.919 | 92.1% | 91.7% | 0.2081 | 42 | best.pt (1400) | `bench_har_mhealth__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 0.561 | 56.1% | 58.2% | 1.0323 | 42 | best.pt (15800) | `bench_har_mhealth__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 0.456 | 47.8% | 51.1% | 1.4346 | 42 | best.pt (20000) | `bench_har_mhealth__normwear_base__finetune__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 | probe | 0.734 | 74.0% | 74.9% | 0.5279 | 42 | best.pt (12000) | `bench_har_mhealth__vd_ek1_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 | probe | 0.776 | 78.0% | 78.9% | 0.4005 | 42 | best.pt (10000) | `bench_har_mhealth__vd_ek1_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 | probe | 0.768 | 77.1% | 78.6% | 0.5329 | 42 | best.pt (7400) | `bench_har_mhealth__vd_ek5_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 | probe | 0.784 | 78.6% | 79.2% | 0.4615 | 42 | best.pt (13800) | `bench_har_mhealth__vd_ek5_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 | probe | 0.778 | 78.1% | 79.1% | 0.4869 | 42 | best.pt (6800) | `bench_har_mhealth__vd_ek5_f1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 | probe | 0.748 | 75.1% | 75.7% | 0.5513 | 42 | best.pt (17200) | `bench_har_mhealth__vd_ek10_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 | probe | 0.807 | 80.7% | 81.6% | 0.4124 | 42 | best.pt (13800) | `bench_har_mhealth__vd_ek10_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 | probe | 0.791 | 79.0% | 80.0% | 0.4694 | 42 | best.pt (13400) | `bench_har_mhealth__vd_ek10_f1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 | probe | 0.775 | 77.9% | 79.0% | 0.4848 | 42 | best.pt (11400) | `bench_har_mhealth__vd_ek25_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 | probe | 0.805 | 79.9% | 80.6% | 0.4247 | 42 | best.pt (11400) | `bench_har_mhealth__vd_ek25_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 | probe | 0.772 | 77.0% | 78.1% | 0.4518 | 42 | best.pt (6800) | `bench_har_mhealth__vd_ek25_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 | probe | 0.790 | 79.0% | 80.4% | 0.4629 | 42 | best.pt (7400) | `bench_har_mhealth__vd_ek50_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 | probe | 0.811 | 80.5% | 81.9% | 0.4010 | 42 | best.pt (6800) | `bench_har_mhealth__vd_ek50_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 | probe | 0.743 | 73.8% | 74.9% | 0.4517 | 42 | best.pt (6000) | `bench_har_mhealth__vd_ek50_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 | probe | 0.798 | 79.5% | 80.3% | 0.4189 | 42 | best.pt (6800) | `bench_har_mhealth__vd_ek50_f10__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 | probe | 0.784 | 78.9% | 80.1% | 0.5143 | 42 | best.pt (12000) | `bench_har_mhealth__vd_ek100_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 | probe | 0.763 | 76.2% | 76.7% | 0.4650 | 42 | best.pt (12000) | `bench_har_mhealth__vd_ek100_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 | probe | 0.767 | 76.6% | 78.0% | 0.4378 | 42 | best.pt (17200) | `bench_har_mhealth__vd_ek100_f1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 | probe | 0.792 | 79.3% | 80.4% | 0.3945 | 42 | best.pt (17200) | `bench_har_mhealth__vd_ek100_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 | probe | 0.686 | 70.1% | 70.8% | 0.5595 | 42 | best.pt (9600) | `bench_har_mhealth__vd_ek1000_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 | probe | 0.821 | 81.6% | 83.2% | 0.3981 | 42 | best.pt (6800) | `bench_har_mhealth__vd_ek1000_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 | probe | 0.859 | 86.0% | 86.0% | 0.3808 | 42 | best.pt (11400) | `bench_har_mhealth__vd_ek1000_f1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 | probe | 0.791 | 79.6% | 79.6% | 0.4471 | 42 | best.pt (13800) | `bench_har_mhealth__vd_ek1000_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 | probe | 0.851 | 84.7% | 85.8% | 0.3633 | 42 | best.pt (11400) | `bench_har_mhealth__vd_ek1000_f100__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.616 | 62.2% | 62.9% | 0.8795 | 42 | best.pt (12000) | `bench_har_mhealth__vd_ek1_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.619 | 63.2% | 63.9% | 0.8480 | 42 | best.pt (4200) | `bench_har_mhealth__vd_ek1_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.661 | 66.5% | 66.4% | 0.7656 | 42 | best.pt (12800) | `bench_har_mhealth__vd_ek5_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.640 | 65.0% | 66.1% | 0.7433 | 42 | best.pt (6800) | `bench_har_mhealth__vd_ek5_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.613 | 62.4% | 62.8% | 0.7351 | 42 | best.pt (17000) | `bench_har_mhealth__vd_ek5_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.661 | 67.0% | 66.2% | 0.7050 | 42 | best.pt (12200) | `bench_har_mhealth__vd_ek10_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.647 | 65.8% | 66.4% | 0.6839 | 42 | best.pt (11200) | `bench_har_mhealth__vd_ek10_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.650 | 65.6% | 66.8% | 0.6918 | 42 | best.pt (6800) | `bench_har_mhealth__vd_ek10_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.634 | 64.6% | 63.8% | 0.7508 | 42 | best.pt (12800) | `bench_har_mhealth__vd_ek25_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.662 | 68.0% | 67.9% | 0.6583 | 42 | best.pt (11200) | `bench_har_mhealth__vd_ek25_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.669 | 68.0% | 67.3% | 0.6668 | 42 | best.pt (15800) | `bench_har_mhealth__vd_ek25_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.657 | 67.0% | 66.9% | 0.6797 | 42 | best.pt (12800) | `bench_har_mhealth__vd_ek50_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.680 | 68.4% | 67.7% | 0.6666 | 42 | best.pt (11400) | `bench_har_mhealth__vd_ek50_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.627 | 64.0% | 64.3% | 0.7530 | 42 | best.pt (3600) | `bench_har_mhealth__vd_ek50_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.614 | 62.7% | 63.4% | 0.7891 | 42 | best.pt (3600) | `bench_har_mhealth__vd_ek50_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.630 | 64.2% | 65.1% | 0.7405 | 42 | best.pt (12800) | `bench_har_mhealth__vd_ek100_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.658 | 66.3% | 66.8% | 0.6850 | 42 | best.pt (11200) | `bench_har_mhealth__vd_ek100_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.698 | 71.1% | 70.1% | 0.6293 | 42 | best.pt (11200) | `bench_har_mhealth__vd_ek100_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.613 | 63.6% | 62.9% | 0.7876 | 42 | best.pt (3600) | `bench_har_mhealth__vd_ek100_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.666 | 67.8% | 66.6% | 0.7335 | 42 | best.pt (9200) | `bench_har_mhealth__vd_ek1000_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.668 | 68.3% | 66.8% | 0.6930 | 42 | best.pt (1000) | `bench_har_mhealth__vd_ek1000_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.663 | 68.3% | 66.7% | 0.6152 | 42 | best.pt (1000) | `bench_har_mhealth__vd_ek1000_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.687 | 71.4% | 70.3% | 0.5639 | 42 | best.pt (800) | `bench_har_mhealth__vd_ek1000_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 (z-score probe input) | probe | 0.663 | 69.6% | 68.3% | 0.6007 | 42 | best.pt (800) | `bench_har_mhealth__vd_ek1000_f100_zs__probe__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.01% of pool, N=196 | probe | 0.545 | 54.4% | 56.2% | 1.1743 | 42 | best.pt (13400) | `bench_har_mhealth__g5_k1_f0p01__probe__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.456 | 47.2% | 50.4% | 1.3880 | 42 | best.pt (13400) | `bench_har_mhealth__g5_k1_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.01% of pool, N=196 | probe | 0.698 | 69.6% | 68.7% | 0.6782 | 42 | best.pt (9200) | `bench_har_mhealth__g5_k5_f0p01__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.680 | 68.3% | 67.9% | 0.6835 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k5_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 1% of pool, N=19,593 | probe | 0.670 | 66.6% | 66.3% | 0.6956 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k5_f1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.01% of pool, N=196 | probe | 0.642 | 65.8% | 66.4% | 0.7760 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k10_f0p01__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.693 | 69.3% | 68.5% | 0.5933 | 42 | best.pt (2600) | `bench_har_mhealth__g5_k10_f0p1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 1% of pool, N=19,593 | probe | 0.686 | 69.4% | 68.7% | 0.6435 | 42 | best.pt (1800) | `bench_har_mhealth__g5_k10_f1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.01% of pool, N=196 | probe | 0.638 | 64.1% | 64.1% | 0.7875 | 42 | best.pt (15800) | `bench_har_mhealth__g5_k25_f0p01__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.691 | 68.5% | 70.1% | 0.6625 | 42 | best.pt (11200) | `bench_har_mhealth__g5_k25_f0p1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 1% of pool, N=19,593 | probe | 0.689 | 69.7% | 69.4% | 0.6498 | 42 | best.pt (11600) | `bench_har_mhealth__g5_k25_f1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.01% of pool, N=196 | probe | 0.630 | 63.7% | 63.5% | 0.7611 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k50_f0p01__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.632 | 63.3% | 64.3% | 0.7879 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k50_f0p1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 1% of pool, N=19,593 | probe | 0.644 | 65.5% | 66.3% | 0.7360 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k50_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.01% of pool, N=196 | probe | 0.645 | 65.0% | 64.4% | 0.7560 | 42 | best.pt (11200) | `bench_har_mhealth__g5_k100_f0p01__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.669 | 68.6% | 68.0% | 0.6673 | 42 | best.pt (6600) | `bench_har_mhealth__g5_k100_f0p1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 1% of pool, N=19,593 | probe | 0.677 | 69.0% | 68.4% | 0.6414 | 42 | best.pt (11200) | `bench_har_mhealth__g5_k100_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 10% of pool, N=195,929 | probe | 0.642 | 66.9% | 65.7% | 0.6851 | 42 | best.pt (3600) | `bench_har_mhealth__g5_k100_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.653 | 66.7% | 65.2% | 0.6741 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k1000_f0p1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 0.667 | 68.6% | 67.2% | 0.6425 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 0.644 | 66.2% | 65.1% | 0.6917 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 0.670 | 68.5% | 67.0% | 0.6747 | 42 | best.pt (1400) | `bench_har_mhealth__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 0.682 | 70.1% | 68.5% | 0.5745 | 42 | best.pt (1000) | `bench_har_mhealth__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 0.686 | 69.7% | 68.4% | 0.5698 | 42 | best.pt (2200) | `bench_har_mhealth__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 0.718 | 72.6% | 71.7% | 0.5725 | 42 | best.pt (13600) | `bench_har_mhealth__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 0.657 | 66.6% | 65.8% | 0.6223 | 42 | best.pt (1400) | `bench_har_mhealth__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 0.640 | 67.5% | 65.8% | 0.6621 | 42 | best.pt (1400) | `bench_har_mhealth__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.728 | 73.5% | 72.3% | 0.6317 | 42 | best.pt (1400) | `bench_har_mhealth__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.704 | 70.7% | 69.6% | 0.6163 | 42 | best.pt (5200) | `bench_har_mhealth__abl_so3__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | probe | 0.868 | 86.7% | 86.6% | 0.3127 | 42 | best.pt (2000) | `bench_har_mhealth__mae_5src_physin__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | finetune | 0.809 | 81.3% | 80.6% | 0.4633 | 42 | best.pt (200) | `bench_har_mhealth__mae_5src_physin__finetune__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.466 | 48.9% | 51.8% | 1.4353 | 42 | best.pt (12000) | `bench_har_mhealth__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.014 | 8.3% | 9.2% | 2.5465 | 42 | best.pt (2200) | `bench_har_mhealth__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 0.663 | 67.5% | 66.6% | 0.7973 | 42 | best.pt (1000) | `bench_har_mhealth__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 0.649 | 68.0% | 66.3% | 0.6975 | 42 | best.pt (800) | `bench_har_mhealth__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 0.688 | 69.7% | 68.5% | 0.7721 | 42 | best.pt (1400) | `bench_har_mhealth__tx5_odom__probe__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 0.654 | 67.4% | 66.4% | 0.8579 | 42 | best.pt (800) | `bench_har_mhealth__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 0.689 | 69.9% | 69.2% | 0.6972 | 42 | best.pt (3000) | `bench_har_mhealth__mt_body_gesture__probe__s42` | bench |

## HAR · User-study watch — `har_userwatch`

_User-study smartwatch, 60 fine-grained everyday activities (10 participants)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.727 | 72.2% | 73.0% | 1.0929 | 42 | best.pt (2600) | `bench_har_userwatch__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.861 | 85.4% | 85.9% | 0.5593 | 42 | best.pt (600) | `bench_har_userwatch__mae_5src__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 0.730 | 71.4% | 72.7% | 1.1948 | 42 | best.pt (16600) | `bench_har_userwatch__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 0.786 | 78.1% | 79.8% | 0.8469 | 42 | best.pt (600) | `bench_har_userwatch__mae_5src_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 0.695 | 69.0% | 70.0% | 1.3254 | 42 | best.pt (6400) | `bench_har_userwatch__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 0.772 | 76.6% | 77.8% | 0.9421 | 42 | best.pt (600) | `bench_har_userwatch__mae_5src_acconly__finetune__s42` | bench |
| FOCAL (5-source) | probe | 0.470 | 48.1% | 48.9% | 2.0571 | 42 | best.pt (16400) | `bench_har_userwatch__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 0.683 | 67.5% | 68.9% | 1.3417 | 42 | best.pt (1000) | `bench_har_userwatch__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 0.800 | 80.2% | 81.8% | 0.9832 | 42 | best.pt (800) | `bench_har_userwatch__focal_5src_native__probe__s42` | bench |
| SimMTM (5-source) | probe | 0.668 | 67.2% | 68.7% | 1.6055 | 42 | best.pt (16600) | `bench_har_userwatch__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 0.737 | 74.3% | 74.6% | 1.0703 | 42 | best.pt (600) | `bench_har_userwatch__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 0.514 | 51.3% | 53.9% | 2.1124 | 42 | best.pt (17600) | `bench_har_userwatch__simmtm_5src_native__probe__s42` | bench |
| RelCon (5-source) | probe | 0.516 | 53.1% | 53.3% | 2.0659 | 42 | best.pt (18200) | `bench_har_userwatch__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 0.790 | 78.3% | 79.1% | 0.8685 | 42 | best.pt (600) | `bench_har_userwatch__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 0.649 | 66.5% | 69.3% | 1.3896 | 42 | best.pt (16000) | `bench_har_userwatch__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 0.711 | 70.3% | 71.4% | 1.2217 | 42 | best.pt (16600) | `bench_har_userwatch__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 0.758 | 76.3% | 78.3% | 0.9259 | 42 | best.pt (600) | `bench_har_userwatch__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 0.674 | 66.6% | 67.2% | 1.3501 | 42 | best.pt (17400) | `bench_har_userwatch__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 0.759 | 75.6% | 78.0% | 0.9325 | 42 | best.pt (600) | `bench_har_userwatch__sslwear_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.735 | 73.4% | 74.4% | 1.1724 | 42 | best.pt (1200) | `bench_har_userwatch__scratch__scratch__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.716 | 72.9% | 72.8% | 1.3207 | — | best.pt | `deepconvlstm_har_userwatch` | legacy† |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 0.467 | 45.5% | 49.8% | 2.3499 | 42 | best.pt (19800) | `bench_har_userwatch__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 0.887 | 87.9% | 88.6% | 0.4568 | 42 | best.pt (3600) | `bench_har_userwatch__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 0.337 | 34.6% | 39.8% | 2.4798 | 42 | best.pt (19200) | `bench_har_userwatch__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 0.103 | 13.4% | 20.4% | 3.1349 | 42 | best.pt (19800) | `bench_har_userwatch__normwear_base__finetune__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 0.646 | 64.3% | 64.9% | 1.4145 | 42 | best.pt (17200) | `bench_har_userwatch__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 0.710 | 69.8% | 70.6% | 1.3131 | 42 | best.pt (13000) | `bench_har_userwatch__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 0.681 | 68.1% | 68.3% | 1.4683 | 42 | best.pt (16600) | `bench_har_userwatch__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 0.720 | 71.6% | 72.6% | 1.1533 | 42 | best.pt (6200) | `bench_har_userwatch__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 0.750 | 74.6% | 74.7% | 1.0990 | 42 | best.pt (7600) | `bench_har_userwatch__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 0.723 | 72.2% | 73.0% | 1.1653 | 42 | best.pt (10800) | `bench_har_userwatch__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 0.656 | 65.2% | 66.0% | 1.4861 | 42 | best.pt (14200) | `bench_har_userwatch__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 0.657 | 65.2% | 65.9% | 1.3584 | 42 | best.pt (15400) | `bench_har_userwatch__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.725 | 71.4% | 72.4% | 1.1364 | 42 | best.pt (13000) | `bench_har_userwatch__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.693 | 68.8% | 70.1% | 1.2362 | 42 | best.pt (7200) | `bench_har_userwatch__abl_so3__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | probe | 0.844 | 84.5% | 85.6% | 0.7227 | 42 | best.pt (1800) | `bench_har_userwatch__mae_5src_physin__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | finetune | 0.888 | 88.3% | 88.7% | 0.5057 | 42 | best.pt (400) | `bench_har_userwatch__mae_5src_physin__finetune__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.097 | 12.1% | 17.7% | 3.3085 | 42 | best.pt (19400) | `bench_har_userwatch__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.001 | 1.7% | 4.6% | 4.0335 | 42 | best.pt (15800) | `bench_har_userwatch__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 0.812 | 80.1% | 80.9% | 0.7873 | 42 | best.pt (1000) | `bench_har_userwatch__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 0.706 | 70.1% | 71.7% | 1.2153 | 42 | best.pt (1600) | `bench_har_userwatch__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 0.778 | 77.0% | 76.2% | 1.0730 | 42 | best.pt (19800) | `bench_har_userwatch__tx5_odom__probe__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 0.798 | 78.6% | 78.8% | 1.0097 | 42 | best.pt (10000) | `bench_har_userwatch__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 0.761 | 75.7% | 76.8% | 1.0445 | 42 | best.pt (1800) | `bench_har_userwatch__mt_body_gesture__probe__s42` | bench |

## HAR · PRISM — `har_prism`

_PrISM-Tracker (Arakawa et al.) — PRISM activity recognition from one wrist IMU (5 global ADL classes, 31 subjects)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint, re-evaluated with pooled predictions (bench.score.har).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.279 | 27.2% | 67.0% | 0.8894 | 42 | best.pt (3800) | `bench_har_prism__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.368 | 36.2% | 67.8% | 0.8758 | 42 | best.pt (400) | `bench_har_prism__mae_5src__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 0.265 | 26.1% | 66.4% | 0.9005 | 42 | best.pt (12800) | `bench_har_prism__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 0.362 | 35.3% | 70.0% | 0.8881 | 42 | best.pt (600) | `bench_har_prism__mae_5src_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 0.280 | 27.2% | 66.5% | 0.9116 | 42 | best.pt (12800) | `bench_har_prism__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 0.298 | 29.5% | 67.5% | 0.9099 | 42 | best.pt (200) | `bench_har_prism__mae_5src_acconly__finetune__s42` | bench |
| FOCAL (5-source) | probe | 0.219 | 23.2% | 65.9% | 0.9429 | 42 | best.pt (8200) | `bench_har_prism__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 0.288 | 27.9% | 68.5% | 0.9056 | 42 | best.pt (1400) | `bench_har_prism__focal_5src__finetune__s42` | bench |
| FOCAL (5-source, native frequency-domain input) | probe | 0.338 | 32.6% | 70.8% | 0.7876 | 42 | best.pt (3800) | `bench_har_prism__focal_5src_native__probe__s42` | bench |
| SimMTM (5-source) | probe | 0.267 | 26.3% | 66.9% | 0.9108 | 42 | best.pt (12800) | `bench_har_prism__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 0.308 | 30.6% | 67.3% | 0.8985 | 42 | best.pt (800) | `bench_har_prism__simmtm_5src__finetune__s42` | bench |
| SimMTM (5-source, native normalisation) | probe | 0.249 | 25.1% | 66.7% | 0.9154 | 42 | best.pt (12800) | `bench_har_prism__simmtm_5src_native__probe__s42` | bench |
| RelCon (5-source) | probe | 0.183 | 21.2% | 64.8% | 0.9642 | 42 | best.pt (14200) | `bench_har_prism__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 0.289 | 28.4% | 68.4% | 0.9180 | 42 | best.pt (200) | `bench_har_prism__relcon_5src__finetune__s42` | bench |
| RelCon (5-source, native normalisation) | probe | 0.294 | 29.1% | 69.9% | 0.8074 | 42 | best.pt (1400) | `bench_har_prism__relcon_5src_native__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 0.251 | 25.1% | 66.0% | 0.9012 | 42 | best.pt (12800) | `bench_har_prism__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 0.295 | 29.2% | 64.9% | 0.9368 | 42 | best.pt (200) | `bench_har_prism__limubert_5src__finetune__s42` | bench |
| SSL-Wearables (5-source) | probe | 0.276 | 27.1% | 66.3% | 0.8892 | 42 | best.pt (8200) | `bench_har_prism__sslwear_5src__probe__s42` | bench |
| SSL-Wearables (5-source) | finetune | 0.328 | 32.5% | 67.4% | 0.9063 | 42 | best.pt (600) | `bench_har_prism__sslwear_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.278 | 27.2% | 68.5% | 0.9274 | 42 | best.pt (1400) | `bench_har_prism__scratch__scratch__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.426 | 51.0% | 54.4% | 1.1287 | 42 | best.pt (2800) | `bench_har_prism__mae_nhanes2x2_a__probe__s42` | bench |
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.254 | 25.9% | 68.5% | 0.8755 | 42 | best.pt (9400) | `bench_har_prism__mae_recipe_rraw__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.317 | 30.7% | 70.2% | 0.8132 | 42 | best.pt (4600) | `bench_har_prism__mae_div60_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.428 | 51.2% | 55.3% | 1.1085 | 42 | best.pt (2800) | `bench_har_prism__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.464 | 52.9% | 70.4% | 1.0006 | 42 | best.pt (600) | `bench_har_prism__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.432 | 50.3% | 54.1% | 1.1784 | 42 | best.pt (2800) | `bench_har_prism__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.398 | 42.7% | 61.6% | 1.0549 | 42 | best.pt (1000) | `bench_har_prism__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.435 | 52.3% | 53.3% | 1.1424 | 42 | best.pt (10200) | `bench_har_prism__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.446 | 54.5% | 65.9% | 1.0348 | 42 | best.pt (600) | `bench_har_prism__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.414 | 52.7% | 46.9% | 1.1431 | 42 | best.pt (1800) | `bench_har_prism__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.459 | 53.1% | 63.2% | 1.0517 | 42 | best.pt (400) | `bench_har_prism__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.382 | 39.9% | 62.2% | 1.0796 | 42 | best.pt (3800) | `bench_har_prism__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.422 | 49.5% | 66.4% | 1.0210 | 42 | best.pt (600) | `bench_har_prism__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.382 | 41.5% | 58.2% | 1.1642 | 42 | best.pt (3800) | `bench_har_prism__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.403 | 44.5% | 61.1% | 1.0558 | 42 | best.pt (1800) | `bench_har_prism__focal_4src__finetune__s42` | bench |
| RelCon (4-source pool) | probe | 0.440 | 52.8% | 63.3% | 1.0374 | 42 | best.pt (1000) | `bench_har_prism__relcon_4src__probe__s42` | bench |
| RelCon (4-source pool) | finetune | 0.475 | 53.2% | 70.9% | 0.9940 | 42 | best.pt (600) | `bench_har_prism__relcon_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.453 | 52.8% | 63.3% | 1.0259 | 42 | best.pt (8200) | `bench_har_prism__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.440 | 46.0% | 58.5% | 1.0692 | 42 | best.pt (400) | `bench_har_prism__acconly_4src__finetune__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.356 | 34.0% | 70.6% | 0.8272 | — | best.pt | `deepconvlstm_har_prism` | legacy† |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | probe | 0.235 | 24.5% | 67.4% | 0.8896 | 42 | best.pt (12200) | `bench_har_prism__harnet5__probe__s42` | bench |
| HarNet-5s (Yuan et al. 2024, UK Biobank) | finetune | 0.373 | 35.0% | 72.2% | 0.7476 | 42 | best.pt (400) | `bench_har_prism__harnet5__finetune__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | probe | 0.192 | 21.7% | 64.6% | 0.9538 | 42 | best.pt (1400) | `bench_har_prism__normwear_base__probe__s42` | bench |
| NormWear (Luo et al. 2024, released checkpoint) | finetune | 0.183 | 21.2% | 64.6% | 0.9600 | 42 | best.pt (10600) | `bench_har_prism__normwear_base__finetune__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 | probe | 0.332 | 32.2% | 71.6% | 0.7828 | 42 | best.pt (10400) | `bench_har_prism__vd_ek1_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 | probe | 0.349 | 33.3% | 72.8% | 0.7472 | 42 | best.pt (8200) | `bench_har_prism__vd_ek1_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 | probe | 0.336 | 32.1% | 72.0% | 0.7825 | 42 | best.pt (10400) | `bench_har_prism__vd_ek5_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 | probe | 0.339 | 32.3% | 71.6% | 0.7643 | 42 | best.pt (12800) | `bench_har_prism__vd_ek5_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 | probe | 0.341 | 32.5% | 71.8% | 0.7724 | 42 | best.pt (8200) | `bench_har_prism__vd_ek5_f1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 | probe | 0.332 | 31.8% | 71.7% | 0.7847 | 42 | best.pt (10400) | `bench_har_prism__vd_ek10_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 | probe | 0.335 | 32.2% | 71.5% | 0.7694 | 42 | best.pt (8200) | `bench_har_prism__vd_ek10_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 | probe | 0.351 | 34.0% | 71.9% | 0.7635 | 42 | best.pt (10400) | `bench_har_prism__vd_ek10_f1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 | probe | 0.326 | 31.2% | 71.1% | 0.7967 | 42 | best.pt (2600) | `bench_har_prism__vd_ek25_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 | probe | 0.352 | 34.0% | 71.7% | 0.7729 | 42 | best.pt (10400) | `bench_har_prism__vd_ek25_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 | probe | 0.337 | 32.3% | 71.6% | 0.7646 | 42 | best.pt (8200) | `bench_har_prism__vd_ek25_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 | probe | 0.341 | 33.0% | 71.6% | 0.7688 | 42 | best.pt (10400) | `bench_har_prism__vd_ek50_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 | probe | 0.337 | 32.3% | 71.5% | 0.7749 | 42 | best.pt (9400) | `bench_har_prism__vd_ek50_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 | probe | 0.343 | 32.8% | 72.0% | 0.7682 | 42 | best.pt (8200) | `bench_har_prism__vd_ek50_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 | probe | 0.331 | 31.9% | 71.6% | 0.7822 | 42 | best.pt (8200) | `bench_har_prism__vd_ek50_f10__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 | probe | 0.335 | 32.2% | 71.6% | 0.7761 | 42 | best.pt (10400) | `bench_har_prism__vd_ek100_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 | probe | 0.343 | 32.8% | 72.0% | 0.7657 | 42 | best.pt (8200) | `bench_har_prism__vd_ek100_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 | probe | 0.341 | 32.7% | 71.9% | 0.7688 | 42 | best.pt (9400) | `bench_har_prism__vd_ek100_f1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 | probe | 0.344 | 33.1% | 72.2% | 0.7707 | 42 | best.pt (9400) | `bench_har_prism__vd_ek100_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 | probe | 0.334 | 32.7% | 71.2% | 0.7920 | 42 | best.pt (4600) | `bench_har_prism__vd_ek1000_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 | probe | 0.344 | 33.4% | 71.3% | 0.7715 | 42 | best.pt (10400) | `bench_har_prism__vd_ek1000_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 | probe | 0.323 | 31.1% | 71.1% | 0.7748 | 42 | best.pt (12800) | `bench_har_prism__vd_ek1000_f1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 | probe | 0.348 | 33.7% | 72.0% | 0.7600 | 42 | best.pt (10400) | `bench_har_prism__vd_ek1000_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 | probe | 0.333 | 32.0% | 71.8% | 0.7688 | 42 | best.pt (8200) | `bench_har_prism__vd_ek1000_f100__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.234 | 24.0% | 65.5% | 0.9391 | 42 | best.pt (1800) | `bench_har_prism__vd_ek1_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.264 | 26.3% | 66.6% | 0.9107 | 42 | best.pt (3800) | `bench_har_prism__vd_ek1_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.235 | 24.3% | 65.2% | 0.9336 | 42 | best.pt (1200) | `bench_har_prism__vd_ek5_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.269 | 26.6% | 66.8% | 0.9015 | 42 | best.pt (9400) | `bench_har_prism__vd_ek5_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.273 | 26.8% | 67.2% | 0.9022 | 42 | best.pt (9400) | `bench_har_prism__vd_ek5_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.243 | 24.7% | 65.5% | 0.9398 | 42 | best.pt (1200) | `bench_har_prism__vd_ek10_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.260 | 25.9% | 66.9% | 0.9020 | 42 | best.pt (9400) | `bench_har_prism__vd_ek10_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.261 | 26.1% | 66.4% | 0.9126 | 42 | best.pt (2800) | `bench_har_prism__vd_ek10_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.246 | 25.1% | 65.0% | 0.9391 | 42 | best.pt (3800) | `bench_har_prism__vd_ek25_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.251 | 25.3% | 66.3% | 0.9240 | 42 | best.pt (2800) | `bench_har_prism__vd_ek25_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.282 | 27.5% | 67.2% | 0.9013 | 42 | best.pt (12800) | `bench_har_prism__vd_ek25_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.267 | 26.3% | 66.3% | 0.9216 | 42 | best.pt (12800) | `bench_har_prism__vd_ek50_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.281 | 27.3% | 67.7% | 0.8942 | 42 | best.pt (9400) | `bench_har_prism__vd_ek50_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.272 | 26.7% | 66.7% | 0.9023 | 42 | best.pt (12800) | `bench_har_prism__vd_ek50_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.261 | 26.0% | 66.7% | 0.9123 | 42 | best.pt (9400) | `bench_har_prism__vd_ek50_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.226 | 23.6% | 64.8% | 0.9404 | 42 | best.pt (1200) | `bench_har_prism__vd_ek100_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.264 | 26.2% | 66.4% | 0.9120 | 42 | best.pt (2800) | `bench_har_prism__vd_ek100_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.258 | 25.7% | 66.5% | 0.9048 | 42 | best.pt (9400) | `bench_har_prism__vd_ek100_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.255 | 25.4% | 66.2% | 0.9081 | 42 | best.pt (9400) | `bench_har_prism__vd_ek100_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 0.257 | 25.5% | 66.4% | 0.9328 | 42 | best.pt (14200) | `bench_har_prism__vd_ek1000_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 0.249 | 25.4% | 65.1% | 0.9308 | 42 | best.pt (1200) | `bench_har_prism__vd_ek1000_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 0.268 | 26.4% | 66.7% | 0.9051 | 42 | best.pt (9400) | `bench_har_prism__vd_ek1000_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 0.258 | 25.9% | 65.1% | 0.9330 | 42 | best.pt (1200) | `bench_har_prism__vd_ek1000_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 (z-score probe input) | probe | 0.255 | 25.4% | 65.8% | 0.9210 | 42 | best.pt (1600) | `bench_har_prism__vd_ek1000_f100_zs__probe__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.01% of pool, N=196 | probe | 0.183 | 21.4% | 64.3% | 0.9886 | 42 | best.pt (1200) | `bench_har_prism__g5_k1_f0p01__probe__s42` | bench |
| Grid 5-source pool, 1 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.180 | 21.1% | 64.4% | 0.9917 | 42 | best.pt (1200) | `bench_har_prism__g5_k1_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.01% of pool, N=196 | probe | 0.239 | 24.4% | 65.0% | 0.9453 | 42 | best.pt (1200) | `bench_har_prism__g5_k5_f0p01__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.239 | 24.6% | 65.1% | 0.9408 | 42 | best.pt (1200) | `bench_har_prism__g5_k5_f0p1__probe__s42` | bench |
| Grid 5-source pool, 5 largest motion type(s), 1% of pool, N=19,593 | probe | 0.232 | 24.1% | 65.0% | 0.9476 | 42 | best.pt (1200) | `bench_har_prism__g5_k5_f1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.01% of pool, N=196 | probe | 0.247 | 24.9% | 65.0% | 0.9398 | 42 | best.pt (1200) | `bench_har_prism__g5_k10_f0p01__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.259 | 25.8% | 66.1% | 0.9196 | 42 | best.pt (1200) | `bench_har_prism__g5_k10_f0p1__probe__s42` | bench |
| Grid 5-source pool, 10 largest motion type(s), 1% of pool, N=19,593 | probe | 0.255 | 25.5% | 65.4% | 0.9274 | 42 | best.pt (1200) | `bench_har_prism__g5_k10_f1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.01% of pool, N=196 | probe | 0.239 | 24.5% | 65.2% | 0.9348 | 42 | best.pt (3800) | `bench_har_prism__g5_k25_f0p01__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.254 | 25.3% | 66.2% | 0.9146 | 42 | best.pt (9400) | `bench_har_prism__g5_k25_f0p1__probe__s42` | bench |
| Grid 5-source pool, 25 largest motion type(s), 1% of pool, N=19,593 | probe | 0.252 | 25.2% | 66.4% | 0.9064 | 42 | best.pt (9400) | `bench_har_prism__g5_k25_f1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.01% of pool, N=196 | probe | 0.259 | 25.8% | 66.4% | 0.9182 | 42 | best.pt (9400) | `bench_har_prism__g5_k50_f0p01__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.257 | 25.8% | 66.1% | 0.9241 | 42 | best.pt (1200) | `bench_har_prism__g5_k50_f0p1__probe__s42` | bench |
| Grid 5-source pool, 50 largest motion type(s), 1% of pool, N=19,593 | probe | 0.264 | 26.2% | 66.8% | 0.9077 | 42 | best.pt (9400) | `bench_har_prism__g5_k50_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.01% of pool, N=196 | probe | 0.248 | 25.0% | 65.5% | 0.9318 | 42 | best.pt (3800) | `bench_har_prism__g5_k100_f0p01__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.237 | 24.4% | 65.6% | 0.9236 | 42 | best.pt (1200) | `bench_har_prism__g5_k100_f0p1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 1% of pool, N=19,593 | probe | 0.264 | 26.1% | 66.8% | 0.9007 | 42 | best.pt (3800) | `bench_har_prism__g5_k100_f1__probe__s42` | bench |
| Grid 5-source pool, 100 largest motion type(s), 10% of pool, N=195,929 | probe | 0.246 | 24.8% | 65.7% | 0.9155 | 42 | best.pt (3800) | `bench_har_prism__g5_k100_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 0.1% of pool, N=1,959 | probe | 0.254 | 25.5% | 66.1% | 0.9169 | 42 | best.pt (1200) | `bench_har_prism__g5_k1000_f0p1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 1% of pool, N=19,593 | probe | 0.251 | 25.1% | 66.3% | 0.9099 | 42 | best.pt (12800) | `bench_har_prism__g5_k1000_f1__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 10% of pool, N=195,929 | probe | 0.251 | 25.3% | 65.7% | 0.9136 | 42 | best.pt (1200) | `bench_har_prism__g5_k1000_f10__probe__s42` | bench |
| Grid 5-source pool, 1000 largest motion type(s), 100% of pool, N=1,959,288 | probe | 0.246 | 24.9% | 65.9% | 0.9206 | 42 | best.pt (2800) | `bench_har_prism__g5_k1000_f100__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 1% of pool, N=19,593, 150K steps | probe | 0.280 | 27.3% | 67.5% | 0.8915 | 42 | best.pt (12800) | `bench_har_prism__g5_k1000_f1_150k__probe__s42` | bench |
| Grid 5-source pool, 1000 motion types, 100% of pool, N=1,959,288, 150K steps | probe | 0.266 | 26.2% | 66.1% | 0.8997 | 42 | best.pt (3800) | `bench_har_prism__g5_k1000_f100_150k__probe__s42` | bench |
| Corpus ablation 'ref' (per-source counts identical to the reference corpus (HAR/SONIC/Motion-X in full) | probe | 0.271 | 26.6% | 66.9% | 0.8893 | 42 | best.pt (12800) | `bench_har_prism__corpus_ref__probe__s42` | bench |
| Corpus ablation 'uniform' (uniform random draw from the 5-source pool) | probe | 0.246 | 24.7% | 65.3% | 0.9210 | 42 | best.pt (3800) | `bench_har_prism__corpus_uniform__probe__s42` | bench |
| Corpus ablation 'temp03' (source temperature sampling p_s ~ n_s^0.3 (Motion-X in full)) | probe | 0.251 | 25.1% | 66.0% | 0.9071 | 42 | best.pt (12800) | `bench_har_prism__corpus_temp03__probe__s42` | bench |
| Design ablation 'norot' (no rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.285 | 27.7% | 67.0% | 0.8977 | 42 | best.pt (11000) | `bench_har_prism__abl_norot__probe__s42` | bench |
| Design ablation 'so3' (continuous SO(3) rotation augmentation; otherwise the corpus 'ref' arm) | probe | 0.265 | 26.3% | 66.3% | 0.9198 | 42 | best.pt (1200) | `bench_har_prism__abl_so3__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | probe | 0.356 | 33.7% | 72.6% | 0.7416 | 42 | best.pt (9400) | `bench_har_prism__mae_5src_physin__probe__s42` | bench |
| DIAGNOSTIC -- the reference MAE (mae_5src, z-score pretrained) probed / fine-tuned on PHYSICAL-unit input | finetune | 0.383 | 37.1% | 74.3% | 0.7098 | 42 | best.pt (600) | `bench_har_prism__mae_5src_physin__finetune__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.161 | 20.2% | 64.1% | 0.9920 | 42 | best.pt (1200) | `bench_har_prism__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.156 | 20.0% | 64.2% | 1.0243 | 42 | best.pt (1200) | `bench_har_prism__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 0.308 | 29.4% | 68.0% | 0.8717 | 42 | best.pt (3800) | `bench_har_prism__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | finetune | 0.358 | 33.3% | 70.2% | 0.8404 | 42 | best.pt (200) | `bench_har_prism__tx5_fullbody__finetune__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 0.283 | 27.5% | 66.8% | 0.8961 | 42 | best.pt (3800) | `bench_har_prism__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | finetune | 0.295 | 28.4% | 68.9% | 0.8920 | 42 | best.pt (200) | `bench_har_prism__tx5_gesture__finetune__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 0.260 | 26.2% | 66.4% | 0.9155 | 42 | best.pt (8200) | `bench_har_prism__tx5_odom__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | finetune | 0.342 | 32.0% | 70.4% | 0.8451 | 42 | best.pt (200) | `bench_har_prism__tx5_odom__finetune__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | probe | 0.281 | 27.4% | 66.2% | 0.9049 | 42 | best.pt (4000) | `bench_har_prism__tx5_pd__probe__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | finetune | 0.322 | 30.6% | 69.5% | 0.8768 | 42 | best.pt (200) | `bench_har_prism__tx5_pd__finetune__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | probe | 0.289 | 28.0% | 66.9% | 0.8926 | 42 | best.pt (3800) | `bench_har_prism__tx5_language__probe__s42` | bench |
| MAE-5src fine-tuned on IMU-text retrieval | finetune | 0.382 | 36.7% | 70.5% | 0.8302 | 42 | best.pt (400) | `bench_har_prism__tx5_language__finetune__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | probe | 0.274 | 27.0% | 66.4% | 0.9335 | 42 | best.pt (3800) | `bench_har_prism__txs_fullbody__probe__s42` | bench |
| Trained from scratch on full-body pose (no pretraining) | finetune | 0.327 | 30.7% | 69.1% | 0.8927 | 42 | best.pt (200) | `bench_har_prism__txs_fullbody__finetune__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | probe | 0.217 | 23.0% | 65.0% | 0.9679 | 42 | best.pt (2400) | `bench_har_prism__txs_gesture__probe__s42` | bench |
| Trained from scratch on microgesture (no pretraining) | finetune | 0.275 | 27.4% | 65.7% | 0.9701 | 42 | best.pt (200) | `bench_har_prism__txs_gesture__finetune__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | probe | 0.250 | 25.3% | 65.9% | 0.9396 | 42 | best.pt (3800) | `bench_har_prism__txs_odom__probe__s42` | bench |
| Trained from scratch on odometry (wrist-frame velocity) (no pretraining) | finetune | 0.324 | 30.9% | 69.0% | 0.8798 | 42 | best.pt (200) | `bench_har_prism__txs_odom__finetune__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | probe | 0.213 | 22.6% | 64.7% | 0.9736 | 42 | best.pt (11000) | `bench_har_prism__txs_pd__probe__s42` | bench |
| Trained from scratch on Parkinson's disease (no pretraining) | finetune | 0.297 | 30.2% | 65.5% | 0.9774 | 42 | best.pt (600) | `bench_har_prism__txs_pd__finetune__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | probe | 0.251 | 25.2% | 65.1% | 0.9584 | 42 | best.pt (3800) | `bench_har_prism__txs_language__probe__s42` | bench |
| Trained from scratch on IMU-text retrieval (no pretraining) | finetune | 0.292 | 29.7% | 66.2% | 0.9604 | 42 | best.pt (600) | `bench_har_prism__txs_language__finetune__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | probe | 0.270 | 26.6% | 66.4% | 0.9206 | 42 | best.pt (3800) | `bench_har_prism__txs_armpose__probe__s42` | bench |
| Trained from scratch on arm pose (no pretraining) | finetune | 0.324 | 30.9% | 68.6% | 0.8870 | 42 | best.pt (200) | `bench_har_prism__txs_armpose__finetune__s42` | bench |
| Multi-task (full-body pose + odometry, 1:1) | probe | 0.271 | 26.7% | 66.9% | 0.9033 | 42 | best.pt (3800) | `bench_har_prism__mt_body_odom__probe__s42` | bench |
| Multi-task (full-body pose + microgesture, 10:1) | probe | 0.296 | 28.5% | 66.9% | 0.9004 | 42 | best.pt (4600) | `bench_har_prism__mt_body_gesture__probe__s42` | bench |

## Fall detection · SmartFallMM — `fall_smartfallmm`

_SmartFallMM right-wrist IMU → fall vs activity of daily living (31 participants)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.759 | 76.2% | 80.8% | 0.4462 | 42 | best.pt (1800) | `bench_fall_smartfallmm__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.784 | 78.7% | 82.8% | 0.6674 | 42 | best.pt (200) | `bench_fall_smartfallmm__mae_5src__finetune__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 0.786 | 79.2% | 82.8% | 0.4336 | 42 | best.pt (6400) | `bench_fall_smartfallmm__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | finetune | 0.779 | 78.2% | 82.3% | 0.5352 | 42 | best.pt (200) | `bench_fall_smartfallmm__mae_5src_noaltattn__finetune__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 0.743 | 74.5% | 79.5% | 0.4543 | 42 | best.pt (5400) | `bench_fall_smartfallmm__mae_5src_acconly__probe__s42` | bench |
| WristMAE (5-source, acc-only) | finetune | 0.763 | 77.5% | 80.5% | 0.6520 | 42 | best.pt (200) | `bench_fall_smartfallmm__mae_5src_acconly__finetune__s42` | bench |
| FOCAL (5-source) | probe | 0.664 | 65.4% | 75.5% | 0.5032 | 42 | best.pt (6000) | `bench_fall_smartfallmm__focal_5src__probe__s42` | bench |
| FOCAL (5-source) | finetune | 0.709 | 72.7% | 75.3% | 0.4881 | 42 | best.pt (200) | `bench_fall_smartfallmm__focal_5src__finetune__s42` | bench |
| SimMTM (5-source) | probe | 0.740 | 73.9% | 79.5% | 0.4564 | 42 | best.pt (9600) | `bench_fall_smartfallmm__simmtm_5src__probe__s42` | bench |
| SimMTM (5-source) | finetune | 0.718 | 75.5% | 74.9% | 0.5236 | 42 | best.pt (200) | `bench_fall_smartfallmm__simmtm_5src__finetune__s42` | bench |
| RelCon (5-source) | probe | 0.654 | 64.3% | 75.1% | 0.5694 | 42 | best.pt (8400) | `bench_fall_smartfallmm__relcon_5src__probe__s42` | bench |
| RelCon (5-source) | finetune | 0.715 | 73.1% | 76.0% | 0.4900 | 42 | best.pt (200) | `bench_fall_smartfallmm__relcon_5src__finetune__s42` | bench |
| LIMU-BERT (5-source) | probe | 0.760 | 75.3% | 81.5% | 0.4144 | 42 | best.pt (9600) | `bench_fall_smartfallmm__limubert_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | finetune | 0.787 | 78.6% | 83.2% | 0.3990 | 42 | best.pt (200) | `bench_fall_smartfallmm__limubert_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.727 | 70.1% | 81.7% | 0.4705 | 42 | best.pt (600) | `bench_fall_smartfallmm__scratch__scratch__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.626 | 61.5% | 75.2% | 0.5658 | 42 | best.pt (14000) | `bench_fall_smartfallmm__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.422 | 50.0% | 72.9% | 0.6669 | 42 | best.pt (7000) | `bench_fall_smartfallmm__control_raw__probe__s42` | bench |

## Arm pose · rotated frame — `arm_pose_nymeria_rot7`

_Nymeria real right-wrist IMU → 5-joint right-arm chain rotations — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split._ Scorer: `bench.score.arm_pose`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **pos err (cm)**: FK joint-position error at the 5 arm joints with GT ancestors (scripts/eval_arm_pose_position_error.py).

| Encoder | Protocol | **pos err (cm)** | ang err (°) | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|
| Scratch (no pretraining) | scratch | 8.25 | 9.98 | 0.01704 | 42 | best.pt (60000) | `bench_arm_pose_nymeria_rot7__scratch__scratch__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 8.44 | 10.24 | 0.01770 | 42 | best.pt (9000) | `bench_arm_pose_nymeria_rot7__mae_recipe_r0__probe__s42` | bench |

## Microgesture · rotated frame — `microgesture_openwatch_rot7`

_OpenWatch wrist IMU → 5 gestures + background (paper protocol, real recordings) — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split._ Scorer: `bench.score.microgesture`. Selection: {'split': 'val', 'monitor': 'macro_f1', 'mode': 'max', 'note': 'window-level val macro-F1 (paper §3.3.3), inverse-frequency class-weighted CE'}. Reported split: **test**.

| Encoder | Protocol | **clip F1 (test)** | clip acc (test) | win F1 (test) | win acc (test) | clip F1 (val) | win F1 (val) | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Scratch (no pretraining) | scratch | 0.505 | 0.609 | 0.425 | 0.529 | 0.683 | 0.550 | 42 | best.pt (4200) | `bench_microgesture_openwatch_rot7__scratch__scratch__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.483 | 0.540 | 0.422 | 0.492 | 0.598 | 0.508 | 42 | best.pt (1400) | `bench_microgesture_openwatch_rot7__mae_recipe_r0__probe__s42` | bench |

## Full-body pose (probe decoder 0x0) — `full_body_pose_nymeria_dec0x0`

_Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)._ Scorer: `bench.score.full_body_pose`. Selection: {'split': 'val', 'monitor': 'mpjpe_mm', 'mode': 'min'}. Reported split: **val**.
Primary metric **MPJPE (mm)**: root-relative FK joint position error, joints 1-23 (imu_lm/objectives/pose_1d.py::_compute_val_mpjpe), at the selected checkpoint.

| Encoder | Protocol | **MPJPE (mm)** | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 104.92 | 0.02032 | 42 | best_mpjpe.pt (64000) | `bench_full_body_pose_nymeria_dec0x0__mae_5src__probe__s42` | bench |
| FOCAL (5-source) | probe | 113.61 | 0.02283 | 42 | best_mpjpe.pt (85000) | `bench_full_body_pose_nymeria_dec0x0__focal_5src__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 114.59 | 0.02346 | 42 | best_mpjpe.pt (87000) | `bench_full_body_pose_nymeria_dec0x0__control_random__probe__s42` | bench |

## Full-body pose (probe decoder 1x0) — `full_body_pose_nymeria_dec1x0`

_Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)._ Scorer: `bench.score.full_body_pose`. Selection: {'split': 'val', 'monitor': 'mpjpe_mm', 'mode': 'min'}. Reported split: **val**.
Primary metric **MPJPE (mm)**: root-relative FK joint position error, joints 1-23 (imu_lm/objectives/pose_1d.py::_compute_val_mpjpe), at the selected checkpoint.

| Encoder | Protocol | **MPJPE (mm)** | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 97.32 | 0.01937 | 42 | best_mpjpe.pt (21000) | `bench_full_body_pose_nymeria_dec1x0__mae_5src__probe__s42` | bench |
| FOCAL (5-source) | probe | 108.32 | 0.02163 | 42 | best_mpjpe.pt (24000) | `bench_full_body_pose_nymeria_dec1x0__focal_5src__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 104.88 | 0.02068 | 42 | best_mpjpe.pt (27000) | `bench_full_body_pose_nymeria_dec1x0__control_random__probe__s42` | bench |

## Full-body pose (probe decoder 1x4) — `full_body_pose_nymeria_dec1x4`

_Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)._ Scorer: `bench.score.full_body_pose`. Selection: {'split': 'val', 'monitor': 'mpjpe_mm', 'mode': 'min'}. Reported split: **val**.
Primary metric **MPJPE (mm)**: root-relative FK joint position error, joints 1-23 (imu_lm/objectives/pose_1d.py::_compute_val_mpjpe), at the selected checkpoint.

| Encoder | Protocol | **MPJPE (mm)** | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 93.02 | 0.01970 | 42 | best_mpjpe.pt (37000) | `bench_full_body_pose_nymeria_dec1x4__mae_5src__probe__s42` | bench |
| FOCAL (5-source) | probe | 98.32 | 0.02110 | 42 | best_mpjpe.pt (48000) | `bench_full_body_pose_nymeria_dec1x4__focal_5src__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 93.25 | 0.01997 | 42 | best_mpjpe.pt (70000) | `bench_full_body_pose_nymeria_dec1x4__control_random__probe__s42` | bench |

## Microgesture (release raw-ID split, leaky) — `microgesture_openwatch_releasesplit`

_OpenWatch wrist IMU → 5 gestures + background (paper protocol, real recordings)._ Scorer: `bench.score.microgesture`. Selection: {'split': 'val', 'monitor': 'macro_f1', 'mode': 'max', 'note': 'window-level val macro-F1 (paper §3.3.3), inverse-frequency class-weighted CE'}. Reported split: **test**.

| Encoder | Protocol | **clip F1 (test)** | clip acc (test) | win F1 (test) | win acc (test) | clip F1 (val) | win F1 (val) | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.570 | 0.586 | 0.492 | 0.515 | 0.614 | 0.494 | 42 | best.pt (16000) | `bench_microgesture_openwatch_releasesplit__mae_5src__probe__s42` | bench |
| WristMAE (5-source) ⚠ eval-overlap pool | finetune | 0.647 | 0.717 | 0.552 | 0.643 | 0.786 | 0.653 | 42 | best.pt (1400) | `bench_microgesture_openwatch_releasesplit__mae_5src__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.548 | 0.667 | 0.476 | 0.581 | 0.687 | 0.605 | 42 | best.pt (4800) | `bench_microgesture_openwatch_releasesplit__scratch__scratch__s42` | bench |

## Arm pose (deep probe decoder) — `arm_pose_nymeria_deepdec`

_Nymeria real right-wrist IMU → 5-joint right-arm chain rotations._ Scorer: `bench.score.arm_pose`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **pos err (cm)**: FK joint-position error at the 5 arm joints with GT ancestors (scripts/eval_arm_pose_position_error.py).

| Encoder | Protocol | **pos err (cm)** | ang err (°) | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) ⚠ eval-overlap pool | probe | 8.17 | 9.81 | 0.01638 | 42 | best.pt (9000) | `_deepdec/vit1d_arm_pose_probe_nymeria_mae_multidata` | legacy† |
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 8.18 | 9.92 | 0.01687 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__mae_5src__probe__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 8.17 | 9.97 | 0.01685 | 42 | best.pt (9000) | `_deepdec/vit1d_arm_pose_probe_nymeria_mae_noaltattn_multidata` | legacy† |
| WristMAE (5-source, no alt-attn) | probe | 8.12 | 9.95 | 0.01685 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_5src_noaltattn__probe__s42` | bench |
| Ours (MAE, acc-only) | probe | 8.78 | 10.68 | 0.01888 | 42 | best.pt (6000) | `_deepdec/vit1d_arm_pose_probe_nymeria_mae_acconly_multidata` | legacy† |
| WristMAE (5-source, acc-only) | probe | 8.69 | 10.50 | 0.01856 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__mae_5src_acconly__probe__s42` | bench |
| FOCAL | probe | 8.73 | 10.75 | 0.01916 | 42 | best.pt (11000) | `_deepdec/vit1d_arm_pose_probe_nymeria_focal_multidata` | legacy† |
| FOCAL (5-source) | probe | 8.51 | 10.40 | 0.01803 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__focal_5src__probe__s42` | bench |
| SimMTM | probe | 8.56 | 10.39 | 0.01813 | 42 | best.pt (10000) | `_deepdec/vit1d_arm_pose_probe_nymeria_simmtm_multidata` | legacy† |
| SimMTM (5-source) | probe | 8.37 | 10.12 | 0.01730 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__simmtm_5src__probe__s42` | bench |
| RelCon | probe | 8.52 | 10.41 | 0.01826 | 42 | best.pt (8000) | `_deepdec/vit1d_arm_pose_probe_nymeria_relcon_multidata` | legacy† |
| RelCon (5-source) | probe | 8.49 | 10.29 | 0.01797 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__relcon_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 8.15 | 9.82 | 0.01632 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__limubert_5src__probe__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 8.44 | 10.27 | 0.01774 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 8.45 | 10.30 | 0.01779 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 8.25 | 10.02 | 0.01674 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 8.16 | 10.02 | 0.01674 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 8.45 | 10.28 | 0.01779 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__mae_recipe_r0__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 8.46 | 10.29 | 0.01775 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_mr40__probe__s42` | bench |
| MAE mask-ratio 0.50 | probe | 8.41 | 10.27 | 0.01767 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 8.16 | 9.87 | 0.01644 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__mae_4src_z__probe__s42` | bench |
| MAE mask-ratio 0.60 | probe | 8.53 | 10.29 | 0.01775 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 8.30 | 10.12 | 0.01718 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_4src_dc__probe__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | probe | 8.09 | 9.86 | 0.01637 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__mae_4src_nocls__probe__s42` | bench |
| MAE mask-ratio 0.85 | probe | 8.47 | 10.33 | 0.01784 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 8.26 | 10.07 | 0.01685 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_4src_phys__probe__s42` | bench |
| MAE mask-ratio 0.90 | probe | 8.53 | 10.39 | 0.01803 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 8.30 | 10.09 | 0.01718 | 42 | best.pt (10000) | `_deepdec/bench_arm_pose_nymeria__mae_4src_physnp__probe__s42` | bench |
| MAE frame-canonical (no rotation) | probe | 8.42 | 10.21 | 0.01756 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__mae_frame_canon__probe__s42` | bench |
| SimMTM (4-source pool) | probe | 8.12 | 9.91 | 0.01672 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__simmtm_4src__probe__s42` | bench |
| FOCAL (4-source pool) | probe | 8.25 | 10.18 | 0.01741 | 42 | best.pt (11000) | `_deepdec/bench_arm_pose_nymeria__focal_4src__probe__s42` | bench |
| RelCon (4-source pool) | probe | 8.23 | 10.07 | 0.01714 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__relcon_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | probe | 8.49 | 10.27 | 0.01773 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__acconly_4src__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 8.44 | 10.32 | 0.01776 | 42 | best.pt (11000) | `_deepdec/bench_arm_pose_nymeria__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 8.46 | 10.39 | 0.01790 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__inertia1_art__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 8.27 | 10.13 | 0.01713 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 8.47 | 10.41 | 0.01781 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__inertia1_art_large__probe__s42` | bench |
| VolDiv single-source, narrow motion band, N=12,500 | probe | 8.00 | 9.79 | 0.01627 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_nar_n12k__probe__s42` | bench |
| VolDiv single-source, narrow motion band, N=50,000 | probe | 8.07 | 9.87 | 0.01650 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_nar_n50k__probe__s42` | bench |
| VolDiv single-source, narrow motion band, N=200,000 | probe | 8.18 | 9.88 | 0.01652 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_nar_n200k__probe__s42` | bench |
| VolDiv single-source, narrow motion band, N=800,000 | probe | 8.17 | 9.89 | 0.01659 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_nar_n800k__probe__s42` | bench |
| VolDiv single-source, uniform random, N=12,500 | probe | 8.05 | 9.80 | 0.01648 | 42 | best.pt (16000) | `_deepdec/bench_arm_pose_nymeria__vd_one_n12k__probe__s42` | bench |
| VolDiv single-source, uniform random, N=50,000 | probe | 8.14 | 9.82 | 0.01644 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_one_n50k__probe__s42` | bench |
| VolDiv single-source, uniform random, N=200,000 | probe | 8.21 | 9.94 | 0.01681 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_one_n200k__probe__s42` | bench |
| VolDiv single-source, uniform random, N=800,000 | probe | 8.13 | 9.86 | 0.01652 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_one_n800k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=12,500 | probe | 8.04 | 9.92 | 0.01655 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_nat_n12k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=50,000 | probe | 8.16 | 9.92 | 0.01676 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_nat_n50k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=200,000 | probe | 8.09 | 9.81 | 0.01654 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_nat_n200k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=800,000 | probe | 8.17 | 9.88 | 0.01655 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_nat_n800k__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=12,500 | probe | 8.11 | 9.81 | 0.01643 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_bal_n12k__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=50,000 | probe | 8.14 | 9.96 | 0.01673 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_bal_n50k__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=200,000 | probe | 8.15 | 9.91 | 0.01665 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_bal_n200k__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=800,000 | probe | 8.05 | 9.81 | 0.01634 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_bal_n800k__probe__s42` | bench |
| VolDiv 4-source, natural proportions, N=1,738,094 | probe | 8.13 | 9.87 | 0.01657 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_nat_nfull__probe__s42` | bench |
| VolDiv 4-source, source- and type-balanced, N=1,738,094 | probe | 8.22 | 9.91 | 0.01667 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_bal_nfull__probe__s42` | bench |
| VolDiv src1, N=50,000 | probe | 8.11 | 9.90 | 0.01670 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_src1_n50k__probe__s42` | bench |
| VolDiv src2, N=50,000 | probe | 8.11 | 9.85 | 0.01644 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_src2_n50k__probe__s42` | bench |
| VolDiv src3, N=50,000 | probe | 8.16 | 9.85 | 0.01646 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_src3_n50k__probe__s42` | bench |
| VolDiv src4, N=50,000 | probe | 8.19 | 9.95 | 0.01666 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_src4_n50k__probe__s42` | bench |
| VolDiv ocdetect only, 3 T-phys type(s), water-filled, N=12,500 | probe | 8.11 | 9.88 | 0.01674 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_typ3_n12k__probe__s42` | bench |
| VolDiv ocdetect only, 30 T-phys type(s), water-filled, N=12,500 | probe | 8.13 | 9.91 | 0.01667 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_typ30_n12k__probe__s42` | bench |
| VolDiv ocdetect only, 300 T-phys type(s), water-filled, N=12,500 | probe | 8.09 | 9.88 | 0.01649 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_typ300_n12k__probe__s42` | bench |
| VolDiv ocdetect only, every T-phys type(s), water-filled, N=12,500 | probe | 8.17 | 9.93 | 0.01657 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_typall_n12k__probe__s42` | bench |
| VolDiv ocdetect only, 1 subject(s), type-matched, N=50,000 | probe | 8.15 | 9.89 | 0.01651 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_sub1_n50k__probe__s42` | bench |
| VolDiv ocdetect only, 3 subject(s), type-matched, N=50,000 | probe | 8.14 | 9.83 | 0.01659 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_sub3_n50k__probe__s42` | bench |
| VolDiv ocdetect only, 10 subject(s), type-matched, N=50,000 | probe | 8.16 | 9.86 | 0.01668 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_sub10_n50k__probe__s42` | bench |
| VolDiv ocdetect only, every subject(s), type-matched, N=50,000 | probe | 8.21 | 9.86 | 0.01640 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_suball_n50k__probe__s42` | bench |
| VolDiv MAE-space types, 3 T-enc type(s), ocdetect only, N=12,500 | probe | 8.07 | 9.89 | 0.01643 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_etyp3_n12k__probe__s42` | bench |
| VolDiv MAE-space types, 30 T-enc type(s), ocdetect only, N=12,500 | probe | 8.11 | 9.84 | 0.01626 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_etyp30_n12k__probe__s42` | bench |
| VolDiv MAE-space types, 300 T-enc type(s), ocdetect only, N=12,500 | probe | 8.06 | 9.77 | 0.01634 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_etyp300_n12k__probe__s42` | bench |
| VolDiv MAE-space types, every T-enc type(s), ocdetect only, N=12,500 | probe | 8.12 | 9.89 | 0.01654 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_etypall_n12k__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 | probe | 8.06 | 9.82 | 0.01667 | 42 | best.pt (16000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 | probe | 8.15 | 9.85 | 0.01633 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 | probe | 8.11 | 9.92 | 0.01666 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__vd_ek5_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 | probe | 8.15 | 9.86 | 0.01647 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek5_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 | probe | 8.22 | 9.91 | 0.01650 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek5_f1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 | probe | 8.13 | 9.85 | 0.01645 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek10_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 | probe | 8.12 | 9.85 | 0.01669 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_ek10_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 | probe | 8.17 | 9.88 | 0.01656 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek10_f1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 | probe | 8.17 | 9.96 | 0.01673 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek25_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 | probe | 8.07 | 9.82 | 0.01622 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek25_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 | probe | 8.08 | 9.86 | 0.01671 | 42 | best.pt (16000) | `_deepdec/bench_arm_pose_nymeria__vd_ek25_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 | probe | 8.07 | 9.87 | 0.01667 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_ek50_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 | probe | 8.14 | 9.85 | 0.01668 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_ek50_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 | probe | 8.09 | 9.83 | 0.01657 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_ek50_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 | probe | 8.10 | 9.85 | 0.01641 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek50_f10__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 | probe | 8.17 | 9.97 | 0.01678 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek100_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 | probe | 8.09 | 9.88 | 0.01642 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_ek100_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 | probe | 8.14 | 9.89 | 0.01655 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek100_f1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 | probe | 8.08 | 9.82 | 0.01623 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek100_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 | probe | 8.17 | 9.96 | 0.01676 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 | probe | 8.11 | 9.90 | 0.01662 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 | probe | 8.13 | 9.86 | 0.01645 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 | probe | 8.05 | 9.83 | 0.01656 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 | probe | 8.10 | 9.88 | 0.01654 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f100__probe__s42` | bench |
| VolDiv 4-source pool, 1% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 8.09 | 9.90 | 0.01665 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_pk1_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 1% of T-phys types (random, source-stratified, gyro-unit fixed), N=16,113 | probe | 8.17 | 9.92 | 0.01660 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_pk1_n1__probe__s42` | bench |
| VolDiv 4-source pool, 10% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 8.10 | 9.83 | 0.01638 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_pk10_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 10% of T-phys types (random, source-stratified, gyro-unit fixed), N=17,381 | probe | 8.05 | 9.89 | 0.01660 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_pk10_n1__probe__s42` | bench |
| VolDiv 4-source pool, 10% of T-phys types (random, source-stratified, gyro-unit fixed), N=173,809 | probe | 8.15 | 9.87 | 0.01647 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_pk10_n10__probe__s42` | bench |
| VolDiv 4-source pool, 50% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 8.06 | 9.84 | 0.01651 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_pk50_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 50% of T-phys types (random, source-stratified, gyro-unit fixed), N=17,381 | probe | 8.21 | 9.96 | 0.01681 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_pk50_n1__probe__s42` | bench |
| VolDiv 4-source pool, 50% of T-phys types (random, source-stratified, gyro-unit fixed), N=173,809 | probe | 8.19 | 9.93 | 0.01669 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_pk50_n10__probe__s42` | bench |
| VolDiv 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 8.11 | 9.84 | 0.01638 | 42 | best.pt (7000) | `_deepdec/bench_arm_pose_nymeria__vd_pk100_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=17,381 | probe | 8.11 | 9.84 | 0.01639 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_pk100_n1__probe__s42` | bench |
| VolDiv 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=173,809 | probe | 8.06 | 9.79 | 0.01644 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_pk100_n10__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 8.34 | 10.13 | 0.01731 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 8.33 | 10.10 | 0.01726 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 8.28 | 10.08 | 0.01745 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_ek5_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 8.30 | 10.03 | 0.01693 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek5_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 8.29 | 10.08 | 0.01705 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_ek5_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 8.31 | 10.08 | 0.01731 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek10_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 8.24 | 10.09 | 0.01722 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_ek10_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 8.19 | 10.07 | 0.01702 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_ek10_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 8.36 | 10.13 | 0.01742 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek25_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 8.15 | 10.05 | 0.01701 | 42 | best.pt (7000) | `_deepdec/bench_arm_pose_nymeria__vd_ek25_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 8.18 | 9.98 | 0.01690 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek25_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 8.37 | 10.20 | 0.01755 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_ek50_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 8.20 | 10.02 | 0.01727 | 42 | best.pt (9000) | `_deepdec/bench_arm_pose_nymeria__vd_ek50_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 8.16 | 9.98 | 0.01700 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek50_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 8.18 | 9.97 | 0.01699 | 42 | best.pt (8000) | `_deepdec/bench_arm_pose_nymeria__vd_ek50_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 8.51 | 10.22 | 0.01767 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek100_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 8.27 | 10.01 | 0.01697 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek100_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 8.25 | 10.03 | 0.01696 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek100_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 8.24 | 10.00 | 0.01721 | 42 | best.pt (16000) | `_deepdec/bench_arm_pose_nymeria__vd_ek100_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 8.32 | 10.11 | 0.01736 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 8.20 | 10.09 | 0.01727 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 8.23 | 9.97 | 0.01690 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 8.31 | 10.06 | 0.01729 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 (z-score probe input) | probe | 8.22 | 10.00 | 0.01702 | 42 | best.pt (6000) | `_deepdec/bench_arm_pose_nymeria__vd_ek1000_f100_zs__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 8.31 | 10.16 | 0.01734 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 8.47 | 10.22 | 0.01740 | 42 | best.pt (5000) | `_deepdec/bench_arm_pose_nymeria__control_raw__probe__s42` | bench |

## Full-body pose (deep probe decoder) — `full_body_pose_nymeria_deepdec`

_Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)._ Scorer: `bench.score.full_body_pose`. Selection: {'split': 'val', 'monitor': 'mpjpe_mm', 'mode': 'min'}. Reported split: **val**.
Primary metric **MPJPE (mm)**: root-relative FK joint position error, joints 1-23 (imu_lm/objectives/pose_1d.py::_compute_val_mpjpe), at the selected checkpoint.

| Encoder | Protocol | **MPJPE (mm)** | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) ⚠ eval-overlap pool | probe | 97.63 | 0.01981 | 42 | best_mpjpe.pt (110000) | `_deepdec/vit1d_full_body_pose_probe_nymeria_mae_multidata` | legacy† |
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 90.99 | 0.01928 | 42 | best_mpjpe.pt (32000) | `_deepdec/bench_full_body_pose_nymeria__mae_5src__probe__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 97.43 | 0.01940 | 42 | best_mpjpe.pt (45000) | `_deepdec/vit1d_full_body_pose_probe_nymeria_mae_noaltattn_multidata` | legacy† |
| WristMAE (5-source, no alt-attn) | probe | 92.48 | 0.02054 | 42 | best_mpjpe.pt (92000) | `_deepdec/bench_full_body_pose_nymeria__mae_5src_noaltattn__probe__s42` | bench |
| Ours (MAE, acc-only) | probe | 103.37 | 0.02082 | 42 | best_mpjpe.pt (56000) | `_deepdec/vit1d_full_body_pose_probe_nymeria_mae_acconly_multidata` | legacy† |
| WristMAE (5-source, acc-only) | probe | 97.63 | 0.02096 | 42 | best_mpjpe.pt (27000) | `_deepdec/bench_full_body_pose_nymeria__mae_5src_acconly__probe__s42` | bench |
| FOCAL | probe | 108.29 | 0.02208 | 42 | best_mpjpe.pt (105000) | `_deepdec/vit1d_full_body_pose_probe_nymeria_focal_multidata` | legacy† |
| FOCAL (5-source) | probe | 96.08 | 0.02116 | 42 | best_mpjpe.pt (66000) | `_deepdec/bench_full_body_pose_nymeria__focal_5src__probe__s42` | bench |
| SimMTM | probe | 103.48 | 0.02072 | 42 | best_mpjpe.pt (50000) | `_deepdec/vit1d_full_body_pose_probe_nymeria_simmtm_multidata` | legacy† |
| SimMTM (5-source) | probe | 92.98 | 0.02018 | 42 | best_mpjpe.pt (48000) | `_deepdec/bench_full_body_pose_nymeria__simmtm_5src__probe__s42` | bench |
| RelCon | probe | 102.47 | 0.02031 | 42 | best_mpjpe.pt (32000) | `_deepdec/vit1d_full_body_pose_probe_nymeria_relcon_multidata` | legacy† |
| LIMU-BERT (5-source) | probe | 90.07 | 0.01968 | 42 | best_mpjpe.pt (132000) | `_deepdec/bench_full_body_pose_nymeria__limubert_5src__probe__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 101.55 | 0.02006 | 42 | best_mpjpe.pt (48000) | `_deepdec/bench_full_body_pose_nymeria__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 92.26 | 0.02043 | 42 | best_mpjpe.pt (66000) | `_deepdec/bench_full_body_pose_nymeria__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 100.20 | 0.01979 | 42 | best_mpjpe.pt (30000) | `_deepdec/bench_full_body_pose_nymeria__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 100.71 | 0.01978 | 42 | best_mpjpe.pt (29000) | `_deepdec/bench_full_body_pose_nymeria__mae_nhanes2x2_d__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 102.24 | 0.02014 | 42 | best_mpjpe.pt (25000) | `_deepdec/bench_full_body_pose_nymeria__mae_div60_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 91.45 | 0.01846 | 42 | best_mpjpe.pt (16000) | `_deepdec/bench_full_body_pose_nymeria__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 92.98 | 0.01991 | 42 | best_mpjpe.pt (29000) | `_deepdec/bench_full_body_pose_nymeria__mae_4src_dc__probe__s42` | bench |
| MAE 4-source, no CLS token (mean-pooled) | probe | 91.59 | 0.01985 | 42 | best_mpjpe.pt (44000) | `_deepdec/bench_full_body_pose_nymeria__mae_4src_nocls__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 100.34 | 0.02000 | 42 | best_mpjpe.pt (41000) | `_deepdec/bench_full_body_pose_nymeria__mae_4src_phys__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 92.57 | 0.01981 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__mae_4src_physnp__probe__s42` | bench |
| SimMTM (4-source pool) | probe | 93.34 | 0.02013 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__simmtm_4src__probe__s42` | bench |
| FOCAL (4-source pool) | probe | 105.50 | 0.02133 | 42 | best_mpjpe.pt (64000) | `_deepdec/bench_full_body_pose_nymeria__focal_4src__probe__s42` | bench |
| RelCon (4-source pool) | probe | 93.58 | 0.02023 | 42 | best_mpjpe.pt (41000) | `_deepdec/bench_full_body_pose_nymeria__relcon_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | probe | 101.87 | 0.02057 | 42 | best_mpjpe.pt (34000) | `_deepdec/bench_full_body_pose_nymeria__acconly_4src__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 101.06 | 0.02002 | 42 | best_mpjpe.pt (59000) | `_deepdec/bench_full_body_pose_nymeria__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 101.54 | 0.02052 | 42 | best.pt (55000) | `_deepdec/bench_full_body_pose_nymeria__inertia1_art__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 97.93 | 0.01950 | 42 | best_mpjpe.pt (44000) | `_deepdec/bench_full_body_pose_nymeria__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 101.85 | 0.02056 | 42 | best_mpjpe.pt (29000) | `_deepdec/bench_full_body_pose_nymeria__inertia1_art_large__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 | probe | 92.95 | 0.01993 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 | probe | 92.36 | 0.01939 | 42 | best_mpjpe.pt (24000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 | probe | 92.58 | 0.01949 | 42 | best_mpjpe.pt (26000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek5_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 | probe | 92.25 | 0.01989 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek5_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 | probe | 92.54 | 0.01971 | 42 | best_mpjpe.pt (28000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek5_f1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 | probe | 91.67 | 0.01950 | 42 | best_mpjpe.pt (32000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek10_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 | probe | 92.89 | 0.01996 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek10_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 | probe | 92.60 | 0.02066 | 42 | best_mpjpe.pt (73000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek10_f1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 | probe | 91.93 | 0.02011 | 42 | best_mpjpe.pt (46000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek25_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 | probe | 92.29 | 0.01999 | 42 | best_mpjpe.pt (33000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek25_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 | probe | 92.57 | 0.01946 | 42 | best_mpjpe.pt (23000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek25_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 | probe | 91.87 | 0.02006 | 42 | best_mpjpe.pt (46000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek50_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 | probe | 92.10 | 0.01997 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek50_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 | probe | 92.26 | 0.02007 | 42 | best_mpjpe.pt (39000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek50_f1__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 | probe | 92.62 | 0.02014 | 42 | best_mpjpe.pt (46000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek50_f10__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 | probe | 92.83 | 0.02026 | 42 | best_mpjpe.pt (41000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek100_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 | probe | 91.30 | 0.02027 | 42 | best_mpjpe.pt (63000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek100_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 | probe | 92.30 | 0.02037 | 42 | best_mpjpe.pt (66000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek100_f1__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 | probe | 92.45 | 0.01964 | 42 | best_mpjpe.pt (29000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek100_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 | probe | 92.61 | 0.01985 | 42 | best_mpjpe.pt (33000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f0p01__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 | probe | 91.74 | 0.01990 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f0p1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 | probe | 91.73 | 0.01986 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f1__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 | probe | 92.12 | 0.01931 | 42 | best_mpjpe.pt (23000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f10__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 | probe | 92.40 | 0.01962 | 42 | best_mpjpe.pt (27000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f100__probe__s42` | bench |
| VolDiv 4-source pool, 1% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 92.70 | 0.01957 | 42 | best_mpjpe.pt (25000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk1_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 1% of T-phys types (random, source-stratified, gyro-unit fixed), N=16,113 | probe | 91.78 | 0.02021 | 42 | best_mpjpe.pt (41000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk1_n1__probe__s42` | bench |
| VolDiv 4-source pool, 10% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 91.63 | 0.01992 | 42 | best_mpjpe.pt (50000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk10_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 10% of T-phys types (random, source-stratified, gyro-unit fixed), N=17,381 | probe | 91.77 | 0.02003 | 42 | best_mpjpe.pt (41000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk10_n1__probe__s42` | bench |
| VolDiv 4-source pool, 10% of T-phys types (random, source-stratified, gyro-unit fixed), N=173,809 | probe | 92.13 | 0.01927 | 42 | best_mpjpe.pt (27000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk10_n10__probe__s42` | bench |
| VolDiv 4-source pool, 50% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 91.61 | 0.01978 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk50_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 50% of T-phys types (random, source-stratified, gyro-unit fixed), N=17,381 | probe | 91.91 | 0.02030 | 42 | best_mpjpe.pt (63000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk50_n1__probe__s42` | bench |
| VolDiv 4-source pool, 50% of T-phys types (random, source-stratified, gyro-unit fixed), N=173,809 | probe | 92.24 | 0.02015 | 42 | best_mpjpe.pt (46000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk50_n10__probe__s42` | bench |
| VolDiv 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 | probe | 91.19 | 0.01915 | 42 | best_mpjpe.pt (25000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk100_n0p1__probe__s42` | bench |
| VolDiv 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=17,381 | probe | 91.94 | 0.02003 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk100_n1__probe__s42` | bench |
| VolDiv 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=173,809 | probe | 92.39 | 0.01989 | 42 | best_mpjpe.pt (43000) | `_deepdec/bench_full_body_pose_nymeria__vd_pk100_n10__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 91.54 | 0.02022 | 42 | best_mpjpe.pt (115000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 91.85 | 0.02014 | 42 | best_mpjpe.pt (76000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 91.97 | 0.01981 | 42 | best_mpjpe.pt (48000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek5_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 92.93 | 0.01973 | 42 | best_mpjpe.pt (43000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek5_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 5 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 91.11 | 0.02023 | 42 | best_mpjpe.pt (122000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek5_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 92.76 | 0.02061 | 42 | best_mpjpe.pt (83000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek10_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 92.06 | 0.01998 | 42 | best_mpjpe.pt (48000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek10_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 10 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 91.37 | 0.02020 | 42 | best_mpjpe.pt (87000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek10_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 91.92 | 0.02005 | 42 | best_mpjpe.pt (69000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek25_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 92.00 | 0.02005 | 42 | best_mpjpe.pt (69000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek25_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 25 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 92.52 | 0.01900 | 42 | best_mpjpe.pt (18000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek25_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 91.20 | 0.02020 | 42 | best_mpjpe.pt (116000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek50_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 91.94 | 0.02040 | 42 | best_mpjpe.pt (100000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek50_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 92.35 | 0.01998 | 42 | best_mpjpe.pt (49000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek50_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 50 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 92.78 | 0.01938 | 42 | best_mpjpe.pt (22000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek50_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 92.54 | 0.01953 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek100_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 92.08 | 0.02031 | 42 | best_mpjpe.pt (69000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek100_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 92.33 | 0.02052 | 42 | best_mpjpe.pt (89000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek100_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 100 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 92.33 | 0.02043 | 42 | best_mpjpe.pt (102000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek100_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=174 (z-score probe input) | probe | 91.71 | 0.02015 | 42 | best_mpjpe.pt (90000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f0p01_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738 (z-score probe input) | probe | 92.61 | 0.01998 | 42 | best_mpjpe.pt (41000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f0p1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=17,381 (z-score probe input) | probe | 91.69 | 0.01960 | 42 | best_mpjpe.pt (37000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f1_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=173,809 (z-score probe input) | probe | 93.09 | 0.01954 | 42 | best_mpjpe.pt (25000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f10_zs__probe__s42` | bench |
| VolDiv 4-source pool, 1000 largest T-enc type(s), water-filled, N=1,738,094 (z-score probe input) | probe | 91.63 | 0.02006 | 42 | best_mpjpe.pt (79000) | `_deepdec/bench_full_body_pose_nymeria__vd_ek1000_f100_zs__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 91.77 | 0.01940 | 42 | best_mpjpe.pt (39000) | `_deepdec/bench_full_body_pose_nymeria__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 92.17 | 0.02003 | 42 | best_mpjpe.pt (66000) | `_deepdec/bench_full_body_pose_nymeria__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | probe | 91.34 | 0.01980 | 42 | best_mpjpe.pt (43000) | `_deepdec/bench_full_body_pose_nymeria__tx5_prism__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 91.38 | 0.01981 | 42 | best_mpjpe.pt (49000) | `_deepdec/bench_full_body_pose_nymeria__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on odometry (wrist-frame velocity) | probe | 91.36 | 0.01845 | 42 | best_mpjpe.pt (17000) | `_deepdec/bench_full_body_pose_nymeria__tx5_odom__probe__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | probe | 91.89 | 0.02002 | 42 | best_mpjpe.pt (56000) | `_deepdec/bench_full_body_pose_nymeria__tx5_pd__probe__s42` | bench |

## Odometry (deep probe decoder) — `odometry_nymeria_local_deepdec`

_Nymeria real right-wrist IMU → pelvis velocity in the wrist (FLU) frame._ Scorer: `bench.score.odometry_local`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **ATE (m)**: mean absolute trajectory error over all val windows; predicted FLU velocities are rotated to the world frame with the ground-truth start orientation propagated by raw-gyro integration, then summed (bench/score/odometry_local.py, = scripts/eval_odom_nymeria_altattn.py::reconstruct_and_score).

| Encoder | Protocol | **ATE (m)** | drift (%) | vel MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 0.1784 | 50.12 | 0.02293 | 42 | best.pt (116000) | `_deepdec/bench_odometry_nymeria_local__mae_5src__probe__s42` | bench |
| WristMAE (5-source, no alt-attn) | probe | 0.1862 | 51.11 | 0.02483 | 42 | best.pt (138000) | `_deepdec/bench_odometry_nymeria_local__mae_5src_noaltattn__probe__s42` | bench |
| WristMAE (5-source, acc-only) | probe | 0.2042 | 54.96 | 0.02881 | 42 | best.pt (73000) | `_deepdec/bench_odometry_nymeria_local__mae_5src_acconly__probe__s42` | bench |
| FOCAL (5-source) | probe | 0.1995 | 55.02 | 0.02725 | 42 | best.pt (127000) | `_deepdec/bench_odometry_nymeria_local__focal_5src__probe__s42` | bench |
| SimMTM (5-source) | probe | 0.1797 | 50.68 | 0.02319 | 42 | best.pt (144000) | `_deepdec/bench_odometry_nymeria_local__simmtm_5src__probe__s42` | bench |
| LIMU-BERT (5-source) | probe | 0.1845 | 50.55 | 0.02464 | 42 | best.pt (101000) | `_deepdec/bench_odometry_nymeria_local__limubert_5src__probe__s42` | bench |
| Random-init frozen encoder (same architecture as MAE, never trained) | probe | 0.1671 | 47.74 | 0.02090 | 42 | best.pt (262000) | `_deepdec/bench_odometry_nymeria_local__control_random__probe__s42` | bench |
| Raw input + random frozen patch projection (0-layer encoder) | probe | 0.2239 | 61.47 | 0.03131 | 42 | best.pt (120000) | `_deepdec/bench_odometry_nymeria_local__control_raw__probe__s42` | bench |
| MAE-5src fine-tuned on PRISM activity recognition | probe | 0.1744 | 49.08 | 0.02232 | 42 | best.pt (137000) | `_deepdec/bench_odometry_nymeria_local__tx5_prism__probe__s42` | bench |
| MAE-5src fine-tuned on full-body pose | probe | 0.1524 | 43.98 | 0.01828 | 42 | best.pt (113000) | `_deepdec/bench_odometry_nymeria_local__tx5_fullbody__probe__s42` | bench |
| MAE-5src fine-tuned on microgesture | probe | 0.1762 | 49.21 | 0.02278 | 42 | best.pt (163000) | `_deepdec/bench_odometry_nymeria_local__tx5_gesture__probe__s42` | bench |
| MAE-5src fine-tuned on Parkinson's disease | probe | 0.1752 | 49.18 | 0.02270 | 42 | best.pt (158000) | `_deepdec/bench_odometry_nymeria_local__tx5_pd__probe__s42` | bench |

## Arm pose (2-layer probe decoder) — `arm_pose_nymeria_dec2l`

_Nymeria real right-wrist IMU → 5-joint right-arm chain rotations._ Scorer: `bench.score.arm_pose`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **pos err (cm)**: FK joint-position error at the 5 arm joints with GT ancestors (scripts/eval_arm_pose_position_error.py).

| Encoder | Protocol | **pos err (cm)** | ang err (°) | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|
| WristMAE (5-source) ⚠ eval-overlap pool | probe | 8.20 | 9.93 | 0.01659 | 42 | best.pt (17000) | `bench_arm_pose_nymeria_dec2l__mae_5src__probe__s42` | bench |

## Arm pose (4-layer probe decoder) — `arm_pose_nymeria_dec4l`

_Nymeria real right-wrist IMU → 5-joint right-arm chain rotations._ Scorer: `bench.score.arm_pose`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **pos err (cm)**: FK joint-position error at the 5 arm joints with GT ancestors (scripts/eval_arm_pose_position_error.py).

| Encoder | Protocol | **pos err (cm)** | ang err (°) | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|

## Full-body pose (2-layer probe decoder) — `full_body_pose_nymeria_dec2l`

_Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)._ Scorer: `bench.score.full_body_pose`. Selection: {'split': 'val', 'monitor': 'mpjpe_mm', 'mode': 'min'}. Reported split: **val**.
Primary metric **MPJPE (mm)**: root-relative FK joint position error, joints 1-23 (imu_lm/objectives/pose_1d.py::_compute_val_mpjpe), at the selected checkpoint.

| Encoder | Protocol | **MPJPE (mm)** | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---|---|---|

## Full-body pose (4-layer probe decoder) — `full_body_pose_nymeria_dec4l`

_Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)._ Scorer: `bench.score.full_body_pose`. Selection: {'split': 'val', 'monitor': 'mpjpe_mm', 'mode': 'min'}. Reported split: **val**.
Primary metric **MPJPE (mm)**: root-relative FK joint position error, joints 1-23 (imu_lm/objectives/pose_1d.py::_compute_val_mpjpe), at the selected checkpoint.

| Encoder | Protocol | **MPJPE (mm)** | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---|---|---|

## Odometry (2-layer probe decoder) — `odometry_nymeria_local_dec2l`

_Nymeria real right-wrist IMU → pelvis velocity in the wrist (FLU) frame._ Scorer: `bench.score.odometry_local`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **ATE (m)**: mean absolute trajectory error over all val windows; predicted FLU velocities are rotated to the world frame with the ground-truth start orientation propagated by raw-gyro integration, then summed (bench/score/odometry_local.py, = scripts/eval_odom_nymeria_altattn.py::reconstruct_and_score).

| Encoder | Protocol | **ATE (m)** | drift (%) | vel MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|

## Odometry (4-layer probe decoder) — `odometry_nymeria_local_dec4l`

_Nymeria real right-wrist IMU → pelvis velocity in the wrist (FLU) frame._ Scorer: `bench.score.odometry_local`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **ATE (m)**: mean absolute trajectory error over all val windows; predicted FLU velocities are rotated to the world frame with the ground-truth start orientation propagated by raw-gyro integration, then summed (bench/score/odometry_local.py, = scripts/eval_odom_nymeria_altattn.py::reconstruct_and_score).

| Encoder | Protocol | **ATE (m)** | drift (%) | vel MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|

## Caveats that travel with the numbers

* Pose / odometry / PD have **no held-out test split** in v1: the checkpoint is selected on val and val is reported (identical for every row, but optimistic in absolute terms). Microgesture and language report a genuine test split. A v2 split policy (carving test from the fixed val session lists) is the first protocol change to make before publishing absolute numbers.
* Full-body pose MPJPE is the trainer's own validation MPJPE at the selected checkpoint (min over the val curve under the bench selection rule) — a noise-mined statistic; multi-seed cells are what to trust.
* PADS: primary is **subject macro-F1** over 94 val subjects (14 HC / 58 PD / 22 DD); ±0.05 is noise. It replaced subject balanced accuracy on 2026-08-18 because bal-acc rewards degenerate classifiers on this split (a probe that never predicts PD scored 0.54 while classifying 31 % of subjects correctly); the `#classes predicted` column flags such cells. Every frozen probe on PADS is at/near chance — only the finetune column carries signal. Checkpoints are still selected on val window bal-acc (v1); selecting on macro-F1 is the next protocol change and needs re-runs.
* Language: fixed 8k-step budget, final checkpoint (`latest.pt`); §13e showed val-loss selection under-selects. Seed sd ≈ 0.7 MRR — single-seed differences < 1.5 pp are not real.
* External rows come from their own scripts (see `source` in the JSON) and are context, not competitors, under the encoder ranking.
* **HAR pretrain/eval overlap:** every ViT1D row pretrained on the `realdata` MAE source, which contains pamap2, recofit, utd_mhad and wisdm — 4 of the 6 HAR evaluation datasets (HHAR and MHEALTH are clean). External rows (Inertia-1, DeepConvLSTM) never saw them. Treat the HAR column as favourable to the ViT1D rows until the leakage-free re-pretrain (source rebuilt as `realdataclean` = samosa/shoaib/ut_watch, 2026-08-18) lands.
* HAR enters the main tables as ONE aggregate column (mean window macro-F1 over WISDM, UTD-MHAD, PAMAP2, RecoFit, HHAR, MHEALTH; present only when all six are scored). Selection is min val loss and val is reported (no test split); scores are re-evaluated from the checkpoint with pooled predictions, not read from training logs.
