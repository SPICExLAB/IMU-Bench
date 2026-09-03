# IMU-Bench Leaderboard

_Generated 2026-09-01T14:45:35-05:00 from `bench/results/` (647 scored results, git 1b95aa3-dirty). Regenerate with `python -m bench.leaderboard`. Every number links to a result JSON produced by a `bench/score/*` scorer — nothing is hand-typed._

**Controlled protocol (v1):** one encoder architecture (ViT1D, 21M), one recipe per task shared by every encoder (`bench/registry/tasks.yaml`), seed 42 (+43/44 where shown), checkpoint selected on val by the task's rule, scored by one scorer per task. `†` = harvested from a pre-harness run whose config is recipe-equivalent (same template/LR/selection); all other cells were trained by the harness. Bold = best per column. Lower-is-better metrics are marked ↓.

## Frozen probe — primary metric per task

| Encoder | Arm pose<br><sub>pos err (cm) ↓</sub> | Full-body pose<br><sub>MPJPE (mm) ↓</sub> | Odometry<br><sub>ATE (m) ↓</sub> | Microgesture<br><sub>clip F1 (test) ↑</sub> | Language alignment<br><sub>R@1 G=100 (mean i2t/t2i) ↑</sub> | Parkinson's disease<br><sub>subject macro-F1 ↑</sub> | HAR (10 datasets)<br><sub>mean macro-F1 (10 sets) ↑</sub> | Mean rank | Tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours (MAE, no alt-attn) | 8.17<sup>†</sup> | **97.43**<sup>†</sup> | 0.1585 | **0.647**<sup>†</sup> | **28.6** | 0.386 | **0.619**<sup>†</sup> | 2.43 | 7/7 |
| MAE 4-source (no HAR pool), per-window z-score | 8.14 | 98.09 | **0.1425** | 0.613 | 24.1 | **0.463** | 0.612 | 2.57 | 7/7 |
| Ours (MAE, 11-source) | 8.17<sup>†</sup> | 97.63<sup>†</sup> | 0.1493 | 0.606<sup>†</sup> | 27.1 | 0.388 | 0.612<sup>†</sup> | 2.86 | 7/7 |
| Ours (MAE, 5-source) | **8.14**<sup>†</sup> | 98.11<sup>†</sup> | 0.1564 | 0.647 | 27.9 | 0.340 | 0.613<sup>†</sup> | 3.14 | 7/7 |
| MAE 4-source (no HAR pool), physical rescale | 8.26 | 100.34 | 0.1563 | 0.556 | 17.0 | 0.334 | 0.557 | 5.43 | 7/7 |
| MAE 4-source, physical input + patch-normalised loss | 8.30 | 100.43 | 0.1588 | 0.492 | 16.5 | 0.357 | 0.545 | 6.71 | 7/7 |
| Ours (MAE, acc-only) | 8.78<sup>†</sup> | 103.37<sup>†</sup> | 0.1822 | 0.586<sup>†</sup> | 19.3 | 0.397 | 0.546<sup>†</sup> | 6.86 | 7/7 |
| SimMTM | 8.56<sup>†</sup> | 103.48<sup>†</sup> | 0.1793 | 0.532<sup>†</sup> | 16.7 | 0.368 | 0.525<sup>†</sup> | 7.86 | 7/7 |
| MAE 4-source (no HAR pool), raw/DC-preserving | 8.30 | 101.94 | 0.1696 | 0.472 | 8.3 | 0.317 | 0.508 | 8.43 | 7/7 |
| RelCon | 8.52<sup>†</sup> | 102.47<sup>†</sup> | 0.1717 | 0.419<sup>†</sup> | 14.5 | 0.285 | 0.342<sup>†</sup> | 9.14 | 7/7 |
| FOCAL | 8.73<sup>†</sup> | 108.29<sup>†</sup> | 0.2081 | 0.376<sup>†</sup> | 8.3 | 0.254 | 0.430<sup>†</sup> | 10.57 | 7/7 |
| _Scratch (no pretraining) — reference_ | 8.15<sup>†</sup> | 92.43<sup>†</sup> | 0.1400<sup>†</sup> | 0.488<sup>†</sup> | 23.2 | 0.340 | 0.535<sup>†</sup> | ref | 7/7 |

## Full finetune — primary metric per task

| Encoder | Arm pose<br><sub>pos err (cm) ↓</sub> | Full-body pose<br><sub>MPJPE (mm) ↓</sub> | Odometry<br><sub>ATE (m) ↓</sub> | Microgesture<br><sub>clip F1 (test) ↑</sub> | Language alignment<br><sub>R@1 G=100 (mean i2t/t2i) ↑</sub> | Parkinson's disease<br><sub>subject macro-F1 ↑</sub> | HAR (10 datasets)<br><sub>mean macro-F1 (10 sets) ↑</sub> | Mean rank | Tasks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAE 4-source (no HAR pool), per-window z-score | **7.99** | 89.22 | **0.1079** | **0.635** | 37.5 | 0.590 | **0.649** | 2.29 | 7/7 |
| Ours (MAE, 11-source) | 8.00<sup>†</sup> | **87.46**<sup>†</sup> | 0.1276<sup>†</sup> | 0.580 | 38.8<sup>†</sup> | 0.542 | 0.627<sup>†</sup> | 3.00 | 7/7 |
| Ours (MAE, 5-source) | 8.17<sup>†</sup> | 88.11 | 0.1285<sup>†</sup> | 0.577 | 38.8 | 0.597 | 0.624<sup>†</sup> | 3.71 | 7/7 |
| Ours (MAE, no alt-attn) | 8.16 | 88.28 | 0.1311 | 0.544 | **39.4** | 0.458 | 0.620 | 4.71 | 7/7 |
| MAE 4-source, physical input + patch-normalised loss | 8.16 | 91.19 | 0.1140 | 0.561 | 29.2 | 0.254 | 0.615 | 6.29 | 7/7 |
| RelCon | 8.56 | 91.25 | 0.1419 | 0.618 | 30.7 | **0.619** | 0.545 | 6.43 | 7/7 |
| SimMTM | 8.45 | 93.27 | 0.1371 | 0.579 | 31.3 | 0.515 | 0.578 | 6.86 | 7/7 |
| MAE 4-source (no HAR pool), physical rescale | 8.29 | 92.00 | 0.1136 | 0.543 | 30.1 | 0.313 | 0.613 | 6.86 | 7/7 |
| MAE 4-source (no HAR pool), raw/DC-preserving | 8.42 | 90.00 | 0.1188 | 0.504 | 22.9 | 0.364 | 0.595 | 7.57 | 7/7 |
| Ours (MAE, acc-only) | 8.81 | 96.20 | 0.1613 | 0.548 | 29.7 | 0.606 | 0.571 | 8.29 | 7/7 |
| FOCAL | 8.76 | 97.80 | 0.1517 | 0.531 | 23.5 | 0.331 | 0.563 | 10.00 | 7/7 |
| _Scratch (no pretraining) — reference_ | 8.15<sup>†</sup> | 92.43<sup>†</sup> | 0.1400<sup>†</sup> | 0.488<sup>†</sup> | 23.2 | 0.340 | 0.535<sup>†</sup> | ref | 7/7 |

## Task cards — unified input / output

| Task | Dataset / split | Input | Output | Selection | Reported split | Primary metric |
|---|---|---|---|---|---|---|
| **Arm pose**<br><sub>Nymeria real right-wrist IMU → 5-joint right-arm chain rotations</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=5, 6]`<br><sub>6D-continuous local rotations of SMPL joints [14,17,19,21,23] (collar, shoulder, elbow, wrist, hand), one per 0.32 s patch</sub> | val_loss (min) on val | val | pos err (cm) ↓ |
| **Full-body pose**<br><sub>Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=23, 6]`<br><sub>6D local rotations for SMPL joints 1..23, one per 0.32 s patch (sub_patches=1)</sub> | mpjpe_mm (min) on val | val | MPJPE (mm) ↓ |
| **Odometry**<br><sub>Nymeria real right-wrist IMU → pelvis velocity in a per-window canonical world frame</sub> | Nymeria (real wrist IMU + Project-Aria pelvis trajectory)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, FLU | `[P_t=8, K=4, 3]`<br><sub>3-D pelvis velocity (m/s) in the window's canonical world frame (heading of net displacement → +X), 4 sub-patches per 0.32 s patch (0.08 s resolution)</sub> | val_loss (min) on val | val | ATE (m) ↓ |
| **Microgesture**<br><sub>OpenWatch wrist IMU → 5 gestures + background (paper protocol, real recordings)</sub> | OpenWatch (Bonazzi et al.), open_watch_without_augmentations<br><sub>contamination-safe posture-stratified subject split over canonical identities, seed 42, 80/10/10 (33/5/5 subjects; val 102 clips / test 87 clips)</sub> | `[6, 128]` @ 100 Hz, 1.0 s, FLU | `[6]`<br><sub>class logits per 1 s window; clip label = first run of k=3 identical window predictions, else majority (imu_lm/utils/clip_voting.py)</sub> | macro_f1 (max) on val | test | clip F1 (test) ↑ |
| **Language alignment**<br><sub>HumanML3D synthetic wrist IMU (UniMTS_data) ↔ text retrieval</sub> | HumanML3D / UniMTS_data sim (right-wrist channel), aug_texts captions<br><sub>HumanML3D official train/val/test lists (val 1,197 clips; test 3,701 clips)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[512]`<br><sub>L2-normalised IMU embedding in CLIP ViT-B/32 text space (ContrastiveCLIPHead); text tower trainable, i2t InfoNCE, τ=0.1 fixed</sub> | fixed_budget (latest) on none | test | R@1 G=100 (mean i2t/t2i) ↑ |
| **Parkinson's disease**<br><sub>PADS smartwatch IMU → {Healthy, PD, Differential diagnosis} per subject</sub> | PADS (PhysioNet 1.0.0), 469 patients, 11 tasks × 2 wrists<br><sub>subject-disjoint 80/20 (375 / 94 subjects), splits.group_key=subject_id, seed 0</sub> | `[6, 128]` @ 50 Hz, 2.56 s, FLU | `[3]`<br><sub>class logits per 2.56 s window; subject prediction = argmax of mean softmax over all of the subject's windows</sub> | bal_acc (max) on val | val | subject macro-F1 ↑ |
| **HAR · WISDM**<br><sub>WISDM activity recognition from one wrist/arm IMU (18 classes)</sub> | WISDM: smartwatch (Weiss et al. 2019), 18 activities, 51 subjects; wrist acc+gyro at 20 Hz -> 50 Hz canonical<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=18]`<br><sub>class logits over the 18 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · UTD-MHAD**<br><sub>UTD-MHAD activity recognition from one wrist/arm IMU (21 classes)</sub> | UTD-MHAD: wrist inertial sensor (Chen et al. 2015), 21 actions, 8 subjects; acc+gyro 50 Hz<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=21]`<br><sub>class logits over the 21 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · PAMAP2**<br><sub>PAMAP2 activity recognition from one wrist/arm IMU (18 classes)</sub> | PAMAP2: wrist IMU (Reiss & Stricker 2012), 18 activities, 9 subjects; acc+gyro 100 Hz -> 50 Hz<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=18]`<br><sub>class logits over the 18 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · RecoFit**<br><sub>RecoFit activity recognition from one wrist/arm IMU (13 classes)</sub> | RecoFit: arm-worn IMU gym exercises (Morris et al. 2014), 13 classes; acc+gyro 50 Hz<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=13]`<br><sub>class logits over the 13 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · HHAR**<br><sub>HHAR activity recognition from one wrist/arm IMU (6 classes)</sub> | HHAR: smartwatch acc+gyro (Stisen et al. 2015), 6 activities, 9 users x 4 watches; asof-aligned, 50 Hz canonical<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=6]`<br><sub>class logits over the 6 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · MHEALTH**<br><sub>MHEALTH activity recognition from one wrist/arm IMU (12 classes)</sub> | MHEALTH: right-lower-arm Shimmer2 acc+gyro (Banos et al. 2014), 12 activities, 10 subjects, 50 Hz; raw gyro is bias-dominated<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=12]`<br><sub>class logits over the 12 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · SAMoSA**<br><sub>SAMoSA activity recognition from one wrist IMU (27 raw labels, 20 subjects)</sub> | SAMoSA: kitchen/household micro-activities, 20 subjects, 125 sessions; 6.7 h, 125 sessions, 27 dataset_activity_id labels (head size = the trainer's own label map)<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K]`<br><sub>class logits over the dataset's activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · UT-Watch**<br><sub>UT-Watch activity recognition from one wrist IMU (24 raw labels, 20 subjects)</sub> | UT-Watch: smartwatch daily activities, 20 subjects; 45.3 h, 4 sessions, 24 dataset_activity_id labels (head size = the trainer's own label map)<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K]`<br><sub>class logits over the dataset's activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · PRISM**<br><sub>PrISM-Tracker (Arakawa et al.) — PRISM activity recognition from one wrist IMU (96 raw labels, 31 subjects)</sub> | PRISM: held-out probe dataset (base.yaml splits.probe_dataset), 31 subjects, never in any pretraining pool; 16.3 h, 20 sessions, 96 dataset_activity_id labels (head size = the trainer's own label map)<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K]`<br><sub>class logits over the dataset's activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **Arm pose · rotated frame**<br><sub>Nymeria real right-wrist IMU → 5-joint right-arm chain rotations — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split</sub> | Nymeria (real wrist IMU + Project-Aria mocap)<br><sub>fixed pre-partitioned session directories (session-disjoint)</sub> | `[6, 512]` @ 50 Hz, 10.24 s, FLU | `[P_t=32, J=5, 6]`<br><sub>6D-continuous local rotations of SMPL joints [14,17,19,21,23] (collar, shoulder, elbow, wrist, hand), one per 0.32 s patch</sub> | val_loss (min) on val | val | pos err (cm) ↓ |
| **Microgesture · rotated frame**<br><sub>OpenWatch wrist IMU → 5 gestures + background (paper protocol, real recordings) — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split</sub> | OpenWatch (Bonazzi et al.), open_watch_without_augmentations<br><sub>contamination-safe posture-stratified subject split over canonical identities, seed 42, 80/10/10 (33/5/5 subjects; val 102 clips / test 87 clips)</sub> | `[6, 128]` @ 100 Hz, 1.0 s, FLU | `[6]`<br><sub>class logits per 1 s window; clip label = first run of k=3 identical window predictions, else majority (imu_lm/utils/clip_voting.py)</sub> | macro_f1 (max) on val | test | clip F1 (test) ↑ |
| **HAR · WISDM · rotated frame**<br><sub>WISDM activity recognition from one wrist/arm IMU (18 classes) — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split</sub> | WISDM: smartwatch (Weiss et al. 2019), 18 activities, 51 subjects; wrist acc+gyro at 20 Hz -> 50 Hz canonical<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=18]`<br><sub>class logits over the 18 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |
| **HAR · PAMAP2 · rotated frame**<br><sub>PAMAP2 activity recognition from one wrist/arm IMU (18 classes) — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split</sub> | PAMAP2: wrist IMU (Reiss & Stricker 2012), 18 activities, 9 subjects; acc+gyro 100 Hz -> 50 Hz<br><sub>subject-disjoint 20 % val split (splits.val_ratio 0.2, seed 42); val is reported (no test split in v1)</sub> | `[6, 128]` @ 50 Hz, 2.56 s, sensor | `[K=18]`<br><sub>class logits over the 18 activity labels, one per 2.56 s window</sub> | val_loss (min) on val | val | macro-F1 ↑ |

## Coverage of the controlled grid

✅ scored · 🔄 running · ⏳ queued · 🟡 trained but not yet scored · ⬜ not launched

| Encoder | Protocol | Arm pose | Full-body pose | Odometry | Microgesture | Language alignment | Parkinson's disease | HAR · WISDM | HAR · UTD-MHAD | HAR · PAMAP2 | HAR · RecoFit | HAR · HHAR | HAR · MHEALTH | HAR · SAMoSA | HAR · UT-Watch | HAR · PRISM | Arm pose · rotated frame | Microgesture · rotated frame | HAR · WISDM · rotated frame | HAR · PAMAP2 · rotated frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MAE | probe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE | finetune | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-5src | probe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-5src | finetune | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-noAA | probe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-noAA | finetune | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-acc | probe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| MAE-acc | finetune | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| FOCAL | probe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| FOCAL | finetune | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| SimMTM | probe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| SimMTM | finetune | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| RelCon | probe | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| RelCon | finetune | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ⬜ | ⬜ | ⬜ | ⬜ |
| Scratch | scratch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 🟡 trained, unscored | 🟡 trained, unscored | ✅ | ✅ | ✅ | ✅ |

## HAR (10 datasets) — `har_mean`

_mean window macro-F1 over 10 wrist HAR datasets (WISDM, UTD-MHAD, PAMAP2, RecoFit, HHAR, MHEALTH, SAMoSA, Shoaib, UT-Watch, PRISM)._ Scorer: `bench.leaderboard.inject_aggregates`. Selection: {'split': 'val', 'monitor': 'per member task', 'mode': '-'}. Reported split: **val**.
Primary metric **mean macro-F1 (10 sets)**: mean of the members' primary metric over 9 datasets: har_wisdm, har_utd_mhad, har_pamap2, har_recofit, har_hhar, har_mhealth, har_samosa, har_ut_watch, har_prism.

| Encoder | Protocol | **mean macro-F1 (10 sets)** | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.612 | 42 |  | — | legacy† |
| Ours (MAE, 11-source) | finetune | 0.627 | 42 |  | — | legacy† |
| Ours (MAE, 5-source) | probe | 0.613 | 42 |  | — | legacy† |
| Ours (MAE, 5-source) | finetune | 0.624 | 42 |  | — | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.619 | 42 |  | — | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.620 | 42 |  | — | aggregate |
| Ours (MAE, acc-only) | probe | 0.546 | 42 |  | — | legacy† |
| Ours (MAE, acc-only) | finetune | 0.571 | 42 |  | — | aggregate |
| FOCAL | probe | 0.430 | 42 |  | — | legacy† |
| FOCAL | finetune | 0.563 | 42 |  | — | aggregate |
| SimMTM | probe | 0.525 | 42 |  | — | legacy† |
| SimMTM | finetune | 0.578 | 42 |  | — | aggregate |
| RelCon | probe | 0.342 | 42 |  | — | legacy† |
| RelCon | finetune | 0.545 | 42 |  | — | aggregate |
| Scratch (no pretraining) | scratch | 0.535 | 42 |  | — | legacy† |
| Inertia-1 PatchTST-small arch, from scratch (no pretraining) | scratch | 0.612 | 42 |  | — | aggregate |
| Inertia-1 PatchTST-small arch, random init, frozen (probe control) | probe | 0.440 | 42 |  | — | aggregate |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.513 | 42 |  | — | aggregate |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.530 | 42 |  | — | aggregate |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.614 | 42 |  | — | aggregate |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.614 | 42 |  | — | aggregate |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.498 | 42 |  | — | aggregate |
| MAE recipe R1 (physical scaling + yaw rotation + lr 3e-4) | probe | 0.449 | 42 |  | — | aggregate |
| MAE recipe R2 (R1 + band-weighted loss (4/2/1/0.5)) | probe | 0.459 | 42 |  | — | aggregate |
| MAE recipe R3 (R2 + modality masking p=0.3 (replace-mode loss)) | probe | 0.416 | 42 |  | — | aggregate |
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.277 | 42 |  | — | aggregate |
| MAE mask-ratio 0.40 | probe | 0.539 | 42 |  | — | aggregate |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.453 | 42 |  | — | aggregate |
| MAE mask-ratio 0.50 | probe | 0.532 | 42 |  | — | aggregate |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.612 | 42 |  | — | aggregate |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.649 | 42 |  | — | aggregate |
| MAE mask-ratio 0.60 | probe | 0.548 | 42 |  | — | aggregate |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.508 | 42 |  | — | aggregate |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.595 | 42 |  | — | aggregate |
| MAE mask-ratio 0.85 | probe | 0.505 | 42 |  | — | aggregate |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.557 | 42 |  | — | aggregate |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.613 | 42 |  | — | aggregate |
| MAE mask-ratio 0.90 | probe | 0.497 | 42 |  | — | aggregate |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.545 | 42 |  | — | aggregate |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.615 | 42 |  | — | aggregate |
| SimMTM (4-source pool) | probe | 0.549 | 42 |  | — | aggregate |
| SimMTM (4-source pool) | finetune | 0.634 | 42 |  | — | aggregate |
| FOCAL (4-source pool) | probe | 0.472 | 42 |  | — | aggregate |
| FOCAL (4-source pool) | finetune | 0.595 | 42 |  | — | aggregate |
| MAE acc-only (4-source pool) | probe | 0.588 | 42 |  | — | aggregate |
| MAE acc-only (4-source pool) | finetune | 0.637 | 42 |  | — | aggregate |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.539 | — |  | — | legacy† |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.639 | 42 |  | — | aggregate |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.684 | 42 |  | — | aggregate |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.669 | 42 |  | — | aggregate |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.663 | 42 |  | — | aggregate |

## Arm pose — `arm_pose_nymeria`

_Nymeria real right-wrist IMU → 5-joint right-arm chain rotations._ Scorer: `bench.score.arm_pose`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **pos err (cm)**: FK joint-position error at the 5 arm joints with GT ancestors (scripts/eval_arm_pose_position_error.py).

| Encoder | Protocol | **pos err (cm)** | ang err (°) | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 8.17 | 9.81 | 0.01638 | 42 | best.pt (9000) | `vit1d_arm_pose_probe_nymeria_mae_multidata` | legacy† |
| Ours (MAE, 11-source) | finetune | 8.00 | 9.89 | 0.01599 | 42 | best.pt (3000) | `vit1d_arm_pose_finetune_nymeria_mae_multidata` | legacy† |
| Ours (MAE, 5-source) | probe | 8.14 | 9.86 | 0.01643 | 42 | best.pt (9000) | `vit1d_arm_pose_probe_nymeria_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, 5-source) | finetune | 8.17 | 9.91 | 0.01617 | 42 | best.pt (3000) | `vit1d_arm_pose_finetune_nymeria_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 8.17 | 9.97 | 0.01685 | 42 | best.pt (9000) | `vit1d_arm_pose_probe_nymeria_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 8.16 | 9.98 | 0.01630 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 8.78 | 10.68 | 0.01888 | 42 | best.pt (6000) | `vit1d_arm_pose_probe_nymeria_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 8.81 | 10.90 | 0.01892 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 8.73 | 10.75 | 0.01916 | 42 | best.pt (11000) | `vit1d_arm_pose_probe_nymeria_focal_multidata` | legacy† |
| FOCAL | finetune | 8.76 | 10.85 | 0.01915 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__focal__finetune__s42` | bench |
| SimMTM | probe | 8.56 | 10.39 | 0.01813 | 42 | best.pt (10000) | `vit1d_arm_pose_probe_nymeria_simmtm_multidata` | legacy† |
| SimMTM | finetune | 8.45 | 10.38 | 0.01787 | 42 | best.pt (4000) | `bench_arm_pose_nymeria__simmtm__finetune__s42` | bench |
| RelCon | probe | 8.52 | 10.41 | 0.01826 | 42 | best.pt (8000) | `vit1d_arm_pose_probe_nymeria_relcon_multidata` | legacy† |
| RelCon | finetune | 8.56 | 10.49 | 0.01771 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 8.15 | 9.94 | 0.01672 | 42 | best.pt (6000) | `vit1d_arm_pose_supervised_nymeria_altattn_scratch` | legacy† |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 8.44 | 10.27 | 0.01774 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 8.45 | 10.30 | 0.01779 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 8.25 | 10.02 | 0.01674 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 8.16 | 10.02 | 0.01674 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 8.45 | 10.28 | 0.01779 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__mae_recipe_r0__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 8.46 | 10.29 | 0.01775 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__mae_mr40__probe__s42` | bench |
| MAE mask-ratio 0.50 | probe | 8.41 | 10.27 | 0.01767 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 8.14 | 9.88 | 0.01638 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 7.99 | 9.82 | 0.01584 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__mae_4src_z__finetune__s42` | bench |
| MAE mask-ratio 0.60 | probe | 8.53 | 10.29 | 0.01775 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 8.30 | 10.12 | 0.01718 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 8.42 | 10.26 | 0.01752 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__mae_4src_dc__finetune__s42` | bench |
| MAE mask-ratio 0.85 | probe | 8.47 | 10.33 | 0.01784 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 8.26 | 10.07 | 0.01685 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 8.29 | 10.21 | 0.01695 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__mae_4src_phys__finetune__s42` | bench |
| MAE mask-ratio 0.90 | probe | 8.53 | 10.39 | 0.01803 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 8.30 | 10.09 | 0.01718 | 42 | best.pt (10000) | `bench_arm_pose_nymeria__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 8.16 | 10.08 | 0.01700 | 42 | best.pt (5000) | `bench_arm_pose_nymeria__mae_4src_physnp__finetune__s42` | bench |
| MAE frame-canonical (no rotation) | probe | 8.42 | 10.21 | 0.01756 | 42 | best.pt (9000) | `bench_arm_pose_nymeria__mae_frame_canon__probe__s42` | bench |
| SimMTM (4-source pool) | probe | 8.10 | 9.89 | 0.01662 | 42 | best.pt (17000) | `bench_arm_pose_nymeria__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 8.16 | 9.94 | 0.01653 | 42 | best.pt (4000) | `bench_arm_pose_nymeria__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 8.25 | 10.18 | 0.01741 | 42 | best.pt (11000) | `bench_arm_pose_nymeria__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 8.38 | 10.50 | 0.01800 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__focal_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 8.45 | 10.30 | 0.01743 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 8.50 | 10.38 | 0.01739 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__acconly_4src__finetune__s42` | bench |
| *MobilePoser (Xu et al.)* | external | 9.74 | 24.28 | — | — |  | — | harvest:Experiments_table.md §1 / baseli |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 8.44 | 10.32 | 0.01776 | 42 | best.pt (11000) | `bench_arm_pose_nymeria__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 8.51 | 10.38 | 0.01755 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 8.46 | 10.39 | 0.01790 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 8.35 | 10.31 | 0.01744 | 42 | best.pt (5000) | `bench_arm_pose_nymeria__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 8.27 | 10.13 | 0.01713 | 42 | best.pt (8000) | `bench_arm_pose_nymeria__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 8.27 | 10.17 | 0.01696 | 42 | best.pt (3000) | `bench_arm_pose_nymeria__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 8.47 | 10.41 | 0.01781 | 42 | best.pt (6000) | `bench_arm_pose_nymeria__inertia1_art_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 8.55 | 10.51 | 0.01755 | 42 | best.pt (2000) | `bench_arm_pose_nymeria__inertia1_art_large__finetune__s42` | bench |

## Full-body pose — `full_body_pose_nymeria`

_Nymeria real right-wrist IMU → 23-joint SMPL body (root excluded)._ Scorer: `bench.score.full_body_pose`. Selection: {'split': 'val', 'monitor': 'mpjpe_mm', 'mode': 'min'}. Reported split: **val**.
Primary metric **MPJPE (mm)**: root-relative FK joint position error, joints 1-23 (imu_lm/objectives/pose_1d.py::_compute_val_mpjpe), at the selected checkpoint.

| Encoder | Protocol | **MPJPE (mm)** | val 6D-MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 97.63 | 0.01981 | 42 | best_mpjpe.pt (110000) | `vit1d_full_body_pose_probe_nymeria_mae_multidata` | legacy† |
| Ours (MAE, 11-source) | finetune | 87.46 | 0.01800 | 42 | best_mpjpe.pt (21000) | `vit1d_full_body_pose_finetune_nymeria_mae_multidata_lr1e-3` | legacy† |
| Ours (MAE, 5-source) | probe | 98.11 | 0.01951 | 42 | best_mpjpe.pt (56000) | `vit1d_full_body_pose_probe_nymeria_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, 5-source) | finetune | 88.11 | 0.01817 | 42 | best_mpjpe.pt (21000) | `bench_full_body_pose_nymeria__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 97.43 | 0.01940 | 42 | best_mpjpe.pt (45000) | `vit1d_full_body_pose_probe_nymeria_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 88.28 | 0.01873 | 42 | best_mpjpe.pt (42000) | `bench_full_body_pose_nymeria__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 103.37 | 0.02082 | 42 | best_mpjpe.pt (56000) | `vit1d_full_body_pose_probe_nymeria_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 96.20 | 0.02017 | 42 | best_mpjpe.pt (11000) | `bench_full_body_pose_nymeria__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 108.29 | 0.02208 | 42 | best_mpjpe.pt (105000) | `vit1d_full_body_pose_probe_nymeria_focal_multidata` | legacy† |
| FOCAL | finetune | 97.80 | 0.02083 | 42 | best_mpjpe.pt (24000) | `bench_full_body_pose_nymeria__focal__finetune__s42` | bench |
| SimMTM | probe | 103.48 | 0.02072 | 42 | best_mpjpe.pt (50000) | `vit1d_full_body_pose_probe_nymeria_simmtm_multidata` | legacy† |
| SimMTM | finetune | 93.27 | 0.01947 | 42 | best_mpjpe.pt (16000) | `bench_full_body_pose_nymeria__simmtm__finetune__s42` | bench |
| RelCon | probe | 102.47 | 0.02031 | 42 | best_mpjpe.pt (32000) | `vit1d_full_body_pose_probe_nymeria_relcon_multidata` | legacy† |
| RelCon | finetune | 91.25 | 0.01892 | 42 | best_mpjpe.pt (17000) | `bench_full_body_pose_nymeria__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 92.43 | 0.02047 | 42 | best_mpjpe.pt (370000) | `vit1d_full_body_pose_supervised_nymeria_altattn_scratch` | legacy† |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 101.55 | 0.02006 | 42 | best_mpjpe.pt (48000) | `bench_full_body_pose_nymeria__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 101.08 | 0.02016 | 42 | best_mpjpe.pt (56000) | `bench_full_body_pose_nymeria__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 100.20 | 0.01979 | 42 | best_mpjpe.pt (30000) | `bench_full_body_pose_nymeria__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 100.71 | 0.01978 | 42 | best_mpjpe.pt (29000) | `bench_full_body_pose_nymeria__mae_nhanes2x2_d__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 102.24 | 0.02014 | 42 | best_mpjpe.pt (25000) | `bench_full_body_pose_nymeria__mae_div60_dc__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | finetune | 90.54 | 0.01919 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__mae_div60_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 98.09 | 0.01937 | 42 | best_mpjpe.pt (38000) | `bench_full_body_pose_nymeria__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 89.22 | 0.01881 | 42 | best_mpjpe.pt (16000) | `bench_full_body_pose_nymeria__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 101.94 | 0.02030 | 42 | best_mpjpe.pt (87000) | `bench_full_body_pose_nymeria__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 90.00 | 0.01876 | 42 | best_mpjpe.pt (15000) | `bench_full_body_pose_nymeria__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 100.34 | 0.02000 | 42 | best_mpjpe.pt (41000) | `bench_full_body_pose_nymeria__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 92.00 | 0.01934 | 42 | best_mpjpe.pt (8000) | `bench_full_body_pose_nymeria__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 100.43 | 0.02014 | 42 | best_mpjpe.pt (62000) | `bench_full_body_pose_nymeria__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 91.19 | 0.01933 | 42 | best_mpjpe.pt (13000) | `bench_full_body_pose_nymeria__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 100.83 | 0.02025 | 42 | best_mpjpe.pt (68000) | `bench_full_body_pose_nymeria__simmtm_4src__probe__s42` | bench |
| FOCAL (4-source pool) | probe | 105.50 | 0.02133 | 42 | best_mpjpe.pt (64000) | `bench_full_body_pose_nymeria__focal_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | probe | 101.87 | 0.02057 | 42 | best_mpjpe.pt (34000) | `bench_full_body_pose_nymeria__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 96.01 | 0.02073 | 42 | best_mpjpe.pt (10000) | `bench_full_body_pose_nymeria__acconly_4src__finetune__s42` | bench |
| *MobilePoser (Xu et al.)* | external | 91.88 | — | — |  | — | harvest:Experiments_table.md §2 / baseli |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 101.06 | 0.02002 | 42 | best_mpjpe.pt (59000) | `bench_full_body_pose_nymeria__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 93.61 | 0.01970 | 42 | best_mpjpe.pt (10000) | `bench_full_body_pose_nymeria__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 101.54 | 0.02052 | 42 | best.pt (55000) | `bench_full_body_pose_nymeria__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 94.12 | 0.01991 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 97.93 | 0.01950 | 42 | best_mpjpe.pt (44000) | `bench_full_body_pose_nymeria__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 92.74 | 0.01972 | 42 | best_mpjpe.pt (12000) | `bench_full_body_pose_nymeria__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 101.85 | 0.02056 | 42 | best_mpjpe.pt (29000) | `bench_full_body_pose_nymeria__inertia1_art_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 94.77 | 0.02035 | 42 | best_mpjpe.pt (14000) | `bench_full_body_pose_nymeria__inertia1_art_large__finetune__s42` | bench |

## Odometry — `odometry_nymeria_worldframe`

_Nymeria real right-wrist IMU → pelvis velocity in a per-window canonical world frame._ Scorer: `bench.score.odometry`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **ATE (m)**: mean absolute trajectory error of the cumulative-sum reconstruction over all val windows (scripts/eval_odom_nymeria_worldframe.py::reconstruct_and_score).

| Encoder | Protocol | **ATE (m)** | drift (%) | vel MSE | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.1493 | 41.34 | 0.02091 | 42 | best.pt (94000) | `bench_odometry_nymeria_worldframe__mae_11src__probe__s42` | bench |
| Ours (MAE, 11-source) | finetune | 0.1276 | 33.95 | 0.01724 | 42 | best.pt (19000) | `vit1d_odom_worldframe_finetune_nymeria_mae_multidata` | legacy† |
| Ours (MAE, 5-source) | probe | 0.1564 | 42.71 | 0.02220 | 42 | best.pt (88000) | `bench_odometry_nymeria_worldframe__mae_5src__probe__s42` | bench |
| Ours (MAE, 5-source) | finetune | 0.1285 | 34.93 | 0.01747 | 42 | best.pt (27000) | `vit1d_odom_worldframe_finetune_nymeria_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.1585 | 45.99 | 0.02181 | 42 | best.pt (63000) | `bench_odometry_nymeria_worldframe__mae_noaltattn__probe__s42` | bench |
| Ours (MAE, no alt-attn) | finetune | 0.1311 | 36.69 | 0.01762 | 42 | best.pt (17000) | `bench_odometry_nymeria_worldframe__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.1822 | 51.27 | 0.02615 | 42 | best.pt (63000) | `bench_odometry_nymeria_worldframe__mae_acconly__probe__s42` | bench |
| Ours (MAE, acc-only) | finetune | 0.1613 | 45.99 | 0.02203 | 42 | best.pt (27000) | `bench_odometry_nymeria_worldframe__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.2081 | 62.44 | 0.02805 | 42 | best.pt (175000) | `bench_odometry_nymeria_worldframe__focal__probe__s42` | bench |
| FOCAL | finetune | 0.1517 | 42.69 | 0.02074 | 42 | best.pt (62000) | `bench_odometry_nymeria_worldframe__focal__finetune__s42` | bench |
| SimMTM | probe | 0.1793 | 49.62 | 0.02504 | 42 | best.pt (102000) | `bench_odometry_nymeria_worldframe__simmtm__probe__s42` | bench |
| SimMTM | finetune | 0.1371 | 37.58 | 0.01868 | 42 | best.pt (39000) | `bench_odometry_nymeria_worldframe__simmtm__finetune__s42` | bench |
| RelCon | probe | 0.1717 | 47.64 | 0.02380 | 42 | best.pt (198000) | `bench_odometry_nymeria_worldframe__relcon__probe__s42` | bench |
| RelCon | finetune | 0.1419 | 38.30 | 0.01938 | 42 | best.pt (25000) | `bench_odometry_nymeria_worldframe__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.1400 | 37.69 | 0.01911 | 42 | best.pt (36000) | `vit1d_odom_worldframe_supervised_nymeria_altattn_scratch` | legacy† |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.1730 | 48.31 | 0.02421 | 42 | best.pt (82000) | `bench_odometry_nymeria_worldframe__mae_div60_dc__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | finetune | 0.1220 | 32.97 | 0.01627 | 42 | best.pt (44000) | `bench_odometry_nymeria_worldframe__mae_div60_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.1425 | 40.63 | 0.01931 | 42 | best.pt (134000) | `bench_odometry_nymeria_worldframe__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.1079 | 29.18 | 0.01427 | 42 | best.pt (38000) | `bench_odometry_nymeria_worldframe__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.1696 | 50.66 | 0.02297 | 42 | best.pt (156000) | `bench_odometry_nymeria_worldframe__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.1188 | 31.56 | 0.01590 | 42 | best.pt (47000) | `bench_odometry_nymeria_worldframe__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.1563 | 46.53 | 0.02122 | 42 | best.pt (119000) | `bench_odometry_nymeria_worldframe__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.1136 | 30.49 | 0.01502 | 42 | best.pt (36000) | `bench_odometry_nymeria_worldframe__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.1588 | 47.09 | 0.02106 | 42 | best.pt (182000) | `bench_odometry_nymeria_worldframe__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.1140 | 31.73 | 0.01514 | 42 | best.pt (44000) | `bench_odometry_nymeria_worldframe__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.1655 | 49.52 | 0.02233 | 42 | best.pt (182000) | `bench_odometry_nymeria_worldframe__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.1079 | 29.32 | 0.01431 | 42 | best.pt (55000) | `bench_odometry_nymeria_worldframe__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.1911 | 57.56 | 0.02647 | 42 | best.pt (337000) | `bench_odometry_nymeria_worldframe__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.1236 | 34.53 | 0.01640 | 42 | best.pt (63000) | `bench_odometry_nymeria_worldframe__focal_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.1596 | 45.88 | 0.02244 | 42 | best.pt (137000) | `bench_odometry_nymeria_worldframe__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.1431 | 40.67 | 0.01928 | 42 | best.pt (25000) | `bench_odometry_nymeria_worldframe__acconly_4src__finetune__s42` | bench |
| *MobilePoser (Xu et al.)* | external | 0.2355 | 59.42 | 0.02963 | — | checkpoints/nymeria_rw/best.pt | — | bench |
| *TLIO* | external | 0.1510 | 38.79 | 0.02419 | — |  | — | harvest:Experiments_table.md §4b / basel |
| *Air-IO* | external | 0.1444 | 38.56 | 0.02461 | — |  | — | harvest:Experiments_table.md §4b / basel |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.1626 | 45.53 | 0.02256 | 42 | best.pt (175000) | `bench_odometry_nymeria_worldframe__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.1518 | 42.01 | 0.02116 | 42 | best.pt (39000) | `bench_odometry_nymeria_worldframe__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.1667 | 48.00 | 0.02327 | 42 | best.pt (75000) | `bench_odometry_nymeria_worldframe__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.1507 | 42.33 | 0.02077 | 42 | best.pt (24000) | `bench_odometry_nymeria_worldframe__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 0.1651 | 48.48 | 0.02242 | 42 | best.pt (69000) | `bench_odometry_nymeria_worldframe__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 0.1437 | 39.60 | 0.01968 | 42 | best.pt (16000) | `bench_odometry_nymeria_worldframe__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 0.1806 | 53.26 | 0.02463 | 42 | best.pt (46000) | `bench_odometry_nymeria_worldframe__inertia1_art_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 0.1434 | 39.29 | 0.02017 | 42 | best.pt (25000) | `bench_odometry_nymeria_worldframe__inertia1_art_large__finetune__s42` | bench |

## Microgesture — `microgesture_openwatch`

_OpenWatch wrist IMU → 5 gestures + background (paper protocol, real recordings)._ Scorer: `bench.score.microgesture`. Selection: {'split': 'val', 'monitor': 'macro_f1', 'mode': 'max', 'note': 'window-level val macro-F1 (paper §3.3.3), inverse-frequency class-weighted CE'}. Reported split: **test**.

| Encoder | Protocol | **clip F1 (test)** | clip acc (test) | win F1 (test) | win acc (test) | clip F1 (val) | win F1 (val) | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.606 | 0.598 | 0.490 | 0.508 | 0.674 | 0.561 | 42 | best.pt (1200) | `vit1d_gesture_probe_openwatch_6class_mae_windowed_strat` | legacy† |
| Ours (MAE, 11-source) | finetune | 0.580 | 0.598 | 0.519 | 0.563 | 0.564 | 0.548 | 42 | best.pt (200) | `bench_microgesture_openwatch__mae_11src__finetune__s42` | bench |
| Ours (MAE, 5-source) | probe | 0.647 | 0.678 | 0.494 | 0.536 | 0.729 | 0.544 | 42 | best.pt (8400) | `bench_microgesture_openwatch__mae_5src__probe__s42` | bench |
| Ours (MAE, 5-source) | finetune | 0.577 | 0.655 | 0.524 | 0.607 | 0.634 | 0.573 | 42 | best.pt (3400) | `bench_microgesture_openwatch__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 0.647 | 0.655 | 0.524 | 0.552 | 0.647 | 0.552 | 42 | best.pt (9000) | `vit1d_gesture_probe_openwatch_6class_mae_noaltattn_windowed_strat` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.544 | 0.644 | 0.481 | 0.586 | 0.620 | 0.579 | 42 | best.pt (3400) | `bench_microgesture_openwatch__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.586 | 0.586 | 0.478 | 0.496 | 0.625 | 0.508 | 42 | best.pt (3600) | `vit1d_gesture_probe_openwatch_6class_mae_acconly_windowed_strat` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.548 | 0.621 | 0.502 | 0.582 | 0.695 | 0.611 | 42 | best.pt (2200) | `bench_microgesture_openwatch__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.376 | 0.414 | 0.337 | 0.379 | 0.478 | 0.393 | 42 | best.pt (4800) | `vit1d_gesture_probe_openwatch_6class_focal_windowed_strat` | legacy† |
| FOCAL | finetune | 0.531 | 0.598 | 0.486 | 0.559 | 0.530 | 0.489 | 42 | best.pt (3200) | `bench_microgesture_openwatch__focal__finetune__s42` | bench |
| SimMTM | probe | 0.532 | 0.552 | 0.447 | 0.471 | 0.607 | 0.502 | 42 | best.pt (9600) | `vit1d_gesture_probe_openwatch_6class_simmtm_windowed_strat` | legacy† |
| SimMTM | finetune | 0.579 | 0.598 | 0.502 | 0.536 | 0.589 | 0.534 | 42 | best.pt (600) | `bench_microgesture_openwatch__simmtm__finetune__s42` | bench |
| RelCon | probe | 0.419 | 0.425 | 0.407 | 0.421 | 0.411 | 0.386 | 42 | best.pt (16800) | `vit1d_gesture_probe_openwatch_6class_relcon_windowed_strat` | legacy† |
| RelCon | finetune | 0.618 | 0.609 | 0.542 | 0.550 | 0.580 | 0.543 | 42 | best.pt (600) | `bench_microgesture_openwatch__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.488 | 0.609 | 0.403 | 0.519 | 0.621 | 0.486 | 42 | best.pt (2800) | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch_windowed_strat` | legacy† |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.501 | 0.563 | 0.440 | 0.487 | 0.655 | 0.544 | 42 | best.pt (7400) | `bench_microgesture_openwatch__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.568 | 0.598 | 0.480 | 0.525 | 0.625 | 0.524 | 42 | best.pt (3400) | `bench_microgesture_openwatch__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.351 | 0.345 | 0.305 | 0.320 | 0.411 | 0.337 | 42 | best.pt (2000) | `bench_microgesture_openwatch__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.355 | 0.345 | 0.325 | 0.331 | 0.363 | 0.295 | 42 | best.pt (3200) | `bench_microgesture_openwatch__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.503 | 0.563 | 0.458 | 0.517 | 0.662 | 0.510 | 42 | best.pt (5400) | `bench_microgesture_openwatch__mae_recipe_r0__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 0.536 | 0.586 | 0.473 | 0.523 | 0.641 | 0.534 | 42 | best.pt (7400) | `bench_microgesture_openwatch__mae_mr40__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.394 | 0.425 | 0.385 | 0.410 | 0.521 | 0.473 | 42 | best.pt (8800) | `bench_microgesture_openwatch__mae_div60_dc__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | finetune | 0.595 | 0.667 | 0.460 | 0.540 | 0.629 | 0.515 | 42 | best.pt (3000) | `bench_microgesture_openwatch__mae_div60_dc__finetune__s42` | bench |
| MAE mask-ratio 0.50 | probe | 0.501 | 0.540 | 0.472 | 0.515 | 0.621 | 0.503 | 42 | best.pt (9400) | `bench_microgesture_openwatch__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.613 | 0.632 | 0.543 | 0.567 | 0.614 | 0.558 | 42 | best.pt (6200) | `bench_microgesture_openwatch__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.635 | 0.701 | 0.574 | 0.646 | 0.665 | 0.598 | 42 | best.pt (7800) | `bench_microgesture_openwatch__mae_4src_z__finetune__s42` | bench |
| MAE mask-ratio 0.60 | probe | 0.550 | 0.575 | 0.441 | 0.498 | 0.575 | 0.524 | 42 | best.pt (7400) | `bench_microgesture_openwatch__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.472 | 0.494 | 0.414 | 0.443 | 0.521 | 0.483 | 42 | best.pt (4200) | `bench_microgesture_openwatch__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.504 | 0.517 | 0.444 | 0.473 | 0.533 | 0.487 | 42 | best.pt (1000) | `bench_microgesture_openwatch__mae_4src_dc__finetune__s42` | bench |
| MAE mask-ratio 0.85 | probe | 0.541 | 0.563 | 0.459 | 0.500 | 0.620 | 0.501 | 42 | best.pt (3400) | `bench_microgesture_openwatch__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.556 | 0.575 | 0.458 | 0.490 | 0.543 | 0.486 | 42 | best.pt (7400) | `bench_microgesture_openwatch__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.543 | 0.598 | 0.492 | 0.557 | 0.638 | 0.586 | 42 | best.pt (4200) | `bench_microgesture_openwatch__mae_4src_phys__finetune__s42` | bench |
| MAE mask-ratio 0.90 | probe | 0.490 | 0.529 | 0.437 | 0.471 | 0.638 | 0.505 | 42 | best.pt (5800) | `bench_microgesture_openwatch__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.492 | 0.517 | 0.456 | 0.489 | 0.523 | 0.477 | 42 | best.pt (12200) | `bench_microgesture_openwatch__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.561 | 0.575 | 0.486 | 0.508 | 0.586 | 0.547 | 42 | best.pt (600) | `bench_microgesture_openwatch__mae_4src_physnp__finetune__s42` | bench |
| MAE frame-canonical (no rotation) | probe | 0.459 | 0.506 | 0.442 | 0.498 | 0.636 | 0.542 | 42 | best.pt (13600) | `bench_microgesture_openwatch__mae_frame_canon__probe__s42` | bench |
| SimMTM (4-source pool) | probe | 0.522 | 0.540 | 0.455 | 0.473 | 0.548 | 0.503 | 42 | best.pt (3800) | `bench_microgesture_openwatch__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.594 | 0.678 | 0.504 | 0.592 | 0.647 | 0.625 | 42 | best.pt (10400) | `bench_microgesture_openwatch__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.369 | 0.437 | 0.339 | 0.406 | 0.483 | 0.426 | 42 | best.pt (6000) | `bench_microgesture_openwatch__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.482 | 0.540 | 0.398 | 0.450 | 0.573 | 0.515 | 42 | best.pt (1000) | `bench_microgesture_openwatch__focal_4src__finetune__s42` | bench |
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

## Language alignment — `language_humanml3d`

_HumanML3D synthetic wrist IMU (UniMTS_data) ↔ text retrieval._ Scorer: `bench.score.language`. Selection: {'split': 'none', 'monitor': 'fixed_budget', 'mode': 'latest', 'note': 'fixed 8k-step budget, final checkpoint (latest.pt); §13e: val-loss best.pt under-selects for retrieval, latest ≥ best on every arm'}. Reported split: **test**.
Primary metric **R@1 G=100 (mean i2t/t2i)**: mean of IMU→Text and Text→IMU Recall@1 over 200 random 100-item galleries, first-caption protocol.

| Encoder | Protocol | **R@1 G=100 (mean i2t/t2i)** | i2t R@1 | t2i R@1 | i2t MRR | t2i MRR | i2t R@1 full | t2i R@1 full | i2t MRR full | t2i MRR full | R@1 G=100 (val) | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 27.1 | 27.6 | 26.6 | 42.1 | 41.1 | 3.3 | 2.9 | 8.5 | 7.5 | 27.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_11src__probe__s42` | bench |
| Ours (MAE, 11-source) | finetune | 38.8 | 39.3 | 38.3 | 54.9 | 53.9 | 5.3 | 4.1 | 12.9 | 10.8 | 40.7 | 42 | latest.pt (8000) | `rtune_lr1e-4_tau0.1fixed` | legacy† |
| Ours (MAE, 5-source) | probe | 27.9 | 28.3 | 27.5 | 42.5 | 41.8 | 3.6 | 3.4 | 9.0 | 8.2 | 27.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_5src__probe__s42` | bench |
| Ours (MAE, 5-source) | finetune | 38.8 | 39.5 | 38.1 | 54.9 | 53.9 | 5.1 | 4.4 | 12.6 | 11.0 | 39.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 28.6 | 29.5 | 27.7 | 43.7 | 42.2 | 3.4 | 2.5 | 8.9 | 7.4 | 27.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_noaltattn__probe__s42` | bench |
| Ours (MAE, no alt-attn) | finetune | 39.4 | 39.8 | 38.9 | 55.2 | 54.2 | 5.3 | 4.4 | 13.2 | 11.6 | 39.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 19.3 | 19.8 | 18.7 | 32.8 | 31.9 | 2.0 | 2.0 | 5.6 | 5.3 | 18.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_acconly__probe__s42` | bench |
| Ours (MAE, acc-only) | finetune | 29.7 | 30.0 | 29.4 | 45.2 | 44.6 | 3.1 | 3.3 | 8.7 | 8.2 | 30.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 8.3 | 8.6 | 8.0 | 19.2 | 18.2 | 0.5 | 0.3 | 2.2 | 1.8 | 7.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__focal__probe__s42` | bench |
| FOCAL | finetune | 23.5 | 23.6 | 23.3 | 38.2 | 37.8 | 1.8 | 1.9 | 6.1 | 6.1 | 22.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__focal__finetune__s42` | bench |
| SimMTM | probe | 16.7 | 17.3 | 16.1 | 29.8 | 28.6 | 1.7 | 1.2 | 4.9 | 4.0 | 17.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm__probe__s42` | bench |
| SimMTM | finetune | 31.3 | 32.0 | 30.6 | 47.4 | 46.0 | 3.9 | 3.1 | 9.9 | 8.6 | 32.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm__finetune__s42` | bench |
| RelCon | probe | 14.5 | 15.3 | 13.6 | 28.2 | 26.1 | 1.1 | 1.1 | 3.8 | 3.4 | 13.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__relcon__probe__s42` | bench |
| RelCon | finetune | 30.7 | 31.2 | 30.3 | 46.3 | 45.5 | 3.5 | 3.2 | 9.5 | 8.5 | 31.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 23.2 | 23.6 | 22.8 | 38.5 | 37.6 | 2.1 | 2.1 | 6.5 | 6.0 | 23.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__scratch__scratch__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 24.1 | 24.5 | 23.7 | 38.6 | 37.7 | 2.6 | 2.5 | 7.2 | 6.7 | 24.2 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 37.5 | 38.3 | 36.7 | 53.5 | 52.2 | 4.4 | 4.7 | 11.8 | 10.8 | 37.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 8.3 | 8.9 | 7.7 | 19.7 | 18.2 | 0.5 | 0.5 | 2.2 | 1.8 | 8.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 22.9 | 23.1 | 22.7 | 38.0 | 37.5 | 2.1 | 2.2 | 6.2 | 6.1 | 22.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 17.0 | 17.3 | 16.6 | 30.4 | 29.6 | 1.8 | 1.5 | 4.9 | 4.6 | 16.3 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 30.1 | 30.7 | 29.5 | 45.8 | 44.6 | 3.5 | 3.4 | 9.2 | 8.6 | 30.1 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 16.5 | 17.0 | 16.0 | 30.0 | 29.0 | 1.3 | 1.5 | 4.4 | 4.1 | 15.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 29.2 | 29.5 | 28.9 | 44.8 | 44.1 | 3.0 | 3.2 | 8.6 | 8.2 | 29.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 16.6 | 16.8 | 16.5 | 29.8 | 29.2 | 1.6 | 1.4 | 4.7 | 4.2 | 18.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 33.7 | 34.0 | 33.5 | 49.3 | 48.5 | 4.6 | 4.5 | 10.9 | 10.1 | 34.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__simmtm_4src__finetune__s42` | bench |
| *UniMTS (official ckpt, zero-shot)* ⚠ invalid | external | 17.6 | 33.6 | 1.6 | 49.3 | 6.4 | — | — | — | — | — | — |  | — | harvest:analysis/retrieval_eval/humanml3 |
| *UniMTS (from scratch, wrist-only)* | external | 13.1 | 24.4 | 1.7 | 39.1 | 6.7 | — | — | — | — | — | — |  | — | harvest:analysis/retrieval_eval/humanml3 |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 19.8 | 20.2 | 19.5 | 34.3 | 33.5 | 1.3 | 1.9 | 5.3 | 5.4 | 21.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 27.7 | 28.2 | 27.2 | 42.8 | 41.8 | 3.6 | 3.1 | 8.9 | 7.9 | 27.5 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 19.8 | 20.5 | 19.1 | 34.3 | 33.1 | 2.0 | 1.6 | 5.8 | 5.1 | 20.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 31.3 | 31.7 | 30.9 | 46.7 | 45.8 | 4.0 | 3.6 | 10.0 | 9.1 | 32.6 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 24.2 | 24.4 | 23.9 | 38.6 | 38.0 | 2.9 | 3.0 | 7.6 | 7.3 | 25.0 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 35.3 | 35.7 | 34.9 | 50.8 | 50.0 | 4.3 | 4.0 | 11.1 | 10.4 | 36.8 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 21.3 | 22.0 | 20.6 | 35.8 | 34.4 | 2.1 | 2.2 | 6.2 | 5.9 | 20.9 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_art_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 37.0 | 37.5 | 36.6 | 52.7 | 51.9 | 4.5 | 4.4 | 11.7 | 11.1 | 37.4 | 42 | latest.pt (8000) | `bench_language_humanml3d__inertia1_art_large__finetune__s42` | bench |

## Parkinson's disease — `pd_pads`

_PADS smartwatch IMU → {Healthy, PD, Differential diagnosis} per subject._ Scorer: `bench.score.pd`. Selection: {'split': 'val', 'monitor': 'bal_acc', 'mode': 'max', 'note': 'window-level val balanced accuracy'}. Reported split: **val**.
Primary metric **subject macro-F1**: mean-prob aggregation over ~180 windows/subject (scripts/eval_pads_subject_level.py), macro-F1 over HC/PD/DD; a class never predicted contributes F1=0. Chance ≈ 0.25-0.33.

| Encoder | Protocol | **subject macro-F1** | subject bal-acc | #classes predicted (of 3) | window bal-acc | window macro-F1 | window acc | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.388 | 40.9% | 3 | 43.4% | 0.441 | 59.8% | 42 | best.pt (5200) | `bench_pd_pads__mae_11src__probe__s42` | bench |
| Ours (MAE, 11-source) | finetune | 0.542 | 54.8% | 3 | 44.9% | 0.441 | 53.4% | 42 | best.pt (1600) | `bench_pd_pads__mae_11src__finetune__s42` | bench |
| Ours (MAE, 5-source) | probe | 0.340 | 37.9% | 2 | 41.9% | 0.425 | 58.3% | 42 | best.pt (600) | `bench_pd_pads__mae_5src__probe__s42` | bench |
| Ours (MAE, 5-source) | finetune | 0.597 | 56.1% | 3 | 45.6% | 0.450 | 54.4% | 42 | best.pt (9600) | `bench_pd_pads__mae_5src__finetune__s42` | bench |
| Ours (MAE, 5-source) ⚠ partial | legacy_valloss | 0.299 | 35.6% | 2 | 38.3% | 0.361 | 63.1% | 42,42 | best.pt (400) | `vit1d_pd_finetune_pads_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.386 | 40.9% | 2 | 43.5% | 0.443 | 61.1% | 42 | best.pt (5200) | `bench_pd_pads__mae_noaltattn__probe__s42` | bench |
| Ours (MAE, no alt-attn) | finetune | 0.458 | 45.0% | 3 | 45.5% | 0.456 | 59.3% | 42 | best.pt (1400) | `bench_pd_pads__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.397 | 41.8% | 2 | 41.1% | 0.410 | 57.2% | 42 | best.pt (600) | `bench_pd_pads__mae_acconly__probe__s42` | bench |
| Ours (MAE, acc-only) | finetune | 0.606 | 55.8% | 3 | 45.8% | 0.453 | 54.7% | 42 | best.pt (1600) | `bench_pd_pads__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.254 | 33.3% | 1 | 33.9% | 0.285 | 59.8% | 42 | best.pt (3400) | `bench_pd_pads__focal__probe__s42` | bench |
| FOCAL | finetune | 0.331 | 37.2% | 3 | 40.1% | 0.398 | 51.5% | 42 | best.pt (5400) | `bench_pd_pads__focal__finetune__s42` | bench |
| FOCAL ⚠ partial | legacy_valloss | 0.254 | 33.3% | 1 | 33.5% | 0.259 | 61.8% | 42,42 | best.pt (800) | `vit1d_pd_finetune_pads_focal` | legacy† |
| SimMTM | probe | 0.368 | 39.6% | 2 | 39.1% | 0.359 | 54.6% | 42 | best.pt (600) | `bench_pd_pads__simmtm__probe__s42` | bench |
| SimMTM | finetune | 0.515 | 48.8% | 3 | 46.8% | 0.465 | 56.7% | 42 | best.pt (2200) | `bench_pd_pads__simmtm__finetune__s42` | bench |
| SimMTM ⚠ partial | legacy_valloss | 0.254 | 33.3% | 1 | 35.6% | 0.304 | 62.7% | 42 | best.pt (10800) | `vit1d_pd_probe_pads_simmtm` | legacy† |
| RelCon | probe | 0.285 | 34.8% | 2 | 35.4% | 0.317 | 59.8% | 42 | best.pt (600) | `bench_pd_pads__relcon__probe__s42` | bench |
| RelCon | finetune | 0.619 | 58.7% | 3 | 45.1% | 0.440 | 51.5% | 42 | best.pt (1600) | `bench_pd_pads__relcon__finetune__s42` | bench |
| RelCon ⚠ partial | legacy_valloss | 0.254 | 33.3% | 1 | 35.3% | 0.297 | 62.4% | 42,42 | best.pt (1000) | `vit1d_pd_finetune_pads_relcon` | legacy† |
| Scratch (no pretraining) | scratch | 0.340 | 37.9% | 2 | 41.6% | 0.421 | 58.2% | 42 | best.pt (2600) | `bench_pd_pads__scratch__scratch__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.463 | 47.3% | 2 | 45.7% | 0.461 | 57.6% | 42 | best.pt (1600) | `bench_pd_pads__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.590 | 55.2% | 3 | 45.0% | 0.443 | 54.5% | 42 | best.pt (5600) | `bench_pd_pads__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.317 | 39.2% | 2 | 38.2% | 0.341 | 51.4% | 42 | best.pt (2400) | `bench_pd_pads__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.364 | 39.4% | 2 | 38.3% | 0.375 | 59.2% | 42 | best.pt (5800) | `bench_pd_pads__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.334 | 37.3% | 2 | 38.8% | 0.384 | 56.1% | 42 | best.pt (1600) | `bench_pd_pads__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.313 | 36.4% | 2 | 41.1% | 0.414 | 54.7% | 42 | best.pt (6200) | `bench_pd_pads__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.357 | 38.8% | 2 | 39.3% | 0.374 | 57.2% | 42 | best.pt (800) | `bench_pd_pads__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.254 | 33.3% | 1 | 41.2% | 0.403 | 52.8% | 42 | best.pt (5600) | `bench_pd_pads__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.386 | 40.9% | 2 | 36.8% | 0.344 | 55.8% | 42 | best.pt (600) | `bench_pd_pads__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.563 | 51.9% | 3 | 44.9% | 0.453 | 57.0% | 42 | best.pt (5200) | `bench_pd_pads__simmtm_4src__finetune__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.254 | 33.3% | 1 | 34.7% | 0.284 | 62.5% | 42 | best.pt (9800) | `bench_pd_pads__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.386 | 40.9% | 2 | 40.2% | 0.403 | 59.8% | 42 | best.pt (9000) | `bench_pd_pads__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.254 | 33.3% | 1 | 35.8% | 0.312 | 62.5% | 42 | best.pt (7600) | `bench_pd_pads__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.386 | 40.9% | 2 | 38.9% | 0.374 | 62.5% | 42 | best.pt (2600) | `bench_pd_pads__inertia1_art__finetune__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | probe | 0.315 | 36.4% | 3 | 39.5% | 0.388 | 61.2% | 42 | best.pt (5600) | `bench_pd_pads__inertia1_patchtst_large__probe__s42` | bench |
| Inertia-1 PatchTST-large (NHANES repro) | finetune | 0.364 | 39.4% | 2 | 38.7% | 0.378 | 60.7% | 42 | best.pt (4800) | `bench_pd_pads__inertia1_patchtst_large__finetune__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | probe | 0.298 | 54.3% | 2 | 42.5% | 0.248 | 25.0% | 42 | best.pt (600) | `bench_pd_pads__inertia1_art_large__probe__s42` | bench |
| Inertia-1 AR-Transformer-large (NHANES repro) | finetune | 0.254 | 33.3% | 1 | 37.4% | 0.359 | 59.6% | 42 | best.pt (600) | `bench_pd_pads__inertia1_art_large__finetune__s42` | bench |

## HAR · WISDM — `har_wisdm`

_WISDM activity recognition from one wrist/arm IMU (18 classes)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.639 | 65.5% | 65.5% | 1.0511 | 42 | best.pt (10800) | `vit1d_har_probe_wisdm_mae_multidata` | legacy† |
| Ours (MAE, 11-source) | finetune | 0.704 | 71.2% | 71.2% | 0.8594 | 42 | best.pt (2200) | `vit1d_har_finetune_wisdm_mae_multidata` | legacy† |
| Ours (MAE, 5-source) | probe | 0.630 | 64.8% | 64.8% | 1.0639 | 42 | best.pt (17200) | `vit1d_har_probe_wisdm_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, 5-source) | finetune | 0.711 | 71.9% | 71.9% | 0.8404 | 42 | best.pt (2200) | `vit1d_har_finetune_wisdm_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.629 | 64.8% | 64.8% | 1.0923 | 42 | best.pt (17200) | `vit1d_har_probe_wisdm_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.707 | 71.6% | 71.6% | 0.8329 | 42 | best.pt (2200) | `bench_har_wisdm__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.535 | 55.1% | 55.1% | 1.3547 | 42 | best.pt (10800) | `vit1d_har_probe_wisdm_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.647 | 65.6% | 65.6% | 1.0856 | 42 | best.pt (4200) | `bench_har_wisdm__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.390 | 42.0% | 42.0% | 1.8537 | 42 | best.pt (19000) | `vit1d_har_probe_wisdm_focal_multidata` | legacy† |
| FOCAL | finetune | 0.655 | 66.3% | 66.3% | 1.0485 | 42 | best.pt (11400) | `bench_har_wisdm__focal__finetune__s42` | bench |
| SimMTM | probe | 0.509 | 53.6% | 53.6% | 1.4945 | 42 | best.pt (17200) | `vit1d_har_probe_wisdm_simmtm_multidata` | legacy† |
| SimMTM | finetune | 0.674 | 68.6% | 68.6% | 0.9613 | 42 | best.pt (6600) | `bench_har_wisdm__simmtm__finetune__s42` | bench |
| RelCon | probe | 0.335 | 38.1% | 38.1% | 2.3599 | 42 | best.pt (19600) | `vit1d_har_probe_wisdm_relcon_multidata` | legacy† |
| RelCon | finetune | 0.670 | 67.7% | 67.7% | 1.0058 | 42 | best.pt (4200) | `bench_har_wisdm__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.606 | 60.9% | 60.9% | 1.2387 | 42 | best.pt (9000) | `vit1d_har_supervised_wisdm_altattn_scratch` | legacy† |
| Inertia-1 PatchTST-small arch, from scratch (no pretraining) | scratch | 0.676 | 68.6% | 68.6% | 1.0037 | 42 | best.pt (18400) | `bench_har_wisdm__inertia1_patchtst_scratch__scratch__s42` | bench |
| Inertia-1 PatchTST-small arch, random init, frozen (probe control) | probe | 0.424 | 45.8% | 45.8% | 1.7609 | 42 | best.pt (18600) | `bench_har_wisdm__inertia1_patchtst_randinit__probe__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.521 | 54.4% | 54.4% | 1.4637 | 42 | best.pt (16800) | `bench_har_wisdm__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.532 | 55.5% | 55.5% | 1.4414 | 42 | best.pt (16800) | `bench_har_wisdm__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.591 | 61.1% | 61.1% | 1.2334 | 42 | best.pt (15800) | `bench_har_wisdm__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.598 | 61.4% | 61.4% | 1.2047 | 42 | best.pt (15800) | `bench_har_wisdm__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.505 | 53.2% | 53.2% | 1.5175 | 42 | best.pt (16800) | `bench_har_wisdm__mae_recipe_r0__probe__s42` | bench |
| MAE recipe R1 (physical scaling + yaw rotation + lr 3e-4) | probe | 0.448 | 47.7% | 47.7% | 1.7327 | 42 | best.pt (20000) | `bench_har_wisdm__mae_recipe_r1__probe__s42` | bench |
| MAE recipe R2 (R1 + band-weighted loss (4/2/1/0.5)) | probe | 0.409 | 43.9% | 43.9% | 1.8175 | 42 | best.pt (19600) | `bench_har_wisdm__mae_recipe_r2__probe__s42` | bench |
| MAE recipe R3 (R2 + modality masking p=0.3 (replace-mode loss)) | probe | 0.378 | 42.4% | 42.4% | 1.8474 | 42 | best.pt (19800) | `bench_har_wisdm__mae_recipe_r3__probe__s42` | bench |
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.298 | 35.6% | 35.6% | 2.2026 | 42 | best.pt (18000) | `bench_har_wisdm__mae_recipe_rraw__probe__s42` | bench |
| MAE mask-ratio 0.30 | probe | 0.363 | 39.5% | 39.5% | 1.8971 | 42 | best.pt (18000) | `bench_har_wisdm__mae_mr30__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 0.511 | 53.7% | 53.7% | 1.4652 | 42 | best.pt (17200) | `bench_har_wisdm__mae_mr40__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.516 | 54.2% | 54.2% | 1.5063 | 42 | best.pt (18000) | `bench_har_wisdm__mae_div60_dc__probe__s42` | bench |
| MAE mask-ratio 0.50 | probe | 0.510 | 53.8% | 53.8% | 1.4560 | 42 | best.pt (17200) | `bench_har_wisdm__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.659 | 67.4% | 67.4% | 1.0693 | 42 | best.pt (16600) | `bench_har_wisdm__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.769 | 77.1% | 77.1% | 0.9109 | 42 | best.pt (12200) | `bench_har_wisdm__mae_4src_z__finetune__s42` | bench |
| MAE mask-ratio 0.60 | probe | 0.500 | 53.0% | 53.0% | 1.4792 | 42 | best.pt (17200) | `bench_har_wisdm__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.464 | 50.2% | 50.2% | 1.6127 | 42 | best.pt (6600) | `bench_har_wisdm__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.730 | 73.5% | 73.5% | 0.9765 | 42 | best.pt (13600) | `bench_har_wisdm__mae_4src_dc__finetune__s42` | bench |
| MAE mask-ratio 0.85 | probe | 0.473 | 50.4% | 50.4% | 1.6232 | 42 | best.pt (18400) | `bench_har_wisdm__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.514 | 55.3% | 55.3% | 1.4134 | 42 | best.pt (4600) | `bench_har_wisdm__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.739 | 74.4% | 74.4% | 1.0239 | 42 | best.pt (14400) | `bench_har_wisdm__mae_4src_phys__finetune__s42` | bench |
| MAE mask-ratio 0.90 | probe | 0.460 | 49.3% | 49.3% | 1.6134 | 42 | best.pt (18400) | `bench_har_wisdm__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.543 | 58.1% | 58.1% | 1.3336 | 42 | best.pt (4600) | `bench_har_wisdm__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.740 | 74.2% | 74.2% | 0.9059 | 42 | best.pt (12200) | `bench_har_wisdm__mae_4src_physnp__finetune__s42` | bench |
| MAE frame-canonical (no rotation) | probe | 0.565 | 58.6% | 58.6% | 1.3392 | 42 | best.pt (17200) | `bench_har_wisdm__mae_frame_canon__probe__s42` | bench |
| SimMTM (4-source pool) | probe | 0.607 | 62.8% | 62.8% | 1.1949 | 42 | best.pt (18400) | `bench_har_wisdm__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.766 | 77.0% | 77.0% | 0.7984 | 42 | best.pt (12200) | `bench_har_wisdm__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.508 | 53.8% | 53.8% | 1.5397 | 42 | best.pt (15600) | `bench_har_wisdm__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.723 | 73.6% | 73.6% | 0.8164 | 42 | best.pt (8000) | `bench_har_wisdm__focal_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.587 | 61.8% | 61.8% | 1.2498 | 42 | best.pt (2200) | `bench_har_wisdm__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.748 | 74.9% | 74.9% | 0.9758 | 42 | best.pt (12200) | `bench_har_wisdm__acconly_4src__finetune__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.700 | 70.5% | 70.5% | 0.9571 | — | best.pt | `deepconvlstm_har_wisdm` | legacy† |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.614 | 63.7% | 63.7% | 1.3264 | 42 | best.pt (18600) | `bench_har_wisdm__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.743 | 74.3% | 74.3% | 0.7941 | 42 | best.pt (9000) | `bench_har_wisdm__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.681 | 70.0% | 70.0% | 0.9480 | 42 | best.pt (15800) | `bench_har_wisdm__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.725 | 73.6% | 73.6% | 0.8195 | 42 | best.pt (6800) | `bench_har_wisdm__inertia1_art__finetune__s42` | bench |

## HAR · UTD-MHAD — `har_utd_mhad`

_UTD-MHAD activity recognition from one wrist/arm IMU (21 classes)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.633 | 63.8% | 63.8% | 1.1398 | 42 | best.pt (400) | `vit1d_har_probe_utd_mhad_mae_multidata` | legacy† |
| Ours (MAE, 11-source) | finetune | 0.727 | 75.0% | 73.9% | 1.0218 | 42 | best.pt (200) | `vit1d_har_finetune_utd_mhad_mae_multidata` | legacy† |
| Ours (MAE, 5-source) | probe | 0.665 | 68.4% | 68.6% | 1.2662 | 42 | best.pt (400) | `vit1d_har_probe_utd_mhad_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, 5-source) | finetune | 0.669 | 70.0% | 70.0% | 1.1203 | 42 | best.pt (200) | `vit1d_har_finetune_utd_mhad_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.704 | 70.1% | 70.0% | 1.0314 | 42 | best.pt (1000) | `vit1d_har_probe_utd_mhad_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.655 | 66.2% | 64.7% | 1.6439 | 42 | best.pt (200) | `bench_har_utd_mhad__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.579 | 59.2% | 60.4% | 1.3574 | 42 | best.pt (600) | `vit1d_har_probe_utd_mhad_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.662 | 66.9% | 67.1% | 1.1260 | 42 | best.pt (200) | `bench_har_utd_mhad__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.504 | 52.1% | 53.6% | 1.4172 | 42 | best.pt (200) | `vit1d_har_probe_utd_mhad_focal_multidata` | legacy† |
| FOCAL | finetune | 0.658 | 66.9% | 66.7% | 1.2474 | 42 | best.pt (200) | `bench_har_utd_mhad__focal__finetune__s42` | bench |
| SimMTM | probe | 0.626 | 64.8% | 64.7% | 1.1063 | 42 | best.pt (400) | `vit1d_har_probe_utd_mhad_simmtm_multidata` | legacy† |
| SimMTM | finetune | 0.739 | 74.2% | 73.9% | 1.1495 | 42 | best.pt (200) | `bench_har_utd_mhad__simmtm__finetune__s42` | bench |
| RelCon | probe | 0.642 | 65.8% | 63.8% | 1.1877 | 42 | best.pt (17600) | `vit1d_har_probe_utd_mhad_relcon_multidata` | legacy† |
| RelCon | finetune | 0.772 | 77.5% | 76.8% | 0.7699 | 42 | best.pt (200) | `bench_har_utd_mhad__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.701 | 71.4% | 71.0% | 1.4460 | 42 | best.pt (400) | `vit1d_har_supervised_utd_mhad_altattn_scratch` | legacy† |
| Inertia-1 PatchTST-small arch, from scratch (no pretraining) | scratch | 0.603 | 64.1% | 64.3% | 1.5564 | 42 | best.pt (200) | `bench_har_utd_mhad__inertia1_patchtst_scratch__scratch__s42` | bench |
| Inertia-1 PatchTST-small arch, random init, frozen (probe control) | probe | 0.460 | 48.0% | 49.8% | 1.7556 | 42 | best.pt (400) | `bench_har_utd_mhad__inertia1_patchtst_randinit__probe__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.551 | 57.9% | 58.5% | 1.3972 | 42 | best.pt (1200) | `bench_har_utd_mhad__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.589 | 60.1% | 60.9% | 1.3610 | 42 | best.pt (1400) | `bench_har_utd_mhad__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.702 | 69.8% | 71.0% | 1.0153 | 42 | best.pt (1000) | `bench_har_utd_mhad__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.650 | 66.9% | 67.1% | 1.0336 | 42 | best.pt (1000) | `bench_har_utd_mhad__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.534 | 55.5% | 54.6% | 1.5714 | 42 | best.pt (1200) | `bench_har_utd_mhad__mae_recipe_r0__probe__s42` | bench |
| MAE recipe R1 (physical scaling + yaw rotation + lr 3e-4) | probe | 0.414 | 42.6% | 43.5% | 1.8963 | 42 | best.pt (8000) | `bench_har_utd_mhad__mae_recipe_r1__probe__s42` | bench |
| MAE recipe R2 (R1 + band-weighted loss (4/2/1/0.5)) | probe | 0.467 | 45.5% | 47.8% | 1.6099 | 42 | best.pt (14800) | `bench_har_utd_mhad__mae_recipe_r2__probe__s42` | bench |
| MAE recipe R3 (R2 + modality masking p=0.3 (replace-mode loss)) | probe | 0.408 | 41.4% | 42.0% | 1.9233 | 42 | best.pt (17400) | `bench_har_utd_mhad__mae_recipe_r3__probe__s42` | bench |
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.355 | 39.3% | 41.1% | 2.1841 | 42 | best.pt (17000) | `bench_har_utd_mhad__mae_recipe_rraw__probe__s42` | bench |
| MAE mask-ratio 0.30 | probe | 0.439 | 46.6% | 46.4% | 1.9527 | 42 | best.pt (2200) | `bench_har_utd_mhad__mae_mr30__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 0.599 | 60.9% | 59.4% | 1.4109 | 42 | best.pt (1000) | `bench_har_utd_mhad__mae_mr40__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.476 | 48.1% | 48.8% | 1.7370 | 42 | best.pt (2600) | `bench_har_utd_mhad__mae_div60_dc__probe__s42` | bench |
| MAE mask-ratio 0.50 | probe | 0.562 | 57.6% | 58.0% | 1.4023 | 42 | best.pt (1000) | `bench_har_utd_mhad__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.748 | 75.6% | 73.4% | 1.8358 | 42 | best.pt (4600) | `bench_har_utd_mhad__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.858 | 86.5% | 85.5% | 0.9013 | 42 | best.pt (11400) | `bench_har_utd_mhad__mae_4src_z__finetune__s42` | bench |
| MAE mask-ratio 0.60 | probe | 0.627 | 64.4% | 63.8% | 1.3653 | 42 | best.pt (1200) | `bench_har_utd_mhad__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.558 | 56.8% | 56.0% | 1.3510 | 42 | best.pt (6800) | `bench_har_utd_mhad__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.676 | 67.9% | 66.7% | 3.0957 | 42 | best.pt (1000) | `bench_har_utd_mhad__mae_4src_dc__finetune__s42` | bench |
| MAE mask-ratio 0.85 | probe | 0.557 | 56.9% | 57.0% | 1.5102 | 42 | best.pt (2200) | `bench_har_utd_mhad__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.667 | 67.3% | 67.1% | 1.4231 | 42 | best.pt (4600) | `bench_har_utd_mhad__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.772 | 78.6% | 77.8% | 1.3823 | 42 | best.pt (5400) | `bench_har_utd_mhad__mae_4src_phys__finetune__s42` | bench |
| MAE mask-ratio 0.90 | probe | 0.496 | 51.9% | 52.2% | 1.7387 | 42 | best.pt (1200) | `bench_har_utd_mhad__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.610 | 63.3% | 63.3% | 1.8274 | 42 | best.pt (10000) | `bench_har_utd_mhad__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.776 | 78.7% | 79.2% | 1.0983 | 42 | best.pt (600) | `bench_har_utd_mhad__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.702 | 70.8% | 69.1% | 0.9569 | 42 | best.pt (600) | `bench_har_utd_mhad__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.734 | 74.1% | 73.4% | 2.2114 | 42 | best.pt (4600) | `bench_har_utd_mhad__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.525 | 55.5% | 53.6% | 4.0959 | 42 | best.pt (3200) | `bench_har_utd_mhad__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.642 | 66.5% | 66.2% | 1.2707 | 42 | best.pt (200) | `bench_har_utd_mhad__focal_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.631 | 63.2% | 65.7% | 1.0887 | 42 | best.pt (800) | `bench_har_utd_mhad__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.820 | 82.3% | 81.6% | 1.0876 | 42 | best.pt (2200) | `bench_har_utd_mhad__acconly_4src__finetune__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.518 | 55.0% | 56.0% | 1.5120 | — | best.pt | `deepconvlstm_har_utd_mhad` | legacy† |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.678 | 68.0% | 69.1% | 0.9000 | 42 | best.pt (5400) | `bench_har_utd_mhad__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.756 | 76.0% | 76.8% | 0.8141 | 42 | best.pt (400) | `bench_har_utd_mhad__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.717 | 72.1% | 72.9% | 0.8977 | 42 | best.pt (600) | `bench_har_utd_mhad__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.758 | 77.0% | 76.3% | 0.8563 | 42 | best.pt (200) | `bench_har_utd_mhad__inertia1_art__finetune__s42` | bench |

## HAR · PAMAP2 — `har_pamap2`

_PAMAP2 activity recognition from one wrist/arm IMU (18 classes)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.542 | 52.9% | 55.6% | 1.3220 | 42 | best.pt (2400) | `vit1d_har_probe_pamap2_mae_multidata` | legacy† |
| Ours (MAE, 11-source) | finetune | 0.413 | 45.1% | 51.3% | 1.7841 | 42 | best.pt (400) | `vit1d_har_finetune_pamap2_mae_multidata` | legacy† |
| Ours (MAE, 5-source) | probe | 0.561 | 55.2% | 56.5% | 1.3444 | 42 | best.pt (2400) | `vit1d_har_probe_pamap2_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, 5-source) | finetune | 0.410 | 44.9% | 50.7% | 1.7899 | 42 | best.pt (400) | `vit1d_har_finetune_pamap2_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.482 | 48.1% | 53.5% | 1.3499 | 42 | best.pt (2400) | `vit1d_har_probe_pamap2_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.387 | 42.8% | 49.4% | 1.6897 | 42 | best.pt (400) | `bench_har_pamap2__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.462 | 46.0% | 47.2% | 1.5436 | 42 | best.pt (2400) | `vit1d_har_probe_pamap2_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.379 | 44.1% | 44.9% | 1.9068 | 42 | best.pt (400) | `bench_har_pamap2__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.269 | 32.4% | 34.3% | 1.9593 | 42 | best.pt (10800) | `vit1d_har_probe_pamap2_focal_multidata` | legacy† |
| FOCAL | finetune | 0.291 | 32.0% | 35.6% | 2.0901 | 42 | best.pt (800) | `bench_har_pamap2__focal__finetune__s42` | bench |
| SimMTM | probe | 0.410 | 42.8% | 47.1% | 1.6274 | 42 | best.pt (10400) | `vit1d_har_probe_pamap2_simmtm_multidata` | legacy† |
| SimMTM | finetune | 0.386 | 44.8% | 49.5% | 1.6421 | 42 | best.pt (800) | `bench_har_pamap2__simmtm__finetune__s42` | bench |
| RelCon | probe | 0.129 | 16.5% | 22.0% | 2.6041 | 42 | best.pt (10400) | `vit1d_har_probe_pamap2_relcon_multidata` | legacy† |
| RelCon | finetune | 0.169 | 23.0% | 27.5% | 2.9159 | 42 | best.pt (400) | `bench_har_pamap2__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.219 | 26.0% | 28.5% | 2.5208 | 42 | best.pt (800) | `vit1d_har_supervised_pamap2_altattn_scratch` | legacy† |
| Inertia-1 PatchTST-small arch, from scratch (no pretraining) | scratch | 0.265 | 31.7% | 40.1% | 2.5043 | 42 | best.pt (800) | `bench_har_pamap2__inertia1_patchtst_scratch__scratch__s42` | bench |
| Inertia-1 PatchTST-small arch, random init, frozen (probe control) | probe | 0.195 | 27.7% | 32.7% | 2.9636 | 42 | best.pt (1200) | `bench_har_pamap2__inertia1_patchtst_randinit__probe__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.441 | 46.6% | 49.6% | 1.5085 | 42 | best.pt (10400) | `bench_har_pamap2__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.436 | 45.6% | 49.2% | 1.5252 | 42 | best.pt (10400) | `bench_har_pamap2__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.470 | 52.0% | 60.3% | 1.4107 | 42 | best.pt (10800) | `bench_har_pamap2__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.397 | 44.4% | 51.3% | 1.7053 | 42 | best.pt (10800) | `bench_har_pamap2__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.391 | 42.0% | 46.5% | 1.5922 | 42 | best.pt (10400) | `bench_har_pamap2__mae_recipe_r0__probe__s42` | bench |
| MAE recipe R1 (physical scaling + yaw rotation + lr 3e-4) | probe | 0.329 | 37.3% | 42.9% | 2.0411 | 42 | best.pt (10800) | `bench_har_pamap2__mae_recipe_r1__probe__s42` | bench |
| MAE recipe R2 (R1 + band-weighted loss (4/2/1/0.5)) | probe | 0.335 | 39.4% | 46.4% | 2.1167 | 42 | best.pt (16800) | `bench_har_pamap2__mae_recipe_r2__probe__s42` | bench |
| MAE recipe R3 (R2 + modality masking p=0.3 (replace-mode loss)) | probe | 0.365 | 40.3% | 45.3% | 1.9659 | 42 | best.pt (10800) | `bench_har_pamap2__mae_recipe_r3__probe__s42` | bench |
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.194 | 23.6% | 29.0% | 2.1440 | 42 | best.pt (10800) | `bench_har_pamap2__mae_recipe_rraw__probe__s42` | bench |
| MAE mask-ratio 0.30 | probe | 0.288 | 32.6% | 35.1% | 1.8958 | 42 | best.pt (10400) | `bench_har_pamap2__mae_mr30__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 0.438 | 46.1% | 48.9% | 1.5456 | 42 | best.pt (10400) | `bench_har_pamap2__mae_mr40__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.342 | 42.5% | 48.3% | 1.6876 | 42 | best.pt (13200) | `bench_har_pamap2__mae_div60_dc__probe__s42` | bench |
| MAE mask-ratio 0.50 | probe | 0.409 | 42.4% | 47.2% | 1.5899 | 42 | best.pt (5200) | `bench_har_pamap2__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.439 | 51.0% | 56.3% | 1.9790 | 42 | best.pt (4800) | `bench_har_pamap2__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.433 | 47.9% | 53.2% | 2.6371 | 42 | best.pt (2800) | `bench_har_pamap2__mae_4src_z__finetune__s42` | bench |
| MAE mask-ratio 0.60 | probe | 0.430 | 45.6% | 49.0% | 1.5457 | 42 | best.pt (10400) | `bench_har_pamap2__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.421 | 47.3% | 52.6% | 1.8234 | 42 | best.pt (10400) | `bench_har_pamap2__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.441 | 46.3% | 49.0% | 1.9369 | 42 | best.pt (800) | `bench_har_pamap2__mae_4src_dc__finetune__s42` | bench |
| MAE mask-ratio 0.85 | probe | 0.387 | 41.4% | 44.3% | 1.6447 | 42 | best.pt (10400) | `bench_har_pamap2__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.459 | 51.1% | 51.1% | 2.0174 | 42 | best.pt (4800) | `bench_har_pamap2__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.378 | 43.6% | 48.5% | 2.5523 | 42 | best.pt (1200) | `bench_har_pamap2__mae_4src_phys__finetune__s42` | bench |
| MAE mask-ratio 0.90 | probe | 0.369 | 40.8% | 44.1% | 1.6504 | 42 | best.pt (10400) | `bench_har_pamap2__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.342 | 44.9% | 46.1% | 2.5862 | 42 | best.pt (9600) | `bench_har_pamap2__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.405 | 43.7% | 47.3% | 2.7707 | 42 | best.pt (1200) | `bench_har_pamap2__mae_4src_physnp__finetune__s42` | bench |
| MAE frame-canonical (no rotation) | probe | 0.382 | 41.1% | 44.7% | 1.6885 | 42 | best.pt (10400) | `bench_har_pamap2__mae_frame_canon__probe__s42` | bench |
| SimMTM (4-source pool) | probe | 0.442 | 52.3% | 54.3% | 1.8015 | 42 | best.pt (5600) | `bench_har_pamap2__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.486 | 53.5% | 52.5% | 1.9051 | 42 | best.pt (1200) | `bench_har_pamap2__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.299 | 40.7% | 42.1% | 2.2848 | 42 | best.pt (4000) | `bench_har_pamap2__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.416 | 48.5% | 50.4% | 1.8022 | 42 | best.pt (1200) | `bench_har_pamap2__focal_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.464 | 50.3% | 49.7% | 2.2213 | 42 | best.pt (1200) | `bench_har_pamap2__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.460 | 47.9% | 51.8% | 1.8836 | 42 | best.pt (800) | `bench_har_pamap2__acconly_4src__finetune__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.292 | 35.0% | 37.0% | 1.8867 | — | best.pt | `deepconvlstm_har_pamap2` | legacy† |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.437 | 47.5% | 55.0% | 1.5566 | 42 | best.pt (21200) | `bench_har_pamap2__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.431 | 47.0% | 50.2% | 1.6222 | 42 | best.pt (800) | `bench_har_pamap2__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.444 | 50.0% | 57.8% | 1.4458 | 42 | best.pt (2400) | `bench_har_pamap2__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.453 | 47.2% | 52.7% | 1.6667 | 42 | best.pt (400) | `bench_har_pamap2__inertia1_art__finetune__s42` | bench |

## HAR · RecoFit — `har_recofit`

_RecoFit activity recognition from one wrist/arm IMU (13 classes)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.630 | 61.9% | 78.9% | 0.7050 | 42 | best.pt (27600) | `vit1d_har_probe_recofit_mae_multidata` | legacy† |
| Ours (MAE, 11-source) | finetune | 0.768 | 77.5% | 88.5% | 0.4118 | 42 | best.pt (2400) | `vit1d_har_finetune_recofit_mae_multidata` | legacy† |
| Ours (MAE, 5-source) | probe | 0.657 | 64.7% | 79.7% | 0.6771 | 42 | best.pt (28800) | `vit1d_har_probe_recofit_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, 5-source) | finetune | 0.756 | 76.4% | 87.9% | 0.4521 | 42 | best.pt (2400) | `vit1d_har_finetune_recofit_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.649 | 63.7% | 80.0% | 0.6862 | 42 | best.pt (27600) | `vit1d_har_probe_recofit_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.746 | 75.2% | 87.8% | 0.4213 | 42 | best.pt (2400) | `bench_har_recofit__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.569 | 57.2% | 70.0% | 0.9517 | 42 | best.pt (8400) | `vit1d_har_probe_recofit_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.680 | 69.1% | 82.0% | 0.5895 | 42 | best.pt (2400) | `bench_har_recofit__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.437 | 44.9% | 55.7% | 1.3341 | 42 | best.pt (8400) | `vit1d_har_probe_recofit_focal_multidata` | legacy† |
| FOCAL | finetune | 0.660 | 66.3% | 81.9% | 0.5958 | 42 | best.pt (8400) | `bench_har_recofit__focal__finetune__s42` | bench |
| SimMTM | probe | 0.541 | 54.1% | 69.5% | 0.9841 | 42 | best.pt (33200) | `vit1d_har_probe_recofit_simmtm_multidata` | legacy† |
| SimMTM | finetune | 0.715 | 71.1% | 83.1% | 0.5598 | 42 | best.pt (5600) | `bench_har_recofit__simmtm__finetune__s42` | bench |
| RelCon | probe | 0.343 | 32.6% | 49.6% | 1.5457 | 42 | best.pt (34400) | `vit1d_har_probe_recofit_relcon_multidata` | legacy† |
| RelCon | finetune | 0.690 | 68.9% | 84.8% | 0.5362 | 42 | best.pt (3600) | `bench_har_recofit__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.663 | 66.6% | 81.0% | 0.6663 | 42 | best.pt (7200) | `vit1d_har_supervised_recofit_altattn_scratch` | legacy† |
| Inertia-1 PatchTST-small arch, from scratch (no pretraining) | scratch | 0.656 | 65.0% | 80.3% | 0.6620 | 42 | best.pt (15200) | `bench_har_recofit__inertia1_patchtst_scratch__scratch__s42` | bench |
| Inertia-1 PatchTST-small arch, random init, frozen (probe control) | probe | 0.435 | 44.6% | 60.1% | 1.2878 | 42 | best.pt (32800) | `bench_har_recofit__inertia1_patchtst_randinit__probe__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.542 | 54.7% | 69.5% | 0.9819 | 42 | best.pt (34400) | `bench_har_recofit__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.547 | 54.7% | 71.1% | 0.9667 | 42 | best.pt (27600) | `bench_har_recofit__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.625 | 62.0% | 76.0% | 0.8171 | 42 | best.pt (32400) | `bench_har_recofit__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.627 | 62.4% | 75.5% | 0.8380 | 42 | best.pt (36000) | `bench_har_recofit__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.513 | 51.8% | 66.6% | 1.0618 | 42 | best.pt (34400) | `bench_har_recofit__mae_recipe_r0__probe__s42` | bench |
| MAE recipe R1 (physical scaling + yaw rotation + lr 3e-4) | probe | 0.392 | 41.4% | 58.3% | 1.3068 | 42 | best.pt (26400) | `bench_har_recofit__mae_recipe_r1__probe__s42` | bench |
| MAE recipe R2 (R1 + band-weighted loss (4/2/1/0.5)) | probe | 0.426 | 43.7% | 59.1% | 1.2593 | 42 | best.pt (34400) | `bench_har_recofit__mae_recipe_r2__probe__s42` | bench |
| MAE recipe R3 (R2 + modality masking p=0.3 (replace-mode loss)) | probe | 0.366 | 38.2% | 54.3% | 1.4043 | 42 | best.pt (26400) | `bench_har_recofit__mae_recipe_r3__probe__s42` | bench |
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.329 | 32.3% | 46.4% | 1.6871 | 42 | best.pt (26400) | `bench_har_recofit__mae_recipe_rraw__probe__s42` | bench |
| MAE mask-ratio 0.30 | probe | 0.405 | 41.1% | 56.0% | 1.2828 | 42 | best.pt (32800) | `bench_har_recofit__mae_mr30__probe__s42` | bench |
| MAE mask-ratio 0.40 | probe | 0.554 | 55.8% | 71.1% | 0.9576 | 42 | best.pt (35600) | `bench_har_recofit__mae_mr40__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.509 | 51.5% | 65.6% | 1.1392 | 42 | best.pt (34400) | `bench_har_recofit__mae_div60_dc__probe__s42` | bench |
| MAE mask-ratio 0.50 | probe | 0.545 | 54.7% | 70.4% | 0.9468 | 42 | best.pt (27600) | `bench_har_recofit__mae_mr50__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.647 | 71.0% | 75.9% | 0.8767 | 42 | best.pt (24400) | `bench_har_recofit__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.739 | 79.0% | 87.2% | 0.8204 | 42 | best.pt (11600) | `bench_har_recofit__mae_4src_z__finetune__s42` | bench |
| MAE mask-ratio 0.60 | probe | 0.537 | 53.3% | 69.8% | 0.9770 | 42 | best.pt (27600) | `bench_har_recofit__mae_mr60__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.485 | 53.8% | 61.6% | 1.4017 | 42 | best.pt (21200) | `bench_har_recofit__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.717 | 75.3% | 82.0% | 0.9509 | 42 | best.pt (14000) | `bench_har_recofit__mae_4src_dc__finetune__s42` | bench |
| MAE mask-ratio 0.85 | probe | 0.506 | 51.5% | 66.2% | 1.0755 | 42 | best.pt (34400) | `bench_har_recofit__mae_mr85__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.558 | 62.4% | 67.2% | 1.1614 | 42 | best.pt (7600) | `bench_har_recofit__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.718 | 76.2% | 85.0% | 0.7639 | 42 | best.pt (11600) | `bench_har_recofit__mae_4src_phys__finetune__s42` | bench |
| MAE mask-ratio 0.90 | probe | 0.471 | 46.8% | 63.6% | 1.1402 | 42 | best.pt (27200) | `bench_har_recofit__mae_mr90__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.607 | 67.3% | 72.0% | 0.9708 | 42 | best.pt (28800) | `bench_har_recofit__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.706 | 75.5% | 80.2% | 0.8496 | 42 | best.pt (11200) | `bench_har_recofit__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.539 | 62.3% | 64.6% | 1.1667 | 42 | best.pt (6000) | `bench_har_recofit__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.756 | 80.2% | 87.0% | 0.6143 | 42 | best.pt (11200) | `bench_har_recofit__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.544 | 59.7% | 65.1% | 1.2877 | 42 | best.pt (34800) | `bench_har_recofit__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.693 | 74.5% | 79.2% | 0.7919 | 42 | best.pt (14000) | `bench_har_recofit__focal_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.635 | 69.5% | 74.5% | 0.9195 | 42 | best.pt (17600) | `bench_har_recofit__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.727 | 76.1% | 85.2% | 0.9029 | 42 | best.pt (14000) | `bench_har_recofit__acconly_4src__finetune__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.721 | 72.1% | 86.0% | 0.4900 | — | best.pt | `deepconvlstm_har_recofit` | legacy† |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.601 | 59.4% | 76.1% | 0.8842 | 42 | best.pt (39200) | `bench_har_recofit__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.666 | 67.7% | 83.4% | 0.5246 | 42 | best.pt (4400) | `bench_har_recofit__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.648 | 64.7% | 80.8% | 0.6323 | 42 | best.pt (23600) | `bench_har_recofit__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.659 | 67.3% | 82.2% | 0.5416 | 42 | best.pt (4400) | `bench_har_recofit__inertia1_art__finetune__s42` | bench |

## HAR · HHAR — `har_hhar`

_HHAR activity recognition from one wrist/arm IMU (6 classes)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.679 | 68.2% | 67.0% | 0.9045 | 42 | best.pt (3200) | `vit1d_har_probe_hhar_mae_multidata` | legacy† |
| Ours (MAE, 11-source) | finetune | 0.628 | 63.5% | 61.7% | 1.1546 | 42 | best.pt (200) | `vit1d_har_finetune_hhar_mae_multidata` | legacy† |
| Ours (MAE, 5-source) | probe | 0.614 | 62.2% | 61.8% | 0.9704 | 42 | best.pt (200) | `vit1d_har_probe_hhar_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, 5-source) | finetune | 0.614 | 62.4% | 61.7% | 1.1349 | 42 | best.pt (200) | `vit1d_har_finetune_hhar_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.661 | 66.1% | 65.5% | 0.9308 | 42 | best.pt (7400) | `vit1d_har_probe_hhar_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.599 | 60.6% | 59.6% | 1.1973 | 42 | best.pt (200) | `bench_har_hhar__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.600 | 60.7% | 61.4% | 0.9892 | 42 | best.pt (1400) | `vit1d_har_probe_hhar_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.569 | 58.0% | 58.0% | 1.1880 | 42 | best.pt (200) | `bench_har_hhar__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.500 | 51.5% | 51.1% | 1.2454 | 42 | best.pt (11000) | `vit1d_har_probe_hhar_focal_multidata` | legacy† |
| FOCAL | finetune | 0.588 | 58.3% | 56.8% | 1.1406 | 42 | best.pt (600) | `bench_har_hhar__focal__finetune__s42` | bench |
| SimMTM | probe | 0.573 | 57.7% | 56.5% | 1.1478 | 42 | best.pt (7800) | `vit1d_har_probe_hhar_simmtm_multidata` | legacy† |
| SimMTM | finetune | 0.479 | 52.5% | 51.0% | 1.1589 | 42 | best.pt (200) | `bench_har_hhar__simmtm__finetune__s42` | bench |
| RelCon | probe | 0.283 | 33.2% | 35.4% | 1.6737 | 42 | best.pt (13600) | `vit1d_har_probe_hhar_relcon_multidata` | legacy† |
| RelCon | finetune | 0.454 | 47.2% | 46.7% | 1.2264 | 42 | best.pt (200) | `bench_har_hhar__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.504 | 51.0% | 50.1% | 1.2415 | 42 | best.pt (800) | `vit1d_har_supervised_hhar_altattn_scratch` | legacy† |
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
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.663 | 67.3% | 64.0% | 1.0825 | 42 | best.pt (3600) | `bench_har_hhar__mae_4src_dc__probe__s42` | bench |
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
| MAE acc-only (4-source pool) | probe | 0.704 | 70.6% | 68.3% | 0.9448 | 42 | best.pt (6000) | `bench_har_hhar__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.700 | 70.6% | 69.9% | 1.2029 | 42 | best.pt (1000) | `bench_har_hhar__acconly_4src__finetune__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.448 | 46.6% | 44.8% | 1.2853 | — | best.pt | `deepconvlstm_har_hhar` | legacy† |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.713 | 73.1% | 69.7% | 0.8740 | 42 | best.pt (15400) | `bench_har_hhar__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.785 | 78.5% | 75.8% | 0.6651 | 42 | best.pt (800) | `bench_har_hhar__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.717 | 71.9% | 68.8% | 0.8406 | 42 | best.pt (7000) | `bench_har_hhar__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.670 | 67.6% | 64.9% | 0.8168 | 42 | best.pt (200) | `bench_har_hhar__inertia1_art__finetune__s42` | bench |

## HAR · MHEALTH — `har_mhealth`

_MHEALTH activity recognition from one wrist/arm IMU (12 classes)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.689 | 69.9% | 68.4% | 0.5826 | 42 | best.pt (1400) | `vit1d_har_probe_mhealth_mae_multidata` | legacy† |
| Ours (MAE, 11-source) | finetune | 0.627 | 66.7% | 65.6% | 1.0757 | 42 | best.pt (200) | `vit1d_har_finetune_mhealth_mae_multidata` | legacy† |
| Ours (MAE, 5-source) | probe | 0.693 | 70.6% | 68.8% | 0.6027 | 42 | best.pt (600) | `vit1d_har_probe_mhealth_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, 5-source) | finetune | 0.683 | 70.4% | 68.8% | 0.8415 | 42 | best.pt (200) | `vit1d_har_finetune_mhealth_mae_realdata_ocdetect_eatingspeed_motionx_sonic` | legacy† |
| Ours (MAE, no alt-attn) | probe | 0.699 | 70.5% | 68.9% | 0.5657 | 42 | best.pt (2200) | `vit1d_har_probe_mhealth_mae_noaltattn_multidata` | legacy† |
| Ours (MAE, no alt-attn) | finetune | 0.726 | 73.3% | 72.3% | 0.9625 | 42 | best.pt (400) | `bench_har_mhealth__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.661 | 68.5% | 67.2% | 0.7463 | 42 | best.pt (800) | `vit1d_har_probe_mhealth_mae_acconly_multidata` | legacy† |
| Ours (MAE, acc-only) | finetune | 0.628 | 66.0% | 64.4% | 1.1452 | 42 | best.pt (200) | `bench_har_mhealth__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.644 | 65.5% | 65.1% | 0.8744 | 42 | best.pt (12800) | `vit1d_har_probe_mhealth_focal_multidata` | legacy† |
| FOCAL | finetune | 0.705 | 71.2% | 70.4% | 0.8267 | 42 | best.pt (600) | `bench_har_mhealth__focal__finetune__s42` | bench |
| SimMTM | probe | 0.605 | 62.2% | 62.5% | 0.9021 | 42 | best.pt (1400) | `vit1d_har_probe_mhealth_simmtm_multidata` | legacy† |
| SimMTM | finetune | 0.574 | 63.3% | 64.6% | 1.0800 | 42 | best.pt (200) | `bench_har_mhealth__simmtm__finetune__s42` | bench |
| RelCon | probe | 0.401 | 42.0% | 44.4% | 1.7410 | 42 | best.pt (12000) | `vit1d_har_probe_mhealth_relcon_multidata` | legacy† |
| RelCon | finetune | 0.602 | 65.1% | 65.1% | 0.8966 | 42 | best.pt (200) | `bench_har_mhealth__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.659 | 68.5% | 67.6% | 0.8376 | 42 | best.pt (600) | `vit1d_har_supervised_mhealth_altattn_scratch` | legacy† |
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
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.772 | 77.7% | 78.0% | 0.4858 | 42 | best.pt (7400) | `bench_har_mhealth__mae_4src_dc__probe__s42` | bench |
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
| MAE acc-only (4-source pool) | probe | 0.848 | 84.9% | 85.1% | 0.3949 | 42 | best.pt (4000) | `bench_har_mhealth__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.892 | 89.6% | 89.3% | 0.5978 | 42 | best.pt (1600) | `bench_har_mhealth__acconly_4src__finetune__s42` | bench |
| *DeepConvLSTM (Ordóñez & Roggen)* | external | 0.556 | 59.5% | 60.2% | 0.8779 | — | best.pt | `deepconvlstm_har_mhealth` | legacy† |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.893 | 89.5% | 89.3% | 0.3359 | 42 | best.pt (19200) | `bench_har_mhealth__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.896 | 89.7% | 89.6% | 0.3246 | 42 | best.pt (800) | `bench_har_mhealth__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.919 | 91.9% | 91.9% | 0.2088 | 42 | best.pt (2600) | `bench_har_mhealth__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.851 | 85.6% | 85.7% | 0.3519 | 42 | best.pt (200) | `bench_har_mhealth__inertia1_art__finetune__s42` | bench |

## HAR · SAMoSA — `har_samosa`

_SAMoSA activity recognition from one wrist IMU (27 raw labels, 20 subjects)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint, re-evaluated with pooled predictions (bench.score.har).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Ours (MAE, 11-source) | probe | 0.474 | 46.4% | 48.7% | 1.7905 | 42 | best.pt (9800) | `bench_har_samosa__mae_11src__probe__s42` | bench |
| Ours (MAE, 11-source) | finetune | 0.521 | 50.2% | 53.5% | 1.6368 | 42 | best.pt (400) | `bench_har_samosa__mae_11src__finetune__s42` | bench |
| Ours (MAE, 5-source) | probe | 0.467 | 46.3% | 47.5% | 1.8319 | 42 | best.pt (11400) | `bench_har_samosa__mae_5src__probe__s42` | bench |
| Ours (MAE, 5-source) | finetune | 0.521 | 51.5% | 53.4% | 1.6811 | 42 | best.pt (400) | `bench_har_samosa__mae_5src__finetune__s42` | bench |
| Ours (MAE, no alt-attn) | probe | 0.509 | 50.4% | 52.5% | 1.6388 | 42 | best.pt (12800) | `bench_har_samosa__mae_noaltattn__probe__s42` | bench |
| Ours (MAE, no alt-attn) | finetune | 0.521 | 51.1% | 54.2% | 1.5638 | 42 | best.pt (400) | `bench_har_samosa__mae_noaltattn__finetune__s42` | bench |
| Ours (MAE, acc-only) | probe | 0.418 | 41.7% | 43.9% | 1.9447 | 42 | best.pt (12800) | `bench_har_samosa__mae_acconly__probe__s42` | bench |
| Ours (MAE, acc-only) | finetune | 0.435 | 43.4% | 45.7% | 1.9087 | 42 | best.pt (400) | `bench_har_samosa__mae_acconly__finetune__s42` | bench |
| FOCAL | probe | 0.263 | 27.0% | 30.5% | 2.4267 | 42 | best.pt (15200) | `bench_har_samosa__focal__probe__s42` | bench |
| FOCAL | finetune | 0.380 | 37.9% | 40.5% | 2.0679 | 42 | best.pt (800) | `bench_har_samosa__focal__finetune__s42` | bench |
| SimMTM | probe | 0.410 | 40.6% | 42.9% | 1.9721 | 42 | best.pt (12800) | `bench_har_samosa__simmtm__probe__s42` | bench |
| SimMTM | finetune | 0.476 | 46.0% | 49.3% | 1.6982 | 42 | best.pt (400) | `bench_har_samosa__simmtm__finetune__s42` | bench |
| RelCon | probe | 0.260 | 24.2% | 29.7% | 2.4698 | 42 | best.pt (18400) | `bench_har_samosa__relcon__probe__s42` | bench |
| RelCon | finetune | 0.459 | 44.4% | 48.4% | 1.7237 | 42 | best.pt (400) | `bench_har_samosa__relcon__finetune__s42` | bench |
| Scratch (no pretraining) | scratch | 0.395 | 39.2% | 39.9% | 2.1485 | 42 | best.pt (800) | `bench_har_samosa__scratch__scratch__s42` | bench |
| MAE 2x2 arm A (400 h NHANES + z-score) | probe | 0.408 | 40.5% | 43.2% | 2.0030 | 42 | best.pt (12800) | `bench_har_samosa__mae_nhanes2x2_a__probe__s42` | bench |
| MAE 2x2 arm B (20k h NHANES + z-score) | probe | 0.408 | 40.5% | 43.0% | 1.9666 | 42 | best.pt (12800) | `bench_har_samosa__mae_nhanes2x2_b__probe__s42` | bench |
| MAE 2x2 arm C (400 h NHANES + physical scaling) | probe | 0.532 | 53.7% | 55.7% | 1.4269 | 42 | best.pt (14400) | `bench_har_samosa__mae_nhanes2x2_c__probe__s42` | bench |
| MAE 2x2 arm D (20k h NHANES + physical scaling) | probe | 0.531 | 52.2% | 55.4% | 1.4502 | 42 | best.pt (15200) | `bench_har_samosa__mae_nhanes2x2_d__probe__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.376 | 37.0% | 39.9% | 2.0559 | 42 | best.pt (19000) | `bench_har_samosa__mae_recipe_r0__probe__s42` | bench |
| MAE recipe R1 (physical scaling + yaw rotation + lr 3e-4) | probe | 0.321 | 33.4% | 38.4% | 2.1486 | 42 | best.pt (18400) | `bench_har_samosa__mae_recipe_r1__probe__s42` | bench |
| MAE recipe R2 (R1 + band-weighted loss (4/2/1/0.5)) | probe | 0.348 | 36.5% | 41.4% | 2.0089 | 42 | best.pt (18400) | `bench_har_samosa__mae_recipe_r2__probe__s42` | bench |
| MAE recipe R3 (R2 + modality masking p=0.3 (replace-mode loss)) | probe | 0.238 | 25.4% | 30.8% | 2.2926 | 42 | best.pt (18400) | `bench_har_samosa__mae_recipe_r3__probe__s42` | bench |
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.174 | 19.0% | 23.7% | 2.5400 | 42 | best.pt (18400) | `bench_har_samosa__mae_recipe_rraw__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.374 | 38.0% | 41.8% | 1.8977 | 42 | best.pt (19200) | `bench_har_samosa__mae_div60_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.559 | 59.3% | 56.8% | 1.2514 | 42 | best.pt (11200) | `bench_har_samosa__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.605 | 62.7% | 60.5% | 2.1101 | 42 | best.pt (3400) | `bench_har_samosa__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.385 | 43.8% | 41.0% | 1.8464 | 42 | best.pt (5800) | `bench_har_samosa__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.497 | 54.9% | 52.4% | 1.4996 | 42 | best.pt (400) | `bench_har_samosa__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.458 | 49.7% | 47.7% | 1.6364 | 42 | best.pt (15000) | `bench_har_samosa__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.561 | 58.5% | 57.8% | 1.7058 | 42 | best.pt (1200) | `bench_har_samosa__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.489 | 53.5% | 51.5% | 1.4727 | 42 | best.pt (16200) | `bench_har_samosa__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.545 | 56.3% | 56.0% | 2.8488 | 42 | best.pt (4600) | `bench_har_samosa__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.524 | 56.2% | 54.6% | 1.3735 | 42 | best.pt (12000) | `bench_har_samosa__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.607 | 61.6% | 60.8% | 2.5083 | 42 | best.pt (7800) | `bench_har_samosa__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.360 | 41.3% | 39.8% | 1.9654 | 42 | best.pt (15600) | `bench_har_samosa__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.505 | 51.8% | 52.1% | 2.3280 | 42 | best.pt (2000) | `bench_har_samosa__focal_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.555 | 58.9% | 56.5% | 1.4025 | 42 | best.pt (10400) | `bench_har_samosa__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.527 | 57.5% | 54.9% | 1.4966 | 42 | best.pt (600) | `bench_har_samosa__acconly_4src__finetune__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | probe | 0.534 | 53.1% | 57.2% | 1.4800 | 42 | best.pt (19400) | `bench_har_samosa__inertia1_patchtst__probe__s42` | bench |
| Inertia-1 PatchTST-small (NHANES repro) | finetune | 0.513 | 51.3% | 54.1% | 1.4688 | 42 | best.pt (1200) | `bench_har_samosa__inertia1_patchtst__finetune__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | probe | 0.558 | 56.0% | 58.2% | 1.3331 | 42 | best.pt (5200) | `bench_har_samosa__inertia1_art__probe__s42` | bench |
| Inertia-1 AR-Transformer-small (NHANES repro) | finetune | 0.526 | 53.6% | 55.9% | 1.3868 | 42 | best.pt (600) | `bench_har_samosa__inertia1_art__finetune__s42` | bench |

## HAR · UT-Watch — `har_ut_watch`

_UT-Watch activity recognition from one wrist IMU (24 raw labels, 20 subjects)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint, re-evaluated with pooled predictions (bench.score.har).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| MAE recipe Rraw (R0 + raw/DC-preserving input) | probe | 0.250 | 27.8% | 43.1% | 1.2261 | 42 | best.pt (14400) | `bench_har_ut_watch__mae_recipe_rraw__probe__s42` | bench |
| MAE diversity-pool (~850 h) 60% mask, DC-preserving | probe | 0.340 | 34.2% | 48.9% | 1.1599 | 42 | best.pt (13000) | `bench_har_ut_watch__mae_div60_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | probe | 0.485 | 52.7% | 53.0% | 1.1990 | 42 | best.pt (10800) | `bench_har_ut_watch__mae_4src_z__probe__s42` | bench |
| MAE 4-source (no HAR pool), per-window z-score | finetune | 0.419 | 46.7% | 45.7% | 1.3083 | 42 | best.pt (1000) | `bench_har_ut_watch__mae_4src_z__finetune__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | probe | 0.389 | 44.1% | 43.1% | 1.3744 | 42 | best.pt (2000) | `bench_har_ut_watch__mae_4src_dc__probe__s42` | bench |
| MAE 4-source (no HAR pool), raw/DC-preserving | finetune | 0.413 | 42.9% | 42.8% | 1.3862 | 42 | best.pt (1200) | `bench_har_ut_watch__mae_4src_dc__finetune__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | probe | 0.457 | 48.3% | 51.6% | 1.2695 | 42 | best.pt (11800) | `bench_har_ut_watch__mae_4src_phys__probe__s42` | bench |
| MAE 4-source (no HAR pool), physical rescale | finetune | 0.360 | 44.5% | 45.7% | 1.2858 | 42 | best.pt (400) | `bench_har_ut_watch__mae_4src_phys__finetune__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | probe | 0.414 | 47.2% | 44.6% | 1.3086 | 42 | best.pt (7600) | `bench_har_ut_watch__mae_4src_physnp__probe__s42` | bench |
| MAE 4-source, physical input + patch-normalised loss | finetune | 0.402 | 45.7% | 44.3% | 1.2999 | 42 | best.pt (800) | `bench_har_ut_watch__mae_4src_physnp__finetune__s42` | bench |
| SimMTM (4-source pool) | probe | 0.412 | 47.4% | 46.2% | 1.2660 | 42 | best.pt (3200) | `bench_har_ut_watch__simmtm_4src__probe__s42` | bench |
| SimMTM (4-source pool) | finetune | 0.416 | 47.4% | 44.0% | 1.3163 | 42 | best.pt (1000) | `bench_har_ut_watch__simmtm_4src__finetune__s42` | bench |
| FOCAL (4-source pool) | probe | 0.371 | 43.0% | 41.5% | 1.3532 | 42 | best.pt (1600) | `bench_har_ut_watch__focal_4src__probe__s42` | bench |
| FOCAL (4-source pool) | finetune | 0.458 | 47.7% | 49.0% | 1.3469 | 42 | best.pt (1600) | `bench_har_ut_watch__focal_4src__finetune__s42` | bench |
| MAE acc-only (4-source pool) | probe | 0.414 | 50.3% | 48.2% | 1.2593 | 42 | best.pt (3400) | `bench_har_ut_watch__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.419 | 44.5% | 47.8% | 2.0534 | 42 | best.pt (2000) | `bench_har_ut_watch__acconly_4src__finetune__s42` | bench |

## HAR · PRISM — `har_prism`

_PrISM-Tracker (Arakawa et al.) — PRISM activity recognition from one wrist IMU (96 raw labels, 31 subjects)._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint, re-evaluated with pooled predictions (bench.score.har).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
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
| MAE acc-only (4-source pool) | probe | 0.453 | 52.8% | 63.3% | 1.0259 | 42 | best.pt (8200) | `bench_har_prism__acconly_4src__probe__s42` | bench |
| MAE acc-only (4-source pool) | finetune | 0.440 | 46.0% | 58.5% | 1.0692 | 42 | best.pt (400) | `bench_har_prism__acconly_4src__finetune__s42` | bench |

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

## HAR · WISDM · rotated frame — `har_wisdm_rot7`

_WISDM activity recognition from one wrist/arm IMU (18 classes) — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Scratch (no pretraining) | scratch | 0.628 | 62.8% | 62.8% | 1.1692 | 42 | best.pt (9000) | `bench_har_wisdm_rot7__scratch__scratch__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.495 | 52.0% | 52.0% | 1.5480 | 42 | best.pt (16800) | `bench_har_wisdm_rot7__mae_recipe_r0__probe__s42` | bench |

## HAR · PAMAP2 · rotated frame — `har_pamap2_rot7`

_PAMAP2 activity recognition from one wrist/arm IMU (18 classes) — identical task, re-expressed in ONE fixed random SO(3) frame (frame_perturb seed 7), applied to every split._ Scorer: `bench.score.har`. Selection: {'split': 'val', 'monitor': 'val_loss', 'mode': 'min', 'note': 'same convention as Experiments_table §3 (metrics read at the min-val-loss checkpoint)'}. Reported split: **val**.
Primary metric **macro-F1**: window-level macro-F1 on the val split at the min-val-loss checkpoint (trainer eval, imu_lm/objectives/supervised.py).

| Encoder | Protocol | **macro-F1** | bal-acc | acc | val loss | seeds | ckpt (step) | run | src |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Scratch (no pretraining) | scratch | 0.238 | 27.7% | 30.4% | 2.3437 | 42 | best.pt (800) | `bench_har_pamap2_rot7__scratch__scratch__s42` | bench |
| MAE recipe R0 (baseline recipe (z-score, octahedral rotation, full_mse, lr 1e-4)) | probe | 0.371 | 39.7% | 43.9% | 1.6440 | 42 | best.pt (10400) | `bench_har_pamap2_rot7__mae_recipe_r0__probe__s42` | bench |

## Caveats that travel with the numbers

* Pose / odometry / PD have **no held-out test split** in v1: the checkpoint is selected on val and val is reported (identical for every row, but optimistic in absolute terms). Microgesture and language report a genuine test split. A v2 split policy (carving test from the fixed val session lists) is the first protocol change to make before publishing absolute numbers.
* Full-body pose MPJPE is the trainer's own validation MPJPE at the selected checkpoint (min over the val curve under the bench selection rule) — a noise-mined statistic; multi-seed cells are what to trust.
* PADS: primary is **subject macro-F1** over 94 val subjects (14 HC / 58 PD / 22 DD); ±0.05 is noise. It replaced subject balanced accuracy on 2026-08-18 because bal-acc rewards degenerate classifiers on this split (a probe that never predicts PD scored 0.54 while classifying 31 % of subjects correctly); the `#classes predicted` column flags such cells. Every frozen probe on PADS is at/near chance — only the finetune column carries signal. Checkpoints are still selected on val window bal-acc (v1); selecting on macro-F1 is the next protocol change and needs re-runs.
* Language: fixed 8k-step budget, final checkpoint (`latest.pt`); §13e showed val-loss selection under-selects. Seed sd ≈ 0.7 MRR — single-seed differences < 1.5 pp are not real.
* External rows come from their own scripts (see `source` in the JSON) and are context, not competitors, under the encoder ranking.
* **HAR pretrain/eval overlap:** every ViT1D row pretrained on the `realdata` MAE source, which contains pamap2, recofit, utd_mhad and wisdm — 4 of the 6 HAR evaluation datasets (HHAR and MHEALTH are clean). External rows (Inertia-1, DeepConvLSTM) never saw them. Treat the HAR column as favourable to the ViT1D rows until the leakage-free re-pretrain (source rebuilt as `realdataclean` = samosa/shoaib/ut_watch, 2026-08-18) lands.
* HAR enters the main tables as ONE aggregate column (mean window macro-F1 over WISDM, UTD-MHAD, PAMAP2, RecoFit, HHAR, MHEALTH; present only when all six are scored). Selection is min val loss and val is reported (no test split); scores are re-evaluated from the checkpoint with pooled predictions, not read from training logs.
