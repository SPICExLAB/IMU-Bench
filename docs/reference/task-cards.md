# Task cards

`bench/registry/tasks.yaml` holds one card per task: dataset and split rule, input window, output and decoder,
optimisation recipe per protocol, selection rule, metrics with their direction, scorer. `imu-bench tasks` prints the
summary. The ten tasks of the benchmark:

| Task id | Dataset | Input window | Output | Primary metric | Split |
|---|---|---|---|---|---|
| `arm_pose_nymeria` | Nymeria, real right-wrist IMU + mocap | 10.24 s, 50 Hz | 5 right-arm joint rotations per 0.32 s | position error, cm ↓ | recording session |
| `full_body_pose_nymeria` | Nymeria | 10.24 s | 23 SMPL joint rotations per 0.32 s | root-relative MPJPE, mm ↓ | recording session |
| `odometry_nymeria_local` | Nymeria | 2.56 s | pelvis velocity in the wrist frame, integrated | ATE, m ↓ | recording session |
| `microgesture_openwatch` | OpenWatch (43 canonical identities) | 1 s, 100 Hz | 5 gestures + background, voted per clip | clip macro-F1 ↑ | subject, posture-stratified |
| `language_humanml3d` | HumanML3D, simulated wrist IMU | 10.24 s | alignment with CLIP ViT-B/32 text | mean Recall@1, galleries of 100 ↑ | official test split |
| `pd_pads` | PADS (469 participants) | 2.56 s | healthy / Parkinson's / differential per participant | participant macro-F1 ↑ | subject-disjoint 80/20 |
| `har_hhar` | HHAR (6 classes) | 2.56 s | window label | macro-F1 ↑ | subject-disjoint |
| `har_mhealth` | MHEALTH (12) | 2.56 s | window label | macro-F1 ↑ | subject-disjoint |
| `har_prism` | PrISM (5 global classes) | 2.56 s | window label | macro-F1 ↑ | subject-disjoint |
| `har_userwatch` | TransfHAR (60) | 2.56 s | window label | macro-F1 ↑ | 80/20 in time per recording |

Every input is one wrist, accelerometer x/y/z (m/s², gravity included) then gyroscope x/y/z (rad/s), in a
forward–left–up frame, resampled to the card's rate.

Decoders: linear heads for classification; width-192 attention decoders with two cross-channel and two temporal
layers for arm pose, one cross-channel layer for full-body pose and one temporal layer for odometry; a projection into
the CLIP text space for retrieval. Classification heads are trained with inverse-frequency class-weighted
cross-entropy. The activity column of the leaderboard is the mean macro-F1 over the four activity sets.

Other task ids in `tasks.yaml` (decoder ladders, rotation variants, the release split of OpenWatch, world-frame
odometry, extra activity sets) are ablations and controls from the paper; they are marked `disabled` and need
`--allow-disabled` in `bench.launch`.
