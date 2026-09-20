# Mask-ratio ablation (frozen-encoder probes)

One pretraining arm per `objective.mask_ratio`; every other key held byte-identical
to `configs/nhanes_scale/mae_recipe_R0.yaml` (leak-free 514 h pool, 30k steps, seed 42).
The AltAttn encoder masks BERT-style (all tokens enter the encoder, masked ones replaced
by a learned mask token), so per-step compute is the same for every arm — equal steps is
equal compute here. All cells are frozen-encoder probes at seed 42, scored by `bench.score`.

| mask ratio | wisdm | utd_mhad | pamap2 | recofit | hhar | mhealth | **HAR mean** | arm pose (cm)↓ | gesture clip-F1↑ |
|---|---|---|---|---|---|---|---|---|---|
| **0.30** | 0.363 | 0.439 | 0.288 | 0.405 | · | · | · (4/6) | · | · |
| **0.40** | 0.511 | 0.599 | 0.438 | 0.554 | 0.489 | 0.644 | **0.539** | 8.46 | 0.536 |
| **0.50** | 0.510 | 0.562 | 0.409 | 0.545 | 0.553 | 0.613 | **0.532** | 8.41 | 0.501 |
| **0.60** | 0.500 | 0.627 | 0.430 | 0.537 | 0.525 | 0.668 | **0.548** | 8.53 | 0.550 |
| **0.75** | 0.505 | 0.534 | 0.391 | 0.513 | 0.545 | 0.622 | **0.518** | 8.45 | 0.503 |
| **0.85** | 0.473 | 0.557 | 0.387 | 0.506 | 0.483 | 0.623 | **0.505** | 8.47 | 0.541 |
| **0.90** | 0.460 | 0.496 | 0.369 | 0.471 | 0.544 | 0.643 | **0.497** | 8.53 | 0.490 |

HAR mean = mean window macro-F1 over the six leaderboard HAR sets (blank until all six land).
Arm pose = FK position error, lower is better. Gesture = OpenWatch clip macro-F1 on the test split.

**Best HAR mean: mask ratio 0.60 (0.548).** Spread over completed arms: 0.497–0.548.

## Reconstruction quality of the SAME six arms, at fixed evaluation mask ratios

Every arm rescored at one common mask ratio (identical per-batch mask sequence), so these
MSEs are comparable across arms -- unlike the logged val losses. Compare against the probe
table above: the ordering is REVERSED.

| evaluated at | 0.30 | 0.40 | 0.50 | 0.60 | 0.75 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|---|
| **0.5** | 0.16839 | 0.34535 | 0.37352 | 0.33261 | 0.35711 | 0.28214 | 0.27178 |
| **0.75** | 0.17577 | 0.52722 | 0.49510 | 0.50622 | 0.48491 | 0.40232 | 0.38136 |
| **0.9** | 0.21887 | 0.50559 | 0.43443 | 0.51792 | 0.50445 | 0.48945 | 0.46418 |

Lower is better. The 0.85/0.90 arms reconstruct best at every evaluation ratio while
transferring worst, so reconstruction fidelity is anti-correlated with probe accuracy here:
MAE loss cannot be used to pick a mask ratio, even after the fixed-ratio correction.

## Pretraining loss at a fixed evaluation mask ratio (300k-step Stage-1 arms)

Independent evidence from the earlier 300k-step Stage-1 (acc-only, 11-source) arms.
Each arm is rescored at one COMMON mask ratio with an identical per-batch mask
sequence, because a 0.90-trained arm's logged val loss answers a strictly harder
question than a 0.50-trained arm's.

Evaluated at mask ratio **0.5** (60 val batches):

| trained mask ratio | fixed-ratio MSE↓ | vs predict-zero | recon/truth std | corr |
|---|---|---|---|---|
| 0.50 | 0.752380 | 24.7% better | 0.496 | 0.497 |
| 0.75 | 0.761976 | 23.6% better | 0.484 | 0.486 |
| 0.90 | 0.783533 | 21.7% better | 0.461 | 0.465 |

Evaluated at mask ratio **0.75** (60 val batches):

| trained mask ratio | fixed-ratio MSE↓ | vs predict-zero | recon/truth std | corr |
|---|---|---|---|---|
| 0.50 | 0.813743 | 18.7% better | 0.432 | 0.432 |
| 0.75 | 0.807141 | 19.4% better | 0.443 | 0.440 |
| 0.90 | 0.821424 | 17.9% better | 0.426 | 0.423 |

Evaluated at mask ratio **0.9** (60 val batches):

| trained mask ratio | fixed-ratio MSE↓ | vs predict-zero | recon/truth std | corr |
|---|---|---|---|---|
| 0.50 | 0.900795 | 9.9% better | 0.293 | 0.316 |
| 0.75 | 0.884804 | 11.6% better | 0.357 | 0.340 |
| 0.90 | 0.893102 | 10.7% better | 0.349 | 0.328 |

**Replicate.** Both partition twins of the rescore job ran (9972314 at 10:58, 9972317 at 12:30),
each rebuilding the val loader independently -- an accidental full replicate. Run-1 MSEs were
0.7527/0.7628/0.7871 (eval 0.50), 0.8101/0.8068/0.8195 (eval 0.75), 0.8995/0.8842/0.8922 (eval 0.90)
for trained 0.50/0.75/0.90; max drift between replicates is 0.0036, and the ranking is identical at
every eval ratio in both runs. Gaps below ~0.004 (0.75 vs 0.50 at eval 0.75) are within that drift;
the 0.75-over-0.90 gap at eval 0.90 (0.0080, 0.0083) is not.

## Still missing

`har_hhar/mae_mr30`, `har_mhealth/mae_mr30`, `arm_pose_nymeria/mae_mr30`, `microgesture_openwatch/mae_mr30`
