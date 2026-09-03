# Why do the Inertia-1 (NHANES) encoders beat our MAE on HAR? — diagnostic (2026-08-18)

Setup difference: every ViT1D HAR row (probe / finetune / scratch) sees 24 Hz low-passed, **per-window
z-scored** input (per channel: mean removed, std normalised); the Inertia-1 adapter feeds **raw g-scale
acceleration** (no filter, no z-score). Per-window z-scoring deletes the window mean (gravity direction →
posture) and the window scale (motion intensity). Runs: `runs/diag_har_*`, configs `configs/diag_har_norm/`,
scores `bench/generated/diag_har_norm_results.json`, `bench/generated/diag_gravity_*.json`.

macro-F1 (val, min-val-loss ckpt)               | MHEALTH | HHAR  | WISDM
------------------------------------------------|---------|-------|------
std: MAE-11src probe (z-scored, 6ch)            | 0.689   | 0.679 | 0.639
std: MAE-11src finetune (z-scored, 6ch)         | 0.627   | 0.628 | 0.704
std: MAE acc-only probe (z-scored, 3ch)         | 0.661   | 0.600 | 0.535
std: scratch ViT1D (z-scored, 6ch)              | 0.659   | 0.504 | 0.606
std: Inertia-1 PatchTST-S probe (raw g, 3ch)    | 0.893   | 0.713 | 0.614
**diag** scratch ViT1D, z-score OFF (raw, 6ch)  | 0.608*  | 0.619*| 0.730
**diag** Inertia-1 PT-S probe, z-score ON       | 0.553   | 0.499 | 0.409
**diag** MAE-11src probe, z-score OFF (mismatch)| 0.724   | 0.547 | 0.621
logreg on window mean+std, acc only (6 numbers) | 0.690   | 0.348 | (running)
logreg on window mean+std, acc+gyro (12 numbers)| 0.691   | 0.361 | (running)

\* early-stopped at step 200/400 (val loss minimum reached almost immediately: unnormalised m/s² inputs +
the scratch recipe's lr 1e-4 optimise badly) — treat the raw-scratch MHEALTH/HHAR numbers as lower bounds.

Reading:
1. **Inertia-1's HAR edge disappears completely once its input is z-scored** (MHEALTH 0.89→0.55, HHAR 0.71→0.50,
   WISDM 0.61→0.41) — it then sits well below our MAE probe everywhere. (Partly a train/test mismatch for a
   model pretrained on raw g, symmetric to our MAE-without-z-score control; but the direction is unambiguous.)
2. **The deleted information is worth a whole probe on posture-heavy sets:** 6 window statistics (mean/std of
   raw acc) give 0.69 on MHEALTH = our MAE probe; even our MAE, fed raw input under mismatch, gains on MHEALTH
   (0.689→0.724). On HHAR the stats alone are weak (0.35): there the raw signal must be *processed* to help.
3. **It is not only preprocessing:** a from-scratch ViT1D on raw input reaches 0.61 on MHEALTH, Inertia-1 0.89.
   NHANES pretraining on raw wrist accelerometry (642K h of daily life) genuinely learns posture/intensity
   structure our z-score-pretrained MAE cannot express. WISDM raw scratch (0.73) also beats every MAE row —
   the per-window z-score costs us on WISDM as well.

Implication for the benchmark: our per-window z-score is a self-imposed handicap on HAR-type tasks (posture,
intensity) while being neutral-to-helpful on the kinematic tasks. Two remedies, both protocol changes:
(a) re-pretrain the MAE with a DC-preserving normalisation (fixed per-sensor scaling, e.g. divide by g / a
    global std) — a multi-day re-run of the pretraining stack, but the principled fix;
(b) cheap hybrid: give every probe head the 12 raw window statistics alongside the pooled embedding
    (same for all encoders) — restores the information without re-pretraining; changes the probe protocol.


## Control experiment: same architecture + same raw pipeline, WITHOUT NHANES pretraining (2026-08-18)

`inertia1_patchtst_scratch` = the Inertia-1 PatchTST-small architecture and its exact input pipeline
(acc-only, raw g, no filter/z-score, 20 Hz, 0.5 s patches) trained end-to-end from random init under our
standard HAR recipe; `inertia1_patchtst_randinit` = the same architecture frozen at random init + linear head.
Both go through `bench.score.har` like every other row.

macro-F1                                    | wisdm | utd  | pamap2 | recofit | hhar  | mhealth | mean
--------------------------------------------|-------|------|--------|---------|-------|---------|------
Inertia-1 PT-S NHANES **finetune**          | 0.743 | 0.756| 0.431  | 0.666   | 0.785 | 0.896   | **0.713**
Inertia-1 PT-S NHANES **probe**             | 0.614 | 0.678| 0.437  | 0.601   | 0.713 | 0.893   | 0.656
Inertia-1 PT-S **scratch** (raw, no pretrain)| 0.676 | 0.603| 0.265  | 0.656   | 0.697 | 0.779   | 0.612
Inertia-1 PT-S **random frozen** probe      | 0.424 | 0.460| 0.195  | 0.435   | 0.455 | 0.670   | 0.440
MAE-11src probe (ours, z-scored, 6ch)       | 0.639 | 0.633| 0.542  | 0.630   | 0.679 | 0.689   | 0.635
MAE-11src finetune (ours, z-scored, 6ch)    | 0.704 | 0.727| 0.413  | 0.768   | 0.628 | 0.627   | 0.644
ViT1D scratch (ours, z-scored, 6ch)         | 0.606 | 0.701| 0.219  | 0.663   | 0.504 | 0.659   | 0.559

**Decomposition of the 0.713 vs 0.644 finetune gap (+0.069):**
* input pipeline + architecture, no pretraining at all: 0.612 vs our z-scored scratch 0.559 → **+0.053 from the
  raw-g pipeline** (biggest on HHAR +0.19, MHEALTH +0.12; *negative* on PAMAP2 −0.05 and UTD −0.10);
* NHANES pretraining on top of that pipeline: 0.713 − 0.612 → **+0.101 from pretraining**;
* pretraining value in *linear separability*: probe 0.656 vs random-frozen 0.440 → **+0.216**.

So both factors are real and roughly comparable in size, and neither alone explains the gap. Note also that a
**random** encoder on raw input scores 0.670 on MHEALTH — within noise of our pretrained MAE probe (0.689) —
which is the sharpest single statement of how much MHEALTH is decided by the DC component our z-score removes.


## Step 1 of the fix: MAE embedding + 12 raw window statistics (2026-08-18)

`StatsConcatEncoder` (imu_lm/models/ViT1D/model.py, `stats_concat: {enabled: true}` + `preprocess.normalize.enabled:
false`): the loader passes the filtered but un-z-scored window; the wrapper computes per-channel mean/std of the
physical signal (acc / 9.81 g, gyro rad/s), z-scores the window itself so the pretrained encoder sees its usual
input, and returns `[z_MAE ; mu ; sigma]` (384 + 12 = 396-d) to the linear head. No re-pretraining. 18 runs
(`runs/diag_har_*__stats`), scored by `bench.score.har`.

macro-F1                            | wisdm | utd  | pamap2| recofit| hhar | mhealth| mean
------------------------------------|-------|------|-------|--------|------|--------|------
MAE-11src probe (standard)          | 0.639 | 0.633| 0.542 | 0.630  | 0.679| 0.689  | 0.635
**MAE-11src probe + 12 raw stats**  | 0.687 | 0.664| 0.553 | 0.700  | 0.680| 0.793  | **0.679**
MAE-11src finetune (standard)       | 0.704 | 0.727| 0.413 | 0.768  | 0.628| 0.627  | 0.644
**MAE-11src finetune + 12 raw stats**| 0.706 | 0.731| 0.482 | 0.773  | 0.664| 0.663  | **0.670**
ViT1D scratch (standard)            | 0.606 | 0.701| 0.219 | 0.663  | 0.504| 0.659  | 0.559
ViT1D scratch + 12 raw stats        | 0.619 | 0.661| 0.240 | 0.650  | 0.494| 0.620  | 0.547
Inertia-1 PT-S NHANES probe         | 0.614 | 0.678| 0.437 | 0.601  | 0.713| 0.893  | 0.656
Inertia-1 ART-S NHANES probe        | 0.681 | 0.717| 0.444 | 0.648  | 0.717| 0.919  | 0.688
Inertia-1 PT-S NHANES finetune      | 0.743 | 0.756| 0.431 | 0.666  | 0.785| 0.896  | 0.713

* **Probe: +0.044 (0.635 -> 0.679), never negative on any dataset**; biggest on MHEALTH (+0.104) and RecoFit
  (+0.070) — exactly the posture/intensity-carrying sets. That overtakes the Inertia-1 PatchTST-S probe (0.656,
  **+0.023**) and lands within noise of the ART-S probe (0.688, −0.008).
* **Finetune: +0.026 (0.644 -> 0.670)**, still 0.043 behind Inertia-1's finetune (0.713) — MHEALTH/HHAR remain
  their strongholds (0.66 vs 0.90 / 0.79).
* **Scratch + stats: −0.011** — the statistics only help when the encoder is *pretrained* (frozen or warm-started);
  a from-scratch model can already fit whatever it needs from the z-scored waveform. So this is not "12 extra
  features are free accuracy"; it is specifically the pretrained representation that is missing the DC component.

Conclusion: roughly **half of the HAR gap is recoverable without re-pretraining**, and the remainder is genuine
NHANES-pretraining value on raw posture structure. This supports the planned v2 recipe (fixed physical scaling
instead of per-window z-score, dual-view physical + dynamic streams, auxiliary gravity/amplitude losses), with
the acceptance criterion Delta_HAR > 0 while Delta_pose ~ Delta_odom ~ 0 and Delta_gesture >= 0.
