# Data audit and leakage

Every evaluation set went through two audits before it was split, and every submission is checked the same way.

## Declare the pretraining corpus

The encoder card lists the pretraining corpus dataset by dataset, with the split used (`pretrain_data`). The
evaluation splits of the ten tasks must not be in it. Declare bundles by their member datasets: common
activity-recognition pools contain WISDM, PAMAP2, RecoFit, SAMoSA, Shoaib and UT-Watch, which is why those sets are
not evaluation tasks here and why the four activity tasks (HHAR, MHEALTH, PrISM, TransfHAR) are held out of the
released corpus.

| Overlap | Meaning | Leaderboard |
|---|---|---|
| none | the corpus contains no recording of any evaluation split | ranked |
| exposure | unlabeled evaluation recordings were in the pretraining corpus | listed as *overlap*, not ranked on the affected tasks |
| leakage | evaluation participants or labels were used for training | not listed |

## Audit a release before splitting it

```bash
imu-bench audit <dir> --subject-regex 'id_(\d+)/' [--glob '**/*.csv'] [--acc-cols AccX AccY AccZ]
```

1. **Exact copies.** Every file is hashed; identical files under different participant ids link those ids.
2. **Mirrored or rotated copies** (with `--acc-cols`). The accelerometer magnitude is unchanged by mirroring or
   rotating the sensor axes, so two recordings whose standardised magnitude correlates above 0.999 at a small lag are
   the same motion.

Linked ids are merged into identity groups; a participant-disjoint split must be made over these groups. On OpenWatch
this reduces 50 participant ids to 43 people and links all 35 motion-augmented ids to their sources, which is the
procedure behind the benchmark's gesture split.

## Five checks for any wrist-IMU evaluation

1. **Hash a release before splitting it by participant** (above).
2. **Check the pretraining pool against every evaluation set.** An earlier corpus of ours contained Nymeria and
   HumanML3D training splits and OpenWatch recordings; the released corpus does not.
3. **Run a never-trained encoder through the same decoder.** The `control_random` and `control_raw` rows show what the
   decoder alone achieves; with a 12-layer odometry decoder the untrained encoder beats every pretrained one.
4. **Verify sensor units for every source.** EatingSpeed stores the gyroscope in deg/s and OpenWatch stores raw counts;
   the harness converts units on load and rejects a source whose still windows have a median gyroscope magnitude above
   0.3 rad/s.
5. **Know when an official split is partly transductive.** 437 of HumanML3D's 2,192 test clips share frames with a
   training or validation clip; this affects every encoder equally and is stated with the retrieval results.
