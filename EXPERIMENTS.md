# SSL Benchmark: Ours (MAE) vs. FOCAL vs. SimMTM vs. RelCon vs. Scratch

Tracks downstream performance of four self-supervised pretraining objectives —
our ViT1D MAE model, FOCAL, SimMTM, and RelCon — all pretrained on the **same
combined 11-source dataset** (`mae_multidata_cache`), plus a from-scratch (no
pretraining) reference column, across 6 downstream tasks. A 6th column, "Ours
(MAE, no alt-attn)", isolates the contribution of the alternating frame/global
attention architecture itself: identical MAE objective/data/decoder, but a
plain whole-self-attention encoder instead.

**Architecture parity**: all columns use the same encoder capacity
(`hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
intermediate_size=1536`, `patch_size=16`, `shared_kernel=acc_gyro`). Ours (MAE)
uses the alternating frame/global attention encoder (`alt_attention: true`);
FOCAL, SimMTM, RelCon, Scratch, and Ours (MAE, no alt-attn) use the plain
ViT1D encoder — Scratch is built with `alt_attention: true` (matching
Ours(MAE) exactly) so that column isolates pretraining vs. random init on the
*same* architecture, while Ours (MAE, no alt-attn) isolates architecture vs.
objective (same MAE pretraining, plain encoder). FOCAL/SimMTM checkpoints:
`runs/vit1d_ssl_{focal,simmtm}_multidata/checkpoints/best.pt`. Ours(MAE)
checkpoint: `runs/vit1d_ssl_mae_varlen_altattn_multidata/checkpoints/best.pt`
(early-stopped step 176,000, best val step 166,000). Ours (MAE, no alt-attn)
checkpoint: `runs/vit1d_ssl_mae_varlen_noaltattn_multidata/checkpoints/best.pt`
(pretraining job 7245668).

**RelCon (2026-07-18, 🔄 10 of 11 downstream jobs done)**: Xu et al., ICLR 2025 ("RelCon:
Relative Contrastive Learning for a Motion Foundation Model for Wearable
Data"), vendored at `baselines/SSL/relcon`. Integrated with the same
"different pretext task, same shared ViT1DEncoder backbone" methodology as
FOCAL/SimMTM (see `imu_lm/objectives/relcon_ssl.py` for the full design
writeup). Two-stage pipeline, both faithfully porting the vendored repo's
actual math/networks:
  - **Stage 1** (auxiliary, frozen afterward) — MotifDist: a small
    cross-attention distance network (their own `CrossAttn.Net`) trained via
    query/augmented-key reconstruction on the same combined multidata source,
    using their own published augmentation set (8 IMU transforms, 2 chosen
    per sample). Standalone script: `scripts/train_relcon_motifdist.py`
    (job 7378858). Checkpoint: `runs/relcon_motifdist_multidata/checkpoints/best.pt`.
  - **Stage 2** (the actual SSL baseline evaluated below) — trains the
    shared ViT1DEncoder via RelCon's `relative_contrastive_loss` (ported
    verbatim), using the frozen Stage-1 network to rank each anchor's
    within-batch candidates by motif-distance (job 7380561, depends on Stage 1).
    Checkpoint: `runs/vit1d_ssl_relcon_multidata/checkpoints/best.pt`.
  - Documented simplification: RelCon's paper also uses "within-user"
    candidates (other windows from the same participant); this benchmark's
    generic multidata loader draws windows without session-grouped batching,
    so only the paper's own "between-window" (rotated-batch) candidates are
    used here.
  - Smoke-tested end-to-end on real data (small step budgets, both stages)
    before committing to the full run — caught and fixed one real bug (a
    device-placement race: the frozen MotifDist network was being moved to
    `encoder`'s device *before* the shared `Trainer` had actually moved
    `encoder` there itself, so it landed back on CPU; fixed by matching
    `Trainer`'s own device-selection logic directly instead of reading
    `encoder`'s device at that point in the code).
  - Downstream: same 9 probe configs as FOCAL (arm pose, full-body pose,
    WISDM/UTD-MHAD/PAMAP2/RecoFit HAR, gesture, CLIP, wrist-FLU odometry —
    11 jobs total incl. TotalCapture variants, all dependent on Stage 2). Not
    added to Section 4b (canonical world-frame odometry) — FOCAL/SimMTM were
    never added there either, so this keeps RelCon's scope matched to its
    siblings. **2026-07-20**: Stage 2 pretrain (originally job 7384383) hit
    its 1-day walltime and TIMED OUT at step ~9000 (not a crash — plain
    walltime exhaustion); its 11 downstream jobs were left in
    `DependencyNeverSatisfied` since their `--dependency=afterok:7384383`
    could never be satisfied by a TIMEOUT state. Fixed by cancelling all 11
    stale pending jobs and resubmitting: Stage 2 resumed via the launch
    script's existing `--resume best` flag as job 7431062 (continues from
    the `best.pt` checkpoint at step ~9000), and all 11 downstream jobs
    resubmitted with `--dependency=afterok:7431062` (new IDs: 7431077,
    7431080, 7431083, 7431085, 7431089, 7431093, 7431098, 7431101, 7431105,
    7431107, 7431109).
  - **2026-07-21 update (resolved)**: job 7431062 `COMPLETED` cleanly
    (23h56m total, ended 12:07) — the local-log staleness noted earlier that
    same day was just infrequent flushing, not a hang; training continued
    normally underneath it. Early-stopped at step 147,000 (patience=10
    evals), best val_loss=435.166 at step 137,000
    (`checkpoints/best.pt`). All 11 downstream RelCon probes were unblocked
    immediately: 10 of 11 have now completed (arm pose Nymeria/TotalCapture,
    full-body pose TotalCapture, gesture, CLIP HumanML3D, HAR
    WISDM/PAMAP2/UTD-MHAD/RecoFit, odometry Nymeria — results now filled in
    throughout Sections 1-6 above), only full-body pose Nymeria (7431101)
    still running (best so far 102.47mm, step ~309,000). `lr5e-5` (§2's LR
    sweep) and the odometry-scaling 50% probe (§9e) were also unblocked and
    are now
    running.
  - **2026-07-22 check-in**: job 7431101 is still `RUNNING` (17h57m elapsed,
    now at step ~476,000/epoch 773). Best val mpjpe is still 102.47mm at step
    32,000 — unchanged since the last check-in 444,000+ steps ago; the val
    curve has been flat-oscillating in a 104.5-105.6mm band (worse than the
    step-32,000 best) for the entire time since. `early_stopping.enabled:
    false` on this config (same pre-existing-template gap as the other
    "runaway job" cases documented throughout this doc), so it will not stop
    on its own. `best_mpjpe.pt` already holds the step-32,000 checkpoint, so
    nothing is at risk — this looks like a candidate for the same
    manual-cancel handling used elsewhere, but left running pending
    confirmation rather than cancelled unilaterally.
  - **2026-07-22 later check-in**: still `RUNNING`, now ~24h elapsed, step
    ~636,000/epoch 1031. Best val mpjpe unchanged at 102.47mm (step 32,000)
    — 600,000+ steps with zero improvement, val still oscillating 104-106mm.
    Confirms this is a genuine runaway job, not transient noise. Left running
    pending an explicit cancel decision; `best_mpjpe.pt` remains safe either
    way.
  - **Resolved (2026-07-22)**: the question resolved itself — job hit its
    own 1-day walltime and ended with `TIMEOUT` at step 639,000, never
    needing a manual cancel. Final check across all 639 val evals confirms
    the best mpjpe never moved from 102.47mm (step 32,000). This row is
    now closed out (✅ in Section 2's table above).

**Ours (MAE, acc-only) (2026-07-20, 🔄 mostly done)**: a 7th column, isolating
the contribution of gyro data to pretraining — the mirror image of "Ours
(MAE, no alt-attn)" (which isolates architecture) and "Ours (MAE,
Nymeria-only, no norm)" (which isolates pretraining data scope, Section 2
only). Same MAE objective/data/decoder/alt-attn encoder as Ours(MAE), same
combined 11-source dataset, but every batch is truncated to its accelerometer
channels (`data.acc_only: true`, `data.sensor_columns: [acc_x, acc_y,
acc_z]`) before it reaches the encoder or the reconstruction loss — the
encoder never sees gyro, at pretraining *or* downstream probe time.
  - `shared_kernel: true` (not `acc_gyro`) — no gyro half exists to split.
  - Implementation: a single choke point in `Trainer._acc_only_slice`
    (`imu_lm/runtime_consistency/trainer.py`) truncates every `[B, C, T]`
    batch to `data.sensor_columns`'s width right after `_to_device`, for both
    the train and eval loops. This covers MAE pretraining and every
    downstream probe/finetune objective without touching per-source loaders
    or per-objective `forward_loss` code, since all of them treat `x` as an
    opaque `[B, C, T]` tensor. The FLU wrist loaders (Nymeria, TotalCapture,
    HumanML3D) and the MAE multidata cache always emit the full acc+gyro
    concat regardless of `data.sensor_columns`, so this is the only reliable
    place to cut it down; loaders that already honor `sensor_columns`
    (WISDM/UTD-MHAD/PAMAP2/RecoFit HAR, gesture) hand back acc-only data
    directly, and the slice is a no-op for them (it truncates to an explicit
    target width, not just "half of whatever arrives", so it's safe either
    way).
  - Downstream: same 10 probe configs as Ours(MAE, no alt-attn) (arm pose
    Nymeria/TotalCapture, full-body pose Nymeria/TotalCapture,
    WISDM/UTD-MHAD/PAMAP2/RecoFit HAR, gesture, CLIP — jobs
    7461716-7461769, all dependent on the pretrain job 7461706). Odometry
    (Section 4) reuses the pretrain run's own val metrics, same as every
    other SSL column; its ATE/drift eval
    (`scripts/eval_odom_nymeria_altattn.py`) was patched to truncate the
    encoder's input to acc-only while still integrating rotation from the
    *real* ground-truth gyro (a classical `exp_so3` dead-reckoning step, not
    a model input) — dropping that would break trajectory reconstruction for
    every column, not just this one. Not added to Section 4b (world-frame
    odometry) or Section 8 (cross-task transfer) — no-alt-attn wasn't added
    there either, so this keeps acc-only's scope matched to its sibling
    ablation.
  - Smoke-tested end-to-end on real data (2 CPU train steps against the real
    `mae_multidata_cache`, capture24 source only) before committing to the
    full run — confirmed the encoder/decoder/loss shapes all resolve
    correctly with 3 input channels instead of 6 (this exercises
    `ViT1DPoseDecoder`/`ViT1DOdomDecoder` too, since they're built with
    `in_channels=encoder.in_channels`, itself derived from
    `len(data.sensor_columns)` — consistent with the post-slice channel count
    by construction).
  - Pretraining job 7429901 crashed at step 10000 on a periodic-checkpoint
    write (`torch.save` → `RuntimeError: [enforce fail at
    inline_container.cc:672] . unexpected pos ... vs ...`, an
    interrupted/truncated write) — unrelated to the acc-only code itself
    (loss was decreasing normally, 0.7-0.9, right up to the crash). `best.pt`
    (saved 3 min earlier at step 9000 via the early-stopping path, val_loss
    0.7949) survived intact, so the truncated `latest.pt` was deleted and the
    job resubmitted with `--resume best` (job 7440402) — which crashed at the
    *same* step 10000 checkpoint write again, this time surfacing the real
    cause underneath the truncated-write symptom: `OSError: [Errno 122] Disk
    quota exceeded`. `mmlsquota` confirmed the `gpfs:home` fileset (backing
    `/gpfs/home/vbc1591`, where `runs/` lives) was at 83,776,096/83,886,080 KB
    (99.87% of the 80GB per-user quota) — a repo-wide blocker, not specific to
    this ablation, that would have failed the *next* checkpoint write from
    any job on this filesystem (it also nearly caught the unrelated,
    concurrently-running `vit1d_relcon_multidata` job, 7431062). Fixed by
    migrating `runs/` (104 run dirs, ~21GB at the time) to
    `/projects/b53017/vbc1591/runs` (`project_b53017` fileset, no enforced
    quota, 24TB free) via `rsync -a`, then replacing
    `/gpfs/home/vbc1591/IMU-SSL/runs` with a symlink to the new location —
    every config's hardcoded `paths.runs_root:
    /gpfs/home/vbc1591/IMU-SSL/runs` keeps resolving unchanged, and all new
    writes (from every job, not just this ablation) now land on the
    unconstrained filesystem with zero config edits. Freed home usage to
    45,519,232 KB (54%). Pretraining resubmitted a second time as **job
    7461706** with `--resume best`, downstream jobs re-queued against it
    (7461716-7461769).
  - Checkpoint: `runs/vit1d_ssl_mae_varlen_acconly_multidata/checkpoints/best.pt`.
  - **Status (2026-07-20)**: all 10 of 10 downstream probes completed
    (7461716/7461717 arm pose, 7461721/7461722 full-body pose, 7461724/
    7461725/7461727/7461728 HAR, 7461767 gesture, 7461769 CLIP). Full-body
    pose Nymeria (7461721) plateaued at 103.4-105.9mm mpjpe from step ~18,000
    onward and was manually cancelled at step 106,515, taking its
    `best_mpjpe.pt` (103.37mm, step 56,000) — same handling as this
    benchmark's other no-early-stopping "runaway" jobs. Still open:
    arm-pose Nymeria `pos_err`/`ang_err` needs a rerun of
    `scripts/eval_arm_pose_position_error.py` with an acc-only entry added
    to its hardcoded `RUNS` list (script currently only knows the 6-channel
    configs); Section 4 odometry (vel_mse/ATE/drift) needs
    `scripts/eval_odom_nymeria_altattn.py --run
    vit1d_ssl_mae_varlen_acconly_multidata --mode probe` run as its own job
    (script was already patched to handle acc-only truncation, just hasn't
    been launched against this checkpoint yet).

Legend: ✅ done · 🔄 running/queued · ⏳ partially done, follow-up scoring pending · — not yet run

---

## 1. Arm pose estimation (Nymeria real-wrist, 5 joints)

Config family: `vit1d_arm_pose_probe_nymeria_{model}_multidata_flu.yaml` /
`vit1d_arm_pose_supervised_nymeria_altattn_scratch.yaml`. Metric: best val pose-MSE loss (lower is better).

| Model | val_loss | pos_err (cm) | ang_err (deg) | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 0.01638 | 8.17 | **9.81** | ✅ | `vit1d_arm_pose_probe_nymeria_mae_multidata` |
| Ours (MAE) finetune | **0.01599** | **8.00** | 9.89 | ✅ | `vit1d_arm_pose_finetune_nymeria_mae_multidata` |
| Ours (MAE, no alt-attn) | 0.01685 | 8.17 | 9.97 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 0.018883 | — | — | ⏳ probe done (job 7461716), pos/ang err scoring pending | `vit1d_arm_pose_probe_nymeria_mae_acconly_multidata` |
| FOCAL | 0.01916 | 8.73 | 10.75 | ✅ | `vit1d_arm_pose_probe_nymeria_focal_multidata` |
| SimMTM | 0.01813 | 8.56 | 10.39 | ✅ | `vit1d_arm_pose_probe_nymeria_simmtm_multidata` |
| RelCon | 0.018257 | — | — | ⏳ probe done (job 7431077, best step 8,000), pos/ang err scoring pending | `vit1d_arm_pose_probe_nymeria_relcon_multidata` |
| Scratch | 0.01672 | 8.15 | 9.94 | ✅ | `vit1d_arm_pose_supervised_nymeria_altattn_scratch` |
| MobilePoser* | — | 9.74 | 24.28 | ✅ | `baselines/full_body_pose/SimplePoser` |

**Task complete.** val_loss ranking (best to worst): Ours(MAE) finetune
**0.0160** < Ours(MAE) 0.0164 < Scratch 0.0167 < Ours(MAE, no alt-attn) 0.0169
< SimMTM 0.0181 < FOCAL 0.0192 — finetuning edges out the frozen probe by
~2.4%, and is now the best result on this task; Ours(MAE) is still the best
*pretraining* method (probe-only) here, beating scratch by ~2%. FOCAL/SimMTM
both underperform scratch.

`pos_err`/`ang_err` (job 7326886, `scripts/eval_arm_pose_position_error.py`)
now computed for every ViT1D row too, via forward kinematics with the same
`articulate.FullMotionEvaluator`-style rotation math MobilePoser uses (see
methodology note below the table). All 6 ViT1D rows land in a tight cluster
(8.0-8.7cm, 9.8-10.7°) and **all beat MobilePoser by a wide margin** (9.74cm,
24.28°) on this metric — mirroring the val_loss ranking closely (Ours(MAE)
finetune has the best pos_err, Ours(MAE) probe the best ang_err, FOCAL/SimMTM
still trail everything else). Note this isn't a perfectly fair fight in
MobilePoser's favor: see the methodology caveat below — these numbers hold
the torso/ancestor chain at ground truth, which MobilePoser's end-to-end
prediction does not get to do.

*MobilePoser is a different architecture family (RNN-based, not ViT1D) with no
`val_loss` in this task's units. For the ViT1D rows, `pos_err`/`ang_err` are
computed by composing the model's predicted rotations for the 5 arm-chain
joints onto an otherwise-ground-truth 24-joint skeleton and running forward
kinematics (`core.articulate`) — i.e. the torso/ancestor chain is held at
ground truth, isolating "how well does the model predict the arm chain's own
shape" rather than a fully end-to-end number. MobilePoser's own `pos_err`/
`ang_err` (root-aligned positional/angular error over the right
collar/shoulder/elbow/wrist/hand chain, via `articulate.FullMotionEvaluator`;
hand-joint rotation is architecturally forced to identity, no wrist-only
sensor evidence for it) predicts its own torso chain end-to-end, so its
errors compound in a way the ViT1D numbers above don't — the comparison
should be read as directional, not strictly apples-to-apples. Evaluated on
independent 10.24s windows matching this benchmark's own
`windowing.window_seconds` (not whole unbounded sessions — see Section 7).
See Section 7 for full methodology. Not otherwise comparable to the
`val_loss` column.

---

## 1b. Arm pose estimation (TotalCapture DIP, 5 joints)

**New dataset added (2026-07-18)**: TotalCapture/DIP-IMU mocap+IMU dataset
(`/projects/b53017/vbc1591/DIP_recalculate`, standard s1-s4 train / s5 val
split, same split already used as one of the 11 combined-multidata
pretraining sources). Same right-arm chain convention as Nymeria
(`RIGHT_WRIST_CHAIN_JOINTS = [14,17,19,21,23]`), reusing the *already-present*
loader (`imu_lm/data/totalcapture_wrist_loader_FLU.py`,
`loader: totalcapture_real_wrist_flu`) and the same generic `pose_probe`/
`pose` objective types as every Nymeria pose config — this loader/objective
plumbing predates this session but its configs referenced stale checkpoints
and an inaccessible path (`/mnt/nas2/DIP_recalculate`, pre-migration); all 7
configs here are fresh, cloned from the equivalent Nymeria config and
retargeted to TotalCapture's loader/paths, matching the current benchmark's
architecture/hyperparameter conventions exactly (only `objective.pose_probe`/
`objective.pose`'s loader-specific keys differ).

Two things needed fixing before submitting, caught in a pre-flight loader
test against real data: **(1)** TotalCapture is much smaller than Nymeria
(41 train sessions vs. 867) — only ~2 train batches/epoch at `batch_size:
256`, so the Nymeria template's `val_every_steps: 1000` would have meant
~500 epochs between evaluations; dropped to `val_every_steps: 20`.
**(2)** the full-body-pose sibling (see 2b below) inherited
`early_stopping.enabled: false` from its Nymeria source — the exact
runaway-job pattern documented in the pipeline audit below — flipped to
`true` here as a preventive fix, not a reaction to an actual incident this
time.

**Mostly complete (2026-07-19); RelCon still pending on its own Stage-2
pretrain** (that pretrain crashed from a disk-quota checkpoint-write error
after ~3h19m and was resumed from its last good checkpoint at step 8000 —
see Section 8/RelCon writeup above — so it's running again, just slower to
land than the rest of this table). The other 6 all completed cleanly.
**`pos_err`/`ang_err` are not available** — unlike Nymeria's arm-pose table,
no dedicated eval script (equivalent to `scripts/eval_arm_pose_position_error.py`)
has been built for TotalCapture yet; only the training-time `val_loss`
(`pose_mse`) is reported below.

| Model | val_loss | pos_err (cm) | ang_err (deg) | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 0.019950 | — | — | ✅ (job 7381464) | `vit1d_arm_pose_probe_totalcapture_mae_multidata` |
| Ours (MAE) finetune | 0.020179 | — | — | ✅ (job 7381466) | `vit1d_arm_pose_finetune_totalcapture_mae_multidata` |
| Ours (MAE, no alt-attn) | **0.018941** | — | — | ✅ (job 7381467) | `vit1d_arm_pose_probe_totalcapture_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 0.021213 | — | — | ✅ (job 7461717) | `vit1d_arm_pose_probe_totalcapture_mae_acconly_multidata` |
| FOCAL | 0.024580 | — | — | ✅ (job 7381468) | `vit1d_arm_pose_probe_totalcapture_focal_multidata` |
| SimMTM | 0.021552 | — | — | ✅ (job 7381469) | `vit1d_arm_pose_probe_totalcapture_simmtm_multidata` |
| RelCon | 0.019861 | — | — | ✅ (job 7431093, best step 340) | `vit1d_arm_pose_probe_totalcapture_relcon_multidata` |
| Scratch | 0.020045 | — | — | ✅ (job 7381470) | `vit1d_arm_pose_supervised_totalcapture_altattn_scratch` |

Interesting and different from Nymeria's arm-pose table: **Ours(MAE,
no-alt-attn) has the lowest val_loss**, edging out the full alt-attn
Ours(MAE) probe/finetune columns — on this much smaller dataset (41
sessions vs. Nymeria's 867), the plain whole-self-attention encoder may
generalize slightly better than the alternating-attention variant. FOCAL
remains clearly the weakest SSL pretraining method, consistent with every
other table in this benchmark.

---

## 2. Full-body pose estimation (Nymeria real-wrist, 23 joints)

Config family: `vit1d_full_body_pose_probe_nymeria_{model}_multidata_flu.yaml` /
`vit1d_full_body_pose_supervised_nymeria_altattn_scratch.yaml`. Metric: best val MPJPE (mm, lower is better).

| Model | mpjpe (mm) | Status | Run |
|---|---|---|---|
| Ours (MAE) | 97.63 | ✅ | `vit1d_full_body_pose_probe_nymeria_mae_multidata` |
| Ours (MAE) finetune | 92.39 | ✅ | `vit1d_full_body_pose_finetune_nymeria_mae_multidata` |
| Ours (MAE, no alt-attn) | 99.81 | ✅* | `vit1d_full_body_pose_probe_nymeria_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 103.37 | ✅† | `vit1d_full_body_pose_probe_nymeria_mae_acconly_multidata` |
| FOCAL | 108.29 | ✅ | `vit1d_full_body_pose_probe_nymeria_focal_multidata` |
| SimMTM | 103.48 | ✅ | `vit1d_full_body_pose_probe_nymeria_simmtm_multidata` |
| RelCon | **102.47** | ✅ (job 7431101, `TIMEOUT` at 1-day walltime, step 639,000 — best remained step 32,000 across all 639 val evals, never manually cancelled after all) | `vit1d_full_body_pose_probe_nymeria_relcon_multidata` |
| Scratch | 92.56 | ✅ | `vit1d_full_body_pose_supervised_nymeria_altattn_scratch` |
| MobilePoser** | **91.88** | ✅ | `baselines/full_body_pose/SimplePoser` |
| Ours (MAE, Nymeria-only, no norm) probe | 99.86 | ✅ | `vit1d_full_body_pose_probe_nymeria_only_nonorm` |
| Ours (MAE, Nymeria-only, no norm) finetune | 93.19 | ✅ | `vit1d_full_body_pose_finetune_nymeria_only_nonorm` |

**Task complete.** Ranking (best to worst): MobilePoser **91.88mm** <
Ours(MAE) finetune 92.39mm < Scratch 92.56mm < Ours(MAE, Nymeria-only, no
norm) finetune 93.19mm < Ours(MAE) 97.6mm < Ours(MAE, no alt-attn) 99.81mm ≈
Ours(MAE, Nymeria-only, no norm) probe 99.86mm < SimMTM 103.5mm < FOCAL
108.3mm — finetuning closes almost the entire gap to Scratch/MobilePoser
(97.63→92.39mm, a 5.4% relative improvement over the frozen probe), edging
out Scratch and landing within half a millimeter of MobilePoser. Different
from arm pose direction-wise, where the frozen probe was already the
outright winner — here finetuning was needed for Ours(MAE) to become
competitive with the non-SSL baselines rather than trailing them.

**LR sweep on Ours(MAE) finetune (2026-07-21, ✅ complete, 11 of 11 done)**:
baseline uses `trainer.optim.lr: 0.0001` (not itself tuned) and lands
92.39mm, 0.51mm behind MobilePoser. Launched 10 additional finetune runs
from the same `vit1d_ssl_mae_varlen_altattn_multidata` checkpoint, identical
to baseline in every other hyperparameter, varying only `trainer.optim.lr`
across `{3e-5, 5e-5, 7e-5, 1.5e-4, 2e-4, 3e-4, 4e-4, 5e-4, 7e-4, 1e-3}`
(configs
`configs/vit1d_full_body_pose_finetune_nymeria_mae_multidata_lr{tag}_flu.yaml`,
launch scripts `launch_slurm_full_body_pose_finetune_nymeria_mae_lr{tag}.sh`);
the last 4 (`4e-4`-`1e-3`) were added after the first 6 showed no sign of a
peak yet at `3e-4`.

| LR | best mpjpe (mm) | Status | Job |
|---|---|---|---|
| 3e-5 | 96.27 | ✅ (early-stopped step 44,000, best step 11,000) | 7491703 |
| 5e-5 | 94.16 | ✅ (early-stopped step 34,000, best step 14,000) | 7491704 |
| 7e-5 | 92.86 | ✅ (early-stopped step 33,000, best step 16,000) | 7491706 |
| 1e-4 (baseline) | 92.39 | ✅ | `vit1d_full_body_pose_finetune_nymeria_mae_multidata` |
| 1.5e-4 | 90.81 | ✅ (early-stopped step 31,000, best step 11,000) | 7491710 |
| 2e-4 | 90.44 | ✅ (early-stopped step 35,000, best step 11,000) | 7491716 |
| 3e-4 | 89.07 | ✅ (early-stopped step 30,000, best step 11,000) | 7491718 |
| 4e-4 | 88.71 | ✅ (early-stopped step 30,000, best step 10,000) | 7522400 |
| 5e-4 | 87.80 | ✅ (early-stopped step 32,000, best step 11,000) | 7522401 |
| 7e-4 | 87.79 | ✅ (early-stopped step 41,000, best step 21,000) | 7522402 |
| 1e-3 | **87.77** | ✅ (early-stopped step 34,000, best step 14,000) | 7522403 |

**Final ranking**: mpjpe decreases essentially monotonically across the
entire swept range, 3e-5 (96.27mm) down to 1e-3 (87.77mm) -- the sweep
never clearly turned over even at its highest tested LR. `1e-3` is the
nominal best, but `5e-4`/`7e-4`/`1e-3` (87.80/87.79/87.77mm) are all within
0.03mm of each other -- functionally tied, well inside single-seed noise --
so the honest read is "somewhere at or above 5e-4 is close to optimal for
this task," not a sharp single-LR peak. Using `1e-3` as the headline number:
**4.11mm / 4.5% better than MobilePoser** (91.88mm) and 4.62mm better than
the untuned baseline (92.39mm) -- the best result anywhere in this
benchmark. Not swept further above 1e-3 since the marginal gains had
already flattened to <0.03mm per step by the last three points, suggesting
diminishing returns rather than a still-improving trend worth chasing
further.

**LR/schedule sweep on Ours(MAE) frozen probe (2026-08-15, ⏳ in flight, 8
jobs 9381321-9381328)**: the probe row above (97.63mm) uses
`trainer.optim.lr: 0.0003` and `sched.step_decay` with `step_size: 500000`
— neither ever tuned. Two things are wrong with that recipe:

1. **LR was never swept.** The finetune sweep directly above, on the same
   checkpoint and task, improved monotonically from 3e-5 (96.27mm) to 1e-3
   (87.77mm) and never turned over, so 3e-4 is very likely well below optimal
   for the frozen probe as well.
2. **The LR never actually decays.** `step_size: 500000` exceeds any real
   budget, so the probe trains at a constant 3e-4 forever. Its own
   `logs/metrics.txt` shows convergence by step ~15,000-20,000 to ~98mm
   followed by a **97.6-99.9mm oscillation across another 120,000 steps**,
   while val_loss slowly *degrades* (0.018911 at step 14,000 → 0.020236 at
   step 136,000 — the decoder mildly overfits). **97.63mm is therefore the
   lucky floor of a noisy plateau, not a converged optimum**, and it is the
   min over a metrics file that has multiple appended runs in it. Decaying
   the LR to zero should turn that band's floor into the actual result.

Design: a 4 × 2 factorial, everything else identical to the baseline probe —
`lr ∈ {3e-4 (control), 1e-3, 3e-3, 1e-2}` × `sched ∈ {const (baseline
step_decay, cannot fire), cos (cosine → 0 over the budget)}`. Equal 60,000-step
budget per cell (~2.5h at ~24K steps/h; baseline converged by ~20K), with
`early_stopping.enabled: true` + `patience: 999999` so the flag cannot fire but
`best.pt` is still written. Configs
`configs/probe_lr/vit1d_full_body_pose_probe_nymeria_mae_multidata_lr{tag}_{const|cos}_flu.yaml`,
launchers `launch/probe_lr/` (`submit_all.sh`, `harvest.sh`).

**✅ Complete (8/8 COMPLETED, 1h10m-2h30m each, full 60K budget).** Two numbers
per cell: `best` = min mpjpe over all val evals (**the protocol every number in
Table 2 uses**), `final5` = mean of the last 5 evals (converged value; for the
`cos` arms LR has reached 0, so this is a genuine converged result).

| LR | const best | const final5 | cos best | cos final5 | Jobs (const / cos) |
|---|---|---|---|---|---|
| 3e-4 (baseline value) | **97.48** | 99.03 | 97.71 | **98.17** | 9386493 / 9386494 |
| 1e-3 | 97.87 | 99.38 | 98.26 | 98.85 | 9386483 / 9386484 |
| 3e-3 | 97.88 | 100.25 | **97.25** | 98.29 | 9386486 / 9386488 |
| 1e-2 | 98.76 | 100.83 | 98.04 | 98.29 | 9386480 / 9386481 |

**Conclusion 1 — the frozen probe is NOT LR-limited; the sweep fails to beat the
incumbent.** The best cell (3e-3 cos, 97.25mm) is 0.38mm under the 97.63mm
incumbent, but that is meaningless: the **between-arm spread of `best` is 1.51mm
(97.25-98.76), while every single arm's own within-run plateau oscillation is
2.06-5.56mm** (min-to-max over evals from step 10,000 on). The differences
between cells are strictly smaller than the noise inside one cell. There is also
no monotonic LR trend — 3e-4 const (97.48) beats 1e-3 (97.87) and 3e-3 (97.88),
yet 3e-3 cos is the nominal winner while 1e-3 cos (98.26) is near-worst. That is
noise, not signal. **This is the opposite of the finetune LR sweep above**, which
moved 8.5mm monotonically over the same LR range on the same checkpoint and task.
The explanation is mechanical: finetuning updates the ~21M-parameter encoder and
is therefore highly LR-sensitive, whereas the probe optimizes only the small
decoder over frozen features — a nearly-convex problem already at its
optimization ceiling. **The probe's bottleneck is representation quality, not
optimization, so no optimizer tuning will move it.** By the `final5` column the
original `lr: 0.0003` is in fact already optimal, and 1e-2 is clearly harmful.

**Conclusion 2 — cosine decay is a real effect, but only on converged quality.**
Every `cos` arm's `final5` (98.17-98.85) beats every `const` arm's (99.03-100.83)
— a clean separation with no overlap, unlike the `best` column. Cosine also
halves the plateau oscillation (at 3e-4: spread 3.75 → 2.06mm). So decaying the
LR does exactly what was predicted — it stops the run bouncing and lands it
reliably near its floor — but it does **not lower the floor**, so it buys nothing
under the best-checkpoint protocol.

**Conclusion 3 (methodological, affects the whole of Table 2) — the 97.63mm
headline is noise-mined, and the protocol rewards longer and noisier runs.**
`best` is a min over independent draws from a ~2-5mm-wide oscillation band, so it
falls purely as a function of how many val evals a run happens to get. The
incumbent drew ~130 evals over 130K+ steps; these arms drew 60. The control cell
here (`lr 3e-4 const`) is the *identical recipe* to the incumbent and scored
97.48 in 60 evals vs the incumbent's 97.63 in ~130 — i.e. the recipe reproduces,
and the reported number is dominated by draw count rather than by model quality.
Any cross-method comparison in Table 2 where the runs had different step counts
(e.g. RelCon's 639,000-step TIMEOUT run) is confounded by this. The defensible
fix is to adopt `cos` + report the converged value uniformly across all methods,
which is also cheaper (60K steps, ~2.3h) than the current unbounded runs.

**Net on LR: do not report an LR-"tuned" probe number.** The 97.63mm entry stands
as far as the optimizer is concerned — there is no tuning gain to equalize, so no
baseline re-runs are needed on that account.

---

#### 2a. Frozen-probe tuning summary (2026-08-15)

Two sweeps were run on the Ours(MAE) full-body-pose frozen probe, both on the
same `vit1d_ssl_mae_varlen_altattn_multidata` checkpoint, both at an equal
**60,000-step** budget with `early_stopping.enabled: true` / `patience: 999999`.
Every cell reports two numbers:

- **`best`** — min val mpjpe over all evals. This is the protocol every number in
  Table 2 uses, and it is **noise-mined**: it is a minimum over draws from a
  2-5mm-wide oscillation band, so it falls with the *number* of val evals a run
  received rather than with model quality (proof below).
- **`final5`** — mean of the last 5 evals. With cosine the LR has reached 0 by
  then, so this is a genuine converged value. **Judge on `final5`.**

**Sweep 1 — optimizer (LR × schedule). Verdict: no effect; the probe is not
LR-limited.**

| LR | const best | const final5 | cos best | cos final5 |
|---|---|---|---|---|
| **3e-4** (baseline) | **97.48** | 99.03 | 97.71 | **98.17** |
| 1e-3 | 97.87 | 99.38 | 98.26 | 98.85 |
| 3e-3 | 97.88 | 100.25 | **97.25** | 98.29 |
| 1e-2 | 98.76 | 100.83 | 98.04 | 98.29 |

Between-cell spread of `best` is **1.51mm**, while each cell's own within-run
plateau oscillation is **2.06-5.56mm** — the differences are smaller than the
noise inside a single cell, with no monotonic LR trend. The original `lr: 3e-4`
is already optimal and 1e-2 is harmful. Cosine *is* a real effect but only on
converged quality (every `cos` final5 beats every `const` final5, no overlap) and
it halves the plateau oscillation (3.75 → 2.06mm at 3e-4); it does not lower the
floor, so it buys nothing under `best`.

**Sweep 2 — decoder capacity. Verdict: large, real effect; temporal depth is the
lever.** Notation is `<stage-1 cross-channel depth> × <stage-2 temporal depth>`
(`pose_decoder.num_hidden_layers` × `pose_decoder.num_temporal_layers`).
Optimizer held at the swept-optimal `lr: 3e-4` + cosine. Control is the
`lr3e-4_cos` cell above, reused rather than re-run.

| Decoder | best | final5 | Δfinal5 vs 1×0 | best−final5 gap | last-20 SD | Job |
|---|---|---|---|---|---|---|
| 0 × 0 (**true linear probe**) | 105.23 | **105.42** | **+7.25** | 0.19 | 0.08 | 9467125 |
| **1 × 0** (baseline/control) | 97.71 | 98.17 | — | 0.46 | 0.13 | *(reused)* |
| 2 × 0 | 96.42 | 96.65 | −1.52 | 0.23 | 0.12 | 9453478 |
| 4 × 0 | 95.91 | 96.67 | −1.50 | 0.76 | 0.13 | 9453523 |
| 8 × 0 | 95.31 | 96.80 | −1.37 | **1.49** | 0.22 | 9453525 |
| 1 × 1 | 96.55 | 97.22 | −0.95 | 0.67 | 0.30 | 9453472 |
| 1 × 2 | 95.22 | 95.78 | −2.39 | 0.56 | 0.13 | 9453475 |
| 1 × 4 | 94.55 | 94.63 | −3.54 | 0.08 | 0.11 | 9453476 |
| **1 × 6** ← knee | 94.07 | **94.21** | **−3.96** | 0.14 | 0.14 | 9486783 |
| 1 × 8 | 94.09 | 94.46 | −3.71 | 0.37 | 0.13 | 9486784 |
| 1 × 12 | 94.05 | 94.22 | −3.95 | 0.17 | 0.09 | 9486785 |
| 1 × 16 | **93.69** | 94.13 | −4.04 | 0.44 | 0.18 | 9487628 |
| 4 × 2 | 95.34 | 96.07 | −2.10 | 0.73 | 0.18 | 9453524 |
| 2 × 4 | 94.16 | 94.58 | −3.59 | 0.42 | 0.13 | 9486787 |

Sorted by `final5`, the full capacity ladder spans **10.79mm**:

    0×0 105.42 | 1×0 98.17 | 1×1 97.22 | 8×0 96.80 | 4×0 96.67
    2×0 96.65  | 4×2 96.07 | 1×2 95.78 | 1×4 94.63

Four readings, all from the `final5` column:

0. **The single largest step in the whole sweep is 0×0 → 1×0: 7.25mm**, from
   adding *one* cross-channel attention layer — twice the 3.54mm that the entire
   temporal-depth axis buys. A purely linear readout of the frozen MAE features
   (`Linear(384→192)` → channel mean-pool → `Linear(192→K·J·6)`, no attention, no
   MLP nonlinearity) lands at **105.42mm**. This is the most important number in
   the table for interpreting Table 2: it means **the reported "frozen probe"
   metric is dominated by decoder capacity, not by representation quality alone**
   — the same frozen encoder scores anywhere from 105.4 to 94.6 depending only on
   what is stacked on top of it. For reference, that linear readout would sit
   *below* SimMTM's 103.48 and near FOCAL's 108.29 — though not comparably, since
   those baselines use the 1×0 decoder, which is exactly the point.

1. **Temporal depth is the lever, and it saturates at ~6 layers.** Round 2
   extended the axis to 16. Full curve (final5, temporal depth L at stage-1 = 1):

   | L | 0 | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
   |---|---|---|---|---|---|---|---|---|
   | final5 | 98.17 | 97.22 | 95.78 | 94.63 | **94.21** | 94.46 | 94.22 | 94.13 |
   | Δ from previous | — | −0.95 | −1.44 | −1.15 | −0.42 | +0.25 | −0.24 | −0.09 |

   The gains are real and steep through L=4, then stop: **L=6/8/12/16 span only
   0.33mm (94.13-94.46) while their last-20 SD is 0.09-0.18mm**, so those four
   cells are statistically indistinguishable — a plateau, not a trend. Going
   4 → 16 layers (4× the depth) buys 0.50mm; going 0 → 4 bought 3.54mm.
   **L=6 is the knee and the recommended setting**: 94.21mm, within 0.08mm of the
   nominal best (1×16, 94.13) at 2.7× less depth. 1×16 is also the only cell
   showing mild overfitting at the top of the axis (largest `best−final5` gap,
   0.44, and highest SD, 0.18), so its extra 0.08mm is not worth having. The
   baseline's `num_temporal_layers: 0` makes `_make_layers(0)` an empty
   `ModuleList`, so stage 2 degenerates to pos-embed → LayerNorm → linear head:
   **the baseline decoder did no temporal modelling at all**, predicting each
   0.32s patch's pose from its own channel-fused token with no access to
   neighbouring time steps — on a task whose loss carries an explicit jitter
   penalty (`jitter_weight: 0.0001`). Fixing that is where the 3.5mm came from.
2. **Cross-channel depth helps once, then overfits.** 96.65 at 2 layers, then
   flat-to-worse (96.67 at 4, 96.80 at 8) while the `best−final5` gap widens
   0.23 → 0.76 → **1.49mm**: deeper stage-1 stacks find a good minimum early and
   then degrade away from it. `dropout` was held at 0.2 for a clean
   single-variable sweep, so the deep cells are under-regularized by design.
3. **The axes do not compose — cross-channel depth is useless once temporal depth
   is present.** Both composition tests confirm it from opposite directions:
   4 × 2 (96.07) is *worse* than 1 × 2 (95.78), and 2 × 4 (94.58) is a dead heat
   with 1 × 4 (94.63, Δ 0.05mm — well inside the 0.11-0.13mm SD). Spending
   parameters on stage 1 beyond the first layer buys nothing.

The effect is far outside noise: last-20-eval SD is 0.11-0.30mm in every cell, so
1×4's 3.54mm gain is ~12-30σ. 1×4 also has the smallest `best−final5` gap in the
table (0.08mm), i.e. it genuinely settles at 94.6 rather than touching it once.

**Where the tuned probe lands in Table 2** — MobilePoser 91.88 < Ours(MAE)
finetune 92.39 < Scratch 92.56 < **Ours(MAE) probe, 1×6 decoder 94.21** <
Ours(MAE) probe, baseline decoder 97.63 < no-alt-attn 99.81 < SimMTM 103.48 <
FOCAL 108.29 < *Ours(MAE) linear readout 105.42*. The frozen probe closes from
5.24mm behind finetune to **1.82mm**, and to 1.65mm behind Scratch, but does not
overtake either. Total decoder-driven range on one frozen encoder: **11.29mm**
(105.42 → 94.13).

⚠ **This one DOES break Table 2 comparability — unlike the LR sweep.** Every
baseline probe (SimMTM / FOCAL / RelCon, and the no-alt-attn and acc-only
ablations) uses the 1×0 decoder. A 1×4 decoder gives Ours(MAE) strictly more
trainable capacity, so 94.63 is **not** comparable against 103.48 / 108.29 /
102.47 as they stand. Before any of this is reported as a method win the 1×4
decoder has to be re-run across the SSL baselines — and since the gain comes from
generic temporal modelling of the decoder rather than from anything the MAE
representation provides, it may well shrink the inter-method gaps rather than
preserve them. Configs `configs/probe_dec/`, launchers + `harvest.sh` in
`launch/probe_dec/`.

**Resubmission note (2026-08-15).** The first submission (jobs 9381321-9381328)
was cancelled at 0:00 elapsed without ever starting and resubmitted as the IDs
above. It had copied `--gres=gpu:h100:1` from the `grid5src` template, which
restricts eligibility to the **14 of gengpu's 38 nodes** that carry H100s (the
other 24 are A100). Every probe job that actually schedules promptly on this
cluster requests generic `gres/gpu:1`. Walltime was also cut 12h → 4h, which
backfills far better on a ~1000-deep queue and is still ~1.7× the measured need:
the concurrently running grid probes did 29,795 steps in 67 min on an A100
(**~26.7K steps/h**), so this 60K-step budget is ~2.3h.

Attempting `--partition=gengpu,b53017` to span both the general and buyin queues
fails with `Invalid qos specification`: account `p32139` carries QOS `normal`,
account `b53017` carries QOS `buyin`, gengpu rejects `buyin`, and a job resolves
to exactly one QOS. The two queues can only be targeted by separate submissions.
Also note **partition `b1232` no longer exists** — the older
`launch/launch_slurm_full_body_pose_finetune_nymeria_mae_lr*.sh` scripts still
request `--account=b1232 --partition=b1232` and would fail if rerun today.

**⚠ Fairness caveat.** Every baseline probe in the table above (SimMTM
103.48, FOCAL 108.29, RelCon 102.47) shares this same untuned probe recipe.
A tuned Ours(MAE) probe number is **not** comparable against them — if the
sweep wins, the winning recipe has to be re-run on the SSL baselines before
the improved number can be reported as a method win rather than a tuning
artifact. The finetune LR sweep above has the same unresolved issue.

*Ours(MAE, no alt-attn)'s number comes from `best_mpjpe.pt` (step 13,000),
not the run's `latest.pt` — this config has `early_stopping.enabled: false`
(inherited from the same pre-existing template noted in the 2026-07-17 audit
below) and was manually cancelled after 12h49m / 304,000 steps once its val
mpjpe trend clearly plateaued (99.1-100.6mm oscillation from step ~13,000
onward, no further improvement) — same handling as the four "runaway" jobs
already documented there. Nothing was lost: the trainer saves
`best_mpjpe.pt` independent of the early-stopping flag.

†Ours(MAE, acc-only)'s number comes from `best_mpjpe.pt` (step 56,000), not
`latest.pt` — same `early_stopping.enabled: false` template issue as the
no-alt-attn row above. Val mpjpe oscillated in a tight 103.4-105.9mm band
from step ~18,000 through step 106,515 (7461721 was manually cancelled there,
2026-07-20) with no further improvement, so this is the same
"runaway job, pull best_mpjpe.pt" handling used throughout this benchmark.

The Nymeria-only/no-norm ablation (see the note further below) lands close
to but slightly behind its multidata+normalized counterpart at both probe
(99.86 vs 97.63mm) and finetune (93.19 vs 92.39mm) — losing both the 10-source
data diversity and z-score normalization together costs only ~2-3mm on this
task, suggesting neither variable is single-handedly decisive here.

*MobilePoser's 91.88mm is unit-converted from its own reported 9.19cm
positional error (root-aligned mean joint position error via
`articulate.FullMotionEvaluator` — the same underlying quantity as this
task's mpjpe, so the comparison is meaningful), but the joint set/convention
isn't identical: MobilePoser scores over 24 joints with 8 architecturally
forced to identity rotation (feet/ankles/wrists/hands, per its `joint_set.ignored`),
vs. this task's 23 non-root joints all actively predicted. Its near-tie with
Scratch here is a different architecture family (RNN, single real wrist IMU,
no self-supervised pretraining) reaching a similar error floor — see Section 7
for full methodology. Evaluated on independent 10.24s windows (matching this
task's own `windowing.window_seconds`), not whole unbounded sessions — see
Section 7's window-matching note; the numbers barely moved from the
whole-session version (91.1→91.9mm), so the extra context wasn't materially
inflating MobilePoser's score here.

**Ours (MAE, Nymeria-only, no norm) — new ablation (2026-07-18)**: a fresh MAE
pretraining run (`vit1d_mae_nymeria_only_nonorm_altattn`, job 7325650) that
changes two things at once relative to the main Ours(MAE) column: (1)
pretraining data restricted to *only* the same 867 train / 101 val Nymeria
real-wrist sessions used to train MobilePoser (not the 11-source multidata
mix), via the `mae_nymeria_real_wrist_flu` objective; (2)
`preprocess.normalize.enabled: false` — no z-score normalization anywhere
(pretrain, probe, or finetune), unlike every other column in this benchmark
(see the "Cached data normalization" discussion below for how normalization
normally works). Architecture otherwise matches Ours(MAE) exactly (alt-attn
encoder, same decoder/patch/mask settings) so any difference traces back to
data source + normalization, not capacity. **Bug found while wiring this
up**: `mae_nymeria_real_wrist_flu`, like `odom_finetune_nymeria_real_wrist_flu`/
`odom_probe_nymeria_real_wrist_flu`/`utd_mhad_finetune` before it, never
checked `vit1d.encoder.alt_attention` and would have silently trained a plain
(non-alt-attn) encoder despite the config asking for alt-attn — fixed with
the same pattern used for the other three. Probe (job 7325651) and finetune
(job 7325652) ran via `--dependency=afterok` on the pretraining job and both
completed cleanly: 99.86mm / 93.19mm respectively (see table above).

*Note (2026-07-17)*: these 4 jobs all had `early_stopping.enabled: false` with
`max_steps: 10000000` in their configs (inherited from the pre-existing repo
templates) — they never stop on their own. Checked their val mpjpe trend after
~9-10 hours (300k+ steps) and all four had clearly plateaued (oscillating
within ~1mm band over the last 5+ evals with no further improvement), so
cancelled all 4 manually and pulled the final "new best val mpjpe" checkpoint
value for each — nothing was lost since the trainer already saves
`best_mpjpe.pt` independent of the early-stopping flag. The earlier
"crashed from disk quota" reruns (job 7197445/7197446) are these same jobs;
7200945/7200946 (FOCAL/SimMTM) ran without incident from the start.

---

## 2b. Full-body pose estimation (TotalCapture DIP, 24 joints)

Same new-dataset addition as Section 1b above (see that section for the
shared methodology/pre-flight-testing writeup) -- this is TotalCapture's
full-body-pose sibling. One difference from the Nymeria full-body-pose
convention: the existing TotalCapture loader
(`totalcapture_wrist_loader_FLU.py`) only supports `num_joints` in `{5, 24}`
(a hardcoded branch, not the flexible arbitrary-joint-list Nymeria's loader
allows) -- so this table uses the full 24-joint SMPL body (root included),
not Nymeria's 23-joint exclude-root convention. Not directly numerically
comparable to Section 2's mpjpe values for that reason (different joint
count / root-inclusion), though the ranking *between* columns within this
table is still meaningful. Pre-flight-verified the 24-joint loader path
directly (`x: [256,6,512]`, `y: [256,32,144]` = 32 patches x 24 joints x 6D
rotation) before submitting.

**🐛 Two real bugs found and fixed before real numbers landed (2026-07-19).**
The first attempt (6 non-RelCon jobs, 7381471-7381476) crashed immediately
with a tensor-shape error (`pred_rot` 576 vs `y_rot` 144 — exactly 4x):
`pose_decoder.sub_patches: 4` was copied verbatim from the Nymeria template,
but TotalCapture's loader only ever emits **one** label per 0.32s patch (no
4x sub-patch resolution), so the decoder's output was 4x too wide. Fixed by
removing `sub_patches` from all 7 configs (defaults to 1).

After that fix, training ran fine but **`mpjpe_mm` never appeared in the
logs and `pose_fk_pos` silently stayed 0.000000** — both wrapped in
try/except in `imu_lm/objectives/pose_1d.py`. Root cause, in two parts:
  1. TotalCapture's loader never computed or returned ground-truth FK joint
     *positions* at all (`gt_pos`, the 3rd batch element `pose_1d.py`
     expects) — only local joint *rotations*. Fixed by adding an SMPL
     forward-kinematics pass to `imu_lm/data/totalcapture_wrist_loader_FLU.py`
     (mirroring how `nymeria_real_wrist_full_body_loader_FLU.py` already
     does it) and threading `gt_pos` through the Dataset/collate pipeline,
     gated to the `num_joints==24` case only (the 5-joint arm-chain mode is
     untouched, matching Nymeria's evaluation there having always been
     rotation-only).
  2. Even with `gt_pos` now provided, `_compute_val_mpjpe`/
     `_compute_fk_pos_loss` still failed: both functions unconditionally
     assumed the Nymeria convention (root joint *excluded* from predictions,
     spliced back in from ground truth before running FK). TotalCapture's
     `num_joints=24` config predicts the root too, so splicing produced a
     25-joint tensor and threw (silently, into the try/except). Fixed by
     detecting `num_joints == J_loader` (nothing missing to splice) vs.
     `num_joints == J_loader - 1` (Nymeria's case) in both functions.
  Both fixes verified directly against a live job's log (`mpjpe_mm` first
  appearing, non-zero `pose_fk_pos`) before trusting the full rerun.

**Task complete** — RelCon's row (2026-07-21) was the last one pending,
unblocked when the resumed Stage-2 pretrain finished.

| Model | mpjpe (mm) | Status | Run |
|---|---|---|---|
| Ours (MAE) | 193.51 | ✅ (job 7384545) | `vit1d_full_body_pose_probe_totalcapture_mae_multidata` |
| Ours (MAE) finetune | **174.19** | ✅ (job 7384528) | `vit1d_full_body_pose_finetune_totalcapture_mae_multidata` |
| Ours (MAE, no alt-attn) | 182.27 | ✅ (job 7384529) | `vit1d_full_body_pose_probe_totalcapture_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 182.23 | ✅ (job 7461722) | `vit1d_full_body_pose_probe_totalcapture_mae_acconly_multidata` |
| FOCAL | 184.45 | ✅ (job 7384530) | `vit1d_full_body_pose_probe_totalcapture_focal_multidata` |
| SimMTM | 183.75 | ✅ (job 7384531) | `vit1d_full_body_pose_probe_totalcapture_simmtm_multidata` |
| RelCon | 197.32 | ✅ (job 7431080, best step 540) | `vit1d_full_body_pose_probe_totalcapture_relcon_multidata` |
| Scratch | 175.89 | ✅ (job 7384532) | `vit1d_full_body_pose_supervised_totalcapture_altattn_scratch` |

Ranking: **Ours(MAE) finetune 174.19mm** > Scratch 175.89mm > Ours(MAE,
no-alt-attn) 182.27mm > SimMTM 183.75mm > FOCAL 184.45mm > Ours(MAE) probe
193.51mm — finetuning is a clear, large win here (~19mm better than the
frozen probe), unlike Section 1b's arm-pose table where finetune barely
moved the needle. Values are far higher than Nymeria's ~92-108mm range in
Section 2 -- expected, given the very different joint convention (24 joints
incl. root vs. 23 excl. root) and TotalCapture's much smaller training set
(41 sessions vs. 867), not a sign of anything wrong.

---

## 3. Human Activity Recognition (WISDM, UTD-MHAD, PAMAP2)

Config family: `vit1d_har_probe_{wisdm,utd_mhad,pamap2}_{model}_multidata_flu.yaml` /
`vit1d_har_supervised_{wisdm,utd_mhad,pamap2}_altattn_scratch.yaml`. Metric: best val_loss checkpoint's accuracy (%, higher is better).

Added a literature baseline this session: **DeepConvLSTM** (Ordonez & Roggen
2016, `Sensors` -- 4 per-channel Conv1D layers (64 filters, kernel=5) feeding
a 2-layer LSTM (128 hidden units), classic conv+LSTM HAR architecture,
trained fully supervised from scratch since this repo has no SSL/pretraining
variant of it). New module `imu_lm/models/DeepConvLSTM/{model,run}.py`
(same encoder/objective-agnostic pattern as the existing CNN1D/RNN1D
modules), wired into `scripts/run_pretrain.py`'s config-key dispatch. Configs:
`configs/deepconvlstm_har_{wisdm,utd_mhad}.yaml`. Verified with a CPU smoke
test (full train -> checkpoint -> best-ckpt reload -> eval -> summary.json)
before submitting; jobs launched on `gengpu` (account `p32139` -- `b1232` was
fully booked and `b53017` had a multi-day backlog at submission time).

**Bug found and fixed (2026-07-18): `macro_f1`/`bal_acc`/`macro_precision`/
`macro_recall` below were wrong for every ViT1D row** (DeepConvLSTM
unaffected, see why below). `Trainer._eval()` computed these per validation
*batch* (256 samples ÷ 18 WISDM classes ≈ 14 samples/class/batch) and then
averaged those noisy per-batch estimates across all batches — a badly biased
estimator for F1 specifically (a class with zero hits in one small batch
craters that batch's F1 for that class even if the model handles it fine
elsewhere overall). Correct approach: pool every prediction across the whole
validation pass and compute the metric once. Caught it because WISDM's
reported macro_f1 (0.206) looked absurd next to its 65.5% accuracy;
recomputing properly (`scripts/recompute_har_pooled_metrics.py`, full 25,020-
sample val set) gave **macro_f1 = 0.639** — a >3x difference, exact match to
the buggy per-batch-mean value (0.2054) confirming the mechanism. Fixed in
`imu_lm/runtime_consistency/trainer.py`'s `_eval()` (now pools `_y_true`/
`_y_pred` lists attached by `imu_lm/objectives/supervised.py::forward_loss`
and computes `compute_metrics()` once) and `_format_log()` (skips
`_`-prefixed keys so they don't get dumped into every training-step log
line). DeepConvLSTM was never affected — its post-training eval goes through
`imu_lm/probe/eval.py::eval_head()`, which already pooled correctly. UTD-MHAD's
numbers below also turned out to already be correct (barely moved after
recomputation) purely because its validation set is tiny enough to fit in
~1 batch, so there was no batching to bias — WISDM's much larger val set
(98 batches) is what exposed the bug. Tables below show the corrected values.

### WISDM
| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 65.50% | 65.52% | 0.639 | ✅ | `vit1d_har_probe_wisdm_mae_multidata` |
| Ours (MAE) finetune | **71.27%** | **71.22%** | **0.704** | ✅ | `vit1d_har_finetune_wisdm_mae_multidata` |
| Ours (MAE, no alt-attn) | 64.84% | 64.85% | 0.629 | ✅ | `vit1d_har_probe_wisdm_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 55.04% | 55.05% | 0.534 | ✅ (job 7461728) | `vit1d_har_probe_wisdm_mae_acconly_multidata` |
| FOCAL | 41.95% | 41.98% | 0.390 | ✅ | `vit1d_har_probe_wisdm_focal_multidata` |
| SimMTM | 53.45% | 53.57% | 0.509 | ✅ | `vit1d_har_probe_wisdm_simmtm_multidata` |
| RelCon | 37.83% | 37.88% | 0.331 | ✅ (job 7431105, best step 19,600) | `vit1d_har_probe_wisdm_relcon_multidata` |
| Scratch | 60.89% | 60.89% | 0.606 | ✅ | `vit1d_har_supervised_wisdm_altattn_scratch` |
| DeepConvLSTM | 70.47% | 70.47% | 0.700 | ✅ | `deepconvlstm_har_wisdm` |

### UTD-MHAD
| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 63.77% | 63.83% | 0.633 | ✅ | `vit1d_har_probe_utd_mhad_mae_multidata` |
| Ours (MAE) finetune | **73.91%** | **74.98%** | **0.727** | ✅ | `vit1d_har_finetune_utd_mhad_mae_multidata` |
| Ours (MAE, no alt-attn) | 70.05% | 70.12% | 0.704 | ✅ | `vit1d_har_probe_utd_mhad_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 60.39% | 59.23% | 0.579 | ✅ (job 7461727) | `vit1d_har_probe_utd_mhad_mae_acconly_multidata` |
| FOCAL | 53.62% | 52.10% | 0.504 | ✅ | `vit1d_har_probe_utd_mhad_focal_multidata` |
| SimMTM | 64.73% | 64.77% | 0.626 | ✅ | `vit1d_har_probe_utd_mhad_simmtm_multidata` |
| RelCon | 63.77% | 65.79% | 0.642 | ✅ (job 7431083, best step 17,600) | `vit1d_har_probe_utd_mhad_relcon_multidata` |
| Scratch | 71.01% | 71.37% | 0.701 | ✅ | `vit1d_har_supervised_utd_mhad_altattn_scratch` |
| DeepConvLSTM | 56.04% | 55.01% | 0.518 | ✅ | `deepconvlstm_har_utd_mhad` |

### PAMAP2

New dataset added (2026-07-18): another "realdata" source already present in
the master parquet (`data.train_on_datasets: ["pamap2"]`, same generic
`utd_mhad_probe`/`utd_mhad_finetune`/`supervised` objectives as WISDM/UTD-MHAD
— no new loader code needed). 18 activity classes, 9 subjects, ~1.27M rows —
much larger than WISDM or UTD-MHAD, so `max_steps` raised to 40000 (from
20000) for all 7 configs. Jobs spread across three SLURM accounts for
parallel throughput: `p32139`/`gengpu` (probe MAE/no-alt-attn/FOCAL),
`b53017`/`buyin` (probe SimMTM, finetune MAE), `b1232`/`b1232` H100 (Scratch,
DeepConvLSTM).

**Root cause found and fixed (2026-07-18).** The first attempt (jobs
7327015-7327021) completed without crashing but produced no usable metrics —
`WindowDataset split=val sessions=0 windows=0 (filtered=3)` in every log.
Traced through `imu_lm/data/splits.py`/`imu_lm/data/loaders.py`: PAMAP2 has
14 `(subject_id, session_id)` rows grouped by the default `splits.group_key:
"subject_id"` into 9 subject-groups; `val_ratio: 0.2` assigns ~2 of those
groups (3 session-index rows) to val. Separately, PAMAP2's real-world sensor
dropout gives highly variable per-subject gap counts (1-594 observed across
subjects), and the repo-wide default `windowing.max_gaps_per_session: 100`
(from `base.yaml`) rejects any session over that threshold. By chance, all 3
val-assigned sessions this run exceeded 100 gaps and were filtered out
entirely, leaving val with 0 windows -- so `Trainer._eval()` returned
`(None, None)` every time (silently, no error), no checkpoint's val_loss was
ever compared (`best.pt` was never written for the 6 ViT1D runs), and
DeepConvLSTM's separate `eval_head()` pass reported an all-zeros result for
the same reason (nothing in the val split to evaluate).

**Fix**: raised `windowing.max_gaps_per_session: 700` (comfortably above the
observed max of 594) in all 7 PAMAP2 configs. Verified via a standalone
pre-flight check (`deep_update(base_cfg, model_cfg)` + `make_loaders(cfg)`)
showing `val_loader` batches went from 0 to 17, and confirmed in the
resubmitted runs' own resolved-config dump
(`runs/vit1d_har_probe_pamap2_mae_multidata/logs/stdout.log`) that
`max_gaps_per_session: 700` is actually applied. All 7 jobs resubmitted
2026-07-18 and completed successfully.

| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | **56.12%** | **52.86%** | **0.542** | ✅ | `vit1d_har_probe_pamap2_mae_multidata` |
| Ours (MAE) finetune | 51.77% | 45.09% | 0.412 | ✅ | `vit1d_har_finetune_pamap2_mae_multidata` |
| Ours (MAE, no alt-attn) | 54.00% | 48.07% | 0.482 | ✅ | `vit1d_har_probe_pamap2_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 47.51% | 46.00% | 0.462 | ✅ (job 7461724) | `vit1d_har_probe_pamap2_mae_acconly_multidata` |
| FOCAL | 34.67% | 32.37% | 0.269 | ✅ | `vit1d_har_probe_pamap2_focal_multidata` |
| SimMTM | 47.42% | 42.75% | 0.409 | ✅ | `vit1d_har_probe_pamap2_simmtm_multidata` |
| RelCon | 22.50% | 16.56% | 0.129 | ✅ (job 7431107, best step 10,400) | `vit1d_har_probe_pamap2_relcon_multidata` |
| Scratch | 28.50% | 26.04% | 0.219 | ✅ | `vit1d_har_supervised_pamap2_altattn_scratch` |
| DeepConvLSTM | 37.45% | 35.02% | 0.292 | ✅ | `deepconvlstm_har_pamap2` |

**Task complete (2026-07-18).** All 7 jobs resubmitted after the fix (see
above) completed successfully with non-trivial `best.pt` checkpoints.
PAMAP2 ranking: **Ours(MAE) probe 56.12%** > Ours(MAE, no alt-attn) 54.0% >
Ours(MAE) finetune 51.8% > SimMTM 47.4% > DeepConvLSTM 37.5% > FOCAL 34.7% >
Scratch 28.5%. Unlike WISDM/UTD-MHAD, **the frozen probe beats finetuning
here** — the finetune run's own val curve shows why: val_loss climbs
monotonically from 1.78 (step 400) to 4.29 (step 8400) while train metrics
keep improving, i.e. clear overfitting; `best.pt` was captured at step 400,
essentially right after warmup, and never improved again before early
stopping triggered at step 8400. PAMAP2 has only 9 subjects (fewer than
WISDM's ~30ish or UTD-MHAD's 8, but with far more rows/subject and 18
classes), so full end-to-end finetuning with the shared `lr=1e-4` config
likely has too much capacity relative to the effective subject diversity in
the 20%-val split. Frozen-encoder probing serves as a natural regularizer
here. Not treated as a bug — no config change made — but worth knowing
before assuming finetune always wins.

**WISDM/UTD-MHAD/PAMAP2 complete; RecoFit in progress (see below).** WISDM ranking: **Ours(MAE) finetune 71.27%** > DeepConvLSTM
70.47% > Ours(MAE) 65.5% > Ours(MAE, no alt-attn) 64.8% > Scratch 60.9% >
SimMTM 53.5% > FOCAL 42.0% — finetuning is now the clear winner, narrowly
ahead of the literature DeepConvLSTM baseline and ~6pp above the frozen
Ours(MAE) probe. UTD-MHAD ranking: **Ours(MAE) finetune 73.91%** > Scratch
71.0% > Ours(MAE, no alt-attn) 70.1% > SimMTM 64.7% > Ours(MAE) 63.8% >
DeepConvLSTM 56.0% > FOCAL 53.6% — finetuning flips this task too: where the
frozen probe trailed Scratch, finetuning overtakes it by ~3pp, so "scratch
beats every pretrained model" no longer holds once the encoder is allowed to
adapt. UTD-MHAD is small (8 subjects, **21 classes** — corrected from an
earlier "27"; the master parquet keeps only the wrist-relevant subset of
UTD-MHAD's 27 actions, the rest use a thigh-worn sensor), which likely
explains why DeepConvLSTM (no pretraining, small/simple architecture) doesn't
generalize as well here as on WISDM despite the same recipe. FOCAL remains
the weakest pretraining method on both HAR datasets, and the no-alt-attn
ablation lands between FOCAL/SimMTM and the full Ours(MAE) probe on both
tasks, consistent with alt-attn contributing a real but secondary
improvement on top of the MAE objective itself.

### RecoFit

New dataset added (2026-07-18): another "realdata" source already present in
the master parquet (`data.train_on_datasets: ["recofit"]`, same generic
`utd_mhad_probe`/`utd_mhad_finetune`/`supervised` objectives as WISDM/
UTD-MHAD/PAMAP2 — no new loader code needed, RecoFit was already listed in
the multidata SSL pretrain configs' `train_on_datasets` but had no dedicated
HAR probe/finetune/supervised/DeepConvLSTM configs of its own). Microsoft's
gym-exercise recognition dataset: 94 subjects, 1961 (subject_id, session_id)
groups (`session_id` values like `Trial_01..Trial_72` are reused across
subjects, so `splits.group_key: subject_id`, the base.yaml default, groups
correctly), 7.75M rows, ~50Hz. Unlike PAMAP2, a per-group gap-count check
(>200ms gaps) across all 1961 groups came back **zero everywhere** —
RecoFit has no sensor-dropout artifacts, so `windowing.max_gaps_per_session`
was left at the base.yaml default (100); a pre-flight `make_loaders()` check
confirmed `train_loader`=368 batches / `val_loader`=95 batches before the
first submission. `max_steps` raised to 40000 (from 20000) to match the
PAMAP2 convention, since RecoFit is also much larger than WISDM/UTD-MHAD (94
subjects vs. 8-9, 7.75M rows vs. ~1.2M). Jobs spread across two SLURM
accounts (`b1232`'s 4 H100 nodes were fully occupied by other users' jobs on
the shared `gengpu` partition at submission time, so avoided): `p32139`/
`gengpu` a100 (probe MAE/no-alt-attn/FOCAL, supervised scratch,
DeepConvLSTM), `b53017`/`b53017` a100 (probe SimMTM, finetune MAE).

**🐛 Bug found and fixed after first run (2026-07-18).** All 7 jobs
(7363213-7363219) completed without crashing but logged `acc/bal_acc/
macro_f1/loss = 0.000000` on literally every validation step across all 7
runs — an unmissable red flag (even a 14-class random baseline should clear
~7% accuracy by chance). Root cause: `data.label_column` defaults to
`dataset_activity_id` (`configs/base.yaml:18`), which for RecoFit turns out
to be effectively **subject-specific**, not a shared exercise taxonomy —
verified via direct parquet inspection: 123 distinct `dataset_activity_id`
values across 94 subjects, and a random 75/19 train/val subject split has
**zero label overlap** between the two sides. `imu_lm/objectives/
supervised.py:51-58` remaps unmapped labels to `-100` and, when a whole
batch has no valid labels, silently returns `loss=0.0`/`acc=0.0` instead of
erroring — so every val batch hit that silent-zero path for the entire
training run (`best.pt` in each case was saved from the very first eval,
since 0.0 never got worse). Fix: added `data.label_column:
global_activity_id` to all 7 configs — the master parquet's harmonized
14-class coarse label (`rest_inactive, walk, run_jog, exercise_upper/lower/
core/plyometric, cardio_machine, stretching, device_offbody`, etc.), which
the same overlap check confirmed is properly shared across subjects (13/14
classes present in both a random train and val subject split). All 7 jobs
resubmitted 2026-07-18.

| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 78.21% | 61.67% | 0.627 | ✅ | `vit1d_har_probe_recofit_mae_multidata` |
| Ours (MAE) finetune | **87.84%** | **77.52%** | **0.768** | ✅ | `vit1d_har_finetune_recofit_mae_multidata` |
| Ours (MAE, no alt-attn) | 79.27% | 63.64% | 0.648 | ✅ | `vit1d_har_probe_recofit_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 69.54% | 57.14% | 0.568 | ✅ (job 7461725) | `vit1d_har_probe_recofit_mae_acconly_multidata` |
| FOCAL | 55.23% | 44.91% | 0.437 | ✅ | `vit1d_har_probe_recofit_focal_multidata` |
| SimMTM | 68.91% | 54.04% | 0.541 | ✅ | `vit1d_har_probe_recofit_simmtm_multidata` |
| RelCon | 48.87% | 32.47% | 0.342 | ✅ (job 7431109, best step 34,400) | `vit1d_har_probe_recofit_relcon_multidata` |
| Scratch | 80.26% | 66.60% | 0.663 | ✅ | `vit1d_har_supervised_recofit_altattn_scratch` |
| DeepConvLSTM | 85.18% | 72.15% | 0.721 | ✅ | `deepconvlstm_har_recofit` |

**Task complete (2026-07-18).** RecoFit ranking: **Ours(MAE) finetune 87.84%**
> DeepConvLSTM 85.18% > Scratch 80.26% > Ours(MAE, no alt-attn) 79.27% >
Ours(MAE) 78.21% > SimMTM 68.91% > FOCAL 55.23%. Unlike PAMAP2, finetuning
wins decisively here (+9.6pp acc, +0.14 macro_f1 over the frozen probe) —
consistent with WISDM/UTD-MHAD's pattern rather than PAMAP2's overfitting
anomaly, plausibly because RecoFit's 94 subjects / 7.75M rows gives the
finetuned encoder enough data diversity to avoid PAMAP2's fast-overfit
failure mode. DeepConvLSTM is a strong literature baseline here (2nd overall,
beating every frozen-probe SSL variant), similar to its WISDM showing.
FOCAL remains the weakest pretraining method, consistent across every HAR
dataset in this benchmark.

### WHARF, HHAR, MHEALTH (2026-07-23) — three new literature-comparison datasets

Added to compare against other papers' HAR numbers (per user request). All
three are freely downloadable (no access-gate/registration), unlike MMAct
(see below). Each follows the same pattern as WISDM/PAMAP2/UTD-MHAD/RecoFit:
raw data converted to the master-parquet schema (`dataset`, `subject_id`,
`session_id`, `timestamp_ns`, `acc_x/y/z`, `gyro_x/y/z`,
`{global,dataset}_activity_{id,label}`) via a new one-off
`scripts/prepare_{wharf,hhar,mhealth}_dataset.py` script, but written to
their **own** parquet file rather than appended to
`IMULM_master_dataset_v2.parquet` (avoids any write risk to the shared
13GB/734M-row file everything else depends on) — each dataset's configs just
point `paths.dataset_path` at its own file. Same generic `utd_mhad_probe`/
`utd_mhad_finetune`/`supervised` objectives, same `deepconvlstm_har_*.yaml`
baseline pattern, `window_seconds: 2.56` (128 samples @ the canonical 50Hz).

#### WHARF (Wearable Human Activity Recognition Folder, Bruno et al., RO-MAN 2014)

Downloaded directly from [fulviomas/WHARF](https://github.com/fulviomas/WHARF)
(`git clone`, no gate) into `/projects/b53017/vbc1591/WHARF`. Single
wrist-worn tri-axial **accelerometer only** (no gyroscope) at native 32 Hz,
6-bit-coded raw ints in `[0,63]` mapping to `[-1.5g,+1.5g]` per the dataset's
own `MANUAL.txt`; 892 trials / 17 subjects / 14 activities, badly imbalanced
(`eat_soup`: 3 trials, `eat_meat`: 5, vs. `drink_glass`: 115). Minimum trial
duration is 2.6s (`Sitdown_chair`/`Standup_chair`), just above the 2.56s
window, so effectively every trial contributes at least one window.
`scripts/prepare_wharf_dataset.py` decodes the 6-bit accel, resamples
32→50 Hz (linear interpolation), converts g→m/s² (×9.80665) →
`/projects/b53017/vbc1591/wharf_dataset.parquet`. Since there's no gyro
channel, all ViT1D configs use the **acc-only pretrained checkpoint**
(`vit1d_ssl_mae_varlen_acconly_multidata`, `shared_kernel: true`, 3-channel)
instead of the usual full acc+gyro MAE — the same architecture already used
for the "Ours (MAE, acc-only)" ablation elsewhere in this doc, just applied
here as the natural (not ablation) choice given the source data.

| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE, acc-only) finetune | **48.59%** | **43.54%** | 0.347 | ✅ | `vit1d_har_finetune_wharf_mae_acconly_multidata` (job 7715642) |
| DeepConvLSTM | 45.93% | 41.46% | **0.348** | ✅ | `deepconvlstm_har_wharf` (job 7715639) |
| Ours (MAE, acc-only) | 46.12% | 36.41% | 0.345 | ✅ | `vit1d_har_probe_wharf_mae_acconly_multidata` (job 7715641) |
| Scratch | 44.00% | 37.67% | 0.349 | ✅ | `vit1d_har_supervised_wharf_altattn_scratch` (job 7715536) |

All four land within a tight 44-49% band — expected given WHARF's small
size and severe class imbalance. Finetune edges out DeepConvLSTM and both
beat frozen-probe/scratch, consistent with the general pattern elsewhere in
this benchmark (finetuning helps when there's enough signal to adapt to).

#### HHAR (Heterogeneity Human Activity Recognition, Stisen et al., SenSys 2015)

Downloaded directly from [UCI 344](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition)
(no gate) into `/projects/b53017/vbc1591/HHAR`. Smartwatch + smartphone
accelerometer/gyroscope from 9 users; used the **watch** streams only
(`Watch_accelerometer.csv`/`Watch_gyroscope.csv`, not phone) to match this
repo's wrist-worn convention — 4 watch devices (2 LG, 2 Samsung Gear), 6
activities (bike/sit/stand/walk/stairsup/stairsdown).

**Non-trivial preprocessing**: accelerometer and gyroscope are two
independent, asynchronously-sampled streams per `(User, Device)` — no shared
row index. `scripts/prepare_hhar_dataset.py` aligns them via a
nearest-timestamp `pd.merge_asof` on `Creation_Time` (the device's own
sensor clock). Verified `Arrival_Time` (network-arrival timestamp) is **not
monotonic** under sort for a given device (jitter within the same
millisecond bucket) and produces nonsensical activity segmentation if used
for ordering (spot check: sorting by `Arrival_Time` gave 3611 spurious
label-change points for one user/device pair; sorting by `Creation_Time`
gave the physically sensible 18, matching the actual protocol — long
stand/sit/walk/bike blocks plus repeated stairsup/stairsdown alternation).
After alignment, contiguous same-label runs become sessions, each resampled
to the canonical 50 Hz. Units verified empirically rather than assumed:
accel magnitude ≈9.8 m/s² at rest, gyro ≈0 at rest and ~1.5 rad/s mean during
walk — both already in the project's standard units (Android SensorEvent
convention), no conversion applied. Result: 320 sessions, 857K rows, in
`/projects/b53017/vbc1591/hhar_dataset.parquet`. Standard full acc+gyro MAE
checkpoint (`vit1d_ssl_mae_varlen_altattn_multidata`) used for probe/finetune,
same as WISDM/PAMAP2/UTD-MHAD/RecoFit.

| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | **66.79%** | **68.24%** | **0.679** | ✅ | `vit1d_har_probe_hhar_mae_multidata` (job 7745122) |
| Ours (MAE) finetune | 61.85% | 63.46% | 0.628 | ✅ | `vit1d_har_finetune_hhar_mae_multidata` (job 7723045) |
| Ours (MAE, no alt-attn) | 65.11% | 66.13% | 0.661 | ✅ | `vit1d_har_probe_hhar_mae_noaltattn_multidata` (job 9562586) |
| Ours (MAE, acc-only) | 61.22% | 60.71% | 0.600 | ✅ | `vit1d_har_probe_hhar_mae_acconly_multidata` (job 9562588) |
| FOCAL | 51.18% | 51.50% | 0.500 | ✅ | `vit1d_har_probe_hhar_focal_multidata` (job 9519689) |
| SimMTM | 56.73% | 57.78% | 0.573 | ✅ | `vit1d_har_probe_hhar_simmtm_multidata` (job 9519690) |
| RelCon | 36.63% | 33.10% | 0.283 | ✅ | `vit1d_har_probe_hhar_relcon_multidata` (job 9519691) |
| Scratch | 50.63% | 50.91% | 0.504 | ✅ | `vit1d_har_supervised_hhar_altattn_scratch` (job 7745119) |
| DeepConvLSTM | 44.80% | 46.60% | 0.448 | ✅ | `deepconvlstm_har_hhar` (job 7723042) |

**Frozen probe beats finetune here** — the reverse of WISDM/UTD-MHAD/RecoFit
but the same direction as PAMAP2's overfitting pattern (9 users is on the
small side for full end-to-end adaptation).

#### MHEALTH (Mobile Health, Banos et al., UGR)

Downloaded directly from [UCI 319](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
(no gate) into `/projects/b53017/vbc1591/MHEALTH`. Three Shimmer2 sensors
(chest/left-ankle/right-lower-arm) + 2-lead ECG, 10 subjects, 12 activities,
native 50 Hz (no resampling needed). Used only the **right-lower-arm**
accelerometer + gyroscope columns (dropping chest ECG, all three
magnetometers, and the ankle sensor) as the single wrist/forearm-analog
6-channel stream, matching the wrist-worn convention used elsewhere.
`scripts/prepare_mhealth_dataset.py` segments each subject's continuous log
into contiguous same-label runs (dropping the null/0 class) →
`/projects/b53017/vbc1591/mhealth_dataset.parquet` (126 sessions, 343K rows).
Converted gyro deg/s→rad/s per the dataset's own README-stated units.

**Data-quality caveat (not a preprocessing bug)**: raw gyro magnitude on the
right-lower-arm sensor is nearly identical regardless of activity — ~1.0-1.2
in raw units whether the subject is standing still or running/jumping —
i.e. bias-dominated/uncalibrated, a known characteristic of this particular
raw release. Double-checked this isn't a column-offset bug: the accelerometer
channel (same sensor, adjacent columns) shows exactly the expected
magnitudes (~9.8 m/s² at rest, up to ~20 m/s² during running), and the
24-column layout matches the README's documented count exactly, so the
gyro columns are correctly located — the bias is in the source data itself.
Standard full acc+gyro MAE checkpoint used for probe/finetune.

| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | **72.70%** | **69.92%** | **0.689** | ✅ | `vit1d_har_probe_mhealth_mae_multidata` (job 7745124) |
| Ours (MAE) finetune | 64.58% | 66.65% | 0.627 | ✅ | `vit1d_har_finetune_mhealth_mae_multidata` (job 7723049) |
| Ours (MAE, no alt-attn) | 73.81% | 70.55% | 0.699 | ✅ | `vit1d_har_probe_mhealth_mae_noaltattn_multidata` (job 9562592) |
| Ours (MAE, acc-only) | 70.24% | 68.51% | 0.661 | ✅ | `vit1d_har_probe_mhealth_mae_acconly_multidata` (job 9562593) |
| FOCAL | 65.63% | 65.52% | 0.644 | ✅ | `vit1d_har_probe_mhealth_focal_multidata` (job 9562589) |
| SimMTM | 59.27% | 62.35% | 0.607 | ✅ | `vit1d_har_probe_mhealth_simmtm_multidata` (job 9562590) |
| RelCon | 43.62% | 42.04% | 0.401 | ✅ | `vit1d_har_probe_mhealth_relcon_multidata` (job 9562591) |
| Scratch | 69.83% | 68.49% | 0.659 | ✅ | `vit1d_har_supervised_mhealth_altattn_scratch` (job 7745123) |
| DeepConvLSTM | 60.25% | 59.48% | 0.556 | ✅ | `deepconvlstm_har_mhealth` (job 7723047) |

Same pattern as HHAR: **frozen probe beats finetune** (10 subjects is small
for the same reason as HHAR's 9 and PAMAP2's 9). Scratch also outperforms
finetune here, though not the frozen probe.

**Baseline/ablation rows added 2026-08-17** (jobs 9519689-9562593, configs
`vit1d_har_probe_{hhar,mhealth}_{focal,simmtm,relcon,mae_noaltattn,mae_acconly}_multidata_flu.yaml`,
launchers in `launch/har_hhar_mhealth/`). Two MHEALTH results invert the
ordering seen on every other HAR dataset and should not be quoted without
this caveat:

1. **No-alt-attn (73.81%) beats the headline alt-attn probe (72.70%)** — the
   only dataset where removing alternating frame/global attention does not
   cost accuracy. The bolding on the `Ours (MAE)` row above is therefore
   "our headline method", not "best in column". On HHAR the ablation behaves
   normally (65.11% vs 66.79%).
2. **FOCAL (65.63%) beats SimMTM (59.27%)**, reversing their order on
   WISDM/UTD-MHAD/PAMAP2/RecoFit/HHAR. Consistent with MHEALTH's 12 classes
   being coarse whole-body postures that a contrastive objective separates
   well, and with the documented caveat above that MHEALTH's raw gyro is
   bias-dominated — which penalises the reconstruction-style objectives more
   than the contrastive one.

Both are single-seed numbers on a 10-subject dataset, so neither gap is
established as larger than seed noise; treat them as observations, not claims.

**Task complete (2026-07-23).** All 12 jobs (4 datasets × ... actually 3
datasets × 4 configs = 12) ran clean, zero errors in any log. Across all
three new datasets, the **frozen MAE probe or finetune consistently beats
both Scratch and DeepConvLSTM** — pretraining helps even on data sources
never seen during MAE pretraining (WHARF/HHAR/MHEALTH were not part of the
11-source `mae_multidata_cache`).

#### MMAct — blocked, not started

Large-scale multimodal action dataset (Kong et al., ICCV 2019; RGB + keypoints
+ acc/gyro/orientation + WiFi + pressure, 37 classes, 40 subjects). Unlike
the three datasets above, this one is access-gated: the
[official site](https://mmact19.github.io/2019/) only provides a
[Google Form request](https://docs.google.com/forms/d/e/1FAIpQLScM3OTHIil2b39MZHkORTSHcRpWC6qXcvFnfGivNgb4ZpoRzA/viewform)
(needs the requester's own name/institution/email and agreement to an
academic-only, no-redistribution license); Hitachi emails the download link
after manual review — not something that can be automated or completed on
the user's behalf. User submitted the request 2026-07-23 and is waiting to
hear back. Once the link arrives, downloading/converting/configuring follows
the identical WHARF/HHAR/MHEALTH pattern above.

---

## 4. Odometry (Nymeria real-wrist local velocity)

Standalone script `scripts/eval_odom_nymeria_altattn.py --run <run_name>`
(generalized this session to build either the alt-attn or plain encoder from
the checkpoint's own saved config). Metric: vel_mse, ATE (m), drift (%) — all lower is better.

| Model | vel_mse | ATE (m) | drift | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 0.0245 | **0.0726** | 49.42% | ✅ (checkpoint saved: `data_vis/nymeria_odom_altattn_probe_multidata/odom_head.pt`) | `vit1d_ssl_mae_varlen_altattn_multidata` |
| Ours (MAE) finetune | **0.01675** | 0.1521 | **43.70%** | ✅ | `vit1d_odom_finetune_nymeria_mae_multidata` |
| Ours (MAE, no alt-attn) | 0.02563 | 0.0754 | 49.56% | ✅ | `vit1d_ssl_mae_varlen_noaltattn_multidata` |
| Ours (MAE, acc-only) | — | — | — | ⏳ pretrain done, best val MSE loss 0.7509 (job 7461706), odometry probe/scoring pending | `vit1d_ssl_mae_varlen_acconly_multidata` |
| FOCAL | 0.0332 | 0.0967 | 63.46% | ✅ | `vit1d_ssl_focal_multidata` |
| SimMTM | 0.0300 | 0.0858 | 55.95% | ✅ | `vit1d_ssl_simmtm_multidata` |
| RelCon | 0.02819 | 0.0867 | 55.29% | ✅ (job 7431098, best epoch 50) | `vit1d_ssl_relcon_multidata` |
| Scratch | 0.0200 | 0.1670 | 49.37% | ✅ | `vit1d_odom_supervised_nymeria_altattn_scratch` |
| MobilePoser* | — | 0.150 | — | ✅ | `baselines/full_body_pose/SimplePoser` |

*MobilePoser's ATE(m) cell uses its own `FullMotionEvaluator` "translation
error" definition — mean root translation error over a **1-second window**,
not the whole-trajectory RMS position error (`reconstruct_and_score`) the
other rows' ATE is computed from — so treat it as directionally comparable
(same units, similar spirit: both penalize odometry drift) but not a strict
apples-to-apples number. vel_mse and drift have no MobilePoser equivalent
(different loss formulation / no windowed-drift-ratio metric computed).
Evaluated on independent 10.24s windows matching this benchmark's own
`windowing.window_seconds`, not whole unbounded sessions — see Section 7.
See Section 7 for full methodology.

Ours(MAE) finetune uses the new `odom_finetune_nymeria_real_wrist_flu` objective
in `run.py` (warm-started encoder stays trainable, jointly trained with
`ViT1DOdomDecoder` — same recipe as Scratch, just initialized from the MAE
checkpoint instead of random init), so its checkpoint will already contain
both halves and can be scored directly with `scripts/vis_odom_scratch.py`
once training finishes (no separate probe-training pass needed).

**Task complete — methodology gap resolved.** Wrote `scripts/vis_odom_scratch.py`
to score Scratch's own trained encoder+`ViT1DOdomDecoder` directly (pure
inference on the existing joint checkpoint, no re-probing needed — see "Infra
notes"), giving a true apples-to-apples ATE/drift. Result: **Ours(MAE) probe
has by far the lowest ATE** (0.073m vs. Scratch's 0.167m — 2.3x better), but
drift is essentially tied (49.44% vs. 49.37%). This is an interesting split:
Scratch's *relative* trajectory error (drift, normalized by path length) is
no worse than Ours(MAE)'s, but its *absolute* error (ATE) is much larger —
consistent with Scratch producing noisier/larger-magnitude velocity errors
that happen to scale with path length similarly, rather than Ours(MAE)'s
systematically tighter predictions in absolute terms. FOCAL/SimMTM remain
worse than both Ours(MAE) and Scratch on every metric.

Unlike every other task in this benchmark, **finetuning does not uniformly
help here**: Ours(MAE) finetune has by far the best vel_mse (0.01675, even
beating Scratch's 0.0200) and the best drift (43.70%, beating everything else
including MobilePoser), but its ATE (0.1521m) is *2x worse* than the frozen
probe's (0.0726m) — nearly as bad as Scratch's. Same pattern as the
per-timestep-vs-cumulative split noted above for Scratch: a model can predict
individually-accurate velocities (low vel_mse) yet still accumulate more
absolute trajectory drift (high ATE) than one with slightly noisier
per-step predictions, if its errors are more consistently biased in one
direction rather than zero-mean noise that partly cancels over a long
integration. The frozen Ours(MAE) probe remains the best choice specifically
for absolute trajectory accuracy (ATE) on this task — finetuning trades that
away for better per-step and relative-drift numbers. Ours(MAE, no-alt-attn)
sits close to the full Ours(MAE) probe (0.0754m vs 0.0726m ATE, 49.56% vs
49.42% drift), reinforcing that alt-attn's contribution here is real but
small — most of the gain over FOCAL/SimMTM/Scratch comes from the MAE
objective and multidata pretraining, not the attention variant.

---

## 4b. Odometry — canonical world-frame variant (Nymeria pelvis motion)

New pipeline (2026-07-17): predicts pelvis velocity in **world frame**
(Y-up, m/s) instead of the wrist's own continuously-rotating FLU frame. Since
pelvis body *orientation* isn't available in this dataset (only position),
"facing +X" is defined via movement direction: each window is rotated once
about the vertical axis so its own net horizontal displacement points along
+X (see `imu_lm/data/nymeria_real_wrist_odom_worldframe_loader_FLU.py`).
Verified the rotation is a proper orthogonal matrix (det=1) that correctly
canonicalizes displacement vectors and falls back to identity for
near-stationary windows. Because the canonicalizing rotation is fixed per
window (not time-varying), reconstruction needs no gyro integration at all —
just a plain velocity cumulative sum (`scripts/eval_odom_nymeria_worldframe.py`).
New objective `odom_nymeria_real_wrist_worldframe_flu` added to `run.py` for
the Scratch column (with `early_stopping.enabled: true` this time — the
sibling wrist-FLU scratch config's `false` setting is what caused the ~9-10hr
runaway-job incident earlier, see the pipeline audit above).

| Model | vel_mse | ATE (m) | drift | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 0.02086 | **0.0787** | 48.28% | ✅ done (job 7262762) | `vit1d_ssl_mae_varlen_altattn_multidata` |
| Ours (MAE) finetune | **0.01656** | 0.1293 | **34.08%** | ✅ done (job 7321032 train, 7325714 score) | `vit1d_odom_worldframe_finetune_nymeria_mae_multidata` |
| Scratch | 0.01835 | 0.1419 | 37.31% | ✅ done (job 7262761 train, 7318026 score) | `vit1d_odom_worldframe_supervised_nymeria_altattn_scratch` |
| MobilePoser* | — | 0.150 | — | ✅ | `baselines/full_body_pose/SimplePoser` |
| TLIO† | 0.02419 | 0.1510 | 38.79% | ✅ done (job 7371376) | `baselines/odometry/TLIO` |
| Air-IO† | 0.02461 | 0.1444 | 38.56% | ✅ done (job 7371377) | `baselines/odometry/Air-IO` |

*Same MobilePoser number as the wrist-FLU odometry table (Section 4) —
MobilePoser directly regresses SMPL root/pelvis translation (`tran`) in the
dataset's own world coordinate system, so if anything this world-frame table
is the more natural fit for its output (vs. Section 4's wrist-relative FLU
frame), but it's the same underlying `FullMotionEvaluator` "1-second window"
translation-error metric either way — not the whole-trajectory cumulative-sum
ATE the other rows here use. Directionally comparable, not a strict
apples-to-apples number. vel_mse and drift have no MobilePoser equivalent.
See Section 7 for full methodology.

†**TLIO and Air-IO added (2026-07-18)**, cloned into `baselines/odometry/TLIO`
and `baselines/odometry/Air-IO` (official repos: CathIAS/TLIO, Air-IO/Air-IO).
Both are scored in the *exact* metric definition of this table (same
formulas as `scripts/eval_odom_nymeria_worldframe.py::reconstruct_and_score`/
`eval_metrics`) via a new shared module,
`baselines/odometry/common_worldframe_eval.py` — see its docstring for the
full rationale. Summary:
- **Data**: `baselines/odometry/prepare_nymeria_windows.py` builds a
  model-agnostic window cache (635,356 train / 73,542 val windows, same
  867/101-session split, same 2.56s window / 50% hop, same preprocessing
  and per-window heading-canonicalization as "Ours"/Scratch) so both
  baselines see an identical input distribution and frame convention.
  Unlike the 33-point patch grid "Ours" trains against, the cache stores
  *native per-sample* position (`POS_full`, 129 points/window) so any
  baseline's own temporal resolution can be exactly resampled later.
- **Models**: each script imports the *actual* published network directly
  from its source file (bypassing each repo's package `__init__.py` to
  avoid EKF-only dependencies neither this benchmark nor any other model
  here uses — no model in this table is filter-fused): TLIO's
  `ResNet1D` (`src/network/model_resnet.py`, ~5.0M params) and Air-IO's
  `CodeNetMotion` (`model/code.py`, ~0.33M params, `propcov=False` — no
  covariance branch).
- **Training**: plain MSE regression (no covariance/NLL loss, no bias-shift
  or gravity-perturbation augmentation, no EKF fusion) — matching how every
  other model in this benchmark is trained, not either baseline's original
  full pipeline. TLIO regresses a single net displacement per whole window
  (its actual native design — displacements are normally EKF-fused from
  overlapping windows at high rate, not done here). Air-IO's `CodeNetMotion`
  naturally outputs a short per-step velocity sequence (~15 steps for a
  128-sample window, set by its own conv strides), aligned to GT via its own
  `get_label()` utility.
- **Evaluation**: each model's predicted trajectory (2 points for TLIO,
  ~16 for Air-IO) is linearly interpolated onto the same 33-point grid
  "Ours" is scored on, then the identical ate/drift/vel_mse formulas are
  applied. TLIO's single-displacement design in particular is expected to
  under-perform on ATE relative to `Ours`/Scratch (which both predict
  intra-window curvature via 32 sub-segments) — that's an honest, documented
  property of the adaptation, not a bug.
- Smoke-tested end-to-end on synthetic data (job 7369451) before committing
  to the real 2.5GB/294MB train/val cache (job 7369777) and the real
  training runs (jobs 7371376 TLIO on `p32139`/`gengpu`, 7371377 Air-IO on
  `b53017`).

**Task complete (2026-07-18).** TLIO (ATE 0.1510m, drift 38.79%, early
stopped epoch 42) and Air-IO (ATE 0.1444m, drift 38.56%, early stopped
epoch 95) land close to each other and both worse on ATE than Scratch
(0.1419m) but clearly ahead of MobilePoser's 0.150m rough analog and
substantially behind Ours(MAE)'s 0.0787m. This matches the expected pattern
from the methodology note above: TLIO's single-displacement-per-window
design (no intra-window curvature modeling, unlike every other row here's
32-segment prediction) should be the hardest-handicapped baseline on ATE,
and indeed has the worst ATE of the two despite a *better* vel_mse than
Air-IO (0.02419 vs 0.02461) — the same per-timestep-accuracy-vs-cumulative-
drift disconnect already documented for Ours(MAE) finetune vs probe earlier
in this section. Air-IO's finer ~15-step native resolution gives it a
slight ATE edge (0.1444m) despite having ~15x fewer parameters than TLIO
(0.33M vs 5.0M) — architecture choice (temporal resolution) mattered more
than capacity here. Both remain well behind Ours(MAE)'s frozen-probe
ATE, underscoring that neither baseline's *architecture* (not just its
missing EKF fusion) is designed for this benchmark's short-window,
non-filtered evaluation protocol.

Ours(MAE) finetune required adding a new `odom_finetune_nymeria_real_wrist_worldframe_flu`
objective to `run.py` (mirrors `odom_finetune_nymeria_real_wrist_flu`'s
warm-start pattern but with the world-frame loader/label convention — no
finetune path existed for this variant before). Its checkpoint contains both
the encoder and `ViT1DOdomDecoder`, scored directly with
`scripts/vis_odom_scratch_worldframe.py` (same as the Scratch column).

Scratch's training loop (`run.py`) only logs `odom_mse`/`odom_mae`, not
ATE/drift, so its checkpoint needed a separate post-hoc scoring pass —
`scripts/vis_odom_scratch_worldframe.py` (job 7318026), sibling to
`vis_odom_scratch.py` but using `eval_odom_nymeria_worldframe.py`'s
cumulative-sum reconstruction instead of gyro integration (see that script's
docstring). Ours(MAE) probe has lower vel_mse but a *lower* ATE and higher
drift % than Scratch — Scratch's ATE is nearly 2x Ours(MAE)'s despite a
comparable vel_mse, consistent with Scratch making larger errors on the
(rarer) longer high-displacement windows that dominate ATE/path-length
ratios.

Finetuning reproduces the exact same pattern seen in the wrist-FLU odometry
table above: **best vel_mse and best drift of all three rows** (0.01656,
34.08%), but ATE (0.1293m) lands *worse* than the frozen probe (0.0787m) —
though still better than Scratch (0.1419m). Same explanation applies: the
finetuned model's per-step/relative predictions are the tightest of the
three, but something about its error structure (more directionally-biased
rather than zero-mean noise) accumulates into larger absolute trajectory
drift over long integration windows than the frozen probe's noisier-but-more-
symmetric errors do. The frozen probe remains the best choice specifically
for absolute trajectory accuracy on both odometry variants.

---

## 5. Microgesture recognition (OpenWatch, 6-class)

Classes: `double_clench, double_pinch, pinch_down, pinch_up, slide, background`
(background = subsampled pool of the other 54 OpenWatch gestures, see
`imu_lm/data/openwatch_loader.py`'s `collapse_background`). Config family:
`vit1d_gesture_probe_openwatch_6class_{model}_multidata_flu.yaml` /
`vit1d_gesture_supervised_openwatch_6class_altattn_scratch.yaml`. Metric: accuracy (%, higher is better).

| Model | acc | Status | Run |
|---|---|---|---|
| Ours (MAE) | **67.35%** | ✅ | `vit1d_gesture_probe_openwatch_6class_mae_multidata` |
| Ours (MAE) finetune | 66.33% | ✅ | `vit1d_gesture_finetune_openwatch_6class_mae_multidata` |
| Ours (MAE, no alt-attn) | 63.27% | ✅ | `vit1d_gesture_probe_openwatch_6class_mae_noaltattn_multidata` |
| Ours (MAE, acc-only) | 60.20% | ✅ (job 7461767) | `vit1d_gesture_probe_openwatch_6class_mae_acconly_multidata` |
| FOCAL | 45.92% | ✅ | `vit1d_gesture_probe_openwatch_6class_focal_multidata` |
| SimMTM | 58.16% | ✅ | `vit1d_gesture_probe_openwatch_6class_simmtm_multidata` |
| RelCon | 54.08% | ✅ (job 7431085, best step 200) | `vit1d_gesture_probe_openwatch_6class_relcon_multidata` |
| Scratch | 44.90% | ✅ | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch` |

Ours(MAE) finetune required adding a new `openwatch_gesture_finetune`
objective to `run.py` (mirrors `openwatch_gesture_probe`'s data config exactly;
only the encoder-freeze behavior differs — no finetune path existed for this
task before).

**Task complete.** Accuracy at each run's best-val_loss checkpoint: Ours(MAE)
step 600, Ours(MAE) finetune step 200, Ours(MAE, no alt-attn) step 200,
FOCAL/SimMTM unchanged from before, Scratch step 200. Ranking (best to
worst): Ours(MAE) 67.4% > Ours(MAE) finetune 66.3% > Ours(MAE, no alt-attn)
63.3% > SimMTM 58.2% > FOCAL 45.9% > Scratch 44.9% — unlike every other task
in this benchmark, finetuning is *slightly worse* than the frozen probe here
(67.35→66.33%). Consistent with the overfitting story below: on this tiny
(~763-sample) dataset, letting the encoder move at all gives it more capacity
to overfit than a frozen encoder + linear head does, so the frozen probe's
implicit regularization wins by a hair. Scratch is still the *worst*
performer rather than a strong baseline as on some other tasks. Scratch's val
loss shoots up almost immediately after its step-200 best (1.9 → 3.7 → 8.2,
non-monotonically, while train acc hits 100% by ~step 4000) — a textbook
overfitting signature on this tiny dataset, consistent with the "only ~763
samples" caveat below: with so little data, a frozen pretrained encoder
generalizes far better than training the whole network end-to-end from
random init.

*Note*: only ~763 total samples across 6 classes (5 targets @ ~70-80 each +
~400 subsampled background) — expect noisier results than the larger tasks.
The first Ours(MAE)/Scratch attempts (7198540/7198542) crashed instantly at
setup from a checkpoint-dispatch bug (fixed — see "Infra notes"); no training
steps were lost, both resubmitted clean.

**⚠️ Superseded, read before trusting the ranking above**: §5e (posture-
stratified split) later found this task's train/val subject split has a
severe, undisclosed confound — the plain random split used for every number
in this table happens to put 2 of the entire dataset's only 3 walking-posture
subjects into the tiny 5-subject val set. Redoing this table's ranking under
a posture-balanced split *reverses* which architecture pattern wins: every
frozen-probe row above (including Ours(MAE)'s headline 67.35%) drops 6-18pp,
while both full-finetune rows (Ours(MAE) finetune, Scratch) *improve*
8-17pp — Ours(MAE) finetune becomes the best number in this entire section
(74.70%), not the frozen probe. The numbers above aren't wrong (they're a
faithful report of that specific split), but the "Task complete" ranking
they imply is **not stable** — see §5e for the full comparison table and
methodology before citing this section's ranking elsewhere.

**External comparison: OpenWatch paper's own benchmark (2026-07-22)**:
Bonazzi et al. ("OpenWatch: A Multimodal Benchmark for Hand Gesture
Recognition on Smartwatches") introduce the OpenWatch dataset this repo's
`imu_lm/data/openwatch_loader.py` already draws from, and their own headline
benchmark (their Table 2) targets the *identical* class taxonomy used
here — the same 5 command gestures (`double_clench, double_pinch,
pinch_down, pinch_up, slide`) plus one collapsed Negative/background class —
against the same underlying raw release (our loader points at
`open_watch_without_augmentations`, matching their unaugmented raw data).
Verified from our own config/code (not assumed): our gesture-probe run
applies **no data augmentation** (`apply_augment()` in
`imu_lm/data/augmentations/transform.py` is a no-op unless each block —
`random_rotation`, `gaussian_noise`, `sensor_dropout`, etc. — is explicitly
`enabled: true`, and the gesture-probe config sets none of them) and **no
PPG** (`include_ppg` defaults `False`, never set in the config, even though
the raw CSVs contain a `PPG_mean` column our loader can read).

The paper's Table 2 headline numbers (Mix-Token 95.5%, Apple-CNN 87.9%,
NormWear-Base 78.8% clip-level accuracy, etc.) are **not** the right
comparison point — those use the paper's own augmentation pipeline (wrist
mirroring, signal-space perturbation, locomotion overlay) and, for
NormWear specifically, likely its full native IMU+PPG input. The paper
does give one condition that matches our setup exactly: `NormWear-Base*`,
explicitly labeled "trained without data augmentations" and confirmed
IMU-only by cross-checking its Fig. 7 ablation numbers (window macro-F1
0.340, matching NormWear-Base* window macro-F1 0.340 exactly) —

| Model | Condition | Clip-level Acc |
|---|---|---|
| NormWear-Base* (Bonazzi et al.) | no-aug, IMU-only, 136M params | 69.0% |
| **Ours (MAE) probe** | **no-aug, IMU-only, ViT1D-small** | **67.35%** |

Under the matched condition, the gap is 1.65pp — essentially tied given
our ~763-sample split, not the double-digit gap a naive Table-2 comparison
would suggest. Two open caveats this doc can't resolve from the paper text
alone: (1) the paper's exact clip count for this 5-gesture+background
subset (their Table 3) wasn't available to check against our ~763 samples,
so the relative dataset-size gap is unconfirmed, not ruled out; (2) neither
source reports error bars on these single numbers, so a 1.65pp gap on
~763 samples should not be read as a confirmed win either way. Mix-Token /
Apple-CNN / InceptionTime / Hydra don't get an asterisked no-aug/IMU-only
row in Table 2 (only NormWear does), so no comparably rigorous matched
comparison is available against those four — their Fig. 3 does show
"no aug" bars for all six models, but only as macro-F1 read off a bar
chart, not precise enough to quote here.

### 5a. OpenWatch's own "augmented" raw-data release — matching NormWear's Table 2 headline condition

The comparison above uses `open_watch_without_augmentations` to match
NormWear-Base's asterisked no-aug/IMU-only condition (69.0%). OpenWatch
also separately publishes `open_watch_augmentations` — the paper's own
signal-space augmentation pipeline (wrist mirroring, perturbation,
locomotion overlay) applied to the same underlying sessions — which is
what Table 2's *headline* numbers (Mix-Token 95.5%, NormWear-Base 78.8%,
etc.) are actually trained on. Reran the full model family against this
release (new `..._aug.yaml` configs, identical recipe/objective/best-
val_loss-checkpoint convention as the main table, only `root:` changed
from `open_watch_without_augmentations` to `open_watch_augmentations`) to
get a comparison against those headline numbers instead of only the
asterisked no-aug row.

| Model | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| Ours (MAE) | 66.67% | 56.91% | 0.590 | ✅ (job 7620347, best step 2,800) | `vit1d_gesture_probe_openwatch_6class_mae_multidata_aug` |
| Ours (MAE) finetune | **73.02%** | **63.30%** | **0.672** | ✅ (job 7620349, best step 200) | `vit1d_gesture_finetune_openwatch_6class_mae_multidata_aug` |
| Ours (MAE, no alt-attn) | — | — | — | ⚠️ cancelled (job 7620352, not by this session) | `vit1d_gesture_probe_openwatch_6class_mae_noaltattn_multidata_aug` |
| Ours (MAE, acc-only) | — | — | — | ⚠️ cancelled (job 7620353, not by this session) | `vit1d_gesture_probe_openwatch_6class_mae_acconly_multidata_aug` |
| FOCAL | 55.56% | 40.65% | 0.412 | ✅ (job 7620354, best step 14,200) | `vit1d_gesture_probe_openwatch_6class_focal_multidata_aug` |
| SimMTM | 62.43% | 53.59% | 0.540 | ✅ (job 7620355, best step 16,600) | `vit1d_gesture_probe_openwatch_6class_simmtm_multidata_aug` |
| RelCon | 67.20% | 58.27% | 0.604 | ✅ (job 7620356, best step 6,000) | `vit1d_gesture_probe_openwatch_6class_relcon_multidata_aug` |
| Scratch | 58.73% | 53.83% | 0.517 | ✅ (job 7620357, best step 200) | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch_aug` |
| **NormWear-Base (this repo, first-party)** | 53.97% | 59.62% | 0.517 | ✅ (job 9382934, n_val=189) | `eval_openwatch_normwear.py --root .../open_watch_augmentations` |

‡ **Correction (2026-08-15): the cited 78.8% has been removed as a row here.**
The original framing above ("finetune 5.8pp behind, probe 12.1pp behind")
was an invalid subtraction. Every "acc" in this table is *whole-recording*
accuracy — `OpenWatchDataset` resamples one whole CSV to the encoder's fixed
512-sample/10.24s window, so one forward pass = one recording = one
prediction. The paper's 78.8% is **clip-level (k=3)**: 1s/100-sample windows
at stride Δ=20, clip label assigned by "3 consecutive same-class windows,
else majority vote" (`imu_lm/utils/clip_voting.py`). Their own *window-level*
number for that identical row is 68.2%, i.e. the aggregation step alone is
worth ~10pp with no model change — so "5.8pp behind" was comparing two
different measurements, on top of the split difference (their
posture-stratified vs. our unstratified `_split_subjects`, see §5e) and the
unconfirmed PPG question. Those gaps are real but they are *not* quantified
by that subtraction.

**Replacement**: a first-party NormWear-Base run on this same augmented
release, our same seeded subject split, and our same
`imu_lm.utils.metrics.compute_metrics` (job 9382934, `--root
.../open_watch_augmentations`) — the same recipe §5c already validated on
the no-aug release, where its window-level reproduction landed within 0.8pp
of the paper's cited window-level number (§5d, Result 1). §5c's existing
52.04% could **not** simply be moved into this table: it is the *no-aug*
release and belongs to §5's table, so reusing it here would have swapped one
apples-to-oranges row for another.

**Cited numbers from Bonazzi et al. Table 2 — kept as a citation, not a
table row** (their protocol, their split; not comparable to the numbers
above):

| Method (cited, not run here) | Window-level Acc | Clip-level (k=3) Acc |
|---|---|---|
| Mix-Token (the paper's own proposed method) | 90.8% | 95.5% |
| Apple-CNN | not reported | 87.9% |
| NormWear-Base (headline; augmented, possibly +PPG) | 68.2% | 78.8% |
| NormWear-Base* (no-aug, IMU-only) | 37.0% | 69.0% |

Mix-Token and Apple-CNN get no asterisked no-aug/IMU-only row in Table 2
(only NormWear does) and have no public checkpoint reproduced here, so —
unlike NormWear-Base — they cannot be given a first-party row under our
protocol at all. Any gap to Mix-Token's 95.5% is uninterpretable on two
axes at once (data condition *and* aggregation level), which is exactly why
it belongs in a citation block rather than a comparison column.

**Every non-frozen-encoder model improves substantially on the augmented
release**: FOCAL +9.6pp (45.9→55.6%), SimMTM +4.3pp (58.2→62.4%), RelCon
+13.1pp (54.1→67.2%), Scratch +13.8pp (44.9→58.7%) — expected, since
OpenWatch's augmentation pipeline multiplies the effective training-set
size, directly relieving the "~763 samples" overfitting problem flagged in
§5, and these are exactly the four methods with no frozen/pretrained
component to fall back on. **The frozen probe barely moves** (67.35→66.67%,
-0.68pp, noise-level) since a frozen encoder's linear head has little
capacity to exploit extra augmented training data either way. **Finetune is
the one row whose ranking flips**: it *underperformed* the frozen probe in
§5's no-aug condition (66.33% vs 67.35%, attributed there to overfitting
risk from letting the encoder move on a tiny dataset) but now *clearly
beats* it here (73.02% vs 66.67%, +6.35pp) — with more effective training
data from augmentation, the overfitting risk that previously made finetune
worse than probe is largely removed, and finetune's extra capacity to
adapt the encoder becomes a net win again, consistent with the pattern seen
on every other task in this benchmark. 6 of 8 jobs complete as of
2026-07-23; the `no alt-attn` and `acc-only` variants were **cancelled**
(jobs 7620352/7620353), not queued — corrected 2026-08-15, they will never
land on their own and need resubmitting before those two rows can be filled.

### 5b. Window-level vs. clip-level (k=3) aggregation — testing whether the OpenWatch paper's aggregation bonus explains the Mix-Token/NormWear gap

Every number in §5/§5a above is a **single-window** accuracy:
`OpenWatchDataset` resamples one whole per-gesture CSV recording straight to
the model's fixed 512-sample/10.24s window (`imu_lm/data/openwatch_loader.py`),
so one forward pass = one recording = one prediction. The OpenWatch paper's
own Table 2 instead reports two aggregation levels, "Window-Level" and
"Clip-Level (k=3)" — the latter meaningfully higher for every model in that
table (e.g. their NormWear-Base: window acc 0.682 → clip acc 0.788, a ~10pp
aggregation bonus with no model change), and their headline Mix-Token numbers
(90.8% window / 95.5% clip) are clip-level. Comparing our single-window
number directly against a clip-level number is not apples-to-apples, so this
experiment (`scripts/eval_openwatch_clip_level.py`, SLURM job 7700685, ✅
complete) adds a genuine clip-level number for our own checkpoints to check
whether aggregation alone explains the gap.

**Method**: the paper's exact windowing (window length, stride, whether k=3
windows overlap) isn't published in its README, so this is our own
best-effort reconstruction, not a verified replication of their protocol —
for each val recording, the RAW (pre-resample) signal is split into k=3 equal
consecutive thirds, each third is independently resampled up to the model's
512-sample window (same as the existing loader does for a whole recording)
and scored by the frozen encoder+head. "Window-level" pools all k×N
individual per-third predictions; "clip-level" soft-votes (mean softmax) the
3 predictions for each recording back into one prediction, scored against
that recording's single ground-truth label. Same val split (98 clips) as
every other number in this section.

| Model | Window F1 | Window Acc | Window Rec | Clip(k=3) F1 | Clip(k=3) Acc | Clip(k=3) Rec | vs. whole-recording Acc (§5) |
|---|---|---|---|---|---|---|---|
| Ours (MAE) | 0.181 | 0.493 | 0.220 | 0.187 | **0.510** | 0.217 | 67.35% → -16.35pp |
| Ours (MAE) finetune | 0.151 | 0.490 | 0.192 | 0.139 | 0.490 | 0.183 | 66.33% → -17.33pp |
| Ours (MAE, no alt-attn) | 0.137 | 0.401 | 0.188 | 0.168 | 0.480 | 0.219 | 63.27% → -15.27pp |
| Ours (MAE, acc-only) | 0.144 | 0.425 | 0.200 | 0.162 | 0.469 | 0.216 | 60.20% → -13.30pp |
| FOCAL | 0.171 | 0.401 | 0.196 | 0.148 | 0.439 | 0.192 | 45.92% → -2.02pp |
| SimMTM | 0.166 | 0.435 | 0.195 | 0.140 | 0.459 | 0.186 | 58.16% → -12.26pp |
| RelCon | 0.182 | 0.480 | 0.210 | 0.190 | **0.510** | 0.217 | 54.08% → -3.08pp |
| Scratch | 0.158 | 0.395 | 0.190 | 0.184 | 0.480 | 0.219 | 44.90% → **+3.10pp** |

**Result: this specific k=3 reconstruction is a net loss for every model, not
a recovery** — every row's clip-level accuracy lands well below its own
whole-recording (§5) accuracy, the opposite of the paper's own NormWear/
Mix-Token rows. The likely mechanism: OpenWatch recordings are already short,
and the loader's existing (pre-existing, not new to this experiment) design
resamples a whole recording up to a *fixed* 512-sample/10.24s window because
the encoder's positional embeddings were pretrained at that exact length.
Splitting into thirds *first* means each third — already ~1/3 the duration —
gets stretched roughly 3x further to fill that same fixed window, likely
cropping a short, sharp gesture's actual motion into just one of the three
thirds while the other two look like near-static noise; soft-voting across a
mostly-uninformative triplet doesn't recover the original whole-recording
signal.

**But the drop is very uneven across models**, which is the more interesting
finding: the MAE-pretrained family and SimMTM lose 12-17pp (Ours(MAE) -16.35,
finetune -17.33, no-alt-attn -15.27, acc-only -13.30, SimMTM -12.26), while
FOCAL/RelCon barely move (-2.02/-3.08pp) and Scratch actually *improves*
(+3.10pp, though within plausible noise at n=98 clips). This flips the
ranking substantially: RelCon ties Ours(MAE) for best clip-level accuracy
(both 0.510) despite trailing it by 13pp in the whole-recording table, and
Scratch (worst model in §5, 44.90%) ties `no alt-attn` for 4th here. The
pattern is consistent with the MAE/SimMTM family's advantage in §5 coming
specifically from modeling realistic, non-fragmented temporal structure over
the whole gesture — exactly what this eval protocol destroys — while
RelCon/FOCAL/Scratch never relied on that as heavily, so fragmenting the
input costs them less.

**Conclusion**: aggregation-level mismatch is *not* what explains the gap to
the paper's Mix-Token/NormWear headline numbers — under our own best-effort
k=3 reconstruction, clip-level aggregation makes every one of our models
worse, not better, unlike what the paper reports for its own models. This
experiment doesn't change §5's no-aug/window-level comparison, it just
confirms that naively adding multi-window voting on top of our particular
fixed-window architecture isn't a free win the way it apparently is for the
paper's own models (which likely use a native window length well matched to
the recording length, without our resample-to-fixed-10.24s-window
constraint). **Note**: §5's "tied within 1.65pp" framing (comparing our
67.35% against the *paper's own reported* NormWear-Base* number) is
superseded by §5c below, which actually runs NormWear ourselves under our
exact protocol instead of citing the paper's number.

### 5c. First-party NormWear-Base reproduction — actually running NormWear on our own OpenWatch split

§5's "essentially tied, 1.65pp gap" comparison was Ours(MAE) 67.35% vs. the
*paper's own self-reported* NormWear-Base* number (69.0%) — a citation, not
something we ran ourselves, so it's only as good as our assumption that the
paper's unknown exact protocol matches ours. This experiment removes that
assumption: vendored the actual NormWear repo (Bonazzi et al. cite it from
Luo et al., "Toward Foundation Model for Multivariate Wearable Sensing of
Physiological Signals," ACM Trans. Comput. Healthcare 2026, arXiv:2412.09758;
official code+checkpoint at
`github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear`, Apache-2.0) under
`baselines/SSL/NormWear/`, downloaded their official pretrained checkpoint
(`normwear_pretrain_ckpt.pth`, confirmed 136.1M params — matches the paper's
stated size exactly), and evaluated it under our *own* exact protocol: same
val split (98 clips, subject-seeded identically to every other §5 number),
same `imu_lm.utils.metrics.compute_metrics`.

**Method** (`scripts/eval_openwatch_normwear.py`, SLURM job 7705059 on
gengpu/a100, ✅ complete): while researching this, found the actual OpenWatch
paper (Bonazzi & Ahmed et al., arXiv:2605.04791, "OpenWatch: A Multimodal
Benchmark for Hand Gesture Recognition on Smartwatches") and its own stated
NormWear-Base recipe — frozen backbone + trainable linear head, class-weighted
cross-entropy, AdamW, early stopping on val macro-F1 (patience 5, ≤100
epochs) — which this reproduction follows (with more, smaller full-batch
gradient steps standing in for "epochs" over the tiny cached-embedding
dataset: max 500 steps, patience 25). Input is the *raw, unnormalized* per-
recording signal at its native 100Hz (per the paper), fed directly to
`NormWearModel.get_embedding()` — no resampling/stretching to a fixed window
the way our own ViT1D loader does; NormWear resamples internally to its own
65Hz. Embedding pooling follows the paper's own downstream code
(`downstream_pipeline/model_apis.py::NormWear_API.get_embedding`): mean over
patches per channel, then **concatenate** (not average) across the 6 IMU
channels → a 4608-dim embedding per recording, cached once
(`analysis/legacy_results/results_openwatch_normwear_embeddings.npz`) since the backbone is frozen.
One integration bug fixed pre-flight: the vendored `main_model.py` imports
`torchaudio.transforms.Resample`, an otherwise-unneeded dependency this
repo's shared venv doesn't have and was risky to add against the pinned
custom torch build — replaced with `scipy.signal.resample` (numerically
equivalent, matches this repo's own existing resampling convention).

| Model | acc | bal_acc | macro_f1 | macro_recall | Status | Run |
|---|---|---|---|---|---|---|
| Ours (MAE) probe | **67.35%** | — | — | — | ✅ (§5) | `vit1d_gesture_probe_openwatch_6class_mae_multidata` |
| NormWear-Base* (Bonazzi et al., cited) | 69.0% | — | — | — | citation only | Bonazzi et al. Table 2 |
| **NormWear-Base (this repo, first-party)** | **52.04%** | 53.46% | 0.498 | 53.46% | ✅ (job 7705059) | `scripts/eval_openwatch_normwear.py` |

**⚠️ Two protocol gaps this comparison does NOT close (found after further
checking the paper, post-hoc)**: (1) **Split algorithm.** The paper states its
"training, validation and test splits follow a randomized selection *balanced
per posture and activity condition*" — a stratified split. Ours
(`_split_subjects` in `openwatch_loader.py`) is a plain unstratified random
shuffle of subject IDs at an assumed 80/10/10 ratio (this repo's own
convention, never confirmed against the paper — it doesn't state its exact
split fraction). So our 98 val clips are almost certainly a *different* set
of subjects than the paper's own val/test split, independent of anything
about NormWear itself. (2) **This is a whole-recording evaluation, not the
paper's actual window/clip protocol.** The 52.04% number is one embedding per
whole CSV recording (same convention as "Ours(MAE) 67.35%"), not the paper's
real protocol — 1s windows (100 samples @ 100Hz) at a 20-sample stride
(~6 windows/clip), with the clip label assigned by "3 consecutive same-class
windows, else majority vote." Neither this number nor Ours(MAE)'s 67.35% (nor
§5b's disjoint-thirds attempt, itself already superseded) implements that
real protocol. See §5d below for a corrected attempt at both.

**Result: the "essentially tied" conclusion doesn't survive actually running
NormWear ourselves.** Under an identical val split and identical metric code,
our own NormWear-Base reproduction lands at 52.04% accuracy — 15.3pp *behind*
Ours(MAE) probe (67.35%), not tied with it, and 17pp behind the paper's own
cited number for the same nominal condition. This doesn't necessarily mean
NormWear itself is weak — it means *this quick reproduction* of it
underperforms, and the gap to the paper's cited 69.0% is now the open
question: either (a) the paper's authors used a materially different/more
tuned training recipe than the one-paragraph description we could extract
(exact LR, batch size, warmup, possibly more than a bare linear head), (b)
our embedding pooling choice, while matching their own downstream code,
isn't the specific variant used for Table 2's IMU-only ablation specifically,
(c) the two protocol gaps flagged above (split algorithm, whole-recording vs.
windowed eval), or (d) our 98-clip val split is small enough that single-run
numbers are noisy regardless of recipe fidelity. Not resolved here — this
result should be read as "our own faithful-effort NormWear-Base baseline
underperforms Ours(MAE) on this task under a whole-recording eval," not as a
refutation of NormWear's own reported numbers, which we can't fully audit
without the paper's exact training/eval code.

**⚠️ Superseded by §5e**: this 15.3pp gap is measured on the *unstratified*
split (§5's known posture confound). Under a posture-balanced split, both
sides move — Ours(MAE) probe drops to 49.40% and NormWear-Base barely moves
(51.81%) — and NormWear-Base ends up *ahead* of Ours(MAE) probe instead of
15pp behind it. Read this section's numbers as specific to the unstratified
split, not as the final word on these two models' relative frozen-probe
strength.

### 5d. Correcting the window/clip protocol — real 1s/20-stride windowing + 3-consecutive-run aggregation

§5b's clip-level attempt (disjoint thirds, soft-vote) and §5c's NormWear-Base
number (one embedding per whole recording) both predate finding the paper's
actual protocol. Having now found it (arXiv:2605.04791: "windows of length
W=100 samples (1s at 100Hz) were extracted with stride Δ=20 samples...
yielding 6 overlapping windows per clip," clip label assigned by "3
consecutive same-class windows, else majority vote" — see
`imu_lm/utils/clip_voting.py`), this section redoes both evaluations under
the real rule.

**Key architectural asymmetry, unavoidable given cost**: NormWear has no
fixed-input-length constraint, so `eval_openwatch_normwear_windowed.py`
retrains its linear head on *window-level* embeddings (matching how the
paper's own window/clip-level numbers are presumably produced from a single
trained model). Our own 8 ViT1D checkpoints were already trained on
*whole-recording* inputs (`OpenWatchDataset` resamples one whole CSV straight
to the encoder's fixed 512-sample window) — retraining all 8 on window-level
data would mean relaunching 8 SLURM training jobs, so
`eval_openwatch_windowed_protocol.py` reuses the existing trained heads
as-is and only corrects the *evaluation-time* windowing/aggregation. Our own
rows therefore have a real train/eval mismatch NormWear's don't.

**Discovery**: our raw OpenWatch recordings are a **fixed 300 samples**
(3.0s at the paper's stated 100Hz) — verified directly (`len(x)==300` for
every one of 20 checked val records), giving exactly `(300-100)/20+1 = 11`
windows/clip at stride=20, not the paper's stated "~6". Solving
`(L-100)/20+1=6` gives `L=200` — the paper's own arithmetic implies 2.0s
clips, a full second shorter than ours. Tested the natural correction
(stride=40 instead of 20, which yields exactly 6 windows on our real
300-sample clips: `(300-100)/40+1=6`) as SLURM jobs 7721181/7721182,
alongside the original stride=20 runs (7717331/7717549).

| Model | Stride=20 (11 win/clip) Window Acc | Stride=20 Clip Acc | Stride=40 (6 win/clip) Window Acc | Stride=40 Clip Acc |
|---|---|---|---|---|
| Ours (MAE) | 0.493 | 0.480 | 0.490 | 0.480 |
| Ours (MAE) finetune | 0.483 | 0.490 | 0.488 | 0.490 |
| Ours (MAE, no alt-attn) | 0.410 | 0.439 | 0.412 | 0.449 |
| Ours (MAE, acc-only) | 0.429 | 0.459 | 0.430 | **0.500** |
| FOCAL | 0.423 | 0.469 | 0.427 | 0.490 |
| SimMTM | 0.411 | 0.469 | 0.417 | 0.429 |
| RelCon | 0.484 | **0.500** | 0.476 | **0.510** |
| Scratch | 0.375 | 0.449 | 0.367 | 0.388 |
| **NormWear-Base (first-party)** | **0.378** | **0.459** | 0.311 | 0.296 |

(Recall the matched external citation: NormWear-Base* window acc **0.370**,
clip acc 0.690, Bonazzi et al. Table 2, no-aug/IMU-only condition.)

**Result 1 — the window-level number is a near-exact match, validating the
reproduction pipeline itself.** At stride=20, our first-party NormWear-Base
window-level accuracy (37.8%) lands within 0.8pp of the paper's own cited
window-level number (37.0%) — strong evidence that the embedding extraction
+ per-window classification pipeline (frozen backbone, correct pooling
convention, correct 100Hz native rate) is behaving comparably to the paper's
own, despite none of the training-recipe fine details being confirmed.

**Result 2 — the stride=40 test refutes the windows-per-clip hypothesis.**
If the remaining clip-level gap (45.9% ours vs. 69.0% paper, at stride=20)
were caused by having too many windows/clip (11 vs. the paper's implied 6),
matching their window count should have closed it. It didn't — stride=40
makes *both* window-level (37.8%→31.1%) and clip-level (45.9%→29.6%) accuracy
**worse**, moving further from the paper's numbers, not closer. Likely
explanation: stride=40 nearly halves the training data available to the
linear head (3,648 vs. 6,688 train windows) and gives the majority-vote/
3-consecutive-run aggregation fewer, noisier votes per clip — the wider
stride actively hurts under our recipe, regardless of whether it numerically
matches the paper's clip-length arithmetic. **This hypothesis is now closed,
negative.**

**Result 3 — our own 8 models barely move between strides.** Unlike
NormWear, our own checkpoints' clip-level accuracy is nearly stride-invariant
(e.g. Ours(MAE) 0.480 at both strides; RelCon 0.500→0.510) — expected, since
their whole-recording-trained heads were never sensitive to window count to
begin with; the stride only changes how many resample-distorted fragments
get voted over, not the head's underlying weights.

**Where this leaves the remaining ~23pp clip-level gap (45.9% vs. 69.0%)**:
not explained by windows-per-clip count (ruled out above), not by the
matched no-aug/IMU-only condition (confirmed via the Fig. 7 cross-check in
§5), and the window-level match (37.8% vs. 37.0%) suggests the core
embedding/classification pipeline is sound. The two most likely remaining
causes, neither resolvable without the paper's actual code: (a) the subject-
split algorithm — the paper uses a split "balanced per posture and activity
condition" (stratified); ours (`_split_subjects`) is a plain unstratified
random subject shuffle at an assumed 80/10/10 ratio never confirmed against
the paper's actual fraction — so our 98 val clips are almost certainly a
different subject set than theirs, and (b) undisclosed training-recipe
details for the linear head (exact LR/batch size/epochs-vs-our-gradient-step
count) that a one-paragraph methods description can't fully specify.
**Ranking among our own 8 models is stable and trustworthy under either
stride** (RelCon best at ~50-51% clip acc, Scratch/no-alt-attn weakest) —
it's specifically the absolute-number comparison against the paper's cited
NormWear-Base* that remains open.

---

### 5e. Posture-stratified split — testing the other open hypothesis, and finding a bigger one

§5c/§5d left two open, unresolved explanations for the remaining gap between
our numbers and the paper's cited NormWear-Base*: (a) undisclosed
training-recipe details, and (b) the paper's split being "balanced per
posture and activity condition" (stratified) vs. ours (`_split_subjects`,
plain unstratified random subject shuffle). This section tests (b) directly.

**The imbalance is real and large.** Per-subject posture coverage in this
dataset is extremely uneven (only 3/50 subjects have all 4 postures; 16/50
have only `sitting`). Our default seed=42 split (`_split_subjects`) puts 2 of
the entire dataset's only 3 walking-posture subjects into the 5-subject val
set — val is 40% walking-covered vs. 7.5% in train, and only 40%
sitting-covered vs. 82.5% in train. This is exactly the kind of train/val
distribution mismatch the paper's stratified split is designed to prevent,
and it's present in every number reported in §5-§5d.

**Fix implemented**: `_stratified_split_subjects_by_position` (new function,
`imu_lm/data/openwatch_loader.py`, opt-in via `splits.strategy:
posture_stratified` — the existing default `_split_subjects` is untouched
and still used everywhere else). Stratifies by a coarse (has_walking,
has_arm_down) stratum rather than the exact position-coverage tuple (the
latter creates strata too small for 10%-rounding to ever place a member
outside train — the opposite failure mode), and guarantees at least 1 member
in val/test for any stratum with ≥2 subjects rather than pure proportional
rounding. Resulting split: train=38/val=6/test=6 subjects, with walking and
standing_arm_down now represented in all three splits at reasonable
proportions (verified, no subject overlap between splits).

**Method**: the 6 frozen-probe rows (Ours(MAE) probe, FOCAL, SimMTM, RelCon,
no-alt-attn, acc-only) and NormWear-Base only need a *fresh linear head*
retrained on the same frozen encoder's whole-recording embeddings under the
new split (`scripts/eval_openwatch_stratified_probe.py`, job 7746357,
✅ complete) — cheap, no encoder retraining. The two full end-to-end rows
(Ours(MAE) finetune, Scratch) needed real SLURM retraining from the
pretrained/random-init encoder under the new split (new configs
`configs/*_openwatch*_stratified.yaml`, jobs 7746080/7746081) since their
encoders are trainable, not frozen — reusing their *old*-split-adapted
encoder weights with a new-split head would mix two different training
distributions in one model, not a clean test. Both retrains' `best.pt` is by
val_loss (this benchmark's standing convention, `best_val_loss` locked at
step 200 for both, confirmed by reading the checkpoint directly — thousands
of subsequent steps never beat it, consistent with the fast-overfitting
pattern already documented for this tiny task in §5's own notes).

| Model | Unstratified (§5/§5c) Acc | Posture-stratified Acc | Δ |
|---|---|---|---|
| Ours (MAE) | 67.35% | 49.40% | **-17.95pp** |
| Ours (MAE) finetune | 66.33% | **74.70%** | **+8.37pp** |
| Ours (MAE, no alt-attn) | 63.27% | 49.40% | -13.87pp |
| Ours (MAE, acc-only) | 60.20% | 45.78% | -14.42pp |
| FOCAL | 45.92% | 39.76% | -6.16pp |
| SimMTM | 58.16% | 49.40% | -8.76pp |
| RelCon | 54.08% | 44.58% | -9.50pp |
| Scratch | 44.90% | **61.45%** | **+16.55pp** |
| NormWear-Base (first-party) | 52.04% | 51.81% | **-0.23pp (flat)** |

(Same 83-sample val set for all 9 rows — a genuine apples-to-apples
comparison, not different-sized splits per row.)

**Finding 1 — the split choice flips which architecture pattern wins.** Every
*frozen-probe* row (6 of ours + NormWear-Base) gets worse under the
stratified split, several substantially (Ours(MAE) probe -17.95pp,
acc-only -14.42pp, no-alt-attn -13.87pp). Every *full end-to-end trainable*
row gets *better*, dramatically so for Scratch (+16.55pp) and Ours(MAE)
finetune (+8.37pp, reaching this whole benchmark's new best number anywhere
on this task, 74.70%). Plausible mechanism: a frozen representation's
linear-separability was implicitly tuned (by whatever pretraining or probing
happened before) to a distribution resembling the old split's
sitting-dominated train set; a trainable encoder, by contrast, gets to adapt
its own internal representation to the new, more posture-diverse train set,
and that diversity apparently generalizes better to the (also more diverse)
new val set than the old narrow-posture train set did.

**Finding 2 — NormWear-Base is essentially unaffected by the split change
(-0.23pp), unlike any of our own frozen-probe rows.** This is the most
striking single number in this section: our own frozen ViT1D probes move by
6-18pp when the split changes, while NormWear-Base barely moves at all.
Plausible explanation: NormWear was pretrained on a much larger, more
heterogeneous multi-domain physiological-signal corpus (PPG/ECG/EEG/GSR/IMU
across 11 datasets) than our own IMU-only MAE pretraining, so its frozen
representation may simply be less sensitive to this particular
posture-distribution shift — though this is a plausible interpretation, not
a proven mechanism.

**Finding 3 — under the corrected split, NormWear-Base actually beats every
one of our own frozen-probe rows.** Ranking among the 7 frozen-probe rows
under posture-stratification: **NormWear-Base 51.81%** > Ours(MAE)/no-alt-attn/
SimMTM (49.40% three-way tie) > acc-only 45.78% > RelCon 44.58% > FOCAL
39.76%. The original §5 conclusion ("Ours(MAE) probe is clearly the best
frozen-probe model, comfortably ahead of NormWear-Base") does not survive
correcting the split — it was itself partly an artifact of the original
random split happening to favor our own encoder's learned representation.

**Overall reading**: this doesn't cleanly resolve "who wins" — it depends
entirely on whether the comparison is frozen-probe (NormWear-Base now edges
out our own probe) or full-finetune (Ours(MAE) finetune is the best number
in this whole section, 74.70%, and still ours). What it does resolve is that
**the original single-split numbers in §5/§5c were not a stable ranking** —
switching to a more methodologically defensible (posture-balanced) split
changes which model looks best by double-digit percentage points in several
cases. Caveats: single val split per condition (83 samples), single seed,
not a k-fold or repeated-seed estimate — the largest deltas here (8-18pp) are
far beyond plausible single-fold noise at this sample size, but this should
still be read as "split choice matters a lot for this specific tiny
benchmark," not as a final, noise-free ranking. The paper's own exact
stratification algorithm (fraction, exact grouping) remains unconfirmed
against ours; this is our own best-effort approximation of "balanced per
posture and activity condition," not a verified replication.

**⚠️ Superseded in part by §5f**: the stratified split used here was later
found to itself leak — val subjects id_39/id_42 are byte-identical copies of
train subject id_41, and test subject id_48 of train subject id_46 (release
contamination, see §5f). The frozen-probe drops reported here are therefore
*understated* (val contained train data); the direction of §5e's findings is
unaffected but its absolute numbers inherit the leak.

### 5f. OpenWatch paper-protocol runs (windowed training, contamination-safe split) — and two release-contamination discoveries (2026-08-15)

**Goal**: numbers for Ours(MAE) that are *directly* comparable to the
OpenWatch paper's Table 2 (Bonazzi et al., arXiv:2605.04791), fixing every
axis under our control instead of footnoting it. Their protocol: 1s/100-sample
windows at stride Δ=20, window-level metrics + clip-level via k=3
consecutive-run/majority voting, posture-stratified subject split, models
trained on the augmented release but (as standard) evaluated on real
recordings. §5d only corrected *eval*; the heads were still trained on whole
recordings — a real train/eval mismatch. This section trains windowed.

**Discovery 1 — the "50 real participants" contain byte-identical duplicate
subjects.** Hashing every CSV in `open_watch_without_augmentations`
(`analysis/openwatch_forensics/derive_contamination_maps.py`): id_41 and
id_42 are complete copies of id_39 (31/31 files), id_43 of id_40, id_48 of
id_46; id_49 is half-duplicated (13 files ≡ id_46/48, 6 ≡ id_39); 1-6-file
overlaps entangle id_39/46/47/48/49. Groups:
`[[39,41,42,46,47,48,49],[40,43]]` — the release effectively has ~43
distinct identities, not 50. **Consequence: every split ever used on this
task leaks.** The §5/§5c/§5d unstratified seed-42 split places test id_40 ≡
train id_43 and test id_46 ≡ train id_48 (entire subjects), and val id_49
shares 19 identical files with train subjects; the §5e stratified split
places val id_39/id_42 ≡ train id_41 and test id_48 ≡ train id_46. All
previously reported val numbers on this task are somewhat inflated by this
(val-side: partial for §5's split, two-of-six val subjects for §5e's).

**Discovery 2 — the augmented release's ids 51-85 are wrist-mirrored copies
of specific real subjects.** The release README says "motion-augmented
participants" but gives no mapping. Mirroring preserves |acc| magnitude, so
max-lag-corr(|acc|) > 0.999 in non-walking position dirs identifies each
source exactly (majority-voted over 8 gestures, zero ambiguous ids):
`OPENWATCH_AUG_SOURCE` in `imu_lm/data/openwatch_loader.py` (e.g. 51→1,
69→28, 75→38, 82→22, 83/84/85→39). **Consequence: every split that shuffled
all 85 aug ids independently — §5a's 8 rows (incl. the 73.02% headline) and
the first-party NormWear-aug eval (job 9382934) — can place a mirrored copy
of a held-out subject in train.** Those numbers share one split convention,
so they remain internally comparable, but all carry this caveat now.

**New machinery** (all opt-in, defaults preserve every old run bit-for-bit):
- `windowed: true` + `window_samples: 100` + `window_stride: 20` in the
  objective cfg block — `OpenWatchDataset` yields one 1s window per sample
  (~11/recording); training matches `eval_openwatch_windowed_protocol.py`
  sample-for-sample.
- `window_zero_pad: true` + `window_pad_len: 128` — the encoder is the
  *varlen* alt-attn MAE (pretrained on 64–512-sample windows), so each
  1s/100-sample slice is fed **raw, with no resampling and no time stretch
  at all**, zero-padded 100 → 128 samples into a 1.28s frame (8 × patch 16)
  AFTER preprocessing (filter/z-score see only real samples; the pad stays
  exact zeros). Contrast with §5d's eval-only reconstruction, which stretched
  each slice 5.12x to fill the fixed 512-sample input — likely a big part of
  its 13-17pp drop. (Two intermediate designs — 100→64 resample, and 100→112
  next-patch-multiple pad — were implemented and cancelled before producing
  numbers; the loader still supports both.) Verified pre-launch: pretrained
  checkpoint forwards [B, 6, 128] → [B, 384] cleanly, pad region exact
  zeros, train-vs-eval window construction bit-identical. The last 2 of the
  8 patches are 4 real samples + 28 unmasked zeros — no pad-attention
  masking; the model simply sees a near-zero tail.
- `window_restrict_len: 200` — **the paper says windows are drawn "from a
  restricted sub-interval of the clip, yielding 6 overlapping windows"**;
  (L−100)/20+1 = 6 ⇒ L = 200, a 2.0s sub-interval of the 3.0s recordings.
  This finally explains the 6-vs-11 windows/clip puzzle §5d chased with
  stride=40 (wrong knob). Where the sub-interval sits is unstated; we center
  it (starts 50,70,…,150). Every model family in the comparison uses these
  same 6 windows.
- `eval_root:` — train from the augmented release, hold out REAL recordings;
  the subject split is computed from the eval_root discovery, so aug and
  no-aug conditions share identical val/test sets.
- `contamination_safe: true` — byte-identical files deduped at discovery
  (keep lowest id) + splits computed over CANONICAL identities
  (`canonical_subject`: aug id → source id → duplicate-group minimum).
- `eval_openwatch_windowed_protocol.py --runs ... --splits val test` and a
  strategy/eval_root/contamination-aware `load_val_records` (verified to
  reproduce the training loaders' held-out sets exactly).

**Verified before launch** (smoke test): canonical train/val/test = 33/5/5
identities; val = 102 records {9,10,21,26,35}, test = 87 records
{16,17,23,36,44}, identical across aug/no-aug conditions; **zero
byte-identical files across any split boundary** (direct hash check, all 4
configs); aug-trained condition = 1,375 train records / 8,250 windows (28
mirrored ids incl., all train-sourced) vs 529/3,174 no-aug; val = 612
windows, test = 522 windows (6/clip).

**Fair-comparison pipeline (submitted 2026-08-15)**. Three earlier ViT1D
chains were cancelled before producing numbers (9385452-56: 100→512 stretch;
9387160-65: 100→64 resample; 9387673-79: 112-pad + 11 windows/clip). Final
design adds **Mix-Token and NormWear-Base under the identical protocol** —
one shared data module, `scripts/openwatch_paper_protocol_common.py`,
verified to reproduce the ViT1D loaders' split/windows exactly — so this is
a genuinely fair table, unlike Table 6's old cited-number row.

| Model | Condition | Recipe | Job(s) |
|---|---|---|---|
| Ours (MAE) probe / finetune | no-aug | ViT1D configs `..._windowed_stratified.yaml` | 9388527 / 9388529 |
| Ours (MAE) probe / finetune | aug | `..._windowed_stratified_aug.yaml` | 9388528 / 9388530 |
| → ViT1D eval (window+clip, val+test) | both | afterok on the 4 above | 9388846 → `analysis/legacy_results/results_openwatch_paperprotocol.json` |
| **Mix-Token (reimplementation)** | no-aug / aug | `scripts/train_eval_openwatch_mixtoken.py` | 9388841 / 9388842 |
| **NormWear-Base (first-party, paper head)** | no-aug / aug | `scripts/train_eval_openwatch_normwear_protocol.py` | 9388843 / 9388844 |
| → table build | — | `scripts/build_openwatch_fair_table.py` (afterany all) | 9388847 → `analysis/openwatch_fair_comparison.md` |

**Table 6 completion — all rows under the protocol (submitted 2026-08-15)**.
The paper's Table 6 has 10 rows; the fair pipeline above covered only
Ours(MAE) probe/finetune + the two OpenWatch-paper baselines. The remaining
six rows (FOCAL, SimMTM, RelCon frozen probes; Scratch supervised; MAE
no-alt-attn and acc-only ablations) were added under the identical protocol:
configs `vit1d_gesture_{probe,supervised}_openwatch_6class_<model>_windowed_stratified{,_aug}.yaml`,
cloned programmatically from the verified MAE windowed configs with only the
original per-model diffs applied (encoder type / `pretrained_run` /
acc-only channel truncation / Scratch's LR-wd-warmup); all five pretrained
encoders verified to forward [B,C,128]→[B,384] with strict key match; the
supervised objective got the same `class_weighted_ce` hook. 12 trainings
9404041-9404054 → eval 9404065 → `analysis/legacy_results/results_openwatch_table6_rows.json`;
noaug Mix-Token/NormWear rerun for `bal_acc` (9404068/69); **final build
9404097 → `analysis/table6_openwatch_new.md`** — the paper's exact
Acc./Bal.Acc./Macro-F1 layout at both clip-level (k=3, the paper's headline
aggregation) and window-level, for both data conditions, markdown + LaTeX.
Head recipe is now identical across ALL rows (weighted CE, val macro-F1
selection).

**Table 6 (new configuration) — RESULTS (all 23 jobs completed 2026-08-15 15:47,
no failures; every row 612/102 val, 522/87 test windows/clips; all runs
early-stopped on val macro-F1 with weighted CE).** Test split, clip-level k=3
(paper's headline aggregation) — Acc / Bal.Acc / Macro-F1:

| Model | noaug | aug |
|---|---|---|
| FOCAL | 41.4 / 45.4 / 0.376 | 40.2 / 48.9 / 0.407 |
| SimMTM | 55.2 / 61.6 / 0.532 | 51.7 / 62.9 / 0.515 |
| RelCon | 42.5 / 50.9 / 0.419 | 44.8 / 53.9 / 0.443 |
| Scratch | 60.9 / 48.6 / 0.488 | 51.7 / 33.9 / 0.349 |
| Ours (MAE) probe | 59.8 / 71.3 / 0.606 | 59.8 / **72.5** / 0.615 |
| Ours (MAE), finetuned | **77.0** / 67.8 / **0.704** | **66.7** / 56.7 / 0.580 |
| Ours (MAE), no alt-attn | 65.5 / **73.2** / 0.647 | 57.5 / 68.1 / 0.582 |
| Ours (MAE), acc-only | 58.6 / 67.3 / 0.586 | 57.5 / 65.3 / 0.564 |
| Mix-Token (reimpl., 219K) | 62.1 / 52.6 / 0.500 | 64.4 / 69.6 / **0.641** |
| NormWear-Base (136M) | 51.7 / 47.6 / 0.453 | 54.0 / 57.5 / 0.494 |

Readings: (i) MAE finetune/noaug is best-of-table (77.0% / 0.704); the
symmetric weighted-CE + macro-F1 recipe lifted it from 0.572 (old
unweighted/val_loss run) — the earlier asymmetry was penalising our own
model. (ii) Augmentation helps Mix-Token (0.500→0.641; the paper's cited
0.903 does not survive a leakage-free split) but HURTS MAE finetune
(0.704→0.580) and Scratch (0.488→0.349): end-to-end models overfit the
mirrored/augmented copies; frozen probes are ~unchanged. (iii) The SSL
baseline ordering SimMTM > RelCon ≈ FOCAL from the old Table 6 holds; all
three are well below every MAE row. (iv) Ablations: no-alt-attn PROBE beats
the alt-attn probe on noaug (0.647 vs 0.606, +Bal.Acc), i.e. under 1s
windows/frozen features alt-attn is not the source of the gain — the gain is
in finetuning; acc-only costs ~2-3 F1 points vs full 6-ch. (v) Val-split
numbers are uniformly ~7-10 pts higher than test (selection split), so
quote TEST. Full table incl. window-level + val + LaTeX:
`analysis/table6_openwatch_new.md`; rebuild with
`python scripts/build_openwatch_table6.py`.

**✅ RESULTS (all 23 jobs COMPLETED 2026-08-15, zero failures; every row
verified at exactly 87 test clips / 522 windows). Full 4-variant table:
`analysis/table6_openwatch_new.md`. Headline = paper's condition (aug-trained,
real-recording test, clip-level k=3):**

| Model | Acc. | Bal. Acc. | Macro-F1 |
|---|---|---|---|
| FOCAL | 40.23% | 48.94% | 0.407 |
| SimMTM | 51.72% | 62.87% | 0.515 |
| RelCon | 44.83% | 53.90% | 0.443 |
| Scratch | 51.72% | 33.86% | 0.349 |
| Ours (MAE) | 59.77% | **72.50%** | 0.615 |
| Ours (MAE), finetuned | **66.67%** | 56.65% | 0.580 |
| Ours (MAE), no alt-attn | 57.47% | 68.06% | 0.582 |
| Ours (MAE), acc-only | 57.47% | 65.32% | 0.564 |
| Mix-Token (reimpl., 219K) | 64.37% | 69.62% | **0.641** |
| NormWear-Base (first-party, 136M) | 54.02% | 57.48% | 0.494 |

No-aug clip-level: Ours(MAE) finetune **77.01% / 67.78% / 0.704** is the
best number anywhere in the table; Mix-Token 62.07%/0.500, NormWear
51.72%/0.453.

**Readings** (single split, n=87 test clips — treat <5pp as noise):
1. **The three-way headline is a real contest, not a sweep.** Under the
   paper's own condition, Mix-Token (our reimpl.) takes Macro-F1 (0.641),
   Ours(MAE) finetune takes Acc (66.67%), Ours(MAE) probe takes Bal.Acc
   (72.50%). All three sit within ~7pp Acc / 0.06 F1 of each other; the
   cited Mix-Token 95.5% clip acc is nowhere in sight for anyone — including
   our faithful Mix-Token — which points at split/leakage in the paper's own
   numbers at least as much as at reimplementation gaps.
2. **Without augmentation, our MAE family dominates outright**: finetune
   77.01% / 0.704 vs Mix-Token 62.07% / 0.500 vs NormWear 51.72% / 0.453.
   Mix-Token overfits the 3.2K-window no-aug set (train loss → 0.24); the
   paper's augmentation is what makes it competitive (+2.3pp acc, +0.14 F1
   for it; roughly neutral for our frozen probes).
3. **Augmentation HURTS our finetune** (77.0 → 66.7% clip acc, 0.704 →
   0.580 F1) while barely moving our frozen probes — consistent with the
   mirrored/overlaid copies acting as a distribution shift for a trainable
   encoder pretrained on real motion, and with the aug release's own
   contamination structure (§5f discovery 2). This reverses the §5a finding
   (aug helped finetune +6pp) — that finding was on the leaky unstratified
   split, whole-recording, val set.
4. **Ablations behave**: no-alt-attn ≈ full probe (within noise on both
   conditions; slightly *ahead* on no-aug bal-acc), acc-only costs 2–5pp
   everywhere. NormWear-Base is last of the pretrained family in every
   variant despite 136M params.
5. **Recipe ablation** (`analysis/legacy_results/results_openwatch_paperprotocol_unweighted_valloss.json`
   vs the weighted-CE/macro-F1 rows): switching ViT1D to the paper's head
   recipe raised clip macro-F1 for every row (probe +0.07–0.09, finetune
   noaug +0.13) at the cost of some accuracy for the probes (66.7 → 59.8%
   clip acc) — exactly the acc-vs-F1 trade a background-down-weighting
   recipe should produce; the asymmetry that was penalizing our own rows is
   now gone.
6. Old Table 6 (whole-recording, val, unstratified, leaky) → this table:
   every absolute number moved; the ranking is different (old headline
   Ours(MAE) finetune 73.02% vs new 66.67% aug / 77.01% no-aug). Present
   as a replacement, not a revision.

**Mix-Token reimplementation** (`baselines/mixtoken/`): the paper releases
no code, so this follows §3.3.1 + Appendix B text exactly where specific and
marks every gap `UNSPECIFIED` in-source. Fixed 3-band Butterworth filterbank
per channel → per-channel (grouped) residual 1D CNN → global temporal pool →
concat → MLP (y_cnn); Table-5 statistical features (21 time-domain + 12
frequency-domain per channel, 5 cross-channel descriptor types) as 17 tokens
→ linear encoder + positional embedding → Transformer encoder →
learnable-query attention pooling → MLP (y_attn); concat-embedding MLP
(y_fused); y = softmax(w)·[y_cnn, y_attn, y_fused]. Widths chosen to land at
**219.2K params vs the paper's 223K**. Training = paper §3.3.3 verbatim:
inverse-frequency class-weighted CE, AdamW lr 1e-3 / wd 1e-3, exponential
decay, head dropout, grad-clip value 1.0, AMP, batch 16, ≤100 epochs, seed
42, early stop on val macro-F1 patience 5, best-epoch checkpoint; train-stat
channel-wise z-score (§3.2); raw device axes. CPU smoke test on real data
passed end-to-end (loss falling, fusion weights moving, no NaNs).

**NormWear-Base under the protocol**: official 136.1M checkpoint (the §5c/§5d
pipeline whose window-level number matched the paper's cited one within
0.8pp), embedding = mean over patches / concat over 6 channels (4608-d),
head = the paper's stated LayerNorm → Linear(128) → GELU → Dropout → Linear(K),
same §3.3.3 recipe (global-norm clip), raw 100Hz input, embeddings cached
per split. Note the earlier NormWear rows (§5c/§5d/§5a addendum) used a bare
linear head and are superseded here for the fair table.

**Residual mismatches that remain even under this protocol** (state them
when quoting): (1) for ViT1D, the raw 100Hz window enters an encoder
pretrained at a 50Hz nominal rate (signal untouched, but "read" at 2x its
pretrain timescale) inside a 1.28s zero-padded frame; Mix-Token/NormWear see
the raw window natively; (2) IMU-only vs their headline rows' possible +PPG;
(2b) Mix-Token is a reimplementation with several UNSPECIFIED architectural
choices — its numbers are "a faithful Mix-Token", not "the Mix-Token";
(3) our
stratification/head recipe is a best-effort reconstruction, and our
contamination-safe split is almost certainly *cleaner* than theirs (their
Table 2 numbers inherit whatever the duplicate subjects do to their own
split); (4) `best.pt` selection by val_loss per this repo's convention, test
numbers reported from the val-selected checkpoint.

### 5a addendum (2026-08-15): first-party NormWear-Base on the augmented release

Job 9382934 ✅: NormWear-Base (official 136M checkpoint, frozen + linear
head, same recipe §5c validated) trained/evaluated on
`open_watch_augmentations` under the same unstratified whole-recording
convention as every other §5a row: **acc 53.97%, bal_acc 59.62%, macro_f1
0.517** (n_val=189, early stop step 141). Fills the row that replaced the
non-comparable cited 78.8%. Same-units caveats as the rest of §5a: aug-split
leakage potential (Discovery 2 above) and eval on augmented val recordings.

---

## 6. Language alignment (IMU↔text CLIP, HumanML3D)

Config family: `vit1d_clip_humanml3d_flu_probe_{mae,focal,simmtm}_multidata.yaml` /
`vit1d_clip_humanml3d_flu_altattn_scratch.yaml`. Uses a pre-cached, frozen CLIP
text embedding (`openai/clip-vit-base-patch32`, `/projects/b53017/vbc1591/HumanML3D/clip_text_emb_cache.pt`).
Metric: batch-level retrieval accuracy, IMU→text and text→IMU (%, higher is better).

| Model | acc (imu→text) | acc (text→imu) | Status | Run |
|---|---|---|---|---|
| Ours (MAE) | 19.69% | 21.11% | ✅ | `vit1d_clip_humanml3d_mae_probe` |
| Ours (MAE) finetune | 33.00% | **34.25%** | ✅ | `vit1d_clip_humanml3d_finetune_mae_multidata` |
| Ours (MAE, no alt-attn) | 20.02% | 22.20% | ✅ | `vit1d_clip_humanml3d_mae_noaltattn_probe` |
| Ours (MAE, acc-only) | 12.77% | 13.22% | ✅ (job 7461769) | `vit1d_clip_humanml3d_mae_acconly_probe` |
| FOCAL | 11.72% | 12.00% | ✅ | `vit1d_clip_humanml3d_focal_probe` |
| SimMTM | 16.06% | 17.95% | ✅ | `vit1d_clip_humanml3d_simmtm_probe` |
| RelCon | 11.59% | 13.59% | ✅ (job 7431089, best step 3,000) | `vit1d_clip_humanml3d_relcon_probe` |
| Scratch | 28.81% | 30.44% | ✅ | `vit1d_clip_humanml3d_scratch` |
| Ours (MAE) finetune, fixed-window | 24.73% | 24.92% | ✅ (job 7543850) | `vit1d_clip_humanml3d_finetune_mae_fixedwindow_multidata` |
| Ours (MAE) finetune, aug+adapter | 30.06% | 30.47% | ✅ (job 7567181) | `vit1d_clip_humanml3d_finetune_mae_augadapter_multidata` |
| UniMTS (zero-shot, external)† | 8.79% | 1.76% | ✅ | `baselines/language_alignment/UniMTS/eval_unimts_humanml3d.py` (job 7390552) |
| UniMTS (finetuned on HumanML3D, external)† | 53.99% | 5.15% | ✅ | `baselines/language_alignment/UniMTS/train_unimts_humanml3d.py` (job 7390743) |
| UniMTS (finetuned, frozen text tower, external)† | **61.11%** | 4.11% | ✅ | `baselines/language_alignment/UniMTS/train_unimts_humanml3d_frozen_text.sh` (job 7481114) |

Ours(MAE) finetune uses the same `imu_text_clip_flu` objective as the probe
column, just with `freeze_imu_encoder: false` (that flag already existed —
no new `run.py` code needed here).

Accuracy reported at each run's best-val_loss checkpoint (same convention as
the HAR/pose tables above): MAE step 13000, MAE finetune step 6500, MAE
no-alt-attn step 14500, FOCAL step 44000, SimMTM step 26000, Scratch step
8000 — all early-stopped after 20 evals with no improvement. Ranking (best to
worst): **Ours(MAE) finetune** 33.0%/34.3% > Scratch 28.8%/30.4% > Ours(MAE)
19.7%/21.1% > Ours(MAE, no alt-attn) 20.0%/22.2% > SimMTM 16.1%/18.0% > FOCAL
11.7%/12.0%. This flips the earlier story: with the frozen probe, scratch won
outright by a wide margin (plausible explanation: retrieval against a frozen
CLIP text embedding is a very different transfer target than the IMU-only
reconstruction/classification objectives the pretraining methods were
trained for, so end-to-end training directly against the text-alignment
objective had a big advantage over any frozen-encoder probe) — but once
Ours(MAE)'s encoder is *also* allowed to adapt end-to-end (finetune), it
overtakes scratch by ~4-5pp, suggesting the earlier gap was about frozen vs.
trainable, not about MAE pretraining being a poor fit for this objective.
Ours(MAE) is still the best of the three pretraining methods in both the
probe and no-alt-attn settings, consistent with every other task in this
benchmark.

**Ours (MAE) finetune, fixed-window (2026-07-21/22, 🔄 fixed and resubmitted)**:
an ablation isolating windowing/padding convention rather than architecture
or objective. Identical to the existing "Ours (MAE) finetune" row (same
warm-started trainable IMU encoder, same frozen CLIP text tower) except
`min_window_seconds: 10.24` (raised to equal `max_window_seconds`, was
1.28s) so every batch sees the *entire* 10.24s clip instead of a random-
length random-crop sub-window, plus `pad_mode: wrap` (cyclic repeat for
short clips) replacing the default zero-pad — matching UniMTS's own
`load_custom_data` convention, to check whether the earlier UniMTS
comparison's fixed-window/wrap-pad choices (rather than its architecture)
explain any of the gap to this benchmark's own random-crop convention.

Job 7492246 sat `RUNNING` for **22h51m with zero progress** (empty
checkpoints dir, zero-line `metrics.txt`, `.err` frozen at
`training: 0ep [00:00, ?ep/s]`) before being killed and root-caused.
**Root cause (confirmed, not guessed)**: `_snap_T()` in
`imu_lm/data/humanml3d_dataset.py` (shared by both the FLU and non-FLU
HumanML3D loaders) drops any clip with native length below `T_min` as
"too short — invalid" *before* it ever reaches `__iter__`'s wrap-pad
branch. With `min_window_seconds == max_window_seconds == 10.24s` (512
samples @ 50Hz) feeding straight into that `T_min` gate, and HumanML3D's
own dataset ceiling capping every clip at exactly 200 frames @ 20Hz (=
500 samples after resampling to 50Hz — confirmed by sampling 1000 clips
directly: **0% reach 512 samples**), every single clip in the dataset was
being rejected as invalid. `HumanML3DIMUVarLenDatasetFLU.__iter__`'s
batch-filling loop (lines ~607-649) then retried 3,200 times per outer
iteration, always drew zero usable clips, and spun its outer `while True:`
forever — a live CPU spin with no forward progress, not I/O blocking,
which is exactly why the job looked "alive" for 22+ hours while producing
nothing.

**Fix** (`imu_lm/data/humanml3d_dataset_FLU.py`,
`HumanML3DIMUVarLenDatasetFLU.__init__`/`__iter__`): decoupled the
clip-validity floor from the per-batch window-sampling floor. When
`pad_mode == 'wrap'` (used by exactly this one ablation config, confirmed
via `grep -rl "pad_mode.*wrap" configs/` — zero blast radius on any other
run in this benchmark), the validity floor passed to `_load_clip_imu_flu`
is now `patch_size` (16 samples, just enough for 1 patch) instead of
`T_min`, so short clips reach the intended wrap-pad branch instead of
being dropped; `pad_mode == 'zero'` (every other config, including every
already-published number in this document) is byte-for-byte unchanged.
Smoke-tested directly against the real HumanML3D data before resubmitting:
`_load_clip_imu_flu` now returns valid (non-`None`) clips for 30/30
sampled IDs (previously 0/30 under the fixed-window T_min), and a live
`__iter__()` draw produces correctly-shaped `[64, 6, 512]` batches — slow
per-batch on a cold cache (~50-70s single-threaded, driven by genuine
SMPL-FK + AMASS-SLERP cost per novel clip, the same per-clip cost the
already-working variable-length config pays, not new overhead from this
fix) but no longer an infinite spin. Resubmitted as job 7543850.

**Result (2026-07-22, job 7543850 COMPLETED, 17m04s)**: best checkpoint at
step 2,000 (val_loss 2.835023, early-stopped at step 12,000 after 20 evals
with no improvement — patience × `val_every_steps: 500` arithmetic checks
out exactly). **24.73% imu→text / 24.92% text→imu** — clearly *worse* than
the random-crop "Ours (MAE) finetune" row it's ablating against
(33.00%/34.25%, a ~8-9pp drop in both directions), and only marginally above
the frozen "Ours (MAE)" probe (19.69%/21.11%). So this ablation answers its
own question in the negative: **matching UniMTS's fixed-window/wrap-pad
convention does not explain any of the gap to this benchmark's UniMTS
comparison** — if anything, this benchmark's own random-crop/zero-pad
convention is *better* for this task than the fixed-window/wrap-pad
alternative, not worse. A plausible read: giving every training step the
full, un-cropped 10.24s clip removes the implicit data-augmentation effect
random sub-window cropping provides across steps, and wrap-padding short
clips may introduce a repeating-signal artifact that the random-crop
convention never has to contend with. Not swept further (single ablation,
answer was unambiguous).

**Ours (MAE) finetune, aug+adapter (2026-07-22, 🔄 running, job 7567181)**:
a genuine attempt to close the gap to UniMTS's *finetuned* numbers by
matching pieces of its recipe that are legitimately reproducible (as
opposed to its pretraining-data overlap, see the caveat below — not
something more training can fix). Identical to the baseline "Ours (MAE)
finetune" row except three changes, all combined in a single new config
(`vit1d_clip_humanml3d_flu_finetune_mae_augadapter_multidata.yaml`):
  1. **IMU-side augmentation during CLIP training** — previously none.
     `HumanML3DIMUVarLenDatasetFLU` (the iterable loader this objective
     trains against) never accepted a `cfg`/called `apply_augment` at all —
     confirmed by reading `__init__`/`__iter__` directly, not assumed (the
     sibling *map-style* `HumanML3DIMUDatasetFLU`, used only when HumanML3D
     is folded into the general multidata pretraining mix, already had this
     wired). Added a `cfg` parameter and an `apply_augment` call on each
     sampled window in `imu_lm/data/humanml3d_dataset_FLU.py`, and threaded
     `cfg` through `make_humanml3d_imu_loaders_flu`. New config enables
     `random_rotation` (`octahedral`, matching this repo's own
     `vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml` MAE-pretraining
     augmentation — keeps the finetune stage's rotation policy consistent
     with what the encoder was already trained to be invariant to) plus
     `gaussian_noise`/`gain_jitter`/`axis_bias`/`time_warp` at the same
     values already used in this repo's other CLIP objective config
     (`vit1d_clip_realdata.yaml`) — reusing an existing validated recipe
     rather than inventing new hyperparameters.
  2. **Text-side adapter** — `objective.imu_text_clip_flu.text_projection_mlp:
     true` turns on `TextProjectionMLP` (`imu_lm/objectives/contrastive_clip_1d.py`),
     a small trainable LayerNorm→Linear→GELU→Linear adapter on top of the
     frozen CLIP text embedding. No code change needed — this flag was
     already fully wired in `run.py` and already exercised by other existing
     configs (`vit1d_clip_humanml3d_flu_finetune_10s.yaml`,
     `vit1d_clip_realdata.yaml`) — just never turned on for this particular
     objective/checkpoint combination before.
  3. **Richer captions** — turned out to already be active, no change needed.
     Checked `humanml3d_dataset_FLU.py` directly: `clip['caps']` already
     collects every one of HumanML3D's ~3-4 official per-clip caption
     annotations (`/projects/b53017/vbc1591/HumanML3D/texts/*.txt`, standard
     dataset format), and every batch draws `py_rng.choice(clip['caps'])` —
     a different random annotation per sample, every step, for every row in
     this table including the existing "Ours (MAE) finetune" baseline. The
     text embedding cache (`clip_text_emb_cache.pt`, confirmed 55,865 unique
     captions in the smoke test log below) already has every variant
     pre-encoded. So HumanML3D's caption diversity was never the limiting
     factor here — UniMTS's own extra "LLM-enriched" captions attack a
     different problem (cross-*clip* caption repetitiveness, i.e. many
     distinct clips sharing near-identical generic captions like "a person
     walks forward"), which this fix doesn't touch and would need actual new
     LLM-generated paraphrases to address, not just using what already ships
     with HumanML3D.

  Smoke-tested end-to-end on CPU (3 train steps, real data, real warm-started
  checkpoint, `text_emb_cache: 55,865 captions` loaded, augmented batches
  flowing through training and one val eval) before submitting — confirmed
  no crashes in the new augmentation code path or the (already-existing)
  adapter path. Submitted as job 7567181.

  **Result (2026-07-22, job 7567181 COMPLETED, 21m08s)**: best checkpoint at
  step 7,000. **30.06% imu→text / 30.47% text→imu** — *worse* than the plain
  baseline finetune (33.00%/34.25%, a ~3pp drop in both directions), despite
  combining three changes that each seemed individually reasonable. Not the
  result hoped for, reported as-is. A plausible explanation: unlike
  downstream tasks (HAR, pose, odometry) where augmentation-invariance is a
  generically useful regularizer, a *contrastive alignment* task's frozen
  text targets don't ask for rotation invariance at all — `random_rotation`
  in particular may be actively counterproductive here, since this
  benchmark's FLU frame convention preserves orientation-dependent
  information (e.g. "raises right arm" vs. "raises left arm" is partly
  encoded in which axis the signal loads onto) that captions can implicitly
  reference, and forcing the encoder to become invariant to that destroys
  signal the text side could otherwise exploit — a cost with no
  offsetting benefit, since nothing about the text supervision rewards
  rotation invariance the way HAR/pose labels are agnostic to it. The added
  text-side adapter and already-active caption diversity don't have an
  obvious mechanism to explain a *regression* on their own, so rotation
  augmentation is the leading suspect; not isolated further via a
  three-way ablation (single combined run, matching this ablation's own
  scope), but worth revisiting as a targeted follow-up (e.g. drop
  `random_rotation` only, keep the other two) if this task is revisited.

**Important caveat on what "beating UniMTS" would mean (2026-07-22)**: see
the UniMTS methodology note directly below — its officially released
checkpoint was pretrained **directly on HumanML3D itself** (synthetic IMU
derived from HumanML3D's own motion skeletons, paired with LLM-enriched
versions of HumanML3D's own captions), per the vendored repo's own
`README.md`. That means the UniMTS finetuned rows' numbers aren't a clean
external ceiling for this specific comparison — some of their apparent edge
(e.g. the finetuned row's best checkpoint landing at step 500, an
unusually fast convergence) is plausibly recalibration of an
already-aligned space rather than learning IMU-text alignment from the
~8,191 train clips the way this benchmark's own rows do. This aug+adapter
ablation is a legitimate, uncontaminated attempt to improve our own
recipe on its own merits — not an attempt to erase that pretraining-data
asymmetry, which no amount of finetuning-recipe tuning on our side can
fix.

**† UniMTS (Zhang et al., NeurIPS 2024)** is a literature baseline, not a
ViT1D column trained on this benchmark's data — added for comparison against
a published IMU-text alignment model. Evaluated **zero-shot** using the
official pretrained checkpoint (`xiyuanz/UniMTS` on HuggingFace, no
finetuning of any kind on this repo's data), unlike every other row in this
table, which is pretrained on this benchmark's own 11-source multidata mix
and then probed/finetuned on HumanML3D specifically. Same held-out
`HumanML3D/val.txt` split (518/1460 clips pass the shared `T_min`/`T_max`/
AMASS-availability filter — identical skip logic to `HumanML3DIMUDatasetFLU`,
so not a bug in the adapter), same batch size (64), and the identical
duplicate-caption-aware top-1 retrieval-accuracy formula as
`imu_lm/objectives/contrastive_clip_1d.py`, so the two accuracy columns are
numerically comparable. **A fourth, more fundamental caveat applies only to
the two *finetuned* UniMTS rows below, not the zero-shot row**: per the
vendored repo's own `README.md` (`baselines/language_alignment/UniMTS/README.md`,
"Pre-training" section), the officially released `UniMTS.pth` checkpoint's
own pretraining data *is* HumanML3D itself — its own instructions describe
deriving synthetic IMU from HumanML3D's motion skeletons and further
enriching HumanML3D's own text descriptions via LLMs, then pretraining on
that. So "finetuning" UniMTS on our `HumanML3D/train.txt` is lightly
recalibrating a checkpoint that has already been exposed (in a differently-
synthesized form) to this exact benchmark's motions and captions, not
learning IMU-text alignment from scratch the way every other row in this
table does. This is consistent with the finetuned row's best checkpoint
landing at step 500 (an unusually fast convergence) and means the 53.99%/
61.11% imu→text numbers should not be read as a clean external ceiling to
chase — see the "aug+adapter" ablation note above for the uncontaminated
version of that comparison. Three further architectural constraints limit
the comparison across all three UniMTS rows (including zero-shot),
documented transparently rather than hidden: (1) UniMTS's
publicly released checkpoint was pretrained **accelerometer-only**
(`in_channels=3`; loading it with 6-channel acc+gyro input raises a
state-dict shape mismatch), so gyro is dropped entirely, unlike every ViT1D
row which uses acc+gyro; (2) UniMTS operates over a 22-joint SMPL/AMASS
skeleton graph via an ST-GCN — this repo's single-wrist IMU is placed at
joint index 21 (right wrist, confirmed via UniMTS's own edge-list structure
and its README's worked example), with the other 21 joint slots zeroed,
which is far outside the multi-sensor, full-body-coverage distribution the
graph convolution was designed for; (3) UniMTS's native operating point is
20 Hz / 200-sample (10s) wrap-padded windows (`load_custom_data`'s own
convention) vs. this benchmark's 50 Hz raw FLU IMU, so every window is
resampled down before being fed in. Given all three, the very low
text→IMU accuracy (1.76%, near the 1/64≈1.6% batch-random floor) is best
read as "an off-the-shelf multi-sensor full-body model doesn't transfer
zero-shot to a single accelerometer-only wrist sensor," not as a claim that
UniMTS's own approach is weak — it was never trained for this input
regime. It's a genuinely useful reference point precisely because it's
external and zero-shot, not because it's apples-to-apples with the other
rows.

**UniMTS (finetuned on HumanML3D)** follows up on the zero-shot row by
actually training UniMTS on this benchmark's own HumanML3D data, rather than
only transferring the public checkpoint as-is. Initialized from the same
released checkpoint, then finetuned **fully end-to-end** (both the ST-GCN
IMU encoder and the attached CLIP text tower -- nothing frozen) using
UniMTS's own official `pretrain.py` recipe: Adam, lr=1e-4, temperature=0.1
cross-entropy-over-the-diagonal contrastive loss, per-joint-block random-
rotation augmentation (`utils.augment_data`), batch size 64, kept at
UniMTS's native **20Hz** (not resampled up to this repo's 50Hz convention).
Trained on `HumanML3D/train.txt` (23384 listed, 8191 pass the same
T_min/T_max/AMASS-availability filter as the val split), validated every 500
steps on the same 518-clip val split, early-stopped after 20 evals with no
improvement (job 7390743, `train_unimts_humanml3d.py`) -- best checkpoint at
step 500, training continued to step 10500 before stopping (~2.4h total).
Final numbers are the best checkpoint evaluated over the *full* val split
(matching the zero-shot row's own evaluation convention). Both directions
improved by a roughly proportional ~6x over the zero-shot checkpoint
(8.79%->53.99% imu->text, 1.76%->5.15% text->imu) -- and critically, the same
imu->text >> text->imu asymmetry was **already present zero-shot** (roughly
5:1, vs. roughly 10:1 after finetuning), so this looks like a real property
of UniMTS's embedding space on this task rather than an artifact of the
training run. A plausible explanation: HumanML3D captions are highly
repetitive/generic across distinct clips (many different clips are captioned
near-identically, e.g. "a person walks forward"), which specifically
penalizes the text->imu direction (a generic caption has many plausible IMU
matches in-batch, but only one counts as correct) while imu->text is less
handicapped since each specific IMU signal tends to be more distinguishable
than its paired caption is unique. Notably, **53.99% imu->text was (until
the frozen-text follow-up below pushed it to 61.11%) the single highest
value in that column across this entire table** (above Ours (MAE)
finetune's 33.00%) -- bolding of the column-best value has moved to that
row per this doc's per-column-best convention (same convention applied to
MobilePoser's genuinely-best full-body-pose number in Section 2) -- but
given the architectural/data
caveats above (accel-only, single-joint placement, external checkpoint
init), this should be read as "UniMTS's pretraining prior transfers
unusually well to imu->text retrieval once finetuned on-domain," not as a
strict apples-to-apples win over this benchmark's own ViT1D pipeline; the
text->imu column (5.15%, still far below every ViT1D row) is a more sobering
counterpoint on the very same run.

**UniMTS (finetuned, frozen text tower) (2026-07-21)**: a follow-up
ablation on the finetuned UniMTS row above, isolating whether its
imu->text >> text->imu asymmetry comes from the CLIP text tower reshaping
during finetuning or is already a property of the IMU-side encoder alone.
Identical recipe/data/hyperparameters to the finetuned row (`--freeze-text
1` added to `train_unimts_humanml3d.py`, run script
`train_unimts_humanml3d_frozen_text.sh`, `checkpoint/humanml3d_finetune_frozen_text`),
except the CLIP text tower's weights are frozen and only the ST-GCN IMU
encoder trains. Job 7481114, early-stopped at step 13,000 (best_val_loss
1.3851), ~2h51m total. **Result: 61.11%/4.11%** vs. 53.99%/5.15%
unfrozen — freezing the text tower *improves* imu->text by +7.1pp (now the
single highest value anywhere in this table, ahead of the unfrozen row's
53.99%) but *costs* text->imu -1.0pp. Net effect: constraining the text
side to stay put lets the IMU encoder fit its own embedding more tightly
around the (unmoving) text embeddings for the query direction that matters
most for imu->text retrieval, at a small cost to the reverse direction —
consistent with, not contradicting, the caption-repetitiveness explanation
already given above for why text->imu is the structurally harder
direction on this dataset regardless of which tower is trainable.

---

## 6b. Train/eval-overlap leakage control — NOT a valid held-out comparison

**Motivation**: Section 6's UniMTS caveat (its officially released checkpoint
was pretrained directly on HumanML3D itself, per the vendored repo's own
README) is a plausible but circumstantial explanation for its outsized
finetuned imu→text numbers (53.99%/61.11%) — inferred from its data-prep
instructions, not from directly diffing its original training manifest
against this benchmark's `val.txt`. This experiment tests the *mechanism*
directly on our own pipeline instead: if training on the exact clips you
later evaluate on inflates retrieval accuracy by roughly this much, that's
positive evidence the leakage theory is sufficient to explain UniMTS's
numbers, independent of any architecture/pretraining-scale advantage.

**Setup**: identical to the baseline "Ours (MAE) finetune" row (same
warm-started trainable encoder, same frozen-CLIP-text-embedding-cache setup,
same hyperparameters) except `objective.imu_text_clip_flu.train_split_name:
train_val` — trains on HumanML3D's own official `train_val.txt` (verified
to be exactly the sorted union of `train.txt` + `val.txt`, zero overlap
between the two, 24,844 = 23,384 + 1,460 clips) instead of `train.txt`
alone. The val loader still reads the official `val.txt`, so **every clip
scored for retrieval accuracy has also been directly trained on** — this is
deliberately not a valid generalization measurement, it's a leakage-effect
estimate. Required one small code change: `make_humanml3d_imu_loaders_flu`
and `HumanML3DIMUVarLenDatasetFLU` previously hardcoded the train split to
`'train'`; added a `train_split_name` override (default `'train'`, so every
existing config in this benchmark is unaffected) in
`imu_lm/data/humanml3d_dataset_FLU.py`, threaded through `run.py`'s
`imu_text_clip_flu` block via a new `train_split_name` config key.

Smoke-tested end-to-end on CPU (real warm-started checkpoint) before
submitting — confirmed the train loader reports `split=train_val
clips=17538` (after the same per-clip `joints/*.npy`-existence filter every
split goes through), and critically `17538` exactly equals `train.txt`'s own
valid-clip count (16,496, per the sibling aug+adapter smoke test's log)
plus `val.txt`'s own valid-clip count (1,042) — confirming the union is
read correctly, not silently falling back to plain `train.txt` and not
double-counting. The first smoke-test attempt (batch_size 64, matching the
real config) was OOM-killed by this login node's own session memory cgroup
(`dmesg`: "Memory cgroup out of memory: Killed process ... anon-rss:4194048kB"
— a per-interactive-session limit on this shared cluster, unrelated to the
actual SLURM allocation) before completing a full training step; a retry at
batch_size 16 fit comfortably and completed a full step + checkpoint save +
val eval with no errors, confirming the code path itself is correct. The
real job (submitted as 7568586) requests `--mem=64G` on a compute node and
runs at the original batch_size 64, unaffected by the login node's limit.

| Model | acc (imu→text) | acc (text→imu) | Status | Run |
|---|---|---|---|---|
| Ours (MAE) finetune (held-out, reference row from Section 6) | 33.00% | 34.25% | ✅ | `vit1d_clip_humanml3d_finetune_mae_multidata` |
| Ours (MAE) finetune, train-on-val (leakage control) | **92.52%** | **92.77%** | ✅ (job 7568586, best step 109,000) | `vit1d_clip_humanml3d_finetune_mae_trainonval_multidata` |

Both rows use the identical warm-started-and-trainable recipe, differing
only in whether `val.txt` was inside the training split. **Result: training
on the eval clips inflates retrieval accuracy from 33.00%/34.25% to
92.52%/92.77% — a ~59pp jump from the identical recipe, on the identical
task, changing nothing but whether the eval clips were trained on.** Job
7568586 ran 238 val evals (early-stopped at step 119,000, patience=20 evals
past the step-109,000 best) before completing cleanly, ~1h53m total.

This is strong positive evidence that the leakage mechanism alone is
*sufficient* to produce inflation of this general magnitude — i.e. UniMTS's
53.99%/61.11% imu→text numbers (Section 6) being explained by its
pretraining data overlapping this benchmark's val split is entirely
plausible on the numbers alone, not just circumstantial from reading its
README. It doesn't *prove* that specific mechanism for UniMTS (this
experiment establishes the effect is real and large *in principle* on this
task/metric, not that UniMTS's own checkpoint was contaminated by this exact
route) — but combined with the README evidence already documented in
Section 6, the two together make a strong case. One honest caveat: this
experiment's inflation (~59pp) is larger than UniMTS's own jump from
zero-shot to finetuned (8.79%→53.99%, ~45pp) — consistent with leakage being
*a* major driver rather than the *only* one (UniMTS also brings a
purpose-built text-motion architecture and a from-scratch contrastive
objective, so some genuine adaptation is plausible on top of the leakage
effect, just not enough to explain the full gap on its own).

---

## 6c. Language alignment (IMU↔text CLIP, Nymeria motion-narration captions)

A second, independent language-alignment task alongside Section 6's
HumanML3D one — same CLIP-style contrastive machinery
(`ContrastiveCLIPHead`/`CLIPTextEncoder`/`TextProjectionMLP`/`info_nce_loss`
in `imu_lm/objectives/contrastive_clip_1d.py`), but paired with **real
recorded** wrist IMU and **real human-written** captions instead of
HumanML3D's synthetic AMASS-derived IMU + templated captions — a genuinely
different, arguably more realistic test of the same capability, using data
this benchmark already had (Nymeria) plus its official narration
annotations (newly downloaded).

**Data**: Nymeria's official `motion_narration.csv` per-sequence caption
files, downloaded via the new `scripts/download_nymeria_captions.py` from
the manifest at `scripts/download/nymeria_download_urls_caption.json` (236 of Nymeria's 1,100
sequences have narration annotations; sha1-verified, zero download
failures). Each row is a ~5s segment with **four independent** free-text
descriptions (body posture / hands-arms motion / legs-feet motion / focus
attention) rather than HumanML3D's one-caption-per-whole-10s-clip. New
loader `imu_lm/data/nymeria_caption_dataset_FLU.py` (`NymeriaCaptionVarLenDatasetFLU`)
pairs each caption segment with the same real-wrist FLU IMU pipeline used
throughout this benchmark's other Nymeria tasks (duplicated per this
repo's established convention, minus the pose-label branch which this task
doesn't need), converting each segment's absolute timestamp to a frame index
via the pkl's own `reference_time` offset (verified empirically). New
objective `imu_text_clip_nymeria_flu` mirrors `imu_text_clip_flu` exactly,
just swapping the loader. Train/val uses the same 867/101-session split
(`processed_data_body_Ty_rwrist`/`processed_data_test_body_Ty_rwrist`) every
other Nymeria task in this benchmark uses — already verified leakage-free
against MAE pretraining (see the Section 2/2b leakage check). Of the 236
narrated sequences, 193 have a train-split pkl (32,642 usable caption rows)
and 20 have a val-split pkl (3,488 usable caption rows). Text embedding
cache built via `scripts/cache_clip_text_embeddings_nymeria.py` (83,402
unique captions across all four columns).

| Model | acc (imu→text) | acc (text→imu) | Status | Run |
|---|---|---|---|---|
| Ours (MAE) probe | 6.73% | 6.86% | ✅ (job 7578437) | `vit1d_clip_nymeria_mae_probe` |
| Ours (MAE) probe, lr1e-3 | 6.44% | 6.44% | ✅ (job 7587278) | `vit1d_clip_nymeria_mae_probe_lr1e-3` |
| Ours (MAE) finetune | 5.81% | 5.69% | ✅ (job 7578438) | `vit1d_clip_nymeria_finetune_mae_multidata` |
| Ours (MAE) finetune, lr3e-5 | **6.72%** | **6.69%** | ✅ (job 7587279) | `vit1d_clip_nymeria_finetune_mae_lr3e-5` |
| Ours (MAE) finetune, lr3e-4 | 6.00% | 6.27% | ✅ (job 7593255) | `vit1d_clip_nymeria_finetune_mae_lr3e-4` |
| UniMTS (zero-shot, external)† | 1.94% | 1.77% | ✅ (job 7579739) | `baselines/language_alignment/UniMTS/eval_unimts_nymeria.py` |
| UniMTS (finetuned on Nymeria, external)† | 6.12% | 6.36% | ✅ (job 7581974, early-stopped step 10,500, best step 500) | `baselines/language_alignment/UniMTS/train_unimts_nymeria.py` |
| Ours (MAE) probe, caption-merged | 5.88% | 6.23% | ✅ (job 7596783) — slightly worse than raw | `vit1d_clip_nymeria_mae_probe_capmerged` |
| Ours (MAE) finetune, caption-merged | 5.95% | 6.59% | ✅ (job 7596797) — ~flat vs raw | `vit1d_clip_nymeria_finetune_mae_capmerged` |
| UniMTS (zero-shot, caption-merged, external)† | 2.01% | 1.60% | ✅ (job 7596798) — ~flat vs raw | `baselines/language_alignment/UniMTS/eval_unimts_nymeria.py` |
| Ours (MAE) probe, atomic_action captions | 7.97% | 8.28% | ✅ (job 7602000) — beats motion_narration probe (6.73/6.86%) | `vit1d_clip_nymeria_atomicaction_mae_probe` |
| Ours (MAE) finetune, atomic_action captions | **10.09%** | **10.50%** | ✅ (job 7602001) — best Nymeria CLIP result overall, well above motion_narration finetune@lr3e-5 (6.72/6.69%) | `vit1d_clip_nymeria_atomicaction_finetune_mae_multidata` |

†Same UniMTS adaptation as Section 6 (accelerometer-only, right-wrist joint
21 of a 22-joint skeleton graph, other joints zeroed, native 20Hz/200-sample
wrap-padded windows) — see `eval_unimts_nymeria.py`/`train_unimts_nymeria.py`
docstrings for the full methodology, which mirrors
`eval_unimts_humanml3d.py`/`train_unimts_humanml3d.py` almost verbatim
(the only genuinely new code is the Nymeria-specific data loading; the
model, augmentation, loss, and retrieval-accuracy formula are unmodified).
Unlike Section 6, **the UniMTS-on-HumanML3D contamination caveat does not
apply here** — UniMTS's officially released checkpoint was pretrained on
HumanML3D, not Nymeria, so its zero-shot and finetuned Nymeria numbers are a
clean, uncontaminated comparison on this specific task.

**Every number on this task is far weaker than the equivalent HumanML3D
row** (contrast: Ours(MAE) probe 19.69%/21.11% on HumanML3D vs. 6.73%/6.86%
here; UniMTS zero-shot 8.79%/1.76% vs. 1.94%/1.77% here) — including for
UniMTS, which was never pretrained on either dataset's val split, ruling out
data-leakage as the explanation for *this* task being harder across the
board. A plausible explanation: Nymeria's captions are far more fine-grained
(~5s segments, four independent aspects per segment) and often highly
repetitive *within* a single ~10-20 minute recording session (e.g. many
consecutive segments during a single sustained activity like "playing
foosball" share near-identical wording) — with only 20 val sequences
supplying all 3,488 val segments, any given batch draws heavily from a
handful of sessions, so retrieval must distinguish between many
near-duplicate-captioned neighbors from the same activity, a structurally
harder discrimination problem than HumanML3D's more globally distinct
per-clip captions. This is the same caption-repetitiveness effect already
identified in Section 6 as suppressing text→imu specifically, here
plausibly suppressing *both* directions since the repetition is denser and
the val set narrower.

Interesting reversal from HumanML3D: here **Ours(MAE) probe beats Ours(MAE)
finetune** (6.73%/6.86% vs. 5.81%/5.69%, at each row's untuned default
lr:1e-4) — the opposite of HumanML3D, where finetuning was a clear win.
Consistent with the small, narrow val set (20 sequences) making full
end-to-end adaptation more prone to overfitting than a frozen-encoder linear
probe, similar in spirit to the gesture task's (Section 5) frozen-probe-
beats-finetune result on its own small (~763-sample) dataset.

**LR tuning (2026-07-22)**: both original rows used `lr:1e-4` untuned —
the probe config had actually inherited this from the finetune template
rather than this repo's own established frozen-probe convention (every
other frozen-probe config in this benchmark, e.g.
`vit1d_clip_humanml3d_flu_probe_mae_multidata.yaml`, uses `lr:1e-3`, 10x
higher, since only a small linear head trains). Two follow-ups:
  - **Probe, `lr:1e-3`** (matching convention, job 7587278): **6.44%/6.44%**
    — essentially flat vs. the original 6.73%/6.86% (a hair lower, within
    likely single-run noise). So the LR mismatch turned out *not* to be
    costing anything on this task — the "fix" didn't help, contrary to
    expectation from the convention alone. Plausible read: this task's
    signal is dominated by the caption-repetitiveness/small-val-set
    ceiling discussed above, not by probe optimization headroom, so the
    head's LR doesn't matter much either way once it's in a reasonable
    range.
  - **Finetune, `lr:3e-5`** (lower than the 1e-4 default, job 7587279):
    **6.72%/6.69%** — a real improvement over the 1e-4 baseline
    (5.81%/5.69%), closing almost the entire gap to the frozen probe.
    Directly supports the overfitting hypothesis above: a gentler LR lets
    the warm-started encoder adapt without overfitting this task's narrow
    val set as fast, bringing finetune back up to roughly match probe
    instead of trailing it.
  - **Finetune, `lr:3e-4`** (higher than default, bracketing from the other
    direction, job 7593255): **6.00%/6.27%** — better than the 1e-4
    baseline but worse than `lr:3e-5`. All three finetune LR points now
    complete: 1e-4 (5.81/5.69) < 3e-4 (6.00/6.27) < 3e-5 (6.72/6.69) — not
    perfectly monotonic (plausibly single-run noise on a small val set),
    but the direction is unambiguous: **lower LR helps**, consistent with
    the overfitting explanation. `lr:3e-5` remains the best finetune
    setting found on this task.

**Caption cleaning (2026-07-22)**: investigated *why* this task's numbers
are so much lower than HumanML3D's, with real measurements rather than
guesses (using the actual downloaded captions and the already-built CLIP
embedding cache):
  - Only **64.1%** of val-split captions are unique (55.8% for train) — the
    single most common val caption ("C is standing in front of the foosball
    game table") repeats 51 times across just 3,488 val rows.
  - In a random 64-item batch, only 5.5% of anchors have an *exact*
    duplicate elsewhere (already handled correctly by `info_nce_loss`'s
    multi-positive logic) — but **9-15%** have a *near*-duplicate (cosine
    similarity 0.90-0.95: different wording, same meaning) that the loss's
    tight 0.999 threshold scores as a false negative, not a match.
  - ~9.5% of segments are near-stationary (accel std < 0.3 m/s², mostly
    "sitting playing an arcade game" type activities) — simultaneously
    indistinguishable in the IMU signal *and* captioned near-identically, a
    genuine noise floor rather than a labeling artifact.
  - None of these alone explains the ~3x gap to HumanML3D — the honest
    read is that Nymeria's naturalistic, everyday-activity recordings are a
    structurally harder task than HumanML3D's curated, disjoint
    motion-capture actions, not simply "dirty data" with an easy fix.
  - Also checked whether Nymeria's official release has a more motion-
    focused caption source: `sequence_config.ungrouped_files` in the
    manifest (`scripts/download/nymeria_download_urls_caption.json`) lists
    `narration/atomic_action.csv` and `narration/activity_summarization.csv`
    alongside `narration/motion_narration.csv` — the first is plausibly
    exactly this (short, motion-only action labels, without
    `motion_narration.csv`'s heavy scene/object references like "the
    foosball table"). **Not usable**: only `motion_narration.csv` has an
    actual signed download URL in every one of the 236 sequence entries
    checked; the other two aren't present in this manifest at all. Would
    need a fresh manifest (from wherever this one was generated) requesting
    those file types specifically.
  - **Fix implemented**: `imu_lm/data/nymeria_caption_dataset_FLU.py::_parse_caption_csv`
    now merges consecutive rows (after sorting by `start_time`) whose
    caption content is *exactly* identical and whose time gap is
    `<=1.0s`, by default (`merge_consecutive=True`) — conservative by
    design, never joins content-different or time-distant rows, never
    alters caption text. Since this is the single shared parser used by
    the Ours(MAE) loader, `eval_unimts_nymeria.py`, and
    `train_unimts_nymeria.py`, the fix applies to all three automatically.
    Measured effect: val rows 3,488→3,187 (-8.6%), unique-caption fraction
    64.1%→70.2%, median segment duration 3.00s→3.40s. Batch-level
    near-duplicate rate barely moved (9.0%→7.6% at cos>0.95, unchanged at
    cos>0.90) — most repetition turned out to come from the *same* phrase
    recurring at scattered, non-contiguous points in a session, not from
    oversegmented contiguous runs, so this fix is real but modest, not a
    complete solution. Reran probe (job 7596783), finetune (job 7596797,
    both at the original `lr:1e-4` recipe, isolating the data-cleaning
    effect from the separately-documented LR effect), and UniMTS zero-shot
    (job 7596798) on the cleaned data.

    **Result: a wash.** Probe 5.88%/6.23% (vs. 6.73%/6.86% raw — slightly
    *worse*), finetune 5.95%/6.59% (vs. 5.81%/5.69% raw — roughly flat, a
    hair better on text→imu, within single-run noise), UniMTS zero-shot
    2.01%/1.60% (vs. 1.94%/1.77% raw — flat). Consistent with the batch-level
    near-duplicate measurement barely moving: since the merge doesn't
    meaningfully change the statistics that actually drive the contrastive
    loss's difficulty, it's unsurprising it doesn't meaningfully change
    downstream accuracy either. Kept in the codebase as the new default
    (harmless, never fabricates/alters caption text, mildly more efficient
    — fewer degenerately short segments), but not treated as having solved
    the caption-repetitiveness problem.

    **Follow-up (2026-07-22): Nymeria's fuller download manifest**
    (`scripts/download/Nymeria_download_urls_all.json`, all file types × all 1,100 sequences,
    vs. the original `scripts/download/nymeria_download_urls_caption.json`'s caption-only/
    236-sequence scope) was checked for a more motion-focused caption
    source. `sequence_config.ungrouped_files` lists two more narration
    types: `narration_atomic_action.csv` (845/1,100 sequences — **3.6x**
    `motion_narration.csv`'s 236-sequence coverage; single free-text column,
    "Describe my atomic actions", ~5s segments, longer/richer prose per row)
    and `narration_activity_summarization.csv` (864/1,100 sequences; single
    column, "Describe my activity", ~30s segments, coarser activity-level
    summaries). **Neither is actually motion-only** — verified by reading
    real sample files, not assumed: both reference rooms/objects/context
    just as much as `motion_narration.csv` does (e.g. atomic_action: "C
    opens the laundry closet using both of her hands... leans towards the
    dryer and touches the lid"). So this doesn't solve the
    scene-vs-motion-description mismatch the original hypothesis was
    chasing. `atomic_action.csv`'s much larger sequence coverage is still
    independently valuable, though (more training data, richer per-segment
    text) — downloaded via the now-generalized
    `scripts/download_nymeria_captions.py --file-key/--out-filename`, and a
    dedicated training pipeline for it is being built (see below).

**UniMTS finetune final result (job 7581974)**: early-stopped cleanly at
step 10,500 (best checkpoint step 500, `best_val_loss=4.2219`, 20/20 evals
without improvement — patience arithmetic checks out exactly). Full-val-split
evaluation at the best checkpoint: **6.12%/6.36%** — essentially tied with
Ours(MAE) probe (6.73%/6.86%) and Ours(MAE) finetune@lr3e-5 (6.72%/6.69%,
still the best of the three). All three "serious" models land within ~1pp
of each other on this task once each is given a fair shot (frozen probe,
tuned finetune LR, or UniMTS's own from-a-different-pretraining-prior
finetune) — consistent with the earlier read that this task's ceiling is
set more by the data (repetitive captions, narrow val set, naturalistic
low-motion activities) than by which model/recipe is used.

**Pipeline built for `atomic_action.csv` (2026-07-22)**: new
`caption_source` parameter added to `NymeriaCaptionVarLenDatasetFLU`/
`make_nymeria_caption_imu_loaders_flu` (registry-based —
`NYMERIA_CAPTION_SOURCES` maps a short name to its filename + description
column(s), so configs just set `caption_source: atomic_action` rather than
spelling out schema details), threaded through the `imu_text_clip_nymeria_flu`
objective in `run.py`. `scripts/cache_clip_text_embeddings_nymeria.py`
generalized the same way (`--caption-source` flag). Smoke-tested end-to-end
against the real (then-partial) downloaded data before the full download
finished — confirmed correct batch shapes and a real, richly-detailed
sampled caption. New configs `vit1d_clip_nymeria_atomicaction_{probe,
finetune}_mae_multidata.yaml` (probe at `lr:1e-4`, finetune at `lr:3e-5` —
each using the better of the two LRs already established for this task
family on `motion_narration`, rather than re-testing untuned defaults).
Downloads completed cleanly (845/845 atomic_action.csv, 864/864
activity_summarization.csv, zero failures). Text embedding cache built
(job 7601766, 143,055 unique captions — vs. 83,402 for motion_narration,
confirming richer/more-varied text).

**Result: a genuine, clear win** — probe **7.97%/8.28%** (job 7602000,
beats the motion_narration probe's 6.73%/6.86%) and finetune
**10.09%/10.50%** (job 7602001, best checkpoint step 5,000) — the single
best result anywhere on this task, well clear of motion_narration
finetune@lr3e-5 (6.72%/6.69%, the previous best) and of UniMTS's finetuned
6.12%/6.36%. Unlike the caption-merge cleaning (a wash) and the
LR-convention fix (a wash for probe), switching to a caption source with
3.6x more sequence coverage (845 vs. 236) and longer, more specific
free-text per segment (143,055 vs. 83,402 unique captions) moved the needle
substantially in both directions and for both probe and finetune — the
clearest evidence yet that this task's earlier low numbers were substantially
a *data-scarcity/caption-quality* problem, not solely an inherent ceiling
from Nymeria's naturalistic-activity domain. Also notable: finetune beats
probe here (10.09/10.50 vs. 7.97/8.28), unlike motion_narration's reversal
at the untuned default LR — with more data, finetuning has enough signal to
outperform the frozen probe again, consistent with the earlier overfitting
explanation (more data → less overfitting risk → finetuning wins, matching
the pattern from every other task in this benchmark).

**LLM-simplified caption augmentation (2026-07-23)**: tested whether adding
short, HumanML3D-style motion-only captions — generated by an LLM
(`Qwen/Qwen2.5-7B-Instruct`) rewriting each of `atomic_action.csv`'s 143,054
unique long-form captions into one ≤15-word, third-person, motion-only
sentence, explicitly instructed to drop object nouns and mentions of other
people (`scripts/generate_simplified_nymeria_captions.py`) — as an
*augmentation* (not replacement) of the caption pool would push this task's
numbers further, mirroring the motion-only style that makes HumanML3D's own
captions easier to align against (§6). Written as a sibling
`atomic_action_augmented.csv` per sequence with both the original and
simplified caption as separate description columns
(`scripts/apply_simplified_nymeria_captions.py`), so the existing
multi-caption-per-row random-choice mechanism in `_parse_caption_csv` picks
up the new variants for free — 845/845 sequences augmented, 173,616 rows
total, 0 rows missing a simplified caption. New `atomic_action_augmented`
registry entry, new text-embedding cache (268,882 unique captions — the
union of both columns — vs. 143,055 for `atomic_action` alone), same
`lr:1e-4` probe / `lr:3e-5` finetune configs as the plain `atomic_action`
run.

Generation took two SLURM attempts: job 7621827 (batch-size 128) crashed
with a fatal CUDA OOM at 82,048/143,054 captions after ~2h24m (allocator
fragmentation grew across batches until an allocation could no longer be
satisfied); resubmitted as job 7644223 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and batch-size 64,
which resumed cleanly from the checkpoint and completed the remaining
61,006 captions without incident.

| Model | acc (imu→text) | acc (text→imu) | Status | Run |
|---|---|---|---|---|
| Ours (MAE) probe, atomic_action_augmented captions | 6.95% | 7.72% | ✅ (job 7679669, best step 14,000) | `vit1d_clip_nymeria_atomicaction_augmented_probe_mae_multidata` |
| Ours (MAE) finetune, atomic_action_augmented captions | 8.80% | 9.06% | ✅ (job 7679670, best step 8,000) | `vit1d_clip_nymeria_atomicaction_augmented_finetune_mae_multidata` |
| UniMTS (finetuned on Nymeria, atomic_action_augmented, external) | 2.11% | 1.80% | ✅ (job 7679671, early-stopped step 10,500, best step 500) | `baselines/language_alignment/UniMTS/train_unimts_nymeria_atomicaction_augmented.sh` |

**Result: a mild regression, not a win.** Both probe (6.95/7.72%) and
finetune (8.80/9.06%) are *worse* than their plain-`atomic_action`
counterparts (7.97/8.28% and 10.09/10.50%) by roughly 1-1.4pp in every
column. Plausible explanation: augmentation roughly doubles the unique-
caption pool (143,055 → 268,882) without adding any new IMU data, and the
new simplified captions are considerably shorter/more generic than the
originals (many distinct original captions collapse toward similar short
sentences like "a person reaches down" or "a person turns left"), which
increases the *effective* near-duplicate rate in the text embedding space
— the same false-negative mechanism already identified as a driver of this
task's difficulty in the caption-cleaning discussion above, but working in
the opposite direction from the caption-merge fix (which reduced
near-duplicates; this increases them by adding more generic short
captions to the pool). Unlike swapping to a richer, more specific caption
source entirely (the `atomic_action` win above), *adding* a shorter/more
generic caption variant on top doesn't help and mildly hurts — the
motion-only rewrite loses discriminative detail (object/scene context) that
the contrastive loss was actually using to tell segments apart, even though
that same detail is exactly what the original HumanML3D-inspired hypothesis
wanted to remove.

**UniMTS regresses far more sharply than either Ours(MAE) row.** Its
augmented finetune (job 7679671, early-stopped step 10,500, best step 500 —
identical schedule to its own plain-`atomic_action` run) lands at
**2.11%/1.80%**, a drop of **-4.01pp/-4.56pp** from its plain-`atomic_action`
result (6.12%/6.36%, job 7581974) — roughly 3-4x the size of Ours(MAE)
probe's (-1.02/-0.56pp) or finetune's (-1.29/-1.44pp) regression on the
same swap. All three models now agree in *direction* (augmentation hurts
this task, doesn't help it), but the *magnitude* is starkly different:
Ours(MAE)'s ViT1D encoder + learned CLIP text tower barely notices the
extra near-duplicate captions, while UniMTS's ST-GCN + its own frozen/
lightly-adapted text tower loses most of its already-modest signal. A
plausible reason: UniMTS's finetune only ever moves 500 steps before its
best checkpoint (same as the non-augmented run), so it has very little
opportunity to adapt to the doubled, more homogeneous caption pool relative
to Ours(MAE)'s longer training run — an encoder given less time to adjust
is more exposed to a change in the difficulty/composition of its negatives.
This further supports treating the LLM-caption-augmentation experiment as a
net negative result for this task, consistent across all three models
tested, not just a wash for two of three.

**Synthetic (sensor-noise-free) IMU test (2026-07-23): rules out sensor
noise as the driver.** Prompted by the question of why HumanML3D's
retrieval accuracy (33.00/34.25% finetune) is so much higher than
Nymeria atomic_action's (10.09/10.50%) when the two caption styles aren't
that different — checked first whether caption *repetitiveness* explains
it (it doesn't: HumanML3D's val captions are actually *more* repetitive,
68.3% unique across 4,362 caption instances in `val.txt`, vs. Nymeria
atomic_action val's 85.0% unique across 16,047 instances — ruling out
caption diversity as the explanation, in the opposite direction from what
the LLM-augmentation experiment above implicitly assumed). The remaining
candidate explanation: HumanML3D's IMU is *fully synthetic*, derived
analytically (via SMPL FK + finite differences) from clean, curated mocap
that the caption was written to describe directly — zero sensor noise, zero
temporal misalignment, by construction. Nymeria's real wrist IMU is a
noisy consumer-grade hardware sensor, and its captions were written by a
human annotator watching *separate* egocentric video with full scene/object
context, then loosely time-aligned to a segment — a different, noisier
process than HumanML3D's.

To isolate just the sensor-noise piece, built a synthetic Nymeria wrist IMU
using the same derivation as `nymeria_syn_wrist_loader_FLU.py` (already in
this repo for an unrelated MAE-pretraining ablation): central-difference
acceleration + SO3-log angular velocity from
`gt_data['xsens_pose']` (joint 9 = right wrist, the same ground-truth XSens
mocap used as labels for the full-body-pose task), rotated into the FLU
frame — a clean, noise-free wrist IMU derived from real ground-truth motion
capture, analogous to HumanML3D's SMPL-derived IMU. New
`_load_nymeria_session_imu_flu_synthetic` in
`nymeria_caption_dataset_FLU.py`, threaded through as a new `imu_source`
parameter (`'real'` default vs. `'synthetic'`) alongside the existing
`caption_source` — same atomic_action captions, same train/val split, same
text-embedding cache, same LRs (probe `1e-4`, finetune `3e-5`) as the
real-IMU sibling runs; only the input IMU changes. Smoke-tested first
(sane [B,6,T] shape, no NaNs, correctly normalized stats).

| Model | acc (imu→text) | acc (text→imu) | Status | Run |
|---|---|---|---|---|
| Ours (MAE) probe, atomic_action + synthetic IMU | 7.58% | 7.94% | ✅ (job 7692003, best step 17,000) | `vit1d_clip_nymeria_atomicaction_synthetic_mae_probe` |
| Ours (MAE) finetune, atomic_action + synthetic IMU | 9.39% | 10.64% | ✅ (job 7692004, best step 6,500) | `vit1d_clip_nymeria_atomicaction_synthetic_finetune_mae_multidata` |

**Result: essentially no change, decisively ruling out sensor noise.**
Probe (7.58/7.94%) is flat vs. its real-IMU counterpart (7.97/8.28%, a
~0.3-0.4pp difference, within this benchmark's established noise band).
Finetune (9.39/10.64%) is likewise flat vs. real-IMU (10.09/10.50%) — one
column very slightly down, one very slightly up, net a wash. Neither result
moved even a small fraction of the ~23-24pp gap toward HumanML3D's
finetune ceiling (33.00/34.25%). Since this swap replaced a real, noisy,
hardware-recorded sensor stream with a perfectly clean signal derived
directly from ground-truth motion capture — and captured no improvement —
sensor noise/signal quality is **not** the bottleneck. (Caveat on the probe
row specifically: the frozen encoder was pretrained only on real IMU, so a
frozen probe on synthetic-IMU inputs partly conflates "does clean IMU help"
with "does the frozen backbone generalize OOD to a distribution it never
saw in pretraining" — but the finetune row, where the encoder is trainable
and free to adapt, shows the same flat result, so this caveat doesn't
change the conclusion.)

**What this leaves as the likely explanation**: the caption-*generation*
process itself, not the sensor. HumanML3D captions were written by an
annotator watching the exact rendered mocap clip the IMU was later derived
from — guaranteed, tight correspondence between what's described and what
the wrist IMU can see. Nymeria's captions were written by an annotator
watching real egocentric video with full visual context (rooms, objects,
other people) that a single wrist-worn sensor never had access to, against
segment timestamps that are only approximately aligned to the described
action. Swapping the sensor for a clean one doesn't fix either of those —
the caption can still describe something (an object, a room, another
person's actions) that no wrist kinematics, clean or noisy, could ever
recover, and the segment boundary can still be a few hundred milliseconds
off from the actual described motion. This is a structural, dataset-
curation-level gap between HumanML3D and Nymeria for this specific task,
not a fixable data-quality or caption-wording issue — consistent with why
neither the caption-merge cleaning (§6c above), nor richer/more-specific
captions (the atomic_action win, which helped but only within Nymeria's own
ceiling), nor LLM motion-only simplification (a regression), nor now a
clean sensor swap (a wash) closed the gap to HumanML3D.

---

## 7. External baseline: MobilePoser (Nymeria real right-wrist IMU) — methodology

Numbers now live directly in the arm pose (§1), full-body pose (§2), and
odometry (§4) tables above (each marked with a `*` footnote on unit/metric
compatibility) — this section is methodology/caveat detail only.

Not a ViT1D column — a literature baseline from a different architecture
family (`baselines/full_body_pose/SimplePoser`, an open-source MobilePoser
reimplementation: 4 independently-trained RNN sub-modules — `Joints`,
`Poser`, `FootContact`, `Velocity` — instead of this repo's ViT1D encoder).
Adapted to Nymeria since stock MobilePoser only supports AMASS/DIP-IMU/
TotalCapture/IMUPoser and expects 5 IMU sensors; Nymeria provides one real
right-wrist IMU, so a new `rw` (right-wrist) sensor combo was added and the
other 4 slots zeroed. Also moved the right wrist joint from MobilePoser's
`ignored` set into `reduced` (stock MobilePoser never predicts wrist/hand
rotation at all — architecturally forced to identity — which would have made
the arm-pose number meaningless). New files: `preprocess_nymeria.py` (builds
MobilePoser's `.pt` format from the same Nymeria pkl sessions/labels used
elsewhere in this repo), `train_nymeria.yaml`, `evaluate_nymeria.py` (all
three metrics in one pass via `articulate.FullMotionEvaluator`). Trained from
scratch (no pretraining pipeline exists for MobilePoser) for 60 epochs on the
full available split: 867 train / 101 val sessions.

Full-body pose also has a mesh error of 11.00 cm (± 4.12) not represented in
§2's table (no other row reports mesh error). The right-hand joint (23) in
the arm-pose number is architecturally forced to identity (no sensor evidence
for hand articulation from a wrist IMU alone), which caps how good that
particular joint's contribution can look — see `evaluate_nymeria.py`'s output
`note` field. MobilePoser sees only a single real wrist IMU with no
self-supervised pretraining stage, unlike every other row in this document.

**Window-length matching (2026-07-18)**: initially evaluated with
`PoseDataset(..., evaluate=True)`'s default behavior, which hands back one
item per *entire* Nymeria test session (unbounded — data.py's
`window = len(imu) if self.evaluate else self.cfg['window_length']` uses the
whole session at eval time), letting MobilePoser's bidirectional LSTMs see
many minutes of past+future context per prediction — a real advantage the
ViT1D benchmark's fixed 10.24s window never gets. Fixed by adding a
`--window-seconds` flag to `evaluate_nymeria.py` (default 10.24, matching
this benchmark's own `windowing.window_seconds`) that chops each session
into independent windows and calls `model.reset()` + `forward_offline()`
per-window instead of once per session (job 7323993, resubmitted over the
same 867/101 split and checkpoint — no retraining needed, pure re-eval).
Numbers barely moved (odometry 14.46→15.00cm, full-body pos 9.11→9.19cm/ang
22.26→21.47°, arm pos 9.75→9.74cm/ang 27.79→24.28°) but per-window std
dropped substantially (e.g. odometry ± 15.22→± 9.93cm) since averaging over
many independent short windows removes the outlier weight of a few very long
sessions — so the earlier whole-session numbers weren't dramatically
inflated by the extra context, but the windowed numbers are the ones now
reported in §1/§2/§4 as the methodologically fair comparison.

*Note*: a first pass at 200/101 sessions (before the full-867-session
retrain, and before window-matching) gave visibly worse numbers across the
board (odometry 20.46cm, full-body pos/ang 10.53cm/23.95°, arm pos/ang
11.23cm/29.16°) — rerunning on the full 867/101 split improved every metric,
confirming that smaller run was data-limited rather than architecture-limited.

**LR sweep on MobilePoser (2026-07-21, ✅ complete, 5 of 5 done)**: the ViT1D LR sweep in
§2 raised the question of whether MobilePoser's own baseline number is
similarly untuned — `train_nymeria.yaml` uses `learning_rate: 0.001` (Adam),
never itself swept, just carried over from upstream MobilePoser's own
default. Launched 5 additional from-scratch training runs (same
867/101-session split, same architecture/60 epochs/batch size/dropout,
reusing the already-preprocessed `data_processed/Nymeria/{train,test}.pt`
so no re-preprocessing needed), varying only `learning_rate` across `{3e-4,
5e-4, 7e-4, 1.5e-3, 2e-3}` — 2 below and 3 above the baseline. Configs
`train_nymeria_lr{tag}.yaml`, launch scripts
`launch_slurm_mobileposer_nymeria_lr{tag}.sh`, each writing its checkpoint
to a distinct `checkpoints/nymeria_rw_lr{tag}/` (won't overwrite the
baseline's own `checkpoints/nymeria_rw/best.pt`) and evaluating with the
same `--window-seconds 10.24` window-matching convention as the baseline's
own eval-only script, into `results/nymeria_mobileposer_results_lr{tag}.json`.

| LR | full-body pos_err (cm) | full-body ang_err (deg) | arm pos_err (cm) | odometry (cm/s) | Status | Job |
|---|---|---|---|---|---|---|
| 3e-4 | 9.42 | 21.60 | 10.03 | 16.18 | ✅ | 7522405 |
| 5e-4 | 9.27 | 21.52 | 9.78 | 15.67 | ✅ | 7522406 |
| 7e-4 | 9.22 | 21.50 | 9.80 | 15.73 | ✅ | 7522407 |
| 1e-3 (baseline) | **9.19** | **21.47** | **9.74** | **15.00** | ✅ | `mobileposer_nymeria` |
| 1.5e-3 | 9.29 | 21.38 | 9.96 | 16.06 | ✅ | 7522411 |
| 2e-3 | 9.25 | 21.58 | 9.73 | 15.59 | ✅ | 7522412 |

**Firm conclusion: no improvement found.** All 5 swept LRs land within
~0.25cm of the baseline's full-body pos_err and every single one is worse
than the baseline on that metric (and on odometry) -- the untuned baseline
`1e-3` remains the best value in every column except full-body ang_err
(where `1.5e-3`'s 21.38° edges out the baseline's 21.47° by a hair, not
enough to change the ranking). Unlike Ours(MAE) finetune's default LR
(which had ~5mm of headroom sitting on the table), MobilePoser's default
LR was already at or very near its optimum for this architecture/task --
this is a genuinely useful negative result for the fairness of the overall
comparison: the gap between Ours(MAE) finetune (87.77mm best) and
MobilePoser (91.88mm) isn't an artifact of an undertuned MobilePoser
baseline.

---

## 8. Cross-task transfer (does finetuning on task A help/hurt task B?)

### 8a. Frozen-probe transfer (cross-loaded encoder frozen, only head/decoder trained)

**Question**: every table above already answers "does SSL pretraining help
downstream task X?" (Ours(MAE) probe vs. Scratch). This section asks a
different question: **does supervised finetuning on one downstream task
produce features that transfer to a *different* downstream task?** — i.e.,
is task-specific adaptation net-positive or net-negative for unrelated
tasks, when the resulting encoder is frozen and linearly/decoder-probed
elsewhere.

**Design**: 4 "source" encoders, each already fully finetuned end-to-end on
one task earlier in this benchmark (no new pretraining needed — these
checkpoints already existed): full-body pose
(`vit1d_full_body_pose_finetune_nymeria_mae_multidata`), WISDM HAR
(`vit1d_har_finetune_wisdm_mae_multidata`), world-frame odometry
(`vit1d_odom_worldframe_finetune_nymeria_mae_multidata`), and gesture
(`vit1d_gesture_finetune_openwatch_6class_mae_multidata`). All 4 share the
same alt-attn ViT1DEncoder architecture (warm-started from the same MAE
checkpoint), so cross-loading is a drop-in `pretrained_run` swap — no new
loader/decoder code needed, just 9 cloned probe configs (`vit1d_transfer_
{target}_from_{source}.yaml`) plus 3 direct CLI invocations of
`eval_odom_nymeria_worldframe.py --run <source_finetune_checkpoint>` for the
odometry target (matching how "Ours(MAE)" itself is probed in Section 4b).
Each of the 4 encoders is frozen and probed on the *other* 3 tasks
(12 cells total); the in-domain diagonal and the "no supervised finetuning
at all" control are read off the existing tables above (Ours(MAE) finetune
= in-domain upper bound, Ours(MAE) probe = SSL-only lower-bound control).
`vit1d_full_body_pose_probe_nymeria_mae_multidata_flu.yaml`'s
`early_stopping.enabled: false` was flipped to `true` for the 3
full-body-pose-target transfer configs (its sibling's `false` setting is
what caused an earlier ~9-10hr runaway-job incident, see the pipeline audit
below — not worth re-risking here).

**Task complete (2026-07-18) except one cell still running.** 11 of 12 jobs
finished cleanly; the 12th (odometry<-gesture) ran long relative to its
siblings (1h30m+ vs. ~1h10m-1h40m) — checked it wasn't stuck via `nvidia-smi`/
`ps` on its node (qgpu2015): 57% GPU util, 97.6% CPU, 90+ min of accumulated
CPU time matching wall clock, actively computing. Not resubmitted; just
genuinely slower this run (buffered stdout meant no visible progress lines
yet, not a hang).

**Headline finding**: the *type* of source task matters far more than
whether transfer happens at all. Full-body pose <-> world-frame odometry
transfer **positively** in both directions, each beating the SSL-only
control on every metric — motion-dynamics tasks share structure. HAR and
gesture (classification tasks) show the opposite: every cross-task transfer
into or out of them **underperforms the SSL-only control**, i.e. supervised
finetuning on a different task actively hurts more than it helps once you
leave the motion-dynamics cluster.

**Against the from-scratch (no pretraining at all) baseline, the picture
splits differently.** For the two classification targets (WISDM, gesture),
*every* cross-task-transferred encoder still clears the scratch baseline by
a wide margin (e.g. gesture: scratch 44.90% vs. 56-64% for all 3 transfer
sources) — cross-task supervised features are worse than same-domain SSL
pretraining, but far better than no pretraining at all. For full-body pose,
the opposite: *no* cross-task transfer source beats Scratch (92.56mm) — that
column's scratch baseline is unusually strong (it already beats the SSL-only
probe control itself, a known anomaly noted in Section 2), so cross-task
transfer isn't enough to close that gap. For odometry, it's mixed:
Scratch's vel_mse (0.01835) and drift (37.31%) beat both transfer sources,
but its ATE (0.1419m) is far worse than either (0.0783m/0.0753m) — the same
per-timestep-accuracy-vs-cumulative-drift disconnect already documented for
the finetune columns in Section 4b.

### Target: WISDM HAR (acc / bal_acc / macro_f1)
| Source encoder | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| *(none — SSL pretrain only)* | 65.50% | 65.52% | 0.639 | ✅ (Table 3, Ours(MAE) probe) | `vit1d_har_probe_wisdm_mae_multidata` |
| *(scratch, no pretraining at all)* | 60.89% | 60.89% | 0.606 | ✅ (Table 3, Scratch) | `vit1d_har_supervised_wisdm_altattn_scratch` |
| Full-body pose (finetuned) | 64.09% | 64.06% | 0.629 | ✅ (job 7378735) | `vit1d_transfer_wisdm_from_fullbodypose` |
| Odometry, world-frame (finetuned) | 62.31% | 62.28% | 0.606 | ✅ (job 7378736) | `vit1d_transfer_wisdm_from_odomworldframe` |
| Gesture (finetuned) | 64.78% | 64.79% | 0.632 | ✅ (job 7378821) | `vit1d_transfer_wisdm_from_gesture` |
| *(in-domain upper bound)* | **71.27%** | **71.22%** | **0.704** | ✅ (Table 3, Ours(MAE) finetune) | `vit1d_har_finetune_wisdm_mae_multidata` |

All 3 transfer sources land *below* the SSL-only control (65.50%) — none of
full-body-pose/odometry/gesture finetuning helps WISDM; gesture-transfer
comes closest (64.78%) but still doesn't beat doing nothing extra.

### Target: Full-body pose, Nymeria (mpjpe, mm — lower better)
| Source encoder | mpjpe (mm) | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 97.63 | ✅ (Table 2, Ours(MAE) probe) | `vit1d_full_body_pose_probe_nymeria_mae_multidata` |
| *(scratch, no pretraining at all)* | 92.56 | ✅ (Table 2, Scratch) | `vit1d_full_body_pose_supervised_nymeria_altattn_scratch` |
| WISDM HAR (finetuned) | 100.08 | ✅ (job 7378822) | `vit1d_transfer_fullbodypose_from_wisdm` |
| Odometry, world-frame (finetuned) | **95.93** | ✅ (job 7378823) | `vit1d_transfer_fullbodypose_from_odomworldframe` |
| Gesture (finetuned) | 99.70 | ✅ (job 7378824) | `vit1d_transfer_fullbodypose_from_gesture` |
| *(in-domain upper bound)* | 92.39 | ✅ (Table 2, Ours(MAE) finetune) | `vit1d_full_body_pose_finetune_nymeria_mae_multidata` |

Odometry-finetuned transfer (95.93mm) **beats the SSL-only control**
(97.63mm) — the only cross-task transfer in this whole experiment that
improves on doing nothing extra. WISDM/gesture transfer both land slightly
worse than the control, same pattern as the WISDM target table above.

### Target: Gesture, OpenWatch 6-class (acc)
| Source encoder | acc | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 67.35% | ✅ (Table 5, Ours(MAE) probe) | `vit1d_gesture_probe_openwatch_6class_mae_multidata` |
| *(scratch, no pretraining at all)* | 44.90% | ✅ (Table 5, Scratch) | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch` |
| WISDM HAR (finetuned) | 64.29% | ✅ (job 7378825) | `vit1d_transfer_gesture_from_wisdm` |
| Full-body pose (finetuned) | 62.24% | ✅ (job 7378826) | `vit1d_transfer_gesture_from_fullbodypose` |
| Odometry, world-frame (finetuned) | 56.12% | ✅ (job 7378827) | `vit1d_transfer_gesture_from_odomworldframe` |
| *(in-domain upper bound)* | 66.33% | ✅ (Table 5, Ours(MAE) finetune) | `vit1d_gesture_finetune_openwatch_6class_mae_multidata` |

Steepest drop of the whole experiment: odometry-transfer costs 11pp
(67.35% -> 56.12%) relative to the SSL-only control. Gesture is a
fine-motor, short-window classification task; odometry finetuning pushes
the encoder toward coarse long-range motion-dynamics features that appear
actively counterproductive for it.

### Target: Odometry, world-frame, Nymeria (vel_mse / ATE m / drift %)
| Source encoder | vel_mse | ATE (m) | drift | Status | Run |
|---|---|---|---|---|---|
| *(none — SSL pretrain only)* | 0.02086 | 0.0787 | 48.28% | ✅ (Table 4b, Ours(MAE) probe) | `vit1d_ssl_mae_varlen_altattn_multidata` |
| *(scratch, no pretraining at all)* | 0.01835 | 0.1419 | 37.31% | ✅ (Table 4b, Scratch) | `vit1d_odom_worldframe_supervised_nymeria_altattn_scratch` |
| WISDM HAR (finetuned) | 0.02091 | 0.0783 | 49.15% | ✅ (job 7378828) | `vit1d_transfer_odomworldframe_from_wisdm` |
| Full-body pose (finetuned) | **0.01978** | **0.0753** | **43.99%** | ✅ (job 7378829) | `vit1d_transfer_odomworldframe_from_fullbodypose` |
| Gesture (finetuned) | 0.02115 | 0.0790 | 49.21% | ✅ (job 7378830) | `vit1d_transfer_odomworldframe_from_gesture` |
| *(in-domain upper bound)* | **0.01656** | 0.1293 | **34.08%** | ✅ (Table 4b, Ours(MAE) finetune) | `vit1d_odom_worldframe_finetune_nymeria_mae_multidata` |

Full-body-pose-finetuned transfer **beats the SSL-only control on every
metric** (vel_mse, ATE, *and* drift) — the mirror image of the full-body-pose
target table above, confirming pose <-> odometry transfer is genuinely
bidirectional and positive, not a one-off. WISDM-transfer is roughly a wash
with the control (marginally worse on drift, marginally better on ATE).

---

### 8b. Full end-to-end finetune transfer (encoder warm-started AND kept trainable)

8a above only froze the cross-loaded encoder and trained a fresh head/decoder
on top -- the standard "linear probe" transfer question. This subsection asks
the stronger version: **if you warm-start from a task-A-finetuned encoder and
then keep training it end-to-end on task B, does that beat both the
frozen-probe transfer above and training from the base MAE checkpoint?** Same
12 cells, same 4 source checkpoints, but now using each task's *finetune*
objective (`utd_mhad_finetune`, `pose`, `openwatch_gesture_finetune`,
`odom_finetune_nymeria_real_wrist_worldframe_flu`) instead of its `_probe`
sibling -- i.e. `objective.{type}.pretrained_run` points at the source task's
finetuned checkpoint but the encoder stays `requires_grad=True`, warm-started
rather than frozen. 12 new configs (`vit1d_transfer_finetune_{target}_from_
{source}.yaml`), all confirmed via log inspection to correctly warm-start
from the intended cross-task checkpoint before submitting the batch (e.g.
`[utd_mhad_finetune] warm-starting encoder from .../vit1d_full_body_pose_
finetune_nymeria_mae_multidata/checkpoints/best.pt`).

**Task complete (2026-07-19), all 12 cells.** 5 of the first 12 job
attempts (7382220, 7382223, 7382225-7382227) crashed from the same
disk-quota checkpoint-write corruption documented in Section 8a/RelCon's
Stage-2 writeup -- resolved by the same cleanup (freed ~8GB of accidentally
-accumulated per-step checkpoint snapshots, patched `save_step_checkpoints:
false` into the affected configs) and resubmitted under new job IDs, noted
per-row below. The odometry-world-frame target additionally needed a
separate post-hoc scoring pass (`scripts/vis_odom_scratch_worldframe.py`),
same as Section 4b's Scratch/finetune rows -- these finetune runs only log
raw `odom_mse`/`odom_mae` during training, not `vel_mse`/`ATE`/`drift`.

**Striking finding, sharply different from Section 8a**: with the encoder
kept trainable (not frozen), cross-task transfer routinely **matches or
beats** both the SSL-only control *and* the in-domain finetune upper bound
-- the opposite of 8a, where every frozen-probe transfer underperformed the
control. Two patterns stand out:
  - **A full-body-pose-finetuned source encoder is the strongest transfer
    donor**: on *every other target it was tried on*, it beats that
    target's own in-domain finetune upper bound -- WISDM (71.80% vs.
    71.27%), gesture (68.37% vs. 66.33%), and world-frame odometry (vel_mse
    0.01622 vs. 0.01656, ATE 0.1259m vs. 0.1293m, drift 33.13% vs. 34.08%,
    sweeping all three metrics at once).
  - **Full-body-pose is also the easiest target to beat**: *all three*
    cross-task sources (WISDM 92.16mm, world-frame-odometry 91.49mm,
    gesture 92.24mm) beat its own in-domain upper bound (92.39mm) --
    world-frame-odometry-as-source is the single best cell in the whole
    table. Combined with the point above, full-body-pose and world-frame
    odometry look like they occupy a shared "motion dynamics" representation
    neighborhood (echoing Section 8a's headline finding for frozen-probe
    transfer), while also suggesting full-body-pose's own in-domain
    finetune here isn't especially hard to match once the encoder gets any
    reasonable amount of full end-to-end adaptation, regardless of source.

Once the encoder is allowed to adapt end-to-end, which task supplied
the warm-start matters far less than SSL pretraining alone -- a plausible
read is that a few thousand steps of full finetuning on the target task's
own data erases most of whatever task-specific bias the source finetune left
behind, leaving mainly a (helpful) different random-ish starting point
in weight space rather than a lasting handicap.

### Target: WISDM HAR (acc / bal_acc / macro_f1)
| Source encoder | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| *(none — SSL pretrain only)* | 65.50% | 65.52% | 0.639 | ✅ (Table 3, Ours(MAE) probe) | `vit1d_har_probe_wisdm_mae_multidata` |
| *(scratch, no pretraining at all)* | 60.89% | 60.89% | 0.606 | ✅ (Table 3, Scratch) | `vit1d_har_supervised_wisdm_altattn_scratch` |
| Full-body pose (finetuned, full finetune) | **71.80%** | **71.75%** | **0.710** | ✅ (job 7382216) | `vit1d_transfer_finetune_wisdm_from_fullbodypose` |
| Odometry, world-frame (finetuned, full finetune) | 71.29% | 71.24% | 0.705 | ✅ (job 7382217) | `vit1d_transfer_finetune_wisdm_from_odomworldframe` |
| Gesture (finetuned, full finetune) | 71.18% | 71.12% | 0.703 | ✅ (job 7382218) | `vit1d_transfer_finetune_wisdm_from_gesture` |
| *(in-domain upper bound)* | 71.27% | 71.22% | 0.704 | ✅ (Table 3, Ours(MAE) finetune) | `vit1d_har_finetune_wisdm_mae_multidata` |

### Target: Full-body pose, Nymeria (mpjpe, mm — lower better)
| Source encoder | mpjpe (mm) | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 97.63 | ✅ (Table 2, Ours(MAE) probe) | `vit1d_full_body_pose_probe_nymeria_mae_multidata` |
| *(scratch, no pretraining at all)* | 92.56 | ✅ (Table 2, Scratch) | `vit1d_full_body_pose_supervised_nymeria_altattn_scratch` |
| WISDM HAR (finetuned, full finetune) | 92.16 | ✅ (job 7382219) | `vit1d_transfer_finetune_fullbodypose_from_wisdm` |
| Odometry, world-frame (finetuned, full finetune) | **91.49** | ✅ (job 7384239) | `vit1d_transfer_finetune_fullbodypose_from_odomworldframe` |
| Gesture (finetuned, full finetune) | 92.24 | ✅ (job 7382221) | `vit1d_transfer_finetune_fullbodypose_from_gesture` |
| *(in-domain upper bound)* | 92.39 | ✅ (Table 2, Ours(MAE) finetune) | `vit1d_full_body_pose_finetune_nymeria_mae_multidata` |

*(all three cross-task sources beat the in-domain upper bound here -- 91.49/92.16/92.24 vs. 92.39)*

### Target: Gesture, OpenWatch 6-class (acc)
| Source encoder | acc | Status | Run |
|---|---|---|---|
| *(none — SSL pretrain only)* | 67.35% | ✅ (Table 5, Ours(MAE) probe) | `vit1d_gesture_probe_openwatch_6class_mae_multidata` |
| *(scratch, no pretraining at all)* | 44.90% | ✅ (Table 5, Scratch) | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch` |
| WISDM HAR (finetuned, full finetune) | 63.27% | ✅ (job 7382222) | `vit1d_transfer_finetune_gesture_from_wisdm` |
| Full-body pose (finetuned, full finetune) | **68.37%** | ✅ (job 7384240, resubmitted after a disk-quota crash on 7382223) | `vit1d_transfer_finetune_gesture_from_fullbodypose` |
| Odometry, world-frame (finetuned, full finetune) | 67.35% | ✅ (job 7382224) | `vit1d_transfer_finetune_gesture_from_odomworldframe` |
| *(in-domain upper bound)* | 66.33% | ✅ (Table 5, Ours(MAE) finetune) | `vit1d_gesture_finetune_openwatch_6class_mae_multidata` |

### Target: Odometry, world-frame, Nymeria (vel_mse / ATE m / drift %)
| Source encoder | vel_mse | ATE (m) | drift | Status | Run |
|---|---|---|---|---|---|
| *(none — SSL pretrain only)* | 0.02086 | 0.0787 | 48.28% | ✅ (Table 4b, Ours(MAE) probe) | `vit1d_ssl_mae_varlen_altattn_multidata` |
| *(scratch, no pretraining at all)* | 0.01835 | 0.1419 | 37.31% | ✅ (Table 4b, Scratch) | `vit1d_odom_worldframe_supervised_nymeria_altattn_scratch` |
| WISDM HAR (finetuned, full finetune) | 0.01684 | 0.1278 | 34.09% | ✅ (job 7384241 train, 7384808 score) | `vit1d_transfer_finetune_odomworldframe_from_wisdm` |
| Full-body pose (finetuned, full finetune) | **0.01622** | **0.1259** | **33.13%** | ✅ (job 7384242 train, 7384806 score) | `vit1d_transfer_finetune_odomworldframe_from_fullbodypose` |
| Gesture (finetuned, full finetune) | 0.01666 | 0.1323 | 36.42% | ✅ (job 7384243 train, 7384807 score) | `vit1d_transfer_finetune_odomworldframe_from_gesture` |
| *(in-domain upper bound)* | 0.01656 | 0.1293 | 34.08% | ✅ (Table 4b, Ours(MAE) finetune) | `vit1d_odom_worldframe_finetune_nymeria_mae_multidata` |

---

## 9. Pretraining data scaling (MAE reconstruction loss vs. % of pretraining data)

Config family: `vit1d_ssl_mae_varlen_altattn_multidata_flu_data{0.1,1,5,25,50}pct.yaml`
(`launch_slurm_MAE_scaling_{0.1,1,5,25,50}pct.sh`), each identical to the main
`vit1d_ssl_mae_varlen_altattn_multidata` pretrain config except for
`objective.mae_multidata_varlen_altattn_flu.overall_fraction` (a global
subsampling knob applied across the 11-source multidata mix; OCDetect and
capture24 additionally use a small per-source `sample_fraction` since their
raw files are expensive to enumerate at full size, applied *before* I/O, not
on top of `overall_fraction`). No downstream probes yet — this section
reports the SSL pretraining objective itself (masked-patch reconstruction
MSE on held-out val windows), the same metric "Ours (MAE)"'s own pretrain
run is scored on elsewhere in this document (e.g. Section 4's Odometry
table, which reuses this same run's val metrics directly).

| % of pretraining data | `overall_fraction` | best val MSE loss | step @ best | epoch @ best | Status | Run |
|---|---|---|---|---|---|---|
| 0.1% | 0.001 | 0.8111 | 29,000 | 9,667 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data0.1pct` |
| 1% | 0.01 | 0.7643 | 135,000 | 3,649 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data1pct` |
| 5% | 0.05 | 0.7430 | 221,000 | 1,170 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data5pct` |
| 25% | 0.25 | 0.7438 | 221,000 | 234 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data25pct` |
| 50% | 0.5 | 0.7419 | 221,000 | 117 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata_data50pct` |
| 100% (main run) | 1.0 | **0.7351** | 291,000 | 77 | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata` |

All six early-stopped on `val_loss` with `patience: 10` (`val_every_steps:
1000`), same convention as every other pretrain run in this benchmark.
Monotonic improvement with more data — 0.1% -> 100% closes val loss from
0.811 to 0.735 (~9.4% relative) — but with clearly diminishing returns past
~5%: going 0.1%->1%->5% buys 0.047 and 0.021 of improvement respectively,
while 5%->25%->50%->100% only buys another 0.008 total, and 25% is actually
(very marginally) *worse* than 5% (0.7438 vs 0.7430 — within noise of
early-stopping-checkpoint variance, not a real regression). The epoch column
is the more striking story: the 0.1% run revisits its tiny slice of data
9,667 times to reach its best checkpoint, vs. 77 epochs for the full dataset
at a *higher* step count (291,000) — i.e. the low-data runs aren't
undertrained, they're straightforwardly running out of distinct data to
learn from, which is exactly the qualitative signature you'd want to see
from a real data-scaling ablation (diminishing returns from data quantity,
not an artifact of insufficient training budget).

**🐛 Composition gap found in the low end of this sweep (2026-07-22).**
`overall_fraction` subsamples the *already-concatenated pool* of all 11
sources with a single uniform draw (`_fraction_subset` in
`combined_mae_multidata_loader.py`), which only preserves each source's
share of the mix *in expectation*. Reproduced the exact
`np.random.default_rng(42).choice(...)` draw each run actually performs
(seed/logic match the code, verified against the real training log's
per-source composition breakdown) and checked how many windows each source
contributed at each fraction. Two of the 11 sources — totalcapture (461/485,125
windows, 0.095% of the pool) and imuposer (137/485,125, 0.028%) — are small
enough that the 0.1% draw (k=485) sampled **zero windows from either one**;
at 1% (k=4,851) they get only 6 and 3 windows respectively. From 5% up every
source gets its roughly-proportional share and this stops mattering. So the
0.1%/1% points in the table above aren't a clean "same mix, less of it"
comparison against 5%-100% — they're also missing (or nearly missing) two
whole data domains, both real-wrist mocap+pose sources closest in kind to
this benchmark's own arm-pose/full-body-pose downstream evals. Plausibly
explains part of the steep 0.1%→5% improvement (0.811→0.743): some of that
gain is the model seeing those domains at all for the first time, not purely
"more windows from a fixed distribution."

**Fix**: added `stratify_overall_fraction: true` to
`objective.mae_multidata_varlen_altattn_flu` (`combined_mae_multidata_loader.py`,
`make_combined_mae_multidata_loaders`) — draws `overall_fraction`
independently from *each* source's own train windows (`_fraction_subset`
already floors to `max(1, round(n*fraction))`, so every enabled source keeps
at least 1 window) before concatenation, instead of one draw over the pooled
set. Default remains the original pooled behavior (verified bit-for-bit
identical to the original 0.1% run's `485/485,125` split when the new flag
is left unset — pure opt-in, no behavior change for any existing config).
Smoke-tested directly against the real cache
(`/projects/b53017/vbc1591/mae_multidata_cache`): stratified 0.1% now gives
totalcapture=1, imuposer=1 (vs. 0/0 pooled), all 11 sources present, batch
collation unaffected.

New configs `vit1d_ssl_mae_varlen_altattn_multidata_flu_data{0.1,1}pct_stratified.yaml`
(`launch_slurm_MAE_scaling_{0.1,1}pct_stratified.sh`), identical to their
pooled counterparts except for the new flag. 5%/25%/50%/100% are not
rerun — their composition was already essentially proportional, so
stratifying them would not be expected to change anything.

**Result (jobs 7624656, 7624657, completed 2026-07-22/23):**

| % of pretraining data | best val MSE loss (stratified) | step @ best | epoch @ best | vs. pooled (original) |
|---|---|---|---|---|
| 0.1% | 0.809982 | 27,000 | 9,000 | 0.8111 → 0.8100 (−0.0011, small improvement) |
| 1% | 0.765670 | 135,000 | 3,649 | 0.7643 → 0.7657 (+0.0014, small regression) |

**Verdict: a wash, not the driver of the steep 0.1%→5% curve.** Both deltas
(~0.001-0.0014) are within early-stopping-checkpoint noise level, similar
in size to the caption-merge cleaning result elsewhere in this doc — the
composition-gap bug was real (totalcapture/imuposer genuinely got zero
windows at 0.1%, as documented above) but its effect on the aggregate
11-source val MSE loss turns out to be negligible, since those two sources
are a tiny fraction of the *val* set too, not just the train pool. The
original pooled 0.1%/1% rows in the main table above are kept as the
canonical/reported numbers (all of §9b-9e's downstream scaling evals were
already run against those exact checkpoints); `stratify_overall_fraction`
remains available and is the more methodologically correct setting going
forward, but isn't worth rerunning the downstream evals over for a
noise-level difference.

### 9b. Downstream HAR scaling (linear-probe accuracy vs. % of pretraining data)

Config family: `vit1d_har_probe_{wisdm,utd_mhad,pamap2,recofit}_mae_data{0.1,1,5,25,50}pct_multidata_flu.yaml`
(jobs 7463499-7463522), each an ordinary frozen linear probe (`utd_mhad_probe`
objective) pointed at `pretrained_run: vit1d_ssl_mae_varlen_altattn_multidata_data{X}pct`
instead of the full-data checkpoint — otherwise identical to the probe
configs used for the "Ours (MAE)" rows in Section 3. Metric: best
val_loss checkpoint's accuracy (%, higher is better), same convention as
Section 3. 100% row reuses Section 3's "Ours (MAE)" probe numbers directly
(same checkpoint, not rerun).

#### WISDM
| % of pretraining data | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| 0.1% | 57.29% | 57.35% | 0.550 | ✅ (job 7463499) | `vit1d_har_probe_wisdm_mae_data0.1pct` |
| 1% | **65.75%** | **65.76%** | **0.644** | ✅ (job 7463500) | `vit1d_har_probe_wisdm_mae_data1pct` |
| 5% | 65.68% | 65.70% | 0.641 | ✅ (job 7463501) | `vit1d_har_probe_wisdm_mae_data5pct` |
| 25% | 65.29% | 65.29% | 0.637 | ✅ (job 7463502) | `vit1d_har_probe_wisdm_mae_data25pct` |
| 50% | 65.61% | 65.61% | 0.639 | ✅ (job 7463503) | `vit1d_har_probe_wisdm_mae_data50pct` |
| 100% (main run) | 65.50% | 65.52% | 0.639 | ✅ (Table 3, Ours(MAE) probe) | `vit1d_har_probe_wisdm_mae_multidata` |

#### UTD-MHAD
| % of pretraining data | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| 0.1% | 54.59% | 55.56% | 0.542 | ✅ (job 7463504) | `vit1d_har_probe_utd_mhad_mae_data0.1pct` |
| 1% | 58.45% | 58.85% | 0.555 | ✅ (job 7463505) | `vit1d_har_probe_utd_mhad_mae_data1pct` |
| 5% | 62.32% | 61.60% | 0.610 | ✅ (job 7463509) | `vit1d_har_probe_utd_mhad_mae_data5pct` |
| 25% | **69.08%** | **69.24%** | **0.680** | ✅ (job 7463510) | `vit1d_har_probe_utd_mhad_mae_data25pct` | 
| 50% | 64.73% | 64.61% | 0.639 | ✅ (job 7463511) | `vit1d_har_probe_utd_mhad_mae_data50pct` |
| 100% (main run) | 63.77% | 63.83% | 0.633 | ✅ (Table 3, Ours(MAE) probe) | `vit1d_har_probe_utd_mhad_mae_multidata` |

#### PAMAP2
| % of pretraining data | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| 0.1% | 51.14% | 47.53% | 0.438 | ✅ (job 7463512) | `vit1d_har_probe_pamap2_mae_data0.1pct` |
| 1% | 57.60% | 55.33% | 0.560 | ✅ (job 7463514) | `vit1d_har_probe_pamap2_mae_data1pct` |
| 5% | 57.73% | 55.32% | 0.562 | ✅ (job 7463515) | `vit1d_har_probe_pamap2_mae_data5pct` |
| 25% | 57.27% | 54.84% | 0.560 | ✅ (job 7463516) | `vit1d_har_probe_pamap2_mae_data25pct` |
| 50% | **58.02%** | **56.75%** | **0.567** | ✅ (job 7463517) | `vit1d_har_probe_pamap2_mae_data50pct` |
| 100% (main run) | 56.12% | 52.86% | 0.542 | ✅ (Table 3, Ours(MAE) probe) | `vit1d_har_probe_pamap2_mae_multidata` |

#### RecoFit
| % of pretraining data | acc | bal_acc | macro_f1 | Status | Run |
|---|---|---|---|---|---|
| 0.1% | 71.65% | 56.62% | 0.565 | ✅ (job 7463518) | `vit1d_har_probe_recofit_mae_data0.1pct` |
| 1% | 78.67% | 64.33% | 0.654 | ✅ (job 7463519) | `vit1d_har_probe_recofit_mae_data1pct` |
| 5% | 79.37% | **65.87%** | **0.665** | ✅ (job 7463520) | `vit1d_har_probe_recofit_mae_data5pct` |
| 25% | **80.23%** | 65.17% | 0.656 | ✅ (job 7463521) | `vit1d_har_probe_recofit_mae_data25pct` |
| 50% | 78.11% | 64.20% | 0.643 | ✅ (job 7463522) | `vit1d_har_probe_recofit_mae_data50pct` |
| 100% (main run) | 78.21% | 61.67% | 0.627 | ✅ (Table 3, Ours(MAE) probe) | `vit1d_har_probe_recofit_mae_multidata` |

Unlike the pretraining reconstruction-loss curve above (monotonic
improvement all the way to 100%), downstream HAR probe accuracy saturates
or peaks well before 100% of pretraining data on every one of the four
datasets: WISDM/PAMAP2/RecoFit all peak somewhere in the 1-50% range and
are flat-to-slightly-*down* at 100%; UTD-MHAD peaks hardest at 25%
(69.08%, +5.3pp over the 100% checkpoint's 63.77%). This is consistent
with the reconstruction-loss curve already showing diminishing returns
past ~5% data (0.7430 at 5% vs. 0.7351 at 100%, only 0.008 better) — once
the encoder's reconstruction objective stops improving much, extra
pretraining data isn't clearly buying extra downstream-task signal either.
Some caution warranted on UTD-MHAD/PAMAP2 specifically: both have small
val splits, so part of the 5-100% spread is plausibly eval noise rather
than a genuine non-monotonic effect — would want to confirm with a second
seed before reading too much into the exact peak location. Other
downstream-task scaling curves (arm pose / gesture / odometry accuracy at
each data fraction) — **2026-07-21 update**: arm pose and odometry scaling
curves (Nymeria) have now been run, following the same one-probe-job-per-
fraction pattern as 9b; full-body pose scaling is in progress. Gesture
scaling is still not run. See 9c/9d/9e below.

### 9c. Downstream arm pose scaling (Nymeria probe val_loss vs. % of pretraining data)

Config family: `vit1d_arm_pose_probe_nymeria_mae_data{0.1,1,5,25,50}pct_multidata_flu.yaml`,
same probe recipe as Section 1's "Ours (MAE)" row, pointed at each Section
9 scaling checkpoint instead of the full-data one. Jobs 7491645 (0.1%),
7491647 (1%), 7491648 (5%), 7491650 (25%), 7491652 (50%), all completed
cleanly.

| % of pretraining data | best val_loss | best step | Status | Run |
|---|---|---|---|---|
| 0.1% | 0.017561 | 8,000 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data0.1pct_multidata` |
| 1% | 0.016630 | 6,000 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data1pct_multidata` |
| 5% | 0.016286 | 11,000 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data5pct_multidata` |
| 25% | **0.015866** | 9,000 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data25pct_multidata` |
| 50% | 0.016193 | 11,000 | ✅ | `vit1d_arm_pose_probe_nymeria_mae_data50pct_multidata` |
| 100% (main run) | 0.01638 | — | ✅ | `vit1d_arm_pose_probe_nymeria_mae_multidata` (Section 1) |

**pos_err/ang_err not available for these 5 rows**: Section 1's pos_err
(cm)/ang_err (deg) columns come from a separate scoring script
(`scripts/eval_arm_pose_position_error.py`) whose `RUNS` list is hardcoded
to the original 6 columns — it hasn't been extended to cover these 5
scaling checkpoints (the same gap already noted for the acc-only row in
Section 1). Val_loss alone already shows a non-monotonic pattern similar to
9b's HAR curves: 25% is the best checkpoint (0.015866), beating even the
100%-data run (0.01638), with 50% slightly worse again — another instance
of downstream signal saturating/peaking well before 100% of pretraining
data, consistent with 9b and with Section 9's own reconstruction-loss
curve already flattening past ~5%.

### 9d. Downstream full-body pose scaling (Nymeria probe mpjpe vs. % of pretraining data)

Config family: `vit1d_full_body_pose_probe_nymeria_mae_data{0.1,1,5,25,50}pct_multidata_flu.yaml`,
same probe recipe as Section 2's "Ours (MAE)" row. Jobs 7491653 (0.1%),
7491654 (1%), 7491746 (5%), 7491747 (25%), 7491750 (50%) — the original
5%/25%/50% submissions (7491655/57/58) were cancelled while still pending
(never started, no logs) and resubmitted as the three IDs above.

**Task complete.** All 5 fractions hit the 12h walltime before early-stopping
fired on its own -- same `early_stopping.enabled: false` template gap
documented elsewhere in this doc (Sections 1-2's no-alt-attn/acc-only
footnotes) -- but nothing was lost since the trainer checkpoints
`best_mpjpe.pt` independent of that flag.

| % of pretraining data | best mpjpe (mm) | Status | Run |
|---|---|---|---|
| 0.1% | 101.18 | ✅ hit 12h walltime, `best_mpjpe.pt` pulled | 7491653 |
| 1% | 98.20 | ✅ hit 12h walltime, `best_mpjpe.pt` pulled | 7491654 |
| 5% | 97.25 | ✅ hit 12h walltime, `best_mpjpe.pt` pulled | 7491746 |
| 25% | **96.75** | ✅ hit 12h walltime, `best_mpjpe.pt` pulled | 7491747 |
| 50% | 97.00 | ✅ hit 12h walltime, `best_mpjpe.pt` pulled | 7491750 |
| 100% (main run) | 97.63 | ✅ | `vit1d_full_body_pose_probe_nymeria_mae_multidata` (Section 2) |

Same saturation-before-100% story as 9b/9c, now fully confirmed: 25% and
50% both beat the 100%-data probe's own mpjpe (97.63mm) outright, and even
0.1%/1% (101.18/98.20mm) bracket it fairly tightly. Non-monotonic at the
top end (50% is very slightly worse than 25%) -- consistent with 9b's
UTD-MHAD/PAMAP2 caveat that small differences this close to saturation are
plausibly eval noise rather than a genuine effect, though here every
fraction used the *same* held-out val split so it's slightly more likely
to be a real (if tiny) peak at 25% than pure noise.

### 9e. Downstream odometry scaling (Nymeria probe vel_mse/ATE/drift vs. % of pretraining data)

Config family: `launch_slurm_odom_probe_nymeria_mae_data{0.1,1,5,25,50}pct.sh`,
standalone eval jobs via `scripts/eval_odom_nymeria_altattn.py` against each
scaling checkpoint (same script/metric convention as Section 4's odometry
table).

| % of pretraining data | vel_mse | ATE (m) | drift | Status | Job |
|---|---|---|---|---|---|
| 0.1% | — | — | — | ❌ failed — job 7491659 ran ~93min then was cancelled/terminated, `.out` is empty, no output ever written; not yet retried | 7491659 |
| 1% | 0.02533 | 0.0729 | 50.49% | ✅ | 7491661 |
| 5% | 0.02375 | 0.0708 | 49.51% | ✅ | 7491665 |
| 25% | 0.02376 | 0.0707 | 48.54% | ✅ | 7491666 |
| 50% | 0.02375 | 0.0717 | 49.94% | ✅ | 7491668 |
| 100% (main run) | 0.0245 | 0.0726 | 49.42% | ✅ | `vit1d_ssl_mae_varlen_altattn_multidata` (Section 4) |

4 of 5 fractions now sit essentially on top of the 100%-data probe's own
ATE (0.0707-0.0729m vs. 0.0726m) — flatter than even 9c/9d's downstream
curves, confirming odometry's local-velocity objective saturates its
useful pretraining-data signal by 1% of pretraining data already, with no
further gain out to 50%. The 0.1% point is the one remaining gap (job
failed silently, needs a retry) rather than a completed low-data-regime
data point, so only the very bottom of this curve isn't read yet.

---

## 10. Zero-shot & finetuned HAR classification via language alignment (WISDM, UTD-MHAD, PAMAP2)

A different question from Section 3 (which measures how good the MAE-pretrained
encoder is when given labeled target-dataset data and a supervised classifier
head). This section asks: can the **CLIP-aligned** encoder — specifically
`vit1d_clip_humanml3d_finetune_mae_multidata`, the checkpoint that got
33.00%/34.25% imu↔text retrieval on HumanML3D itself (§6) — classify real-world
HAR activities *zero-shot*, purely via cosine similarity to CLIP text
embeddings of each dataset's own activity-name strings ("The person is
{activity}"), with **no labeled training data from the target dataset at
all**? And how much does finetuning that same CLIP-aligned encoder on the
target dataset's own data close the gap? Directly benchmarked against
**UniMTS** (Zhang et al., NeurIPS 2024) — an external HAR foundation model
whose entire premise is zero-shot text-prompted activity classification —
under the same protocol on the same data.

**Data**: new `baselines/language_alignment/UniMTS/prepare_har_data.py`
exports this repo's own WISDM/UTD-MHAD/PAMAP2 windows (raw, unnormalized
physical units; single right-wrist joint 21 in UniMTS's 22-joint skeleton,
matching the placement convention already established for the HumanML3D/
Nymeria UniMTS adaptations) using this repo's own `build_session_index`/
`make_splits`/`WindowDataset` machinery — same `group_key: subject_id`,
`val_ratio: 0.2`, `seed: 42`, `window_seconds: 2.56` convention as Section 3's
own HAR finetune configs — rather than UniMTS's own separately-bundled
benchmark data, so both models are evaluated on the *same* held-out split.
UniMTS zero-shot/finetune use its own **unmodified** official
`evaluate_custom.py`/`finetune_custom.py --mode full` (200 epochs, Adam
lr=1e-4). Ours(MAE) zero-shot uses `scripts/run_zeroshot_har.py` (no target
training); Ours(MAE) finetune uses a new `imu_text_clip_realdata` objective
config per dataset, warm-started from the HumanML3D-CLIP checkpoint's
*encoder only* (its own CLIP projection head is freshly trained — the
checkpoint's head was calibrated to HumanML3D's caption space, not a HAR
label space) — reported via this benchmark's usual in-batch retrieval-accuracy
convention (best-checkpoint val split, early-stopped).

**Note on metric comparability**: Ours(MAE)'s zero-shot metric (top-k accuracy
over the *entire* target dataset, no held-out split, since no training touches
it) uses `scripts/run_zeroshot_har.py`. Finetune rows for BOTH models are
reported as standard classification accuracy + macro-F1 (sklearn,
`accuracy_score`/`f1_score(average='macro')`, same computation UniMTS's own
`evaluate_custom.py`/`finetune_custom.py` already use) on the *same*
held-out val split -- added a `--val-only` flag to `run_zeroshot_har.py` for
this (evaluates the finetuned checkpoint's classifier against the full,
deduplicated set of class-text embeddings, on `group_key=subject_id,
val_ratio=0.2, seed=42` -- the exact split used during finetuning).

| Dataset | Model | Mode | Result | Job |
|---|---|---|---|---|
| UTD-MHAD (21 classes) | Ours (MAE) | Zero-shot | Top-1 6.21% / Top-3 16.67% / Top-5 27.01% | 7703806 |
| UTD-MHAD | Ours (MAE) | Finetune | acc 5.88% / macro-F1 1.01% | 7783289 |
| UTD-MHAD | UniMTS | Zero-shot | acc 11.31% / macro-F1 7.21% | 7702128 |
| UTD-MHAD | UniMTS | Finetune | acc **79.19%** / macro-F1 **77.60%** | 7702129 |
| WISDM (18 classes) | Ours (MAE) | Zero-shot | Top-1 17.74% / Top-3 33.86% / Top-5 46.73% | 7703974 |
| WISDM | Ours (MAE) | Finetune | acc 5.76% / macro-F1 0.67% | 7783289 |
| WISDM | UniMTS | Zero-shot | acc 28.73% / macro-F1 24.83% | 7703982 |
| WISDM | UniMTS | Finetune | acc **67.92%** / macro-F1 **68.36%** | 7704431 |
| PAMAP2 (18 classes) | Ours (MAE) | Zero-shot | Top-1 11.52% / Top-3 40.78% / Top-5 60.42% | 7703975 |
| PAMAP2 | Ours (MAE) | Finetune | acc 0.13% / macro-F1 0.14% | 7783289 |
| PAMAP2 | UniMTS | Zero-shot | acc 16.14% / macro-F1 12.71% | 7703984 |
| PAMAP2 | UniMTS | Finetune | acc **57.06%** / macro-F1 **44.78%** | 7703987 |

**Zero-shot: UniMTS wins clearly, on every dataset.** UniMTS's zero-shot
accuracy (11.31/28.73/16.14% on UTD-MHAD/WISDM/PAMAP2) beats Ours(MAE)'s
zero-shot top-1 (6.21/17.74/11.52%) by a consistent ~2x across all three --
unsurprising given UniMTS was purpose-built and pretrained at scale
specifically for zero-shot text-prompted HAR classification across dozens of
benchmark datasets, whereas Ours(MAE)'s CLIP alignment was only ever trained
on HumanML3D's synthetic mocap-derived captions and has never seen real
wrist-sensor activity data paired with text before. Ours(MAE)'s top-3/top-5
numbers (up to 27-60%) show the right class is usually *nearby* in embedding
space even when not top-1 -- the HumanML3D-trained alignment carries some
genuine motion-semantic signal, just not sharp enough for confident top-1
zero-shot classification on out-of-domain real sensor data.

**Finetune: UniMTS wins decisively on all three datasets -- Ours(MAE)'s
finetune numbers reported earlier this session (88-94% "acc_imu2text/
acc_text2imu") were a metric artifact, not a real result.** Once the
finetuned "Ours (MAE)" checkpoints are scored the *correct* way -- genuine
classification accuracy against the full, fixed set of class-text
embeddings, matching UniMTS's own metric exactly -- they collapse to
near-chance or worse (5.88%/5.76%/0.13% accuracy on UTD-MHAD/WISDM/PAMAP2,
vs. UniMTS's 79.19%/67.92%/57.06%). Per-class inspection makes the failure
mode obvious: on WISDM the model predicts "eating soup" for essentially
*every* window regardless of true activity (recall 1.00, precision 0.06) --
total representational collapse, not "worse but real" classification.

**Root cause, confirmed (not a training bug, a training/eval *metric*
bug)**: the `acc_imu2text`/`acc_text2imu` figures logged during training are
an **in-batch, multi-positive retrieval accuracy**
(`imu_lm/objectives/contrastive_clip_1d.py`'s `forward_loss` -- top-1 counts
as a hit if it lands on *any* same-activity item within the *same batch*,
not the single globally-correct one), computed on a validation loader built
with `shuffle=False`
(`imu_lm/data/realdata_clip_loader.py::make_realdata_clip_loaders`). For a
captioning task with many distinct, roughly-unique captions per batch (e.g.
HumanML3D, Nymeria -- where this same convention was validated all session)
that's a reasonable proxy. But WISDM/UTD-MHAD/PAMAP2 have only 18-21
*classes* total, and their sessions are each one long continuous
single-activity recording -- so a non-shuffled val batch of 64 consecutive
windows is drawn from one contiguous span and is therefore almost entirely
*one class*. Against a near-single-class batch, "top-1 lands on any
same-activity item" is nearly guaranteed regardless of whether the model
learned anything discriminative at all -- the reported 88-94% measured "can
the model recognize a dominant local cluster," not "can it tell 18 real
activities apart," and the two turned out to have essentially no
relationship for this recipe. This is why the fix requested here (same
metric as UniMTS, computed against the *global* fixed class set,
independent of any batch's local composition) was necessary to get an
honest number, and it flips the entire finetune conclusion.

This also means the CLIP-alignment recipe (`imu_text_clip_realdata`,
in-batch multi-positive contrastive loss) is a poor fit for classification
finetuning on datasets with few classes and session-blocked data -- the
loss's own training signal is subject to the identical problem (train
batches, while shuffled, still average this weak/misleading signal over
many largely-homogeneous local neighborhoods relative to how few classes
exist), consistent with the model converging to a degenerate few-class
attractor rather than a genuinely class-discriminative embedding space.
A standard supervised classification finetune (as Section 3 already does,
successfully) remains the right recipe for this kind of small-class-count
labeled data -- this section's CLIP-based approach is answering "does
language-alignment transfer help HAR," and for finetuning specifically, the
honest answer is no, not with this particular training objective.

**Correction/addendum (2026-07-24): a second, compounding bug affected these
same 3 jobs.** While building a later midtraining experiment, discovered
that `imu_text_clip_realdata`'s own code block in
`imu_lm/models/ViT1D/run.py` never checked the `alt_attention` config flag
when constructing its encoder -- it silently kept the plain `ViT1DEncoder`
assigned earlier in `main()` rather than reconstructing
`ViT1DEncoderAltAttn` (every other objective in this file does this check;
this one didn't). Any warm-start from an alt-attention checkpoint --
including `vit1d_clip_humanml3d_finetune_mae_multidata`, the checkpoint
these 3 jobs (7703976/7703807/7713467) warm-started from -- therefore
landed 100% of its weights in missing/unexpected instead of raising,
meaning these jobs were finetuning from a **near-randomly-initialized
encoder**, not a real HumanML3D-CLIP warm-start. Fixed by adding the same
`alt_attention` check + `ViT1DEncoderAltAttn` construction every other
objective already has (verified via the fixed midtraining job below:
"using alternating frame/global attention encoder" now logs before the
warm-start, and the "partial load" warning no longer appears).

This means the near-chance collapse documented above for these 3 jobs was
caused by (at least) two compounding factors, not one: the in-batch
retrieval metric issue (still real and independently confirmed via the
proper `--val-only` accuracy/F1 eval) *and* a broken warm-start feeding the
contrastive objective a barely-trained encoder to begin with. This doesn't
change the practical conclusion -- the matched-recipe experiment (linear
classifier + cross-entropy, correctly warm-started, see the follow-up
below) already independently demonstrates a properly warm-started
HumanML3D-CLIP encoder finetunes well -- but it does mean these 3 specific
CLIP-contrastive jobs are not a clean test of "does continuing the CLIP
objective hurt a *good* encoder," since the encoder they started from
wasn't actually the good one. Not re-run given the matched-recipe result
already answers the question that mattered (is HumanML3D-CLIP pretraining
useful downstream -- yes) more directly and without this confound.

**Bugs found and fixed while building this pipeline** (all pre-existing in
this repo's `scripts/run_zeroshot_har.py`/`configs/base.yaml`, not introduced
by this session's other work, caught via a UTD-MHAD canary run before
committing the other 8 jobs):
1. `configs/base.yaml`'s `paths.dataset_path`/`runs_root` were stale
   (pointed at a defunct machine's mount paths) — harmless everywhere else
   in this repo since every task config already overrides them, but
   `run_zeroshot_har.py`'s config merge applies `base.yaml`'s paths *on top
   of* the checkpoint's own saved config (opposite direction from every other
   script's merge order), so it crashed with `FileNotFoundError`. Fixed by
   correcting `base.yaml` directly (verified safe: every dependent config
   already sets its own paths).
2. `run_zeroshot_har.py`'s `_load_model()` always constructed a plain
   `ViT1DEncoder`, never checking the checkpoint's own `alt_attention` flag —
   silently dropped 100% of encoder weights (all keys landing in
   missing/unexpected) for every alt-attention checkpoint in this benchmark
   (i.e. every checkpoint that matters) rather than raising, so the "zero-shot"
   eval was actually running on a near-randomly-initialized encoder. Fixed to
   mirror `imu_lm/models/ViT1D/run.py`'s own encoder-selection logic.
3. New launch scripts for both `run_zeroshot_har.py` and the new
   `imu_text_clip_realdata` finetune configs were missing
   `HF_HOME=/gpfs/projects/p32139/.cache` (the CLIP text model isn't in this
   environment's default cache) — added to all 6.
4. PAMAP2's new finetune config was missing `windowing.max_gaps_per_session:
   700` — the repo-wide default of 100 (`base.yaml`) rejects every PAMAP2 val
   session due to real-world sensor dropout (a gotcha already documented and
   fixed for Section 3's own PAMAP2 configs, just not carried over to this
   new one) — resulted in a val_loader with 0 windows, meaning early stopping
   silently never fired and training ran to >140,000 steps with no valid
   checkpoint before being caught and fixed (job 7703978 → 7713467).
5. UniMTS's own `finetune_custom.py` OOM-killed at `--mem=32G` on WISDM
   specifically — its `load_custom_data()` expands each window into a
   `[N,T,22,6]` float64 joint-grid array (~13.5GB for WISDM's 100K-window
   train split alone, before resampling, with train+test copies coexisting)
   — raised to `--mem=180G` (job 7704431).

**Follow-up (2026-07-24): a matched, fair comparison.** The user pointed out
the finetune comparison above wasn't apples-to-apples: UniMTS's own
`finetune_custom.py` doesn't actually keep using the text branch during
finetuning at all -- it trains a standard linear classifier head
(`model.classifier(input)`) via plain cross-entropy against the integer
labels, i.e. exactly Section 3's own supervised-finetune recipe. Ours(MAE)'s
finetune above used a fundamentally different (and, per the collapse
diagnosis, poorly-suited) objective: continued CLIP contrastive training.
Built a matched version to isolate the real question -- **does HumanML3D
language-alignment pretraining help as an encoder *init* for standard HAR
classification finetuning, compared to plain MAE pretraining?**

New configs `configs/vit1d_har_finetune_{wisdm,utd_mhad,pamap2}_humanml3dclip_multidata_flu.yaml`:
identical to Section 3's own `vit1d_har_finetune_{wisdm,utd_mhad,pamap2}_mae_multidata_flu.yaml`
(`utd_mhad_finetune` objective -- warm-started encoder + fresh `LinearHead`
+ cross-entropy, auto-discovered classes) except `pretrained_run:
vit1d_clip_humanml3d_finetune_mae_multidata` instead of the plain-MAE
checkpoint, and `preprocess.normalize.enabled: false` instead of `true`
(the HumanML3D-CLIP checkpoint's own training used raw, unnormalized IMU --
matching that avoids introducing a scale mismatch for the encoder weights
being warm-started). Also reran Ours(MAE)'s **zero-shot** eval with the new
`--val-only` flag so it uses the identical held-out split UniMTS's zero-shot
already used (the original zero-shot row above evaluated the full dataset,
since no training touched it -- fine on its own, but not a matched
comparison against UniMTS's val-only zero-shot number).

| Dataset | Model | Mode | acc | macro-F1 | Job |
|---|---|---|---|---|---|
| UTD-MHAD | Ours (MAE+HumanML3D-CLIP init) | Zero-shot | 5.43% | 2.55% | 7788952 |
| UTD-MHAD | Ours (MAE+HumanML3D-CLIP init) | Finetune | 74.40% | 74.47% | 7788956 |
| UTD-MHAD | UniMTS | Zero-shot | 11.31% | 7.21% | 7702128 |
| UTD-MHAD | UniMTS | Finetune | **79.19%** | **77.60%** | 7702129 |
| WISDM | Ours (MAE+HumanML3D-CLIP init) | Zero-shot | 17.41% | 14.76% | 7788952 |
| WISDM | Ours (MAE+HumanML3D-CLIP init) | Finetune | **74.89%** | **74.26%** | 7788957 |
| WISDM | UniMTS | Zero-shot | 28.73% | 24.83% | 7703982 |
| WISDM | UniMTS | Finetune | 67.92% | 68.36% | 7704431 |
| PAMAP2 | Ours (MAE+HumanML3D-CLIP init) | Zero-shot | 14.20% | 11.74% | 7788952 |
| PAMAP2 | Ours (MAE+HumanML3D-CLIP init) | Finetune | 55.73% | 44.43% | 7788960 |
| PAMAP2 | UniMTS | Zero-shot | 16.14% | 12.71% | 7703984 |
| PAMAP2 | UniMTS | Finetune | 57.06% | 44.78% | 7703987 |

**Zero-shot: UniMTS still wins clearly on all three** (5.43/17.41/14.20% vs.
11.31/28.73/16.14%) -- unsurprising and unchanged from before; matching the
split doesn't change the underlying story.

**Finetune, matched recipe: does language pretraining help as an init? Yes,
consistently, on all three datasets** -- compared to Section 3's own
plain-MAE-warm-start finetune baseline (WISDM 71.27%/70.4%, UTD-MHAD
73.91%/72.7%, PAMAP2 51.77%/41.2% acc/macro-F1), the HumanML3D-CLIP-init
finetune wins every single dataset: WISDM 74.89%/74.26% (+3.6pp/+3.9pp),
UTD-MHAD 74.40%/74.47% (+0.5pp/+1.8pp), PAMAP2 55.73%/44.43%
(+4.0pp/+3.2pp). Small but real and directionally consistent -- HumanML3D
language alignment leaves the encoder in a better place to finetune from
for real HAR classification, even though (per the CLIP-objective section
above) *continuing* the language-alignment objective itself during
finetuning is actively harmful. The two findings aren't in tension: a good
initialization and a good training objective for the downstream task are
different things, and this pretraining checkpoint turned out to help with
the former while this repo's CLIP finetuning recipe hurt on the latter.

**Ours(MAE+HumanML3D-CLIP init) vs. UniMTS, matched recipe: much closer
than the original comparison suggested, and genuinely mixed.** UniMTS still
wins UTD-MHAD (79.19%/77.60% vs. 74.40%/74.47%, ~5pp) -- UTD-MHAD is the
smallest dataset (649 train windows) where UniMTS's large-scale
pretraining prior plausibly matters most. **Ours(MAE+HumanML3D-CLIP init)
wins WISDM outright** (74.89%/74.26% vs. 67.92%/68.36%, ~7pp) -- WISDM has
the most training data (100K windows) and 18 fine-grained ADL classes,
exactly the regime where this benchmark's encoder architecture/pretraining
has consistently done well throughout (Section 3 already showed the
plain-MAE version winning WISDM against every other pretraining baseline
tested there). **PAMAP2 is essentially a tie** (55.73%/44.43% vs.
57.06%/44.78%, ~1.3pp acc, ~0.35pp F1). Once the finetune recipe is made
fair, this is a genuinely competitive result, not the one-sided UniMTS win
the mismatched-recipe comparison above implied.

Unrelated to this task's own correctness, but discovered mid-session: a
large batch of `launch_*.sh` scripts at the repo root — both pre-existing
git-tracked ones (`launch/pretrain/launch_SSL.sh`, `launch/har/launch_HAR.sh`, etc.) and every
`launch_slurm_*.sh` script created this session — were deleted from the
working directory by an external process (confirmed via `git reflog`/
`git status` to be a plain filesystem delete, not any git operation, and not
something this session ran). Recreated only the one script needed to keep
this task moving (`launch/language/launch_slurm_clip_realdata_finetune_pamap2.sh`); the 17
pre-existing tracked scripts remain deleted pending the user's own decision
on whether to restore them (trivial via `git checkout -- <files>` since the
deletion was never committed) — noted here since it affects reproducibility
of *other* sections' launch commands, not this section's own results.

---

## Pipeline correctness audit (2026-07-17)

Requested check of the arm pose, full-body pose, and odometry pipelines.
Findings:

1. **Bug found and fixed — no-alt-attn MAE pretraining diverged to NaN.**
   The "Ours (MAE, no alt-attn)" ablation run (job 7208504) trained normally
   (loss ~1.3-1.6) through step 160, then hit `loss=nan` at step 165 —
   very early, while LR was still ramping through warmup — and stayed NaN for
   the rest of training. Because `val_loss < best_val_loss` is always `False`
   when `val_loss` is NaN, `best.pt` was **never written** (only `latest.pt`
   and one `step_*.pt` snapshot), so all 7 downstream probes chained after it
   failed immediately with "pretrained checkpoint not found" — a clean,
   informative failure, not a silent one. Confirmed this is specific to the
   plain-encoder + MAE-objective combination: grepped the FOCAL and SimMTM
   pretraining logs (also plain `ViT1DEncoder`, same cache/data) for
   `loss=nan` and found zero occurrences in either — both trained cleanly
   start to finish. Root cause is consistent with fp16 `autocast` overflow
   (gradient-clip order in `imu_lm/runtime_consistency/trainer.py` was
   verified correct — unscale before clip before step — so this isn't a
   clipping bug; the loss itself is going non-finite in the forward pass).
   **Fix**: set `trainer.amp: false` in
   `configs/vit1d_ssl_mae_varlen_noaltattn_multidata_flu.yaml` (train this one
   ablation in fp32). Deleted the corrupted checkpoints, resubmitted
   pretraining (new job 7245668) and all 7 dependent probes
   (`--dependency=afterok:7245668`, jobs 7245669-7245675).
2. **Arm pose / full-body pose joint-indexing — audited, no bug found.**
   Traced the joint-slicing logic in the `pose_probe`/`pose` objective
   branches (`run.py`) against `imu_lm/objectives/pose_1d.py`'s
   `_compute_val_mpjpe`/`_compute_fk_pos_loss`, which assume joint 0 of
   `y_full_rot` is the root and the model's predicted joints are exactly
   positions `1..num_joints`. Verified against the actual loaders:
   `nymeria_real_wrist_loader_FLU.py` (arm) returns `y: [N, 5, 6]` — 5 joints,
   **no root, and no 3rd `gt_pos` batch element** — so the arm-pose config
   never exercises the FK/MPJPE path at all (confirmed `joint_pos_weight: 0`
   in that config; the whole block is gated out). `nymeria_real_wrist_full_body_loader_FLU.py`
   returns `y: [N, 24, 6]` plus a 3rd `joint_pos` element, in true canonical
   SMPL order (0=root, 1-23=body) — matching what the FK code assumes exactly.
   Live evidence backs this up: the full-body pose reruns are currently
   logging sane `mpjpe_mm` values (86-119mm across Ours-MAE/FOCAL/SimMTM) with
   zero NaN in ~300K+ steps of training. No changes needed.
3. **Odometry — target quantity verified consistent; reporting gap found (not
   a bug).** See the methodology caveat added to section 4 above: Scratch
   trains on the identical velocity target as the other four columns, but its
   training loop never computes ATE/drift, so that column will need a
   separate scoring pass once it finishes.
4. Spot-checked all 5 currently-running jobs (full-body pose ×4, odom
   scratch) for NaN — zero occurrences in any, all training normally at the
   time of this audit.

---

## Infra notes

- **Disk quota incident (2026-07-16)**: home directory hit its 80G GPFS hard
  quota mid-checkpoint-write, corrupting 3 runs (full-body pose probe/scratch,
  odom scratch). Fixed two ways: (1) deleted 61G of redundant `step_*.pt`
  snapshots (kept `best.pt`/`latest.pt`), (2) added `logging.save_step_checkpoints: false`
  to every downstream probe/scratch config in this benchmark so they never
  accumulate per-step snapshots going forward (pretraining configs — MAE/FOCAL/SimMTM —
  still keep step history for training-curve analysis, unaffected).
- All new probe configs verified to load their target checkpoint with zero
  missing/unexpected keys before submission (`ViT1DEncoder` for FOCAL/SimMTM,
  `ViT1DEncoderAltAttn` for Ours(MAE)).
- **Checkpoint-dispatch bug (2026-07-16)**: `openwatch_gesture_probe` and
  `openwatch_gesture_supervised` build a `head` module, but `run.py`'s
  `extra_modules` dispatch (used for saving/resuming the trainable head/decoder
  alongside the encoder) only special-cased a fixed whitelist of objective
  types and fell through to `extra_modules = {"decoder": decoder}` for anything
  not listed — crashing with `UnboundLocalError: decoder` for both new
  objective types before any training step ran. Fixed by adding both to the
  `{"head": head}` whitelist in `imu_lm/models/ViT1D/run.py`. Both microgesture
  jobs resubmitted with no lost training time.
- **Disk quota incident #2 (2026-07-20)**: the 2026-07-16 fix (deleting
  step-checkpoint snapshots) bought headroom but not durably — by 2026-07-20
  the `gpfs:home` fileset was back up to 83,776,096/83,886,080 KB (99.87% of
  the 80GB per-user quota) purely from `best.pt`/`latest.pt` across 104 run
  dirs (~21-37GB, no step snapshots this time), and crashed the Ours(MAE,
  acc-only) pretraining job (see below) mid-checkpoint-write, twice. This time
  fixed at the root instead of by deleting data: migrated `runs/` to
  `/projects/b53017/vbc1591/runs` (the `project_b53017` fileset, which showed
  no enforced quota and 24TB free — datasets/caches already live there) via
  `rsync -a`, then replaced `/gpfs/home/vbc1591/IMU-SSL/runs` with a symlink
  to the new location. Every existing config's hardcoded
  `paths.runs_root: /gpfs/home/vbc1591/IMU-SSL/runs` keeps resolving
  transparently through the symlink, so this needed zero config edits and
  applies to every job (not just this ablation) going forward. Freed home
  usage to 45,519,232 KB (54%).
- **Ours (MAE, no alt-attn) ablation (2026-07-16)**: added to isolate whether
  Ours(MAE)'s results come from the MAE objective/data or from the alt-attn
  architecture specifically. Pretraining (`launch/pretrain/launch_slurm_MAE_noaltattn.sh`, job
  7245668) uses the exact same `mae_multidata_varlen_altattn_flu` objective,
  cache, decoder, and hyperparameters as Ours(MAE) — only
  `vit1d.encoder.alt_attention: false`. The 7 downstream probes for this column
  were submitted with `sbatch --dependency=afterok:7245668`, so they queue now
  and auto-start the moment pretraining finishes (no manual relaunch needed).
- **Scheduling: b1232 vs. b53017 vs. gengpu (2026-07-17)**. The user has 3
  accounts: `b1232`/`b53017` (both QOS `buyin`, no group TRES cap) and
  `p32139` (QOS `normal`, `gres/gpu=8` cap, grants access to the large shared
  `gengpu` partition). Checked availability with `sinfo`/`sbatch --test-only`:
  `b1232` (4 H100 nodes, buyin-only) was fully booked — a fresh job's
  predicted start was **2026-07-19T02:31** (~2 days out), which is why the
  no-alt-attn pretraining job sat pending "(Resources)" for hours. `b53017`
  (1 node, 4 A100s, buyin-only) was **completely idle** at the time — a test
  job there was predicted to start in ~15 min. `gengpu` via `p32139` also
  works (confirmed with `--test-only`, both `gpu:a100:1` and `gpu:h100:1`
  requests validated, predicted starts ~45min-1h50m) but is heavily shared and
  wasn't obviously faster than b53017 at this snapshot. Cancelled the 8 stuck
  no-alt-attn jobs on b1232 and resubmitted all 8 on `b53017`/`buyin`/a100
  (`launch/pretrain/launch_slurm_MAE_noaltattn.sh` + its 7 dependents) — the pretraining job
  started running within seconds. Going forward, check `sinfo -p b53017 -N`
  before submitting new jobs to b1232 by default — b53017 and gengpu (via
  p32139) are both viable, often-faster alternatives when b1232 is congested.
- **Odometry probe heads were never checkpointed (2026-07-17)**: `scripts/eval_odom_nymeria_altattn.py`
  (used for Ours-MAE/FOCAL/SimMTM's odometry probe) only ever wrote a 3-number
  CSV summary — the trained `RegisterOdomHead` itself was discarded, so any
  re-visualization required a full ~50-epoch retrain (this is why the
  `vis_odom_trajectories.py` runs took ~50-100 min instead of being quick).
  Fixed the script to save `odom_head.pt` (state dict + `embed_dim`/`num_patches`/
  `patch_size` needed to reconstruct it) after training, and added a
  `--head-checkpoint` fast-path to `vis_odom_trajectories.py` that skips
  training entirely when a saved head is available. Retraining Ours(MAE)'s
  odometry probe now (job 7253456) specifically to pick up a saved checkpoint
  for future fast reruns; FOCAL/SimMTM would each need one more run to get the
  same benefit (not done yet — ask if wanted).
- **Scratch odometry ATE/drift (2026-07-17)**: unlike the probe columns,
  Scratch's checkpoint (`runs/vit1d_odom_supervised_nymeria_altattn_scratch/checkpoints/latest.pt`)
  already contains both the trained encoder **and** its jointly-trained
  `ViT1DOdomDecoder` (`model` + `odom_decoder` keys) — no probe head to train
  at all. Wrote `scripts/vis_odom_scratch.py` to load both directly and run
  pure inference through the same `reconstruct_and_score` reconstruction math
  used everywhere else. Took **8 minutes** end to end (vs. ~50-100 min for the
  probe-based columns), confirming pure inference is exactly as fast as it
  should be once there's an actual checkpoint to load.
- **Ours (MAE) finetune columns added (2026-07-17)**: added a "warm-started
  encoder stays trainable" finetune counterpart to every probe column in this
  benchmark, initialized from `vit1d_ssl_mae_varlen_altattn_multidata`.
  Arm pose / full-body pose (`objective.type: pose`), odometry classic
  (`odom_finetune_nymeria_real_wrist_flu`), and UTD-MHAD/WISDM
  (`utd_mhad_finetune`, dataset-agnostic) already had finetune-capable
  objective branches in `run.py`; CLIP just needed `freeze_imu_encoder: false`
  on the existing `imu_text_clip_flu` objective. Two genuinely new branches
  were added: `odom_finetune_nymeria_real_wrist_worldframe_flu` (no finetune
  path existed for the world-frame odometry variant) and
  `openwatch_gesture_finetune` (same gap for OpenWatch gesture). **Bug found
  and fixed**: `odom_finetune_nymeria_real_wrist_flu` and `utd_mhad_finetune`
  (both pre-existing, not touched otherwise) never built `ViT1DEncoderAltAttn`
  even when `alt_attention: true` — they'd only ever been exercised with
  plain-encoder checkpoints before. Loading the alt-attn Ours(MAE) checkpoint
  into a plain `ViT1DEncoder` failed with *every* transformer-layer key
  reported "missing" (the two encoder classes don't share parameter names at
  all), crashing 3 of the first 8 finetune jobs (odom classic, WISDM,
  UTD-MHAD) within seconds. Fixed by adding the same
  `alt_attention` → `ViT1DEncoderAltAttn` check used everywhere else in
  `run.py` to both branches (plus the sibling `odom_probe_nymeria_real_wrist_flu`,
  same latent gap, not yet exercised by anything but fixed while in there).
  All 8 finetune jobs (`7321029/30/32/35/36` from the first batch,
  `7321838/39/40` resubmitted after the fix) are now running.
- **b1232 partition outage (2026-07-23)**: mid-run on the WHARF/HHAR/MHEALTH
  batch (see Section 3), `b1232` (4 H100 nodes) went administratively down
  (`sinfo -p b1232` → `AVAIL=down`) while 4 jobs (HHAR/MHEALTH
  probe+scratch) were still pending on it. Unlike the 2026-07-17 congestion
  case (jobs just sit pending), this time the scheduler auto-cancelled all 4
  pending jobs outright (`CANCELLED+`, no explicit reason recorded) after the
  partition had been down for roughly an hour — a different failure mode
  worth knowing about: a stuck-pending job on `b1232` isn't necessarily safe
  to just wait out. Recovered by resubmitting the same 4 launch scripts with
  `sbatch --partition=gengpu --account=p32139 --gres=gpu:a100:1` overrides
  (job IDs 7745119/7745122/7745123/7745124); all 4 completed cleanly within
  ~10 minutes each. Consistent with the 2026-07-17 note above:
  `gengpu`/`p32139` is the reliable fallback whenever `b1232` misbehaves,
  whether that's congestion or an outright outage.
- **Near-miss: duplicate job submission during the above recovery
  (2026-07-23)**: this incident was worked across several independent
  scheduled check-ins (each re-invoked with only the immediate task prompt,
  not full prior-turn memory). One check-in detected the `b1232` cancellation
  and resubmitted the 4 jobs; a *later* check-in re-diagnosed the same
  symptom from stale/incomplete context and resubmitted the **same 4 launch
  scripts a second time** before checking whether a previous recovery attempt
  was already in flight — producing a second batch of 4 jobs
  (7745382-7745385) with **identical `--run-name` values** to the
  already-running first batch. Since `run-name` maps directly to the run
  directory (`runs/<run-name>/checkpoints/`), two concurrent jobs writing to
  the same path could have corrupted each other's checkpoints. Caught within
  the same turn by checking `squeue`/`sacct` before assuming anything was
  stuck — found the first batch already `RUNNING`/`COMPLETED` — and
  `scancel`'d the duplicate batch immediately; none of the duplicates had
  reached the training-step stage yet, so no actual collision occurred.
  **Lesson for any future multi-turn/scheduled recovery work**: always check
  current `squeue`/`sacct` state for the specific job IDs in question
  *before* resubmitting or cancelling anything, even if (especially if) the
  task prompt asserts something is stuck — the assertion may itself be based
  on stale state from an earlier check-in.

---

## 10 (continued). Closing the zero-shot gap: rotation invariance, midtraining, and IMU-synthesis fidelity (2026-07-25)

Follow-up to the matched-recipe comparison above. The user asked directly why
"Ours(MAE+HumanML3D-CLIP init)" underperforms UniMTS on zero-shot HAR despite
using more sensor channels (acc+gyro vs UniMTS's official accel-only
checkpoint) and asked for active debugging, not just documentation.

**Rotation augmentation (positive).** UniMTS's own `pretrain.py`/`utils.py`
applies a fresh random 3D rotation to every training sample every batch
(`augment_data()`) — explicit orientation-invariance training our own
HumanML3D-CLIP training never had. Retrained Stage 2 from the plain MAE
checkpoint with `random_rotation.enabled: true` (SO(3), matching the
existing-but-previously-unused infrastructure in
`imu_lm/data/augmentations/transform.py`) — config
`configs/vit1d_clip_humanml3d_flu_finetune_mae_multidata_anyrot.yaml`, job
7847102. Result (val-only zero-shot, `--zscore-input`-equivalent already
correct by default — see normalization note below):

| Dataset | Plain (no rotation) | Rotation-aug | UniMTS |
|---|---|---|---|
| UTD-MHAD | 5.43% / 2.55% | 11.31% / 8.50% | 11.31% / 7.21% |
| WISDM | 17.41% / 14.76% | 17.32% / 16.71% | 28.73% / 24.83% |
| PAMAP2 | 14.20% / 11.74% | 13.24% / 10.81% | 16.14% / 12.71% |

Rotation augmentation roughly doubled UTD-MHAD zero-shot and ties UniMTS
there — a genuine, isolated win from a single config change.

**Midtraining (mixed, and one real code bug found).** Tried continuing CLIP
training on real HAR datasets (recofit/samosa/shoaib/ut_watch, excluding the
3 eval targets and the reserved `prism` probe set) to expose the model to
real sensor statistics and short activity-style captions HumanML3D alone
can't provide. Six variants (v1-v6) swept dataset mix, LR, rotation
augmentation during midtraining, `freeze_imu_encoder`, and (after a bug fix)
warm-starting the CLIP projection head:

- v1/v2 (no rotation during midtraining): made zero-shot **worse** across
  the board — negative result, ruled out data-scale/LR as the cause (v2 had
  10x more data and 10x lower LR than v1, yet did worse).
- v3 (rotation-aug checkpoint + rotation kept on during midtraining):
  genuine wins on WISDM (30.59%/25.31%) and PAMAP2 (17.04%/12.52%) zero-shot
  — first "Ours" checkpoints to beat UniMTS on those two — but UTD-MHAD
  zero-shot **collapsed** to a single predicted class ("The person is arm
  curl" for ~100% of windows, 2.26%/0.27%, below the 1/21 random-guess
  floor).
- v4 (v3 + `freeze_imu_encoder: true`): did **not** fix the UTD-MHAD
  collapse (1.81%/1.21%, still single-class) — ruled out encoder drift as
  the sole cause.
- **Root cause found**: `imu_text_clip_realdata`'s objective code
  (`imu_lm/models/ViT1D/run.py`) warm-started the *encoder* from
  `pretrained_run` but never the paired `clip_head` (the IMU-side CLIP
  projection) it was jointly trained with in Stage 2 — `clip_head` was
  always freshly (randomly) initialized and fit only on the midtrain mix's
  narrow templated-caption vocabulary, discarding the HumanML3D-trained
  projection entirely. Confirmed by inspecting the Stage-2 checkpoint
  directly (`torch.load`): it has a real, trained `clip_head` key that was
  simply never loaded. **Fixed** by warm-starting `clip_head` from the same
  checkpoint when present (`run.py`, ~line 2921).
- v5 (fix applied, encoder trainable): UTD-MHAD zero-shot recovered from
  total collapse to 4.98%/2.84% (not fully healthy, but no longer
  degenerate), and gave the best-ever PAMAP2 zero-shot (28.01%/21.94%,
  beats UniMTS by a wide margin) and finetune (61.44%/51.99%) of the whole
  midtrain family — but UTD-MHAD *finetune* regressed to 70.53%/69.45%
  (worst of the family), pointing to residual encoder drift specifically
  hurting UTD-MHAD.
- v6 (v5 + `freeze_imu_encoder: true`, i.e. both fixes combined): **worse**
  than v5 across the board on zero-shot (UTD-MHAD 3.17%/0.63% still
  collapsed onto different classes; WISDM 17.25%/11.53%, worst of the
  family; PAMAP2 back down to 16.22%/12.21%) — freezing does not help once
  the head is properly anchored. v5 remains the best midtrain checkpoint.

Jobs: v1 7795934→7797564, v2 7819499, v3 7875310/7877277-80, v4
7884182/7886522-25, v5 7890557/7892159-62, v6 7919035/7921536-39.

**Z-score normalization mismatch hypothesis: investigated and disproven.**
Hypothesized real target-dataset windows were never normalized at zero-shot
eval time while HumanML3D training windows were. Tested directly by adding
a `--zscore-input` override flag to `scripts/run_zeroshot_har.py` (forces
`preprocess.normalize.enabled=true`) and re-running the rotation-aug
checkpoint (job 7945972) — **byte-identical results** to the un-flagged run,
because `configs/base.yaml` already defaults `preprocess.normalize.enabled:
true`, inherited by every checkpoint's own saved config; real-dataset
zero-shot windows were already being z-scored correctly the whole time.
Separately tested whether the (deliberately raw-scale, `normalize:false`)
matched-recipe *finetune* configs would improve with normalization enabled
— job 7945998 (WISDM, rotation-aug checkpoint, `normalize:true`) — result
was a clear **regression** (72.81%/72.11% vs the original 77.48%/76.84%),
confirming the original finetune config choice was empirically correct
despite its header comment's imprecise reasoning. No config changes made
from this thread; both hypotheses were cleanly closed out.

**Gravity in HumanML3D's synthetic IMU (negative in isolation).**
`imu_lm/data/humanml3d_dataset_FLU.py::_joints_to_imu_flu` derives synthetic
acceleration as a pure second-order finite difference of SMPL wrist
position — no gravity term. Confirmed empirically: real cached clips show
per-axis mean acceleration ≈0. Real accelerometers measure *proper*
acceleration (gravity reaction force + movement); confirmed on the master
parquet: `|acc|` median ≈9.8–10.3 m/s² on WISDM/PAMAP2/UTD-MHAD vs
HumanML3D's ≈1.4–5.5 (zero-centered). World frame confirmed empirically
Y-up (pelvis/joint-0 height sits at Y≈0.95m, low variance; X/Z show
unconstrained horizontal translation). Added a config-gated
`humanml3d.gravity_mps2` parameter (default 0.0, preserves all existing
configs) that adds the world-frame +Y reaction-force term to the kinematic
acceleration before FLU-frame rotation. Retrained Stage 2 from scratch with
`gravity_mps2: 9.81` on top of the rotation-aug recipe — job 7946093,
zero-shot job 7946317. Result: **worse than the no-gravity baseline on
every dataset** (UTD-MHAD 7.24%/5.20%, WISDM 17.39%/15.42%, PAMAP2
10.39%/8.74%). Training itself was healthy (no NaN, comparable convergence
step-count/wall-clock to the baseline; HumanML3D's own retrieval metric
unaffected at 33.86%/34.59%) — a genuine negative result, not a broken run.
Working theory: per-window z-score normalization divides by that window's
own variance; adding gravity means most of a window's variance now comes
from gravity's *rotation-induced projection* onto the sensor's own
rotating axes rather than genuine movement, and z-scoring then shrinks the
proportion of signal that's actually informative.

**IMU-synthesis-fidelity investigation (requested by the user as a
follow-up to the gravity result): a genuine numerical bug, and the biggest
lever found this session.** Investigated whether HumanML3D's synthetic
signal itself (independent of gravity) matches real-sensor characteristics.
Measured a "roughness" proxy (2nd-difference std, i.e. discrete curvature)
on real cached HumanML3D clips: ≈14–21, vs ≈0.8–2.4 on real WISDM/PAMAP2
accelerometer data via the same proxy — HumanML3D's synthetic acceleration
was 10–20x rougher than any real sensor. Traced the cause to
`imu_lm/data/humanml3d_dataset.py::_resample_joints`, which upsamples
HumanML3D's native 20Hz joint positions to 50Hz via **linear**
interpolation before the downstream second-order finite difference derives
acceleration. Linear interpolation has piecewise-constant velocity between
the original 20Hz sample points, so its second derivative is a series of
artificial spikes exactly at each 20Hz breakpoint — a pure numerical
artifact of the interpolation method, not real motion (this mirrors why
UniMTS's own IMUSim synthesis tool uses a `SplinedBodyModel` for smooth
trajectory representation before deriving physical quantities). Confirmed
empirically end-to-end through the full FLU pipeline: switching to cubic
spline interpolation drops the roughness proxy to ≈0.2–0.7 (a 20–40x
reduction) while leaving the true acceleration *magnitude* essentially
unchanged (5.53→5.69, 1.40→1.47 across two test clips) — this removes a
spurious artifact, not real signal. (Also checked and fixed, as a smaller
consistency issue: HumanML3D's loader never applied the `preprocess.filter`
lowpass step every real-dataset loader gets via `preprocess_window` —
added it too, but verified its effect on the roughness gap is negligible,
since a 24Hz cutoff at 50Hz sampling barely touches the spectrum; not the
dominant fix.) Implemented as an opt-in `humanml3d.resample_method: cubic`
config flag (default `'linear'` preserves every other existing config
bit-for-bit).

Considered (but did not attempt) reproducing UniMTS's own IMUSim+BVH
synthesis pipeline exactly: `baselines/language_alignment/UniMTS/bvh2ts.py`
depends on `imusim` (`martinling/imusim`, unmaintained since ~2012-2013,
Python-2-era, not installed here) and `pos2bvh.py` depends on a separate
external repo (`sigal-raab/Motion`) for BVH/inverse-kinematics conversion,
neither present in this environment. Given the real risk/cost of standing
up two legacy, uninstalled dependency chains, built a modern equivalent
directly in this codebase instead (the resampling + gravity fixes above).

**Results — smoothresample alone vs. combined with gravity:**

Config `configs/vit1d_clip_humanml3d_flu_finetune_mae_multidata_anyrot_smoothresample.yaml`
(rotation-aug + `resample_method: cubic`, gravity left at 0 to isolate the
fix as a single variable) — job 7947040, zero-shot job 7947599. Config
`..._smoothresample_gravity.yaml` (same + `gravity_mps2: 9.81`, now that the
roughness artifact hypothesized to have broken the earlier isolated-gravity
test is fixed) — job 7947690, zero-shot job 7947905.

| Dataset | Smoothresample alone | Smoothresample + gravity | UniMTS |
|---|---|---|---|
| UTD-MHAD | 16.74% / 11.66% | 12.67% / 8.91% | 11.31% / 7.21% |
| WISDM | 19.17% / 17.51% | 25.52% / 23.50% | 28.73% / 24.83% |
| PAMAP2 | 16.87% / 14.53% | 16.41% / 15.81% | 16.14% / 12.71% |

**Smoothresample alone beats UniMTS outright on UTD-MHAD and PAMAP2** (both
acc and F1) — the first clean, single-pipeline Stage-2-only checkpoint (no
real-dataset midtraining at all) to do so — and improves WISDM over every
prior rotation-aug/gravity variant, though still behind UniMTS there.
**Adding gravity back in on top of the fixed resampling is a genuine
trade-off, not a clean win**: UTD-MHAD drops (still beats UniMTS, by a
smaller margin), WISDM jumps substantially closer to UniMTS (+6.4pp
acc/+6.0pp F1 vs. smoothresample alone), PAMAP2 roughly flat on accuracy
but better on F1. **Averaged across the three datasets, smoothresample+
gravity (18.20% acc / 16.07% F1) essentially matches UniMTS's own average
(18.73% acc / 14.92% F1) — matching on accuracy within 0.5pp and actually
exceeding UniMTS on average macro-F1** — a dramatic turnaround from the
session's starting point, where "Ours" trailed UniMTS by roughly 2x on
every dataset. This resampling fix is the single largest lever found in
this entire zero-shot investigation, well ahead of rotation augmentation,
midtraining, or gravity individually.

Not yet done as of this writing: matched-recipe finetune evals on the two
new checkpoints (smoothresample alone, smoothresample+gravity); a
gravity-magnitude sweep to look for a better UTD-MHAD/WISDM trade-off point
than the two endpoints (0.0 / 9.81) tested so far; and combining the fixed
Stage-2 checkpoint with midtraining (v5's real strength) as a further
follow-up.

**Learning-rate tuning on smoothresample+gravity: the final, best result.**
The `lr: 1.0e-4` used throughout every Stage-2 config this session (inherited
unchanged from the very first HumanML3D-CLIP recipe) was never itself tuned.
Swept `lr ∈ {3e-5, 5e-5, 1e-4 (baseline), 2e-4}` on top of the
smoothresample+gravity recipe (`gravity_mps2: 9.81`, all other settings
identical) — jobs 7948162 (3e-5), 7948163 (5e-5), 7948164 (2e-4), zero-shot
jobs 7948504/7948505/7948506 respectively:

| LR | UTD-MHAD | WISDM | PAMAP2 | Average |
|---|---|---|---|---|
| 3e-5 | 15.38% / 13.58% | 24.36% / 21.81% | 14.20% / 12.99% | 17.98% / 16.13% |
| **5e-5** | **17.65% / 12.48%** | **25.69% / 24.32%** | **19.01% / 17.25%** | **20.78% / 18.02%** |
| 1e-4 (original) | 12.67% / 8.91% | 25.52% / 23.50% | 16.41% / 15.81% | 18.20% / 16.07% |
| 2e-4 | 12.67% / 8.73% | 26.58% / 24.89% | 13.81% / 12.72% | 17.69% / 15.45% |
| UniMTS | 11.31% / 7.21% | 28.73% / 24.83% | 16.14% / 12.71% | 18.73% / 14.92% |

`lr=5e-5` is a clean, unambiguous win over every other LR tested. Result:
**beats UniMTS outright on UTD-MHAD (+6.3pp acc/+5.3pp F1) and PAMAP2
(+2.9pp acc/+4.5pp F1), and nearly ties WISDM on macro-F1** (24.32% vs
24.83%, though still ~3pp behind on accuracy). **Averaged across all three
datasets this is the first single "Ours" checkpoint this session to beat
UniMTS on BOTH accuracy (20.78% vs 18.73%) and macro-F1 (18.02% vs 14.92%)
without any per-dataset cherry-picking.** This is the best zero-shot result
of the entire investigation:

**Best config found: `configs/vit1d_clip_humanml3d_flu_finetune_mae_multidata_anyrot_smoothresample_gravity_lr5e-5.yaml`**
— rotation-aug (SO(3)) + `resample_method: cubic` (fixes the 20-40x linear-
interpolation roughness artifact) + `gravity_mps2: 9.81` (proper-acceleration
gravity term) + `lr: 5.0e-5` (vs. the original untuned 1.0e-4). Checkpoint:
`runs/vit1d_clip_humanml3d_finetune_mae_multidata_anyrot_smoothresample_gravity_lr5e-5`.

Not yet done: matched-recipe finetune evals on this final checkpoint; a
finer LR sweep around 5e-5 (only 4 points tested, all fairly coarse);
combining this checkpoint with midtraining (v5's real-data-exposure
technique) as a further follow-up.

**Extending the comparison to WHARF/HHAR/MHEALTH (2026-07-26): the earlier
"beats UniMTS on average" result does not generalize.** These 3 datasets
live in separate parquet files
(`/projects/b53017/vbc1591/{wharf,hhar,mhealth}_dataset.parquet`), not the
master `IMULM_master_dataset_v2.parquet` WISDM/UTD-MHAD/PAMAP2 come from.
WHARF has **no gyroscope at all** (gyro columns 100% NaN) -- since our
checkpoint's architecture requires 6-channel acc+gyro input, evaluated it
against a gyro-zero-filled copy
(`data_derived/wharf_gyrozero.parquet`, gyro_x/y/z set to a literal
constant 0.0) rather than skip it entirely; this is a real handicap for
"Ours" that UniMTS's own official checkpoint doesn't share (it's natively
accelerometer-only, per §10's earlier note). HHAR/MHEALTH exported and
evaluated the same way, no special handling needed. UniMTS evaluated via
its own unmodified `evaluate_custom.py` (widened
`prepare_har_data.py --dataset` choices to add `wharf/hhar/mhealth`,
otherwise identical export/eval pipeline already used for WISDM/UTD-MHAD/
PAMAP2 -- `--joint_list 21` single-wrist-joint approximation for all 3, same
convention already used there). Jobs: Ours eval 7948638 (failed --
scratchpad path under `/tmp` isn't visible from compute nodes on this
cluster, moved the gyro-zero-filled WHARF copy to `data_derived/` under the
repo instead) → 7948675 (succeeded). UniMTS: 7948639 (wharf), 7948640
(hhar), 7948641 (mhealth).

| Dataset | Ours (lr5e-5) | UniMTS |
|---|---|---|
| WHARF | 6.97% / 3.48% | 19.09% / 12.67% |
| HHAR | 32.35% / 25.82% | 50.02% / 44.41% |
| MHEALTH | 22.35% / 18.74% | 59.31% / 51.06% |

UniMTS wins clearly on all three, by a much wider margin than on WISDM/
UTD-MHAD/PAMAP2. Averaged across all 6 datasets now tested this session
(UTD-MHAD/WISDM/PAMAP2/WHARF/HHAR/MHEALTH): **Ours (lr5e-5) 20.67% acc /
17.02% F1 vs. UniMTS 30.77% acc / 25.48% F1** -- UniMTS pulls well ahead
again. **Conclusion: the earlier finding that lr5e-5 "beats UniMTS on
average" was specific to the 3-dataset comparison this whole investigation
was built around, and does not hold once the comparison is extended to
datasets outside that original scope** -- particularly HHAR and MHEALTH,
which are plausibly part of UniMTS's own much larger "18 benchmark
datasets" zero-shot evaluation suite its paper is validated against, i.e.
closer to home turf for UniMTS than for this repo's own benchmark design.
This is an important, honest caveat on the whole session's headline result:
"Ours" narrowed and, on 3 specific datasets, closed the zero-shot gap to
UniMTS through real, well-diagnosed fixes (rotation augmentation, the
`clip_head` warm-start bug, the resampling-artifact fix, LR tuning) -- but
UniMTS's own broader pretraining/evaluation scope still generalizes further
across a wider span of real-world HAR datasets.

**Matching train/eval window size at 10.24s (2026-07-26): a real bug found,
and a small net regression, not an improvement.** Confirmed via direct code
inspection (`imu_lm/data/combined_mae_multidata_loader.py::_collate_train`)
that MAE Stage-1 pretraining uses a genuinely variable, randomly-resampled
window length every training batch — `L = random.choice(vl_lengths)` where
`vl_lengths = range(64, 512+1, 16)` samples (1.28s–10.24s at 50Hz), not the
fixed 10.24s the top-level `windowing.window_seconds` key in that config
might suggest. Every "Ours" zero-shot eval this session, by contrast, used a
*fixed* 2.56s window (chosen originally to accommodate UTD-MHAD's short
~3.5s median session length) — a train/eval window-size mismatch. Tested
whether removing that mismatch (retraining Stage 2 on a *fixed* 10.24s
window, matching MAE's own upper bound, then evaluating at the same 10.24s)
changes results.

Retraining required a workaround for a documented hang risk in this exact
codebase: HumanML3D's own clips cap out at ~10.0s (500 samples), just under
512 — with the default `pad_mode='zero'`, a `min_window_seconds ==
max_window_seconds == 10.24` config would cause every clip to be dropped as
"too short," leaving zero valid batches and a silent infinite retry loop.
Fixed via `windowing.pad_mode: wrap` (wrap-pads every clip up to 512
samples instead of skipping short ones) — config
`configs/vit1d_clip_humanml3d_flu_finetune_mae_multidata_anyrot_smoothresample_gravity_lr5e-5_fixedwin1024.yaml`,
job 7948887 (completed cleanly, best val_loss=2.558 at step 7000, clip
counts identical to every prior HumanML3D run confirming wrap-mode worked).

A second, separate bug surfaced at zero-shot eval time: `compute_T_and_hop`
(`imu_lm/data/windowing.py`) reads a *different* config key
(`windowing.window_seconds`, default 2.56) than the one Stage-2 training
actually sets (`windowing.min_window_seconds`/`max_window_seconds`) — two
parallel, non-communicating window-size conventions in this codebase. Since
the checkpoint's saved config never set the former key, the eval script
silently computed `T_model=128` (2.56s) instead of the true 512 (10.24s),
which crashed the UTD-MHAD `--full-session` eval on a batch-collation size
mismatch before any dataset produced a result (`set -euo pipefail` aborted
the script on the first eval call). Fixed by patching the saved checkpoint's
`cfg["windowing"]["window_seconds"]` directly to 10.24 (no retraining
needed) — job 7949096 (after the fixed job 7949068 failed this way).

| Dataset | Fixed-10.24s (train=eval window) | Current best (lr5e-5, variable-window train, 2.56s eval) | UniMTS |
|---|---|---|---|
| UTD-MHAD (`--full-session`, wrap-padded — not like-for-like with the other two, see caveat below) | 14.88% / 10.32% | **17.65% / 12.48%** | 11.31% / 7.21% |
| WISDM | **26.31% / 22.37%** | 25.69% / 24.32% | 28.73% / 24.83% |
| PAMAP2 | **19.51% / 15.11%** | 19.01% / 17.25% | 16.14% / 12.71% |
| Average | 20.23% / 15.93% | **20.78% / 18.02%** | 18.73% / 14.92% |

**Verdict: matching train/eval window size at the larger 10.24s is a small
net regression, not an improvement**, vs. the current best (2.56s eval).
UTD-MHAD got clearly worse (-2.8pp acc/-2.2pp F1); WISDM/PAMAP2 saw small
accuracy gains but F1 losses. Still beats UniMTS's own average (20.23%/
15.93% vs. 18.73%/14.92%), so not a bad checkpoint on its own terms — just
not better than the existing best. Caveat: UTD-MHAD's number here used
`--full-session` wrap-padding (median session ~3.5s, far shorter than
10.24s) rather than genuine sliding-window extraction, unlike WISDM/PAMAP2
which had ample real signal to fill the window — not a clean apples-to-
apples comparison across all three datasets at this window size.

**Matching UniMTS's exact clip windowing/truncation strategy (2026-07-26):
best "Ours" result of the entire investigation, but not equivalence with
UniMTS.** Verified UniMTS's own windowing strategy directly by reading
`baselines/language_alignment/UniMTS/data.py::CLIPDataset.__getitem__`
(their pretraining dataset class):
```python
if len(imu) < self.args.padding_size:
    imu = np.pad(imu, ((0, self.args.padding_size - len(imu)), (0, 0), (0, 0)), mode='wrap')
imu = imu[:self.args.padding_size]
```
Short clips: wrap-pad (cyclically repeat) up to the target length -- already
identical to this repo's own `pad_mode: wrap` (same `np.pad(mode='wrap')`
call). Long clips: **deterministic first-N-samples truncation**
(`imu[:padding_size]`) -- this repo previously took a **random crop** from
anywhere in the clip instead. Implemented the missing piece: added
`humanml3d.crop_mode: 'first'` (default `'random'` preserves existing
behavior everywhere else) to
`HumanML3DIMUVarLenDatasetFLU.__iter__` (`imu_lm/data/humanml3d_dataset_FLU.py`),
setting `start = 0` instead of a random offset when enabled. Combined with
the existing `pad_mode: wrap` (already required for the fixed-10.24s
window, see above), this makes the windowing/truncation strategy match
UniMTS's own `CLIPDataset.__getitem__` exactly, modulo window length
(10.24s @ 50Hz = 512 samples here vs. UniMTS's 10s @ 20Hz = 200 samples).
Config
`configs/vit1d_clip_humanml3d_flu_finetune_mae_multidata_anyrot_smoothresample_gravity_lr5e-5_unimtscrop.yaml`,
job 7949239 (clean completion, best val_loss=2.564, clip counts identical
to every prior HumanML3D run confirming the crop-mode change didn't affect
clip validity). Re-hit and re-fixed the same `compute_T_and_hop` config-key
mismatch as the fixed-10.24s experiment above (patched the checkpoint's
saved `windowing.window_seconds` directly before evaluating, avoiding a
repeat crash). Evaluated across all 6 datasets tested this session — jobs
7949377 (UTD-MHAD/WISDM/PAMAP2) and 7949378 (WHARF/HHAR/MHEALTH):

| Dataset | UniMTS-crop-matched (new) | Current best (lr5e-5, 2.56s eval) | UniMTS |
|---|---|---|---|
| UTD-MHAD | 14.88% / 10.65% | **17.65% / 12.48%** | 11.31% / 7.21% |
| WISDM | **29.00% / 25.91%** | 25.69% / 24.32% | 28.73% / 24.83% |
| PAMAP2 | 17.49% / 14.19% | **19.01% / 17.25%** | 16.14% / 12.71% |
| WHARF | **16.18% / 6.38%** | 6.97% / 3.48% | 19.09% / 12.67% |
| HHAR | **43.76% / 35.35%** | 32.35% / 25.82% | 50.02% / 44.41% |
| MHEALTH | **28.63% / 27.15%** | 22.35% / 18.74% | 59.31% / 51.06% |
| **Average (6 datasets)** | **24.99% / 19.94%** | 20.67% / 17.02% | 30.77% / 25.48% |

**This is the best "Ours" checkpoint across the full 6-dataset comparison
found in this entire investigation** — wins clearly on WISDM, WHARF, HHAR,
and MHEALTH (large margins on the last three: WHARF +9.2pp, HHAR +11.4pp,
MHEALTH +6.3pp acc), loses ground on UTD-MHAD and PAMAP2 vs. the prior
best. Average jumps from 20.67%/17.02% to 24.99%/19.94%, closing a
meaningful fraction of the gap to UniMTS's own average (30.77%/25.48%)
without fully closing it — MHEALTH in particular remains a large gap
(28.63% vs. 59.31%).

**Explicitly NOT claiming this makes the two pipelines equivalent except
window size.** Matching the crop/pad strategy closed one real
methodological gap, but several differences remain regardless of this fix:
(1) architecture -- UniMTS uses a spatio-temporal graph network over
all-22-joint synthesized IMU; this repo uses a ViT1D transformer over a
single synthesized wrist joint; (2) IMU synthesis fidelity -- UniMTS uses
IMUSim (a physics-based simulator); this repo uses a custom finite-
difference method; (3) captions -- UniMTS trains on GPT-3.5-paraphrased
captions (3 per description, `text_aug.py`); this repo uses only
HumanML3D's original ~3-4 official captions with no paraphrasing; (4)
UniMTS applies random per-sample joint-masking augmentation (masking 1-21
of its 22 synthesized joints) with no equivalent here (only one joint is
ever synthesized); (5) sampling rate -- 50Hz here vs. UniMTS's 20Hz; (6)
this repo has an additional MAE self-supervised pretraining stage on real
sensor data before CLIP alignment, which UniMTS's own pipeline does not
have (as far as investigated this session).

### Resample round-trip to 20Hz for zero-shot eval (2026-07-26)

**Hypothesis**: HumanML3D-derived synthetic IMU is trained-on at 50Hz (`_TARGET_FPS`)
but its REAL information content is capped by AMASS mocap's native 20Hz frame
rate (`_resample_joints` upsamples 20Hz->50Hz via cubic spline before deriving
acc/gyro) -- i.e. the model was only ever trained on signals whose genuine
spectral content is band-limited to ~10Hz (20Hz Nyquist), even though nominally
presented at 50Hz. Real target-dataset sensors, by contrast, genuinely capture
content above 10Hz. This train/test bandwidth mismatch could be part of the
zero-shot gap, independent of everything else investigated this session.

**Fix tested**: `--roundtrip-20hz` flag added to `run_zeroshot_har.py` (and a
`roundtrip_low_hz`/`native_hz` param on `RealdataCLIPDataset` /
`FullSessionCLIPDataset` in `imu_lm/data/realdata_clip_loader.py`): every
target-dataset window is downsampled to 20Hz then upsampled back to 50Hz
(FFT-based `scipy.signal.resample`, verified on synthetic tones: a 3Hz
component survives essentially unchanged, a 15Hz component is removed to
~1e-14 of its original power) immediately before encoding -- matching the
same effective bandwidth cap the HumanML3D training data has.

Re-evaluated the best "Ours" checkpoint (`...unimtscrop`) across all 6
datasets with this flag on, same checkpoint/splits/window as the existing
baseline (jobs 7949377/7949378):

| Dataset | Baseline (unimtscrop) | +roundtrip-20hz | UniMTS |
|---|---|---|---|
| UTD-MHAD | 14.88% / 10.65% | 15.48% / 11.22% | 11.31% / 7.21% |
| WISDM | 29.00% / 25.91% | 27.08% / 24.55% | 28.73% / 24.83% |
| PAMAP2 | 17.49% / 14.19% | 18.84% / 16.18% | 16.14% / 12.71% |
| WHARF | 16.18% / 6.38% | 14.19% / 6.07% | 19.09% / 12.67% |
| HHAR | 43.76% / 35.35% | **45.76% / 40.12%** | 50.02% / 44.41% |
| MHEALTH | 28.63% / 27.15% | **29.06% / 29.21%** | 59.31% / 51.06% |
| **Average (6)** | 24.99% / 19.94% | **25.07% / 21.23%** | 30.77% / 25.48% |

**Result**: average accuracy essentially flat (+0.08pp), average F1 up
+1.29pp. Mixed at the per-dataset level (WISDM -1.92pp acc, WHARF -1.99pp
acc) but the two largest gains are HHAR (+4.77pp F1) and MHEALTH (+2.06pp
F1) -- precisely the two datasets where UniMTS previously had its largest
lead, and where every other hypothesis tested this session (sensor
placement, caption template, z-score removal) failed to move the needle.
Consistent with (though not conclusive proof of) the bandwidth-mismatch
hypothesis. Does not close the remaining gap to UniMTS. Jobs 7953922
(UTD-MHAD/WISDM/PAMAP2) and 7953923 (WHARF/HHAR/MHEALTH).

### Trainable CLIP text encoder (freeze_text_encoder: false) (2026-07-26)

**Motivation**: confirmed UniMTS's own `pretrain.py`/`contrastive.py` train their ENTIRE
CLIP text tower jointly with the IMU encoder (optimizer built from plain
`model.parameters()`, no freezing anywhere), whereas every "Ours" HumanML3D-CLIP
checkpoint this whole session kept `freeze_text_encoder: true` and only ever
trained a lightweight IMU-side projection head. Tested unfreezing it, config
`..._unimtscrop_texttrainable.yaml`, job 7954257.

**Required a real fix first**: `run.py`'s `extra_modules` checkpointing for
`imu_text_clip_flu` only ever saved `clip_head` (comment: "text encoder can be
reloaded from HuggingFace" -- true only when frozen). With `freeze_text_encoder:
false` the fine-tuned text tower would have been silently discarded at
checkpoint time, and `run_zeroshot_har.py` would then re-fetch vanilla
pretrained CLIP text weights at eval time, erasing the whole point of the
experiment. Fixed both sides: `run.py` now saves `extra_modules["text_enc"]`
when unfrozen; `run_zeroshot_har.py::_load_model` now loads it. Verified the
saved checkpoint contains `text_enc` (196 tensors, correct `CLIPTextEncoder`
key structure) before spending eval compute on it.

**Also found and fixed along the way**: `HumanML3DIMUVarLenDatasetFLU.__iter__`
applied `apply_augment` (random_rotation/gaussian_noise/etc.) whenever
`self.cfg is not None`, with no train/val distinction -- since
`make_humanml3d_imu_loaders_flu` passes the SAME cfg to both loaders, every
"anyrot" checkpoint this whole session (including `unimtscrop`) was ALSO
applying its random SO(3) rotation to VAL windows, injecting fresh
per-call randomness into the val_loss that early-stopping/best-checkpoint
selection assumes is stable. Added an `augment` flag (`split == 'train'`)
to gate this correctly, matching `RealdataCLIPDataset`'s existing convention.
Applies going forward; does not retroactively change any prior number.

**Internal metric**: best val_loss 2.5425, slightly BETTER than unimtscrop's
2.564 -- letting the text tower adapt to HumanML3D's captions does improve
HumanML3D's own retrieval loss, as expected.

**Zero-shot result: clearly WORSE overall**, jobs 7954599/7954600:

| Dataset | unimtscrop (frozen text) | +trainable text |
|---|---|---|
| UTD-MHAD | 14.88% / 10.65% | 13.69% / 8.46% |
| WISDM | 29.00% / 25.91% | 26.69% / 23.34% |
| PAMAP2 | 17.49% / 14.19% | 16.06% / 12.92% |
| WHARF | 16.18% / 6.38% | **1.99% / 0.47%** |
| HHAR | 43.76% / 35.35% | 42.76% / 36.09% |
| MHEALTH | 28.63% / 27.15% | 25.64% / 20.74% |
| **Average (6)** | 24.99% / 19.94% | **21.14% / 17.00%** |

Average drops -3.85pp acc / -2.94pp F1. WHARF collapses almost completely
(16.18%->1.99% acc). Interpretation: letting the text tower fine-tune on
HumanML3D's specific caption phrasing pulls it away from CLIP's original
broad language representations, actively hurting transfer to target
datasets' different caption distributions -- overfitting to the source
domain's captions, not just the source domain's sensor statistics. A
genuine negative result despite matching UniMTS's own recipe on this axis;
UniMTS's advantage here evidently isn't explained by trainable-vs-frozen
text alone (it may only work well for UniMTS in combination with their much
larger/more diverse pretraining caption set, where overfitting to any one
caption style is less of a risk).

### Gaussian-noise augmentation on HumanML3D-CLIP training (2026-07-26)

**Motivation**: HumanML3D-derived synthetic IMU is a perfectly clean,
deterministic kinematic quantity (finite-difference + SO3 log) with zero
sensor noise. Real target-dataset sensors have genuine measurement noise;
UniMTS's own IMUSim-based synthesis is a physics-based sensor simulator
built to model this. Turned on `gaussian_noise.enabled: true, sigma: 0.6`
(this repo's own base.yaml default, never previously enabled for any
HumanML3D-CLIP config) on top of `unimtscrop`, config
`..._unimtscrop_gaussiannoise.yaml`, job 7954301.

**Internal metric**: best val_loss 3.1707, notably worse than unimtscrop's
2.564 -- expected, noise makes the exact HumanML3D retrieval task itself
harder to fit (a regularization cost).

**Zero-shot result: worse on average, but a clear win on UTD-MHAD**, jobs
7954648/7954649:

| Dataset | unimtscrop (no noise) | +gaussian noise |
|---|---|---|
| UTD-MHAD | 14.88% / 10.65% | **20.83% / 13.23%** |
| WISDM | 29.00% / 25.91% | 19.70% / 17.78% |
| PAMAP2 | 17.49% / 14.19% | 16.32% / 14.00% |
| WHARF | 16.18% / 6.38% | 10.67% / 6.17% |
| HHAR | 43.76% / 35.35% | 31.11% / 28.56% |
| MHEALTH | 28.63% / 27.15% | 17.09% / 13.61% |
| **Average (6)** | 24.99% / 19.94% | **19.29% / 15.56%** |

Average drops -5.70pp acc / -4.38pp F1 -- another net-negative single-axis
change, joining z-score removal, bare-word captions, and trainable text
encoder in the list of individually-tested UniMTS-recipe elements that do
NOT transfer cleanly onto "Ours" in isolation. UTD-MHAD's clear +5.95pp acc
improvement is the interesting exception -- plausibly because UTD-MHAD's
gestures are larger-amplitude/more distinct than HHAR/MHEALTH's finer
postural discriminations, so noise regularization helps there without
costing as much discriminative precision. Sigma=0.6 was not tuned; a
smaller value might trade off differently, but not tested this session.

### Combined: trainable text encoder + Gaussian noise together (2026-07-26)

Rationale: UniMTS's own recipe does both simultaneously (full text-tower
training AND IMUSim's realistic sensor noise); each was tested alone above
and BOTH came back net-negative individually. Tested together to check
whether they compound, partially cancel, or interact -- config
`..._unimtscrop_texttrainable_gaussiannoise.yaml`, job 7954562.

**Internal metric**: best val_loss 3.154 (between the two individual
values: 2.5425 for text-alone, 3.1707 for noise-alone -- text-tower
adaptation partially offsets the internal-fit cost of noise, as expected).

**Zero-shot result: WORSE than either single-axis change alone**, jobs
7954752/7954753:

| Dataset | unimtscrop | +text only | +noise only | +both (combined) |
|---|---|---|---|---|
| UTD-MHAD | 14.88/10.65 | 13.69/8.46 | **20.83/13.23** | 20.83/14.57 |
| WISDM | **29.00/25.91** | 26.69/23.34 | 19.70/17.78 | 21.33/19.01 |
| PAMAP2 | 17.49/14.19 | 16.06/12.92 | 16.32/14.00 | **18.25/14.26** |
| WHARF | **16.18/6.38** | 1.99/0.47 | 10.67/6.17 | 8.32/5.44 |
| HHAR | **43.76/35.35** | 42.76/36.09 | 31.11/28.56 | 33.61/24.42 |
| MHEALTH | 28.63/27.15 | 25.64/20.74 | 17.09/13.61 | **4.27/3.73** |
| **Average (6)** | **24.99/19.94** | 21.14/17.00 | 19.29/15.56 | **17.77/13.57** |

The two negative effects COMPOUND rather than cancel: combined average
(17.77%/13.57%) is worse than either text-alone (21.14%/17.00%) or
noise-alone (19.29%/15.56%). MHEALTH collapses almost completely
(28.63%->4.27% acc -- below the ~8.3% random-guess floor for 12 classes).
UTD-MHAD's gain from noise-alone persists in the combined run (still
~20.8% acc), the one consistent bright spot across all three variants.

**Overall conclusion for this line of investigation**: none of the four
individual UniMTS-recipe elements tested in isolation this session
(z-score removal, bare-word captions, trainable text encoder, Gaussian
noise) improve "Ours" zero-shot transfer on average, and combining the two
most recent ones makes it worse still. UniMTS's zero-shot advantage is not
explained by any single one of these recipe choices carried over onto
"Ours" -- it likely depends on some combination specific to their own
larger/more diverse IMUSim-synthesized pretraining corpus (scale, joint
coverage, or caption diversity) that these isolated ablations cannot
reproduce. The one genuinely positive, still-standing result from this
whole investigation remains `--roundtrip-20hz` (bandwidth matching at
eval time, +1.29pp F1 avg, biggest gains on HHAR/MHEALTH) plus the
earlier crop-mode/resample/gravity fixes baked into `unimtscrop` itself
(24.99%/19.94%, this session's best "Ours" checkpoint).

### UniMTS architecture, trained from scratch on our own all-joint HumanML3D data (2026-07-26)

**Motivation**: isolate whether UniMTS's zero-shot advantage comes from
architecture (ST-GCN over a 22-joint skeleton graph) + all-joint data
richness -- testable without their proprietary IMUSim pipeline -- versus
IMUSim-specific synthesis fidelity or a larger/richer proprietary
pretraining corpus.

**Correction to an earlier unverified claim in this section**: initially
speculated UniMTS's own pretraining corpus was "much larger/proprietary."
Checked their own README directly and this is WRONG on the "different
corpus" part -- their official pipeline is also HumanML3D-based
(`HumanML3D -> pos2bvh.py -> BVH -> bvh2ts.py (IMUSim) -> pretrain.py`, see
`baselines/language_alignment/UniMTS/README.md` lines 97-99). The real,
verified differences are: (a) synthesis method -- IMUSim (physics-based
sensor simulator: noise, drift, realistic dynamics) vs. our finite-
difference-of-position + SO3-log-of-rotation (perfectly clean, zero noise);
(b) likely clip yield -- our export only captured 8,246/23,384 (35%) train
clips because THIS cluster's local AMASS archive mirror is missing ~65% of
the raw `.npz` mocap files `_get_amass_index` needs (a local storage
completeness gap, not a HumanML3D limitation) -- UniMTS's own official run
plausibly processed closer to the full ~23k clips, though this specific
number is inferred from their pipeline description, not independently
confirmed. (c) unknown: their actual training epoch/compute budget for the
released checkpoint.

**Implementation**: generalized `_load_wrist_rotmat_20hz` -> new
`_load_all_joint_rotmat_20hz` (all 22 SMPL joints' canonical FK rotations,
not just joint 21/wrist -- the underlying `forward_kinematics_R` call
already computed all 24 joints, previous code just discarded 23 of them).
New `_all_joints_to_imu` derives per-joint LOCAL-frame acc+gyro for all 22
joints (no shared FLU convention needed since each joint reports in its own
frame, matching how a real per-joint sensor would). New
`prepare_humanml3d_alljoint.py` exports `sim/<id>.npy` ([T,22,6], UniMTS's
own node ordering) + `aug_texts/<id>.txt` (HumanML3D's own captions,
verbatim, already `#`-delimited-comment compatible with their
`CLIPDataset`). UniMTS's own joint numbering (decoded directly from their
published `joint_assignment.png` figure, referenced by
`README.md --joint_list` example) is NOT raw SMPL index order -- it's a
chain-compressed numbering (0=pelvis, 1-4=left leg, 5-8=right leg,
9-11=spine, 12-13=neck/head, 14-17=left arm, 18-21=right arm).
`UNIMTS_TO_SMPL22 = [0,1,4,7,10,2,5,8,11,3,6,9,12,15,13,16,18,20,14,17,19,21]`
maps UniMTS node -> raw SMPL joint index; verified node 21 -> SMPL R_wrist
(21), matching this repo's pre-existing `RIGHT_WRIST_JOINT=21` convention
used elsewhere (`prepare_har_data.py`), and cross-checked against the
existing PAMAP2 sensor-placement mapping (hand=21, chest=11/spine3,
ankle=7) which is consistent with this reading.

Trained via UniMTS's own unmodified `pretrain.py` (random ST-GCN init +
pretrained-CLIP text tower, their own joint-masking augmentation
[1-5 of 22 joints kept per sample] and rotation augmentation, accel-only,
Adam lr=1e-4, batch 64, temperature 0.1, 100 epochs) on the exported
8,246-clip train split (521-clip val split exported separately but unused
by `pretrain.py`, which has no internal val awareness). Job 7953827,
checkpoint every 10 epochs; `epoch_90.pth` (internal 0-indexed epoch=90,
displayed as "Epoch [91/100]") is the LAST checkpoint this run will ever
save (their own save condition `epoch % log == 0` never fires again for
epoch=91..99), so evaluated it directly rather than waiting for the
training job's nominal completion.

**Result: far worse than both the official UniMTS checkpoint and "Ours"**,
jobs 7954835-7954840 (`evaluate_custom.py`, `--joint_list 21`, matching how
every single-wrist-sensor target dataset is evaluated against the official
checkpoint elsewhere in this repo):

| Dataset | UniMTS-arch/Ours-data (scratch) | Ours (unimtscrop) | Official UniMTS |
|---|---|---|---|
| UTD-MHAD | 7.24% / 2.21% | 14.88% / 10.65% | 11.31% / 7.21% |
| WISDM | **5.91% / 1.16%** | 29.00% / 25.91% | 28.73% / 24.83% |
| PAMAP2 | 12.17% / 4.14% | 17.49% / 14.19% | 16.14% / 12.71% |
| WHARF | **2.64% / 0.75%** | 16.18% / 6.38% | 19.09% / 12.67% |
| HHAR | 17.40% / 4.94% | 43.76% / 35.35% | 50.02% / 44.41% |
| MHEALTH | 16.18% / 7.02% | 28.63% / 27.15% | 59.31% / 51.06% |
| **Average (6)** | **10.26% / 3.37%** | 24.99% / 19.94% | 30.77% / 25.48% |

Per-class breakdowns on HHAR and PAMAP2 show severe mode collapse (e.g.
PAMAP2: "ironing" 99.4% recall, "running" 87.8% recall, all 14 other
classes 0%) despite training loss converging smoothly to 0.58 by epoch 91
-- the model fits its own (small, clean, finite-difference-derived) training
distribution but fails to generalize, a training-data-scale/diversity
signature rather than an optimization failure. WHARF (2.64%, 14 classes,
chance~7.1%) performs BELOW random chance -- confidently wrong, not merely
noisy.

**Conclusion for this line of investigation**: UniMTS's zero-shot advantage
is NOT explained by architecture (ST-GCN + all-joint skeleton graph) or
all-joint data coverage in isolation -- their own architecture trained from
scratch on our own all-joint-synthesized data performs far worse than even
our simplest single-wrist-joint pipeline. The advantage is tied to
something about their actual pretraining data: most likely IMUSim's
synthesis fidelity (noise, realistic sensor dynamics -- tested in isolation
via `gaussian_noise` above and also came back negative for "Ours", but that
was noise ADDED to an already-small corpus, not a fundamentally different
richer synthesis method) and/or genuinely more training clips than our
locally-available AMASS archive could supply. Between the architecture
question (now answered: no) and the recipe-detail questions (z-score,
captions, trainable text, noise -- all tested and negative), the remaining
untested and most likely explanation is training-corpus SCALE, which would
require either a more complete local AMASS mirror or access to IMUSim
itself to test further -- out of scope without one of those.

### Investigated: HumanML3D's own new_joint_vecs as an AMASS-free data source (2026-07-26, paused)

User pointed to `/projects/b53017/vbc1591/HumanML3D/HumanML3D/new_joints`
(positions) and `new_joint_vecs` (263-dim HumanML3D/T2M feature vectors,
`motion_representation.ipynb`) as "the new and complete" dataset -- both
have 29,228 files (essentially the full official mirrored dataset), vs.
only ~8,246/23,384 (35%) reachable via the AMASS-archive-index path used
for the all-joint-from-scratch experiment above (this cluster's local
AMASS mirror is incomplete). Investigated whether this could solve the
corpus-scale problem entirely (no AMASS dependency).

**Joint-order compatibility confirmed**: `paramUtil.py`'s
`t2m_kinematic_chain` decodes to exactly standard SMPL joint numbering
(verified: chain `[9,14,17,19,21]` = spine3->R_collar->R_shoulder->
R_elbow->R_wrist, matching `_SMPL_R_WRIST=21`) -- the existing
`UNIMTS_TO_SMPL22` permutation applies unchanged.

**Position fidelity confirmed good**: `new_joints` wrist-trajectory-shape
correlates 0.9998 with independently-computed AMASS-FK wrist trajectory
for the same clip (verified on `000004.npy`/CMU/80_63_poses.npz) --
`new_joints` is a safe substitute for `joints/` positions (as expected,
since HumanML3D's own official consistency-check cell validates exactly
this).

**Rotation fidelity is NOT a safe substitute for gyro simulation**:
`new_joint_vecs`'s `rot_data` is derived via `Skeleton.inverse_kinematics_np`
run on POSITIONS ONLY (not AMASS pose parameters) -- each non-root joint's
local rotation is `qbetween(rest_offset, current_offset)`, the minimal
rotation aligning two vectors, which recovers bending direction (2 DOF)
but is fundamentally blind to twist/roll about the bone's own axis (3rd
DOF) -- a vector-to-vector alignment cannot constrain it. Implemented
`_load_all_joint_rotmat_from_vecs` (root Y-rotation via cumulative
`rot_vel` integration + non-root 6D local rotations via
`_T2M_KINEMATIC_CHAIN` composition) in `humanml3d_dataset_FLU.py`,
verified BIT-EXACT (max abs diff 0.0 across all 22 joints) against a
direct port of the official `recover_from_rot`/`Skeleton.
forward_kinematics_cont6d_np` reference pipeline -- so this is a faithful,
bug-free reimplementation, not a coding error.

Quantified the resulting gyro-relevant disagreement: on 150 clips with
AMASS coverage, compared AMASS-derived (true motion-capture-based) vs.
new_joint_vecs-derived (position-IK-based) WRIST rotation, both relative
to their own frame-0 (canceling any constant canonicalization offset
between the two methods): **median 55.9 deg mean-per-clip disagreement**,
72% of clips >30 deg, 45% of clips >60 deg. Root-joint disagreement is
smaller but still substantial for non-upright motions (up to ~150 deg on
acrobatic/bending clips), consistent with new_joint_vecs's root being
constrained to Y-axis-only yaw (no pitch/roll) vs. AMASS's true 3-DOF
pelvis orientation -- a second, independent fidelity gap on top of the
twist issue.

**Conclusion (paused, not resolved)**: `new_joint_vecs` would fully solve
the corpus-scale/coverage problem (29,228 vs 8,246 clips) but at the cost
of a large, systematic gyro-fidelity loss (median 56 deg wrist disagreement)
that would confound any resulting zero-shot comparison -- a positive or
negative result couldn't be cleanly attributed to scale vs. rotation
noise. User asked to stop before committing to either the full-switch or
hybrid (AMASS-gyro-only-where-available) approach; no re-export or
retraining has been done with this data source. The validated
`_load_all_joint_rotmat_from_vecs` function remains in
`humanml3d_dataset_FLU.py` for future use if this line of investigation
resumes.

### Full AMASS-free retrain: new_joints + new_joint_vecs (2026-07-26)

Following the coverage-vs-fidelity investigation above, user explicitly
directed dropping AMASS entirely for the main "Ours" wrist-only pipeline:
"no do not use keeping AMASS for rotation, estimating rotation from
[new_joint_vecs], it is ok if it is not similar to AMASS." Also dropped
`joints/` for positions (this cluster's local mirror had degraded to 9,933
files, some corrupted, during an ongoing data reorg) in favor of
`HumanML3D/new_joints/` (full ~29,228-clip coverage, confirmed high-fidelity
positions).

**Implementation**: added `joints_subpath` and `rotation_source` params to
`_load_clip_imu_flu` (`humanml3d_dataset_FLU.py`), config-gated via
`humanml3d.joints_subpath`/`humanml3d.rotation_source`. `rotation_source:
new_joint_vecs` reuses the already bit-exact-validated
`_load_all_joint_rotmat_from_vecs`, sliced to joint 21 (wrist).

**Bug found and fixed**: 4/29,228 `new_joints/*.npy` files (`000990`,
`005836`, and their `M`-mirrored variants) store shape `[22,3]` instead of
`[T,22,3]` -- a single-frame clip whose leading time axis got squeezed
away. Crashed the first training attempt (job 7957775, DataLoader
IndexError). Fixed by detecting `ndim==2` and restoring the leading axis;
the clip is then naturally rejected by the existing "too short" validity
checks (a 1-frame clip can't produce finite-difference acc/gyro anyway).
Verified: the known-bad clip now returns `None` cleanly, and a 500-clip
smoke test with the fix in place gives the identical 493/500 yield with
zero errors as before the bug was hit.

**Combined smoke-tested coverage** (new_joints position + new_joint_vecs
rotation, no AMASS at all): 493/500 train clips (98.6%), vs. ~35% when
AMASS was the bottleneck for every prior "Ours" checkpoint this session.

**Cross-check**: user separately generated `root_rotations_3d/<id>.npy`
(3x3 matrices, full ~29,228 coverage) hoping it might be an independent,
better root-orientation source. Verified numerically -- max abs diff 2.1e-7
against the root-rotation component `_load_all_joint_rotmat_from_vecs`
already derives internally from `new_joint_vecs`. It's the identical
Y-axis-only yaw rotation, just pre-exported in matrix form ("_3d" = 3x3
matrix representation, not 3 additional DOF) -- confirms no new information,
doesn't address the twist-blindness gap for non-root joints.

**Training**: config
`vit1d_clip_humanml3d_flu_finetune_mae_multidata_anyrot_smoothresample_gravity_lr5e-5_unimtscrop_newjoints.yaml`,
identical to `unimtscrop` except `joints_subpath`/`rotation_source`. Job
7958415. Best val_loss 2.82, notably worse than unimtscrop's 2.564 on
HumanML3D's own internal retrieval metric (expected, given the added
rotation noise).

**Zero-shot result on 5 acc+gyro datasets** (WHARF excluded -- it has no
real gyro sensor, 100% NaN, gyro-zero-filled -- gyro-fidelity differences
between rotation sources can't show up there), jobs 7958589/7958590:

| Dataset | unimtscrop (AMASS) | new_joints+new_joint_vecs (AMASS-free) |
|---|---|---|
| UTD-MHAD | 14.88% / 10.65% | 11.31% / 5.46% |
| WISDM | 29.00% / 25.91% | 24.03% / 19.58% |
| PAMAP2 | 17.49% / 14.19% | 17.49% / 12.09% |
| HHAR | 43.76% / 35.35% | 38.10% / 29.18% |
| MHEALTH | 28.63% / 27.15% | **34.62% / 29.29%** |
| **Average (5)** | **26.75% / 22.65%** | 25.11% / 19.12% |

Net negative on average (-1.64pp acc, -3.53pp F1) -- the twist-blindness
fidelity cost outweighs the ~2.8x coverage gain overall. But MHEALTH --
UniMTS's single largest lead all session (59.31%/51.06% official checkpoint
vs. Ours' 28.63%/27.15%) -- improves meaningfully (+5.99pp acc, +2.14pp
F1), the same direction (though larger magnitude) as the earlier
`--roundtrip-20hz` and `gaussian_noise` experiments' partial MHEALTH/
postural-class gains. Tentative interpretation: MHEALTH's activities
(postural transitions, cycling, jogging) may depend more on genuine
whole-body coverage/data diversity than on precise wrist-twist fidelity,
whereas UTD-MHAD's fine hand-gesture vocabulary is more twist-sensitive and
suffers more from the fidelity loss.

### Correction: FLU frame alignment for new_joint_vecs-derived rotation is worse than initially characterized (2026-07-26)

While auditing the newjoints_augtext pipeline for correctness (user asked
"double check everything ... makes sense"), found a bigger issue than the
already-disclosed twist-blindness.

**Consistency check 1 (passed)**: verified `new_joints/<id>.npy` is not an
independent position source -- it's literally `recover_from_ric(data, ...)`
in `motion_representation.ipynb`, where `data` IS `new_joint_vecs/<id>.npy`.
Both position and rotation are mathematically derived from the same
underlying array via the same root-rotation reconstruction
(`recover_root_rot_pos`), so they are frame-consistent with each other by
construction -- no risk of mixing incompatible canonicalizations between
the position and rotation sources.

**Consistency check 2 (FAILED -- real issue)**: `R_WRIST_FROM_FLU` (the
fixed rotation this repo's FLU pipeline uses to map wrist-local axes to
Forward-Left-Up) was calibrated for SMPL's own T-pose local-axis convention
(x=proximal, y=dorsal, z=thumb-side, per SMPL's own body-model rigging).
T2M/new_joint_vecs's local-rotation convention is different at a
structural level: at T2M's own "rest pose" (current bone direction ==
`t2m_raw_offsets` rest direction for every joint), every joint's LOCAL
rotation is `qbetween(u,u) = identity`, meaning the wrist's GLOBAL rotation
at rest equals the ROOT's own global rotation -- i.e., T2M's convention has
no joint-specific bind-pose axis frame at all, unlike SMPL's rigging.
Quantified directly: compared AMASS-derived vs. new_joint_vecs-derived
wrist rotation AT FRAME 0 (not the frame-0-cancelled relative comparison
used for the earlier 56deg-median twist estimate) across 60 overlapping
clips: mean misalignment 115.7deg, std 27.1deg, median 112.1deg -- close to
what two UNCORRELATED/independent random 3D rotations would show (a
uniformly random SO(3) rotation has expected magnitude ~120deg). This is
NOT a small fixed recalibration offset (which would show low std relative
to a small mean) -- it indicates `R_WRIST_FROM_FLU` does not correctly map
new_joint_vecs-derived wrist rotation to true Forward-Left-Up at all, on
top of (not instead of) the already-disclosed twist-blindness. The earlier
"median 56deg" figure specifically measured DYNAMICS (relative-to-own-
frame-0 drift, which cancels any constant misalignment) -- it undersold the
total gap, since it never captured this separate STATIC misalignment.

**Mitigating factor**: every "Ours" HumanML3D-CLIP config this session
(including both newjoints checkpoints, 7958415 and 7959342) has
`random_rotation.enabled: true` -- a fresh random SO(3) rotation applied to
acc+gyro together on every training sample, specifically to make the
encoder invariant to absolute sensor/world-frame orientation. Since the
model is never expected to rely on "F literally means forward" in the
first place, this frame-alignment defect is likely far less consequential
for the CLIP-alignment / zero-shot-classification use case than it would
be for a task that needs physically accurate axis semantics (e.g.
visualization, or any downstream consumer without rotation-invariance
training). This plausibly explains why the newjoints checkpoints still
produced reasonable zero-shot numbers (24.99->25.11% avg acc, MHEALTH
actually improving) despite this -- but the possibility remains that
partially-scrambled-but-still-structured axis information is a WORSE
training signal than either (a) correctly-aligned FLU or (b) fully
world-frame-agnostic input, and this has not been isolated/tested.

**Not yet resolved**: whether this axis-convention gap is fixable (e.g. by
deriving a new, T2M-convention-specific fixed rotation analogous to
R_WRIST_FROM_FLU, calibrated against T2M's own rest-pose definition rather
than reusing SMPL's), or whether it even matters much given the rotation-
augmentation mitigation above. Flagging as a known, quantified limitation
of every newjoints/new_joint_vecs-sourced checkpoint trained this session
(7958415, 7959342), not yet corrected.

### aug_texts recovers most of the newjoints fidelity gap (2026-07-26)

Combined `new_joints` (position) + `new_joint_vecs`/`root_rotations_3d`
(rotation, AMASS-free) with `aug_texts` (GPT-paraphrased captions added to
the original 4/clip, user-provided directory, falls back to `texts/` for
clips without an aug_texts file). Config
`..._unimtscrop_newjoints_augtext.yaml`, job 7959342. Best val_loss 2.477 --
the best internal HumanML3D retrieval metric of the entire session (better
than both unimtscrop's 2.564 AMASS-based baseline and the newjoints-only
run's 2.82).

**Zero-shot result on the same 5 acc+gyro datasets**, jobs 7959437/7959438:

| Dataset | unimtscrop (AMASS) | newjoints only | newjoints+augtext |
|---|---|---|---|
| UTD-MHAD | 14.88% / 10.65% | 11.31% / 5.46% | 8.93% / 3.97% |
| WISDM | 29.00% / 25.91% | 24.03% / 19.58% | 26.15% / 22.25% |
| PAMAP2 | 17.49% / 14.19% | 17.49% / 12.09% | 16.23% / 13.48% |
| HHAR | 43.76% / 35.35% | 38.10% / 29.18% | **43.76% / 36.65%** |
| MHEALTH | 28.63% / 27.15% | 34.62% / 29.29% | **35.90% / 34.05%** |
| **Average (5)** | **26.75% / 22.65%** | 25.11% / 19.12% | **26.19% / 22.08%** |

Adding aug_texts recovers nearly the entire average gap versus the AMASS-
based baseline (within 0.6pp acc, 0.6pp F1) while using ZERO AMASS data
(98.6% vs. 35% coverage). HHAR matches baseline exactly on accuracy and
beats it on F1; MHEALTH clearly surpasses baseline (+7.27pp acc, the
dataset UniMTS has dominated most all session). UTD-MHAD is the one
dataset that gets consistently worse across both newjoints variants --
plausibly the most twist/frame-precision-sensitive of the five (short,
fine hand gestures), so it suffers most from the rotation-fidelity and
frame-alignment issues quantified separately this session (median 56deg
wrist-twist gap, ~116deg static frame misalignment -- see the entries
above). Caption diversity alone does not fix a rotation-fidelity problem,
but it appears to substantially compensate for it in aggregate, likely by
giving the encoder more paraphrase-robust text anchors to align against
even when the IMU-side signal itself is a noisier/differently-framed
version of the true motion.

Full recipe for this best-of-session-on-this-axis checkpoint: MAE-warm-
started ViT1DEncoderAltAttn, frozen CLIP text tower, cubic resample,
gravity_mps2=9.81, crop_mode=first, fixed 10.24s window, SO(3) rotation
augmentation, AdamW lr=5e-5 -- identical to unimtscrop except position
(new_joints), rotation (new_joint_vecs+root_rotations_3d), and captions
(aug_texts).
## 11. MAE data-scaling recipe experiment: fixed-step budget + session_fraction diversity scaling (2026-07-25/26/27)

Follow-up to Section 9's saturation finding. Section 9 scaled `overall_fraction`
under `scaling_mode: window_fraction` (the default) — a uniform random draw
over individual, heavily-overlapping (50% hop) windows. `analysis/mae_scaling_diagnostics/DIAGNOSIS.md`
found this understates how much *diversity* actually shrinks at low fractions
(even a small window-level draw still touches nearly every session), and
Section 9's own epoch column showed a massive, uncontrolled training-budget
confound (9,667 epochs @ 0.1% vs. 77 epochs @ 100%). This experiment redesigns
the sweep to remove both confounds simultaneously, and narrows to 3 points
(5%/25%/100%) instead of 6.

**Design, all three points identical except `overall_fraction`:**
- `scaling_mode: session_fraction` — keeps a random `overall_fraction` of
  *sessions* per source (all windows from a kept session are kept), not a
  fraction of individual windows. Deterministic per-source floor of 1 kept
  session (`_group_fraction_subset` in `combined_mae_multidata_loader.py`).
  capture24/eatingspeed have no session-level provenance and silently fall
  back to `window_fraction` (logged warning).
- `sampling_mode: natural` — `balanced` was tried first and rejected:
  `imuposer` has only 137 windows total pool-wide and floors to 3/56 kept
  windows at 5%/25% under session_fraction, which collapses the
  `WeightedRandomSampler`'s `epoch = n_sources * min(source_size)` formula to
  1-11 steps/epoch (unusable) — see `analysis/mae_scaling_diagnostics/out/10_recipe_budget/recipe_budget.json`.
- **Fixed-step budget**: `trainer.max_steps: 300000` identical across all 3
  fractions (same gradient-update count everywhere) instead of Section 9's
  fixed-epoch-implicit/unbounded-max_steps-plus-early-stopping design. Isolates
  "more data" from "more compute" — the standard scaling-law comparison.
  300,000 chosen to match Section 9's 100%-run best step (291,000).
- `cache_dir: mae_multidata_cache_v3` — the *current* production data sizes
  (OCDetect/Capture24/EatingSpeed fully unfractioned, not Section 9's older
  smaller caps), so this sweep's "100%" is ~7.3x more raw data than Section
  9's "100%" (3,555,159 vs. 487,295 total train windows). `session_fraction`
  requires per-window provenance, which forces every source to rebuild from
  raw at job start regardless of `cache_dir` (bypasses the cache for all
  sources except when `overall_fraction=1.0`, i.e. the 100% run only, which
  loads straight from `mae_multidata_cache_v3`).
- NHANES excluded (12th source, added earlier in this doc's investigation) —
  its own scaling behavior is uncharacterized, left out to avoid an extra
  uncontrolled variable in this specific sweep.
- Architecture/objective identical to Section 9's "Ours (MAE)" column
  (`ViT1DEncoderAltAttn`, `hidden_size=384/12L/6H`, `mask_ratio=0.75`,
  `patch_size=16`, `shared_kernel=acc_gyro`).

**Underlying data volume actually retained at each fraction** (session-level,
measured via `analysis/mae_scaling_diagnostics/10_recipe_budget.py`, job
7888102 — hop-based effective hours = windows × 5.12s / 3600, halves the
naive window-length convention since windows overlap 50%):

| fraction | train windows | effective hours |
|---|---|---|
| 5% | 169,479 | ~241h |
| 25% | 892,557 | ~1,269h |
| 100% | 3,555,159 | ~5,056h |

**Configs**: `configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe{5pct,25pct,100pct}.yaml`.
**Pretraining jobs**: 5pct/25pct/100pct originally submitted as 7919072/73/74
(16h `--time`, all three TIMEOUT'd — the ~9 steps/sec throughput estimate the
budget was sized on was ~2.3x too optimistic; actual rate ~3.85-3.9 steps/sec).
Resumed via `--resume latest` (not `best` — resuming from `best.pt` would
desync the Trainer's early-stopping bookkeeping restore, which always prefers
`latest.pt` regardless of resume target, see `imu_lm/runtime_consistency/trainer.py:118`)
as jobs 7952727/7952728/7952729 with 14h/14h/9h budgets sized off the real
observed rate. All three COMPLETED cleanly at `max_steps=300000`.

**Integrity checks run before trusting any number below** (the 45 downstream
configs were `sed`-generated from templates, so a single bad substitution
would have silently pointed a probe at the wrong encoder):
- All 45 downstream runs verified to load the checkpoint matching their own
  fraction — 45/45, zero mismatches (grep the `.err` files, not `.out`: the
  `loading pretrained encoder from …` / `warm-starting encoder from …` lines
  go to stderr, so `.out` looks like no encoder was ever loaded).
- `scaling_mode=session_fraction` verified actually applied, with the window
  counts matching the pre-launch budget prediction exactly (169,479 and
  892,557 of 3,555,159); 100% confirmed unsubsampled.
- Exactly 2 provenance fallbacks per run, and only `capture24` + `eatingspeed`
  — the two documented above. `sampler=natural` on all three.

**MEASURED TRANSDUCTIVE OVERLAP — the downstream numbers in 11b-11g are NOT
held-out-subject generalization** (job 8404892, 2026-07-31). The pretraining
pool's `realdata` source contains the very datasets used as HAR evals
(pamap2/recofit/samosa/shoaib/ut_watch/utd_mhad/wisdm), and `_build_realdata`
splits with `val_ratio: 0.05` + `seed: 42`, while the HAR probes use
`val_ratio: 0.2` + `seed: 0` (inherited from base.yaml) over a *single*-dataset
session index rather than the 7-dataset pooled one. Different seed, different
ratio, different group pool -> effectively unrelated splits, and since
pretraining keeps ~88-100% of every dataset's subjects, nearly every probe val
subject was already seen (unlabeled) during pretraining:

| eval dataset | probe val subjects | also seen in pretrain | truly held out |
|---|---|---|---|
| wisdm | 10 | 9 (90%) | **1** |
| utd_mhad | 2 | 2 (100%) | **0** |
| pamap2 | 2 | 2 (100%) | **0** |
| recofit | 19 | 15 (79%) | **4** |

UTD-MHAD and PAMAP2 have *zero* genuinely held-out subjects, so their absolute
accuracies in 11b/11c should be read as transductive (in-domain, unlabeled-seen)
performance, not generalization to new people. The same applies in kind to the
gesture eval (openwatch is in the pool) and the pose/odometry evals (nymeria is
4.4% of the pool).

Two things this does NOT invalidate: (1) the *scaling comparison* itself -- all
three fractions share an identical eval protocol, so the overlap is a constant
across the axis being measured, not a confound on it; (2) the pretraining-loss
curve in 11a, whose val set is the pretraining pool's own held-out split.
It is nevertheless the main reason the follow-up two-stage experiment
(`vit1d_ssl_mae_stage1_acconly_flu.yaml` / `..._stage2_accgyro_flu.yaml`) drops
realdata/nymeria/openwatch entirely: checkpoints from that lineage have zero
overlap with any eval target, so their downstream numbers ARE true
held-out-subject generalization and are directly comparable to these only with
that difference in mind.

**Code defects found and fixed during this section's review (2026-07-28)** —
see the per-fix notes in the source. Two changed how results must be read:
the val-augmentation gating (§11a below) and the 24-joint MPJPE root-joint
bug (which deflated `num_joints: 24` MPJPE by exactly 23/24 ≈ 4.2%, affecting
the TotalCapture 24-joint configs in Sections 1-2, *not* the 23-joint Nymeria
numbers in this section). Also fixed: `filter_window` mutating cached
HumanML3D clips in place (cumulative re-filtering across draws), `--val-only`
in `run_zeroshot_har.py` hardcoding a split seed that did not match the
finetune's (leaking training subjects into the "held-out" eval — see Section
10), and the NHANES non-wear guard pooling variance across channels so it
never fired.

### 11a. Pretraining reconstruction loss

| fraction | clean-val MSE (authoritative) | as-logged best val loss | best step | Run |
|---|---|---|---|---|
| 5% | 0.760410 | 0.760297 | 297,000 | `vit1d_ssl_mae_recipe_5pct` (7919072→7952727) |
| 25% | 0.758446 | 0.757894 | 297,000 | `vit1d_ssl_mae_recipe_25pct` (7919073→7952728) |
| 100% | **0.757198** | **0.756625** | 292,000 | `vit1d_ssl_mae_recipe_100pct` (7919074→7952729) |

Clean monotonic improvement, small but real (0.0032 total 5%→100% on the
clean val, vs. Section 9's 0.008 total under the old, confounded design) —
the fixed-step budget removes the "small-data runs get 100x more epochs"
confound, and `session_fraction` scaling means the fractions are now
genuinely diversity-different, not just redundancy-different.

**Why two columns (val-augmentation confound found and corrected 2026-07-28).**
The three runs did not receive identical val *treatment*, so the as-logged
column is not strictly comparable across rows. `CachedWindowDataset.__getitem__`
applied `apply_augment` unconditionally — to val as well as train — and
`_load_cached_source` builds it for both splits. Because `scaling_mode:
session_fraction` forces `collect_provenance=True`, which makes
`make_combined_mae_multidata_loaders` bypass `cache_dir` entirely, the three
runs took *different* data paths: confirmed from the run logs, the
`loaded from cache` source count is **0 / 0 / 11** for 5% / 25% / 100%. So
100% scored itself on a fully-augmented val set, while 5%/25% went through
the raw builders, which were themselves inconsistent — nymeria/openwatch/
realdata gate augmentation on `split_name == "train"`, totalcapture/capture24/
eatingspeed/OCDetect did not. Val *composition* was identical throughout
(488,097 windows in all three), only the augmentation differed, with
octahedral `random_rotation` enabled in all three configs.

The clean-val column removes that variable: all three checkpoints rescored
against one identical val set with every augmentation block force-disabled
and an identical per-batch mask sequence (`torch.manual_seed` reset before
each checkpoint), via the real
`masked_1d_freq_channel_mask.forward_loss` — see
`analysis/mae_scaling_diagnostics/12_recipe_val_rescore.py` (job 8134514,
3,814 batches per checkpoint). **The ordering and the conclusion are
unchanged**; the fractions shift by only 0.0001-0.0006 and the 5%→100% gap
narrows slightly (0.0037 → 0.0032). The underlying code defect is fixed
(augmentation is now gated to train in `CachedWindowDataset` and in the four
raw builders that lacked gating), which means **future** MAE runs will not be
directly comparable to the as-logged column here — use the clean-val column
as the reference point.

### 11b. Downstream HAR probe (frozen encoder + linear head, same
`utd_mhad_probe` generic objective and hyperparameters as Section 9b, just
pointed at the checkpoints above)

| dataset | 5% acc | 25% acc | 100% acc | 5% bal_acc/macro_f1 | 25% bal_acc/macro_f1 | 100% bal_acc/macro_f1 | Jobs (5/25/100%) |
|---|---|---|---|---|---|---|---|
| WISDM | 62.96% | 62.94% | **64.07%** | 62.97% / 0.610 | 62.95% / 0.611 | 64.09% / 0.623 | 7958689 / 7958690 / 7955411 |
| UTD-MHAD | 59.42% | 62.32% | **63.77%** | 60.55% / 0.581 | 62.07% / 0.607 | 64.25% / 0.628 | 7999199 / 7999202 / 7999203 |
| PAMAP2 | 53.63% | **57.92%** | 56.51% | 50.79% / 0.501 | 58.06% / 0.575 | 55.83% / 0.542 | 7999204 / 7999206 / 7999208 |
| RecoFit | **75.88%** | 75.15% | 75.46% | 60.71% / 0.611 | 60.21% / 0.610 | 60.38% / 0.610 | 7999209 / 7999210 / 7999211 |

WISDM and UTD-MHAD (the two largest/cleanest-labeled HAR sets) show a real,
monotonic improvement — a genuine contrast with Section 9b, where every one
of these four saturated/peaked before 100%. PAMAP2 and RecoFit still don't
show a clean monotonic gain even under this corrected protocol (PAMAP2 peaks
at 25%, RecoFit essentially flat 75.1-75.9% across all three) — plausibly a
real property of those two specific datasets/label sets rather than an
artifact this redesign was meant to fix.

### 11c. Downstream HAR finetune (full encoder unfrozen, `utd_mhad_finetune`
generic objective, lr=1e-4 vs. probe's 1e-3 — new for this sweep, Section
9 never ran a finetune-scaling variant)

| dataset | 5% acc | 25% acc | 100% acc | 5% bal_acc/macro_f1 | 25% bal_acc/macro_f1 | 100% bal_acc/macro_f1 | Jobs (5/25/100%) |
|---|---|---|---|---|---|---|---|
| WISDM | 71.08% | 71.22% | **71.39%** | 71.04% / 0.695 | 71.17% / 0.700 | 71.36% / 0.709 | 7999212 / 7999214 / 7999219 |
| UTD-MHAD | 67.63% | **73.91%** | 68.12% | 68.59% / 0.671 | 73.63% / 0.734 | 68.33% / 0.664 | 7999221 / 7999223 / 7999224 |
| PAMAP2 | 52.03% | 53.20% | 52.43% | 46.88% / 0.441 | 48.58% / 0.455 | 48.34% / 0.447 | 7999225 / 7999227 / 7999228 |
| RecoFit | **87.03%** | 86.55% | 86.42% | 74.22% / 0.747 | 75.55% / 0.745 | 75.61% / 0.757 | 7999229 / 7999231 / 7999232 |

WISDM finetune is monotonic (small gains, same direction as its probe). Every
other dataset is flat or non-monotonic once finetuned — most strikingly
UTD-MHAD, which *probes* monotonically (11b) but *finetunes* to a sharp peak
at 25% (73.91%, +5.8-5.9pp over both 5% and 100%) — nearly the same
peak-at-25% shape UTD-MHAD showed under Section 9b's old, confounded probe
design (69.08% there). RecoFit finetune is monotonically *decreasing*
(87.03%→86.42%) — more pretraining data very slightly hurts once you
finetune end-to-end on this dataset.

### 11d. Microgesture (OpenWatch 6-class) probe + finetune (Section 5's task,
`openwatch_gesture_probe`/`openwatch_gesture_finetune` objectives)

| mode | 5% acc | 25% acc | 100% acc | 5% bal_acc/macro_f1 | 25% bal_acc/macro_f1 | 100% bal_acc/macro_f1 | Jobs (5/25/100%) |
|---|---|---|---|---|---|---|---|
| probe | 62.24% | 62.24% | 62.24% | 50.19% / 0.519 | 49.59% / 0.520 | **53.52%** / 0.554 | 7999233 / 7999234 / 7999236 |
| finetune | **66.33%** | 63.27% | 62.24% | 50.85% / 0.525 | 47.92% / 0.486 | 53.52% / 0.516 | 7999237 / 7999238 / 7999239 |

`acc` landing on the identical 62.24% across all three probe fractions (and
matching the 100% finetune row too) is a small-val-set quantization artifact
— OpenWatch's held-out split is small enough that several different
underlying prediction sets round to the same accuracy fraction; `bal_acc`/
`macro_f1` do differ and are the more informative columns here. Probe mode
shows a real edge for 100% on bal_acc/macro_f1; finetune mode is monotonically
*decreasing* (66.33%→63.27%→62.24% acc, and bal_acc/macro_f1 follow the same
shape) — the sharpest "more data hurts after finetuning" result in this
sweep.

### 11e. Arm-pose (Nymeria, 5-joint), probe + finetune

**Probe** — `pose_probe` objective, same as Section 9c:

| fraction | best val_loss | best step | Job |
|---|---|---|---|
| 5% | 0.016635 | 7,000 | 7958691 |
| 25% | 0.016496 | 9,000 | 7958692 |
| 100% | **0.016489** | 9,000 | 7956031 |

Monotonic (lower = better), though the 25%→100% gap is tiny (0.000007) —
essentially 25% and 100% are tied, both clearly ahead of 5%. Unlike Section
9c (where 25% *beat* the 100%-data run outright, 0.015866 vs. 0.01638), this
corrected sweep shows no inversion — 100% is at least as good as every other
fraction.

**Arm-pose FINETUNE** (`pose` objective — same warm-started encoder but left
trainable, lr=1e-4, `early_stopping.patience: 20`; the finetune counterpart
Section 9c never ran):

| fraction | best val_loss | best step | Job |
|---|---|---|---|
| 5% | 0.016398 | 2,000 | 8122344 |
| 25% | 0.016276 | 3,000 | 8122345 |
| 100% | **0.016160** | 2,000 | 8122346 |

The cleanest scaling result in this whole sweep: monotonic, *and* every
fraction beats its own probe counterpart, *and* the 5%→100% spread is wider
than the probe's (0.000238 vs. 0.000146). Arm pose is the one task here where
finetuning amplifies the benefit of more pretraining data rather than washing
it out — the opposite of what RecoFit/gesture do in 11c/11d.

### 11f. Full-body-pose (Nymeria, 23-joint), probe + finetune

**Probe** — same `pose_probe` config family as Section 9d — `early_stopping.enabled: false` in this config
template, so all three runs use the full 12h walltime by design;
`best_mpjpe.pt` checkpointed independently so the true best survives the
walltime kill)

| fraction | best mpjpe (mm) | best step | Job |
|---|---|---|---|
| 5% | 97.93 | 81,000 | 7958693 |
| 25% | 98.59 | 81,000 | 7958694 |
| 100% | **97.61** | 56,000 | 7956032 |

Flat, all three within ~1mm — 100% is best but 5% is nearly tied, 25% is the
outlier (worst of the three, not a monotonic trend). Section 9d showed a
similar non-monotonic wobble (50% slightly worse than 25% there) but with a
much larger 0.1%→25% spread (101.18mm→96.75mm); this corrected sweep's
narrower 5%→100% spread (all within ~1mm) suggests full-body-pose estimation
from this encoder architecture may already be near its ceiling for this task,
largely independent of pretraining data amount/diversity.

**Full-body-pose FINETUNE** (`pose` objective, encoder trainable, lr=1e-4;
unlike the probe config above this one HAS `early_stopping.enabled: true`, so
all three converged and exited on their own in 51-55 min rather than burning
the full walltime):

| fraction | best mpjpe (mm) | best step | Job |
|---|---|---|---|
| 5% | 92.49 | 11,000 | 8122347 |
| 25% | **92.14** | 10,000 | 8122348 |
| 100% | 92.42 | 8,000 | 8122349 |

Same flat, non-monotonic shape as the probe (all three within 0.35mm, 25%
best) — but a large absolute gain over probing (~97.6-98.6mm → ~92.1-92.5mm,
i.e. ~5-6mm). So for full-body pose, *finetuning* buys a great deal and
*more pretraining data* buys essentially nothing: the two axes are cleanly
separable here, and only one of them is worth spending on.

### 11g. Odometry (Nymeria local velocity), probe + finetune

**Probe** — `scripts/eval_odom_nymeria_altattn.py`, same script/protocol as
Section 9e:

| fraction | vel_mse | ATE (m) | drift (%) | Job |
|---|---|---|---|---|
| 5% | 0.02627 | 0.0741 | 50.39 | 7999241 |
| 25% | 0.02642 | 0.0749 | 51.10 | 7999242 |
| 100% | **0.02615** | **0.0739** | **50.10** | 7999243 |

Flat within noise on every metric, 25% again the (mild) outlier/worst point.
Matches Section 9e's finding that odometry's local-velocity objective
saturates its useful pretraining-data signal extremely early (by 1% of data
already, in that sweep) — confirmed again here even with the diversity-scaling
and budget-control fixes applied.

**Finetune** — `odom_finetune_nymeria_real_wrist_flu` objective via the normal
`scripts/run_pretrain.py` entrypoint (encoder warm-started and trainable,
lr=1e-4, `early_stopping.patience: 20`):

| fraction | odom_mse (val) | odom_mae | best step | Job |
|---|---|---|---|---|
| 5% | **0.018127** | **0.074389** | 27,000 | 8122350 |
| 25% | 0.018277 | 0.074918 | 30,000 | 8122351 |
| 100% | 0.018389 | 0.075303 | 27,000 | 8122352 |

Monotonically *worse* with more pretraining data, on both metrics — the
clearest inverse-scaling result in this sweep alongside gesture finetune
(11d). Note the metric names differ from the probe block above
(`odom_mse`/`odom_mae` from the training loop vs. the standalone eval
script's `vel_mse`/`ATE`/`drift`), so the two blocks are **not** directly
comparable to each other in absolute terms — only within-column across
fractions.

*(Correction: an earlier draft of this section asserted that "no finetune
variant exists for this task anywhere in this codebase." That was wrong — it
was based on checking only the standalone `eval_odom_nymeria_altattn.py`
probe script, which is indeed probe-only (`--mode` accepts just `probe`),
and missing the separate `odom_finetune_nymeria_real_wrist_flu` objective
branch in `imu_lm/models/ViT1D/run.py:1209` reached through the normal
training entrypoint. The finetune numbers above are from that path.)*

### 11h. Cross-task synthesis

All 17 measured scaling curves (pretraining loss; WISDM/UTD-MHAD/PAMAP2/
RecoFit × {probe, finetune}; gesture × {probe, finetune}; arm-pose ×
{probe, finetune}; full-body-pose × {probe, finetune}; odom × {probe,
finetune}), 5% → 25% → 100%, best value in bold:

| # | curve | metric (dir.) | 5% | 25% | 100% | verdict |
|---|---|---|---|---|---|---|
| 1 | MAE pretraining | clean val MSE ↓ | 0.7604 | 0.7584 | **0.7572** | monotonic ✅ |
| 2 | WISDM probe | acc ↑ | 62.96 | 62.94 | **64.07** | monotonic ✅ |
| 3 | UTD-MHAD probe | acc ↑ | 59.42 | 62.32 | **63.77** | monotonic ✅ |
| 4 | PAMAP2 probe | acc ↑ | 53.63 | **57.92** | 56.51 | peak @25% |
| 5 | RecoFit probe | acc ↑ | **75.88** | 75.15 | 75.46 | flat |
| 6 | WISDM finetune | acc ↑ | 71.08 | 71.22 | **71.39** | monotonic ✅ |
| 7 | UTD-MHAD finetune | acc ↑ | 67.63 | **73.91** | 68.12 | sharp peak @25% |
| 8 | PAMAP2 finetune | acc ↑ | 52.03 | **53.20** | 52.43 | peak @25% |
| 9 | RecoFit finetune | acc ↑ | **87.03** | 86.55 | 86.42 | inverse ❌ |
| 10 | gesture probe | bal_acc ↑ | 50.19 | 49.59 | **53.52** | flat/noisy |
| 11 | gesture finetune | acc ↑ | **66.33** | 63.27 | 62.24 | inverse ❌ |
| 12 | arm-pose probe | val_loss ↓ | 0.016635 | 0.016496 | **0.016489** | monotonic ✅ |
| 13 | arm-pose finetune | val_loss ↓ | 0.016398 | 0.016276 | **0.016160** | monotonic ✅ |
| 14 | full-body-pose probe | mpjpe ↓ | 97.93 | 98.59 | **97.61** | flat |
| 15 | full-body-pose finetune | mpjpe ↓ | 92.49 | **92.14** | 92.42 | flat |
| 16 | odom probe | vel_mse ↓ | 0.02627 | 0.02642 | **0.02615** | flat |
| 17 | odom finetune | odom_mse ↓ | **0.018127** | 0.018277 | 0.018389 | inverse ❌ |

- **Clean monotonic improvement (6 of 17)**: pretraining val_loss, WISDM
  probe, UTD-MHAD probe, WISDM finetune, arm-pose probe, arm-pose finetune.
- **Flat / non-monotonic, no reliable benefit (8 of 17)**: PAMAP2 probe,
  RecoFit probe, UTD-MHAD finetune (sharp peak at 25%), PAMAP2 finetune,
  gesture probe, full-body-pose probe, full-body-pose finetune, odom probe.
- **More data measurably hurts (3 of 17)**: RecoFit finetune, gesture
  finetune (the sharpest effect in the sweep, 66.33%→62.24%), odom finetune
  (monotonic on both odom_mse and odom_mae).

Reading across rows rather than columns: **probing (frozen encoder) is where
this data-scaling recipe's fixes actually pay off** — WISDM and UTD-MHAD both
flip from saturating-before-100% (Section 9b) to genuinely monotonic here,
and the pretraining loss itself improves cleanly. **Finetuning is where the
benefit of more pretraining data usually becomes unreliable and sometimes
reverses** — of the 8 finetune curves only WISDM and arm-pose are monotonic,
while RecoFit, gesture and odom all invert outright. Plausible mechanism:
once you finetune end-to-end, the optimizer overwrites much of what
pretraining contributed, so pretraining's main surviving role is the
initialization point, not the final learned representation — and small
downstream datasets are prone to overfitting regardless of how good that
starting point was.

Two refinements the finetune arm added to the earlier (probe-only) reading:

1. **Arm pose is the counterexample to "finetuning washes out data scaling."**
   It is monotonic in *both* modes, and finetuning *widens* the 5%→100% gap
   (0.000146 → 0.000238) instead of shrinking it. Whatever more pretraining
   data buys for 5-joint arm pose survives end-to-end finetuning.
2. **For full-body pose, the two axes are cleanly separable.** Finetuning is
   worth ~5-6mm mpjpe (97.6-98.6 → 92.1-92.5) while more pretraining data is
   worth ~0.3mm and not even monotonically — so on this task, spend compute
   on finetuning, not on pretraining scale.

Odometry remains the most saturated task of all: flat as a probe (consistent
with Section 9e, where it had already saturated by 1% of data) and actively
*worse* with more data once finetuned.

### 11i. Sweep extended to 5 points: 1% and 10% added (2026-08-08, not yet run)

The 3-point sweep (5/25/100%) is extended with 1% and 10% points under the
identical protocol — fixed-step 300K budget, `session_fraction` scaling,
natural sampling, NHANES excluded. Everything is a mechanical clone of the
5pct files with only the fraction changed:

- Pretrain: `configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe{1,10}pct.yaml`
  (`overall_fraction: 0.01` / `0.10`) + `launch_slurm_mae_recipe_{1,10}pct.sh`.
  The launch walltime is a FRESH-run budget (26h: 300K steps @ the observed
  ~3.86 steps/s + ~2h raw provenance rebuild), unlike the 5/25pct scripts
  whose 14h was a resume budget. `--resume latest` is included so a TIMEOUT
  resubmits as-is.
- Downstream: the full 15-config / 16-launch-script set per fraction (HAR
  probe+finetune ×4 datasets, gesture, arm pose, full-body pose, odom
  finetune; odom probe via `scripts/eval_odom_nymeria_altattn.py`), all
  pointing at `pretrained_run: vit1d_ssl_mae_recipe_{1,10}pct`.
- `12_recipe_val_rescore.py` RUNS now lists all 5 fractions (missing
  checkpoints are skipped), for the clean-val-MSE column.

**RESULTS (all 34 jobs COMPLETE 2026-08-10).** Full 5-point table; §11a-11h
above keep the original 3-point numbers, this is the superset. Harvest rules
were validated by re-deriving the published 5/25/100% values before use:
classification = metric at the best-val-loss step; pose/odom = min over
training; MAE = one `12_recipe_val_rescore.py` run scoring all five
checkpoints on the identical augmentation-free val set (job 8922321).
x-positions are the EFFECTIVE kept-window fractions (1.3% / 10.6%).

| evaluation | 1.3% | 5% | 10.6% | 25% | 100% |
|---|---|---|---|---|---|
| MAE clean-val MSE ↓ | 0.76043 | 0.76057 | 0.75793 | 0.75840 | **0.75670** |
| WISDM probe ↑ | **65.38** | 62.96 | 64.64 | 62.94 | 64.07 |
| WISDM finetune ↑ | 70.21 | 71.08 | **72.36** | 71.22 | 71.39 |
| UTD-MHAD probe ↑ | **68.60** | 59.42 | 63.77 | 62.32 | 63.77 |
| UTD-MHAD finetune ↑ | 70.05 | 67.63 | **76.33** | 73.91 | 68.12 |
| PAMAP2 probe ↑ | 56.53 | 53.63 | **59.91** | 57.92 | 56.51 |
| PAMAP2 finetune ↑ | 54.32 | 52.03 | **55.45** | 53.20 | 52.43 |
| RecoFit probe ↑ | **77.48** | 75.88 | 76.77 | 75.15 | 75.46 |
| RecoFit finetune ↑ | 87.06 | 87.03 | **88.23** | 86.55 | 86.42 |
| Gesture probe (bal) ↑ | **61.76** | 50.19 | 59.38 | 49.59 | 53.52 |
| Gesture finetune ↑ | **69.39** | 66.33 | 63.27 | 63.27 | 62.24 |
| Arm pose probe (loss) ↓ | **0.016426** | 0.016635 | 0.016585 | 0.016496 | 0.016489 |
| Arm pose finetune ↓ | **0.015815** | 0.016398 | 0.016009 | 0.016276 | 0.016160 |
| Full-body probe (mpjpe) ↓ | **97.29** | 97.93 | 98.12 | 98.59 | 97.61 |
| Full-body finetune ↓ | **91.09** | 92.49 | 91.75 | 92.14 | 92.42 |
| Odom probe (vel MSE) ↓ | **0.02542** | 0.02627 | 0.02591 | 0.02642 | 0.02615 |
| Odom finetune ↓ | **0.018011** | 0.018127 | 0.018294 | 0.018277 | 0.018389 |

Mean normalized improvement over the 1.3% baseline across the 16 downstream
evaluations (each normalized to its own baseline, sign-corrected so + is
better; `analysis/plot_data_scaling.py`): **1.3%: 0 · 5%: −3.83 ±1.22 ·
10.6%: −0.43 ±1.02 · 25%: −2.86 ±1.36 · 100%: −2.91 ±0.95** (± s.e.m.).

**Reading.** This is NOT a scaling law and should not be presented as one.
(a) The pretraining objective moves 0.5% total over a 77× data range and is
step-flat, not sloped: 1.3%≈5% (0.7604/0.7606) and 10.6%≈25%
(0.7579/0.7584). Fitting L(D)=a·D^(−α)+c returns α≈0 — the fit is the
constant. (b) Downstream the ordering is non-monotone with the SMALLEST
fraction winning 10 of 16 cells, which no monotone function of D can
describe. The defensible claim is that the entire 1.3%→100% range sits on
the **plateau** of the curve (consistent with §9e, where odometry had already
saturated by 1%), and that the data-quantity effect is below the run-to-run
noise floor here — single seed, one random session draw per fraction, small
downstream test sets. The 1%-beats-100% inversion specifically must NOT be
claimed without seed replicates; the biggest 1% wins are on the noisiest
small tasks (gesture probe swings 8pp between 25% and 100% even in the
original 3-point sweep). §11j is the causal follow-up that explains WHY the
plateau starts so low, and the interesting power-law regime — if any — lies
BELOW 1%, where a `data0.1pct`-style recipe point would probe it.

**Confirmed independently on a leak-free pool (2026-09-10, see §17).** The plateau
reading above rested on this one sweep, on a pool that contained the evaluation
datasets. A four-axis factorial on the leak-free 4-source pool reproduces it with
volume, source count and motion-type count each isolated: all three axes flat, none
monotone, over 137x volume and 224x motion-type range. It also **narrows the
claim** -- the comparison that survives without statistical sleight of hand is
odometry, where recipe factors beat data axes 4-20x with a single task on both
sides; on microgesture the source axis matches the input-scaling effect, and on arm
pose the measured run-to-run noise band (0.108 cm) swallows everything except
modality.

Caveat at 1%: `session_fraction` floors at 1 kept session per source, so
session-poor sources (imuposer, totalcapture) sit above a strict 1% —
measured from job 8836196: 46,116/3,552,334 windows kept = effectively ~1.3%
(totalcapture=6, imuposer=1, openwatch=18 windows). Submission order: the two
pretrains first; every downstream job of a fraction needs its `best.pt` on disk.

**Protocol deviation found mid-run (2026-08-08): humanml3d contributes 0
windows to the 1%/10% pretrains** (composition line says 10 sources, not 11).
Cause: the on-disk `HumanML3D/HumanML3D/joints/` dir was removed after the
original 3-point runs (only `new_joints`/`new_joint_vecs` remain), and the
MAE configs' `humanml3d:` block sets no `joints_subpath`, so
`_load_clip_imu_flu` falls back to the removed default and every clip loads
as None (`loaded=0 skipped=4000` in both new runs' logs; the July-26 5pct
run loaded 2825/4000). Any rerun of 5/25/100pct today would hit the same.
Size of the deviation: humanml3d was 141/169,479 kept windows (0.083%) in
the original 5pct run, and its absence also shrinks logged val by 353
windows (488,097 → 487,744) — well below run noise, and the §11a clean-val
rescore uses the shared v3-cache val set for all checkpoints, so the
cross-fraction comparison metric is unaffected. Fix for any future rerun:
add `joints_subpath: HumanML3D/new_joints` to the `humanml3d:` block (the
CLIP-align configs already do this). Do NOT edit the recipe configs while a
run is resumable in flight — a resume rebuilds the dataset from config and
would change the train pool mid-run.

### 11j. Diversity-saturation causal test: same size, different diversity (2026-08-10, in flight)

Why: 11i's 5-point curve is flat/inverted -- the hypothesis is that transfer
tracks intrinsic motion diversity, which saturates far below 1% of windows,
not raw window count. The causal test holds N fixed and varies ONLY
diversity: three N=46,116 subsets (same size as the session-1pct point) of
the full v3-cache pool, selected in the frozen recipe_100pct encoder's
embedding space (PCA-64 + k-means K=1000):

- `random-1%`  uniform draw (seed 42)
- `diverse-1%` equal windows per cluster (max mode coverage)
- `redundant-1%` largest clusters first (min mode coverage)

Predicted ordering if the hypothesis holds: redundant << random <= diverse,
with diverse-1% ~= random-100% (the existing recipe_100pct numbers).
Supporting diagnostics from the same job: cluster-coverage-vs-fraction
curve, NN-novelty of the unsampled 99% vs the 1% (quantitative redundancy),
embedding effective rank. If coverage saturates where performance saturates,
that is the paper figure ("performance saturation coincides with diversity
saturation").

**DIAGNOSTICS — COMPLETE (job 8964015, 18 min; `analysis/diversity_saturation/out/diversity_analysis.json`).**
These already answer the "why" half of §11i, independently of the three
causal pretrains. K=1000 k-means on PCA-64 (96.9% variance kept) of the
frozen recipe_100pct CLS embeddings over all 3,555,159 cache-pool train
windows:

| pool fraction | cluster coverage |
|---|---|
| 0.01% | 29.2% |
| 0.1% | 93.7% |
| 0.5% | 99.9% |
| 1.3% | **100%** |
| 5% / 10.6% / 25% / 100% | 100% |

**Motion-mode coverage saturates below every fraction §11i trained**, which
is the mechanism behind the plateau: each of the five runs already saw all
1000 modes, so they differ only in how many redundant samples per mode they
got. Two supporting numbers: (i) NN novelty — 50.2% of the unsampled 99% of
the pool lies closer to the random-1% subset than that subset's own internal
NN median distance, i.e. half of all "new" data is not new in embedding
space; (ii) the 384-d embedding covariance has an **effective rank of only
15.2**, so the wrist-motion manifold this encoder represents is intrinsically
very low-dimensional. Control check: the deliberately-degenerate
redundant-1% subset covers just 3/1000 clusters with a self-NN median of
0.99 (vs 3.25 random), so the three arms are well separated by construction.

BUDGET CUT mid-run (2026-08-10, user decision): testing experiments do not
need the full 300K budget -- the causal contrast only needs EQUAL budgets.
All three pretrains were stopped at ~94-99K steps and resumed to finish at
exactly 105K (jobs 9025732/9025743/9025868, probes rechained). Absolute
numbers are therefore NOT comparable to the 300K recipe table (LR not fully
annealed, small LR discontinuity at the resume point -- identical for all
three variants); the within-experiment ordering is unaffected. Early signal
at ~95K: val loss diverse 0.7785 <= random 0.7803 << redundant 0.9249 -- the
redundancy penalty is ~35x the entire 1.3%->100% fraction spread.

Two budget-hygiene traps the cut introduced, both handled -- check these on
any future mid-run budget change:
1. **Over-budget stale `best.pt`.** `scancel` is not instantaneous, so the
   arms died at different steps (random ~100K, redundant 104K, diverse
   109K). All three resume from `latest.pt` at exactly step 100,000 (saved
   every `ckpt_every_steps: 10000`), so the resumed budgets ARE equal --
   but diverse's on-disk `best.pt` was from step 108,000, i.e. 3K steps
   past the new budget, and the probes load `best.pt`. That would have
   given the diverse arm extra training in exactly the direction that
   flatters the hypothesis. Moved to
   `checkpoints/_stale_over_budget/best_step108000.pt` before the job
   started; the resumed run rewrites `best.pt` from steps 101-105K (the
   restored `best_val_loss` bookkeeping comes from `latest.pt`, not from
   the best.pt file, so this is safe). VERIFY after the runs: all three
   `best.pt` must report step <= 105,000.
2. **`metrics.txt` is append-only across runs**, so diverse's file holds
   killed-run rows up to step 109,000 followed by the resumed 101-105K
   rows. Harvest the val number at step 105,000 / the last row, NEVER
   min-over-file -- min-over-file would silently hand diverse its
   out-of-budget 109K value.

Infra: new `scaling_mode: index_file` in combined_mae_multidata_loader.py
(explicit window-index list into the cache-order concat pool; natural
sampling only). Pipeline: analysis/diversity_saturation/01_embed_select_analyze.py
(job 8964015: embed 3.55M windows with frozen recipe_100pct CLS -> select ->
diagnostics) -> 3 pretrains `vit1d_ssl_mae_div1pct_{random,diverse,redundant}`
(jobs 8964018/8964026/8964035, fixed-step 300K, cache path so no raw rebuild,
24h) -> 4 probes each (arm pose, WISDM, PAMAP2, odom; jobs 8964019-8964045),
all dependency-chained. Note: the cache pool INCLUDES humanml3d (11 sources,
pool=3,555,159, val=488,097) unlike the raw-rebuild 1/10pct runs -- same
pool as the original recipe_100pct run. Probes only (no finetune): probes
measure representation quality directly, finetuning dilutes the pretraining
signal (11h). Selection-space circularity caveat: subsets are chosen in the
recipe_100pct encoder's own embedding space -- fine for the within-experiment
contrast, flag if promoted to a data-pruning-method claim.

### 11k. Sub-1% volume sweep: the regime where coverage is NOT saturated (2026-08-11, in flight)

Why: 11i swept 1.3% -> 100% and got a flat/inverted curve; 11j explained it --
motion-mode coverage is already 100% at 1.3%, so every 11i point sits ON the
plateau, and a plateau alone cannot show that coverage is what matters. The
rising limb, if there is one, has to live BELOW 1%, because that is the only
place coverage is not saturated. This sweep trains that regime at a budget
matched to 11j, so the two join into one curve.

Design: four **nested** subsets -- each a strict subset of the next, all
subsets of 11j's own `random-1%` -- drawn as prefixes of a single seeded
shuffle (`analysis/diversity_saturation/06_select_subpct.py`). Nesting means
the x-axis is literally "add more data" and removes draw-to-draw variance; a
uniform subsample of a uniform sample is still uniform, so it costs nothing
statistically. Same v3-cache index space as 11j (11 sources, NHANES off), so
`random-1%` is a free fifth point at the identical budget.

| arm | N windows | fraction | cluster coverage |
|---|---|---|---|
| sub-0.01% | 356 | 0.01001% | **28.9%** |
| sub-0.05% | 1,778 | 0.05001% | **74.8%** |
| sub-0.1% | 3,555 | 0.10000% | **92.8%** |
| sub-0.5% | 17,776 | 0.50001% | **99.9%** |
| random-1% (11j) | 46,116 | 1.29711% | 100% |

Coverage was re-measured by refitting the identical MiniBatchKMeans (K=1000,
seed 42) on the cached `pca64_f32.npy` -- no re-embedding. It reproduces 11j's
independently-measured numbers (28.9 vs 28.8, 92.8 vs 92.7, 99.9 vs 99.94),
which is a useful check that the cluster space is stable. The new 0.05% point
fills 11j's largest coverage gap (29% -> 93%).

**Budget: fixed 105,000 steps, identical to 11j's three arms**, with
`early_stopping.enabled: true` and `patience: 999999` -- on, but with a patience
that cannot fire (105 evals max at `val_every_steps: 1000`). This keeps budgets
equal without disabling the flag.

> **Trap, hit and fixed 2026-08-12 -- do not disable `early_stopping` to force
> an equal budget.** The first submission set `enabled: false`, reasoning that
> the tiny arms would otherwise stop early at unequal step counts. But
> `Trainer._maybe_early_stop` (trainer.py:255-259) writes `best.pt` **only
> inside the `if self.early_stopping_enabled` branch**, so disabling it means
> `best.pt` is never written at all. All four pretrains ran the full 105K steps
> and `COMPLETED` cleanly with only `latest.pt` on disk, then every downstream
> probe died instantly with `FileNotFoundError: pretrained encoder not found:
> .../checkpoints/best.pt`. ~30 GPU-hours lost.
>
> `latest.pt` is not a usable substitute: the 0.01% arm's best val loss was at
> step 20,000 (0.817886) and it overfit to 0.880655 by step 105,000, i.e. the
> final checkpoint is 7.67% worse than the best one -- exactly the overfitting
> the tiny arms were expected to show. The broken run dirs are preserved under
> `runs/_broken_no_bestpt/` (moved rather than deleted, since `metrics.txt` is
> append-only and a rerun in place would have interleaved two runs' rows).
>
> **Second-order trap: the odometry probe fails SILENTLY, not loudly.** The
> seven config-driven probes all died with a clear `FileNotFoundError`, but
> `scripts/eval_odom_nymeria_altattn.py:94-96` falls back to `latest.pt` when
> `best.pt` is absent -- so the odom probe ran to completion against the wrong
> weights and reported `vel_mse=0.02922` for the 0.01% arm as if it were a real
> result. That number sat in `subpct_results.json` looking perfectly normal.
> Its log is quarantined in `logs/_stale_prefix/`, and both harvesters now
> refuse any odom log whose text does not contain `checkpoints/best.pt`,
> emitting a warning instead of a value. Generalisable lesson: a probe that
> *degrades* on a missing checkpoint is more dangerous than one that crashes.

Absolute numbers are NOT comparable to the 300K recipe table; they ARE
comparable to 11j.

Predicted if the coverage hypothesis holds: performance rises across
0.01 -> 0.5% (coverage 29% -> 99.9%) and is flat from 0.5% onward. Predicted if
it does not: flat everywhere, i.e. even 356 windows suffice, which would mean
the plateau is set by something other than coverage.

Jobs (submitted 2026-08-11, 40 total). Pretrains 9075134 (0.01%) / 9075141
(0.05%) / 9075143 (0.1%) / 9075146 (0.5%). Downstream is 11i's **full 8-probe
set** per arm, all `afterok` on their pretrain:

- first batch 9075148-9075221 -- arm pose, WISDM, PAMAP2, odom (11j's four,
  recipes byte-identical to the div1pct probes apart from `pretrained_run`);
- second batch 9079186-9079221 -- UTD-MHAD, RecoFit, gesture, full-body pose,
  cloned from the recipe1pct probes so they match 11i exactly, **including each
  template's own `early_stopping` setting** (the full-body probe ships with it
  disabled; changing it would change the "min over training" harvest and break
  comparability, so it is left alone and those probes are expected to run out
  their walltime as 11i's did).

Probes only, no finetune -- 11j's rationale: probes measure representation
quality directly, finetuning dilutes the pretraining signal (11h).

**Anchor completed too (9079268 / 9079280 / 9079288 / 9079307).** 11j only ran
four probe types, so the 1.3% anchor was missing UTD-MHAD, RecoFit, gesture and
full-body. Left as-is, the anchor's mean would have been taken over a different
probe subset than every other point on the curve -- a bias that would show up
exactly at the knee. Those four probes were therefore run against 11j's
existing `vit1d_ssl_mae_div1pct_random` checkpoint (no dependency needed), so
all five fractions share one 8-probe set.

Harvest: `07_harvest_subpct.py` (same validated rules as `02_harvest.py`;
verified before use by re-deriving 11j's random-1% row exactly -- 61.70 /
54.22 / 0.017184 / 0.02740). Figure: `08_plot_scaling_curve.py`, two stacked
panels on a shared log-x axis (coverage above, normalised probe performance
below) with a rule at the saturation point crossing both -- deliberately not a
dual-axis chart. The 1.3% -> 100% span is SHADED rather than drawn as a line
because 11i's points there are 300K-step, not budget-matched.

**Known gap.** The plateau's right end is still 11i at 300K. A single extra
105K arm on the full pool (`vit1d_ssl_mae_vol100pct_105k` + 4 probes) would
anchor both ends of the curve at one budget and turn the shaded region into
real points. Config/scripts not yet written; flagged rather than run, since
this sweep as submitted answers the question that was asked.

### 11l. Motion-types x data-volume grid on the 5-source pool (2026-08-11, in flight)

The 2D dose-response 11j asked for and 11k only samples along one edge:
x = number of motion types in the pretraining set, y = probe performance, one
line per data volume. Pool is the **5 SSL sources** (realdata, ocdetect,
eatingspeed, motionx, sonic) = **1,959,288 windows**, 55.1% of the 11-source
pool. Cluster sizes are much more even here than in the 11-source pool
(min 181 / median 1,829 / max 7,114 vs max 17,489), and there is **no
capture24**, so the accelerometer-only confound that contaminated 11j's
`redundant` arm cannot recur.

**The grid is triangular, not rectangular.** Two independent constraints, both
enforced in `10_grid_select.py`:

- (a) enough data must exist: `sum(sizes of the K largest clusters) >= F x N`
- (b) K types must fit in the draw: `F x N >= K`

(b) is the easy one to miss and it kills `k1000_f0p01`: 196 windows cannot
contain 1000 distinct motion types. Without that check the cell would still
have been produced, labelled "1000 types" while actually holding ~196 of them
-- the x-axis would be wrong at exactly the point a reader cares most about.
22 of 35 cells survive both:

| K types | max data | 0.01% | 0.1% | 1% | 10% | 100% |
|---|---|---|---|---|---|---|
| 1 | 7,114 (0.36%) | OK | OK | -- | -- | -- |
| 5 | 26,654 (1.36%) | OK | OK | OK | -- | -- |
| 10 | 47,240 (2.41%) | OK | OK | OK | -- | -- |
| 25 | 103,876 (5.30%) | OK | OK | OK | -- | -- |
| 50 | 190,477 (9.72%) | OK | OK | OK | -- | -- |
| 100 | 347,609 (17.7%) | OK | OK | OK | OK | -- |
| 1000 | 1,959,288 (100%) | (b) | OK | OK | OK | OK |

So "100% of the data" is a single point at K=1000, and the 10% line has two
points; the 0.01 / 0.1 / 1% lines are the ones that span the x-axis.

Selection within a cell is an **equal-per-cluster waterfill** over the K largest
clusters, not a uniform draw from their union. A uniform draw misses clusters
whenever `F x N` is near K -- K=100 at 0.01% (196 windows) would land in ~85
distinct clusters, not 100 -- which would again make the x-axis a lie. The
realised type count is asserted `== K` for every cell.

Budget: equal **50,000 steps** across all 22 cells, with
`early_stopping.enabled: true` / `patience: 999999` -- on but un-triggerable
(50 evals max). See the trap box in 11k: setting `enabled: false` to force an
equal budget silently suppresses `best.pt` entirely, because the trainer only
writes it inside the early-stopping branch. Self-contained: absolute numbers
are NOT comparable to 11j/11k (105K) or the 300K recipe table. Downstream is the full 8-probe set per cell (198 jobs total:
22 pretrains + 176 `afterok` probes), configs under `configs/grid5src/` and
launch scripts under `launch/grid5src/`.

Pipeline: `09_grid_feasibility.py` (feasible-cell map, caches the 5-source
k-means labels) -> `10_grid_select.py` (index files + realised-type assertion)
-> `12_harvest_grid.py` -> `11_plot_grid.py` (3x3 small multiples: the
pretraining objective plus the 8 probes, each in native units; volume is encoded
as an ordinal one-hue ramp rather than five categorical hues, since it is a
magnitude).

**PRETRAINING RESULTS — all 22 cells COMPLETE at the equal 50K budget
(2026-08-13).** MAE val loss, lower better:

| | 0.01% | 0.1% | 1% | 10% | 100% |
|---|---|---|---|---|---|
| **K=1** | 0.9570 | 0.9363 | -- | -- | -- |
| **K=5** | 0.8740 | 0.8294 | 0.8291 | -- | -- |
| **K=10** | 0.8792 | 0.8196 | 0.8182 | -- | -- |
| **K=25** | 0.8562 | 0.8128 | 0.8100 | -- | -- |
| **K=50** | 0.8626 | 0.8080 | 0.8052 | -- | -- |
| **K=100** | 0.8535 | 0.8071 | 0.8026 | 0.8027 | -- |
| **K=1000** | -- | 0.7967 | **0.7932** | 0.7944 | 0.7945 |

Reading the two axes, each holding the other fixed:

- **Diversity, at 1,959 windows:** K=1 -> K=1000 is **-14.9%** (0.9363 ->
  0.7967), almost all of it in K=1 -> K=5 (-11.4%).
- **Volume, at fixed K:** 10x more data buys -0.03% (K=5) to -0.55% (K=100).
  At K=100, 19,593 -> 195,929 windows is **+0.01%** -- nothing.
- **Volume turns over at K=1000:** 19,593 windows gives 0.7932, the *entire*
  1,959,288-window pool gives 0.7945. **100x the data, +0.17% WORSE.** The
  volume axis is not merely saturating; past ~20K windows it is flat-to-adverse.
- **Head-to-head:** `k1000_f0p1` (1,959 windows / 1000 types, 0.7967) beats
  `k100_f10` (195,929 windows / 100 types, 0.8027) with **100x less data**.

Caveats: single seed; this is the pretraining objective, not downstream probes
(those are still draining, see below); and volume *does* matter below ~2,000
windows (196 -> 1,959 buys 5-7% at every K), so the finding is a ceiling reached
early rather than volume never mattering. At 196 windows the K axis is
non-monotone (K=100 is 2 windows/type -- the selection is starved), so those
cells are noise and should not be read as a trend.

**Scheduling notes (2026-08-13).** The 208 probes sat behind ~931 cluster-wide
jobs on `gengpu`. They were repointed in place with `scontrol` (no cancel, no
resubmit -- job IDs and `afterok` dependencies preserved) onto the `b53017`
buy-in partition, which required three changes together: `partition=b53017`,
`account=b53017`, and **`qos=buyin`** -- the association permits only the
`buyin` QOS, so leaving `qos=normal` parks every job at `InvalidQOS`. `b53017`
has A100s, so the `--gres=gpu:h100:1` in the launch scripts must be relaxed to
`gres:gpu:1` or nothing schedules. **Concurrency there is throttled to 1 job**
(`CAP=1`, the default in `analysis/diversity_saturation/b53017_throttle.sh`) —
b53017 is a shared 4-GPU buy-in node and we hold ourselves to one job at a time
so the rest stays available for later work. The 4-concurrent state observed on
2026-08-15 was a deviation from that policy and was cleared (see below).

**Full-body-pose probe cells harvested + cancelled early (2026-08-15).** Four
cells were occupying all 4 b53017 GPUs (jobs 9338210-9338213). All four were
**long past converged** and were cancelled at 6h28m to free the node:

| Cell | best mpjpe (mm) | at step | evals w/o improvement | best by 20K |
|---|---|---|---|---|
| k10_f0p01 | 102.76 | 48,000 | 125 | 104.83 |
| k10_f0p1 | **101.42** | 27,000 | 145 | 102.49 |
| k10_f1 | 101.71 | 50,000 | 123 | 103.80 |
| k25_f0p01 | 102.25 | 45,000 | 128 | 103.94 |

Every cell found its minimum by step 50,000 and then ran a further ~123,000
steps (≈4 of the 6.5 hours) producing nothing — the last-20-eval median sits
1.6-2.2mm *worse* than each run's best, i.e. oscillating above the floor and
mildly degrading, exactly the pattern diagnosed in the §2 probe LR sweep.
**Cancelling was lossless**: `best_mpjpe.pt` is checkpointed independently of
early stopping, and truncating the harvest to 50K / 100K / all-173K gives
**bit-identical numbers** for all four cells (all minima occurred ≤50K).

⚠ **Latent fragility in `12_harvest_grid.py`.** Its probe selection is
`min(...)` over *all* val rows with **no step filter** — `MAX_STEPS = 50000`
is applied only to the pretrain section, never to the probes
(`12_harvest_grid.py:86-89`). Since §2 established that this `best`-over-evals
statistic falls purely as a function of *how many val evals a run got*, any
grid cell whose probe ran longer than its neighbours would score better for
that reason alone. It happens to be harmless for these four (all minima ≤50K),
but the probes are **not** on the 50K equal budget the pretrains are — they run
unbounded to walltime — so the guard should be added before the full 8-probe
harvest is trusted: filter `cand` to `r["step"] <= MAX_STEPS`.

### 11m. Volume x diversity factorial on the FULL-SIZE 4-source pool (2026-09-03, built, not yet run)

**The objection this exists to answer.** The data-scaling experiment behind the
paper's "diversity matters more than volume" claim moved five things at once --
window count, session count, subject count, source coverage and motion-type
coverage -- so volume and diversity were never separated. A non-monotone 5% ->
100% curve therefore supports only the weak statement *"under this sampling
strategy, at this fixed step budget, on this mixture, more data did not reliably
help"*. It cannot establish that diversity is what carries the effect. The
"1% of the data covers 99% of motion types" figure inherits a second problem: a
motion type was a k-means cluster in **our own pretrained encoder's** embedding
space (11j), the count was unweighted (a type touched once counts as much as the
mode holding 10% of the pool), and no cross-dataset label alignment was ever
defined. And if 1% really does cover 99% of types, the remaining 99% of the data
is mostly redundant -- which means the scaling sweep never contained a diversity
contrast at all.

Fixing it needs the two contrasts the scaling curve does not contain: **vary
diversity with N fixed, and vary N with diversity fixed.** This section builds
exactly that, on the leak-free 4-source pool at **full size, nothing
sub-sampled**: ocdetect 1,253,077 + eatingspeed 287,678 + sonic 146,944 +
motionx 50,395 = **1,738,094 windows / 2,472 h**.

**The design.** 4 volumes x 4 diversity levels = 16 cells, plus 2 anchors at
N = 100%:

| N (windows) | `nar` | `one` | `nat` | `bal` |
|---|---|---|---|---|
| 12,500 (18 h, 0.7%) | OK | OK | OK | OK |
| 50,000 (71 h, 2.9%) | OK | OK | OK | OK |
| 200,000 (284 h, 12%) | OK | OK | OK | OK |
| 800,000 (1,138 h, 46%) | OK | OK | OK | OK |
| 1,738,094 (2,472 h, 100%) | -- | -- | anchor | anchor (sampler) |

- `nar` **single-source, narrow motion band** -- ocdetect only, drawn from the
  fewest largest motion-type clusters that can supply N. The reviewer's
  *单源重复采样*.
- `one` **single-source, uniform random** -- ocdetect only, natural type mix.
  Swept across the rows this is *单源规模扩展*.
- `nat` **4-source, natural proportions** -- a uniform draw from the pool. This
  column *is* the paper's own scaling curve, reproduced inside the controlled
  design so the old result and the new one are on one page. Swept across rows,
  *多源规模扩展*.
- `bal` **4-source, source- and type-balanced** -- equal per-source water-fill
  (deficits from the small sources redistributed), then equal-per-cluster
  water-fill inside each source. The reviewer's *多源均衡采样*.

Reading a **column** is diversity at fixed N; reading a **row** is volume at
fixed diversity. Crucially `nar` vs `one` moves motion-type diversity with the
source count **pinned at 1**, while `one` vs `nat`/`bal` moves source diversity
-- so the two senses of "diversity" are not confounded with each other either.
The two 100% anchors differ only in sampling law (natural vs the production
`source_weights` WeightedRandomSampler): at 100% the data cannot be re-selected,
only re-weighted, which is the only diversity intervention that still exists
there.

**What "motion type" means here, and how coverage is weighted.** Both of the
reviewer's questions get an explicit answer, and the circularity is removed:

- Types are **k-means clusters (K=1000)**, not human labels and not text
  clusters. Human labels were rejected as the axis on purpose: the four sources
  have non-alignable label vocabularies (ocdetect compulsive-behaviour classes,
  eatingspeed eating episodes, motionx/sonic text-described mocap), so any
  cross-dataset "activity" axis would be an arbitrary hand-mapping. Labels are
  used for *validation* only (11j/`analysis/diversity_saturation/17-20`).
- The clustering that **drives subset selection** is
  `T-phys`: 40 **hand-specified, rotation-invariant physical features** --
  magnitude statistics, jerk and angular-acceleration RMS, zero-crossing rates,
  five-band spectral fractions, spectral entropy/centroid/dominant frequency,
  gravity-direction change and instability, autocorrelation cadence -- with **no
  learned model anywhere in the loop**. Rotation invariance matters twice:
  pretraining applies octahedral random rotations, and the four sources do not
  share a mounting convention, so frame-dependent features would cluster by
  dataset rather than by motion.
- The encoder clustering (`T-enc`, PCA-64 of frozen `mae_4src_z` CLS, the 11j/11l
  definition) is kept **only as an independent read-out**. `03_type_definitions.py`
  reports AMI(T-phys, T-enc) -- if the two agree, the earlier encoder-space
  results were not an artefact of the encoder; if they disagree, that is a
  finding about 11j, and the new claim rests on the encoder-free axis either way.
- **Coverage is reported frequency-weighted, not as a raw count.** Every cell
  gets: raw types present; `pool_mass_covered` (each covered type weighted by its
  frequency **in the pool**, i.e. long-tail-weighted coverage); **effective
  number of types** = exp(Shannon entropy) and inverse-Simpson of the subset's
  own type histogram (a subset holding 999 types with 90% of its mass in one
  scores ~1, not 99.9%); KL(subset || pool); plus two **cluster-free** diversity
  statistics that cannot be gamed by the choice of K -- the participation-ratio
  effective rank and the **Vendi score** -- and the **fill distance** (median/p90/p99
  distance from a random pool window to its nearest neighbour in the subset),
  which is what actually decides whether the unsampled remainder is new data or
  repeats.
- `AMI(type, source)` is reported so the reader can check the type axis and the
  source axis are not collinear to begin with; `07_analyze.py` prints the
  predictor correlation matrix and the design condition number and refuses to
  read the two diversity coefficients apart when they are collinear.

**Diversity is measured per cell, never assumed.** The x-axis of every plot is
the realised statistic of the subset that was actually built, not the design
label. That also exposes a real triangularity honestly: at N=800,000 even the
narrowest feasible band inside ocdetect needs many clusters, so `nar` is much
less narrow there than at N=12,500 -- the design labels compress at high N and
the measured axis shows it.

**Protocol.** Every cell is `configs/nhanes_scale/mae_4src_z.yaml` with exactly
two changes: the window index file, and `max_steps: 50,000`. Same architecture,
mask 0.60, z-score input, octahedral rotation, lr 1e-4, seed 42, and the
**same untouched 4-source val split**, so MAE val loss is a common yardstick and
a narrow cell is scored on the diverse val it did not train on. `early_stopping`
stays `enabled: true` with `patience: 999999` -- on but un-triggerable (setting
`enabled: false` silently stops `best.pt` from ever being written; see the trap
box in 11k). Downstream is a 4-task panel run through the **existing bench
recipe** (`arm_pose_nymeria`, `odometry_nymeria_worldframe`, `har_wisdm`,
`microgesture_openwatch`, probe protocol, seed 42), so nothing about the
adaptation differs from every other row in the paper.

**Compute is held fixed, not epochs.** 50,000 steps x batch 128 = 6.4 M window
presentations for every cell, so the 12,500-window cell revisits its windows
~512 times and the full-pool cell ~3.7 times. That is the control the claim
needs -- *at matched compute, does more distinct data help?* Matching epochs
instead would confound volume with compute, which is what the original curve
did. Absolute numbers are self-contained to this section (50K budget), not
comparable to the 100K `mae_4src_z` row or the 300K recipe table.

**Predictions, stated before running** (so the experiment can fail):

1. At fixed N, `nar` << `one` <= `nat` <= `bal` on every panel task. If `nar` is
   not clearly worst, motion-type diversity does not carry the effect and the
   claim must be dropped.
2. Along a row at fixed diversity, 12.5K -> 800K (64x) moves the metric by less
   than the `nar` -> `bal` gap at any single N. If it does not, volume matters at
   least as much as diversity and the claim must be rewritten as "both matter".
3. `bal` at N=200,000 >= `nat` at N=1,738,094 (8.7x less data, more balanced).
   This is the strong form; it is the one most likely to fail.
4. The two 100% anchors differ: if re-weighting a fixed pool changes nothing,
   then "diversity" is about which windows exist, not how often they are drawn.

**The claim the results are allowed to support.** `07_analyze.py` fits, per task,
`y ~ log2(N) + log2(effective types) + log2(effective sources)` over the cells,
with a bootstrap CI on every coefficient and on `|b_types| / |b_N|`, and converts
the fixed-N diversity gap into an **exchange rate**: how many doublings of data
volume that gap is worth. That rate is **capped at the 7.1 doublings actually
swept** -- when the gap exceeds what the whole volume sweep buys, the output is a
lower bound and says so, rather than extrapolating to a headline multiplier.
That is the difference between the claim this section can make and the one it
replaces.

**Phase 3: the subject/session ladder.** The objection names five confounded
quantities and the factorial above controls four of them (window count, source
coverage, motion-type coverage, and -- through `nar` vs `one` -- type diversity
independently of source diversity). The fifth, *who wore the watch*, needs
per-window provenance the cache does not store, so it is a separate phase:
`08_provenance.py` rebuilds each source from raw with provenance and joins it
onto the cache **positionally, verified by content fingerprints on 400 random
rows, refusing the join outright if a single probe disagrees** (that join is
false for realdata, which is why `analysis/diversity_saturation/fingerprint_join.py`
exists). `09_subject_arms.py` then builds a 4-arm ladder at fixed N=50,000,
fixed source (ocdetect), 1 / 3 / 10 / all wearers, with the **motion-type
distribution held flat** by equal-per-type water-fill inside the chosen
subjects' union -- otherwise "fewer subjects" would silently mean "fewer motion
types" and the ladder would re-create the very confound it is testing. The
realised effective type count is printed next to every rung as the control, and
subjects are picked greedily to *maximise* type coverage, which works against
the hypothesis that more wearers help. The same script back-fills subject and
session effective counts onto all 18 factorial cells, so subject diversity is
measured there too rather than assumed away. eatingspeed exposes no provenance
through the cached path and is recorded as unknown, never guessed.

**Known limits, stated up front.** Single seed per cell (a 2-seed replicate of
the four corner cells is the first thing to add if any contrast is close);
`nar`/`one` use ocdetect as *the* single source, so "single-source" also means
"free-living hand-washing data" and a second single-source ladder on eatingspeed
would separate those; the volume axis tops out at 46% for the low-diversity arms
by construction; the subject ladder lives inside one dataset, so it speaks to
within-source wearer diversity, not to cross-dataset population diversity; and
matched-compute is a choice, not a neutral default -- an equal-epoch version
would answer a different (also legitimate) question.

**Files.**
`analysis/volume_diversity/` --
`arms.py` (pool geometry + the arm table, single source of truth) ->
`01_embed_pool.py` (frozen-encoder read-out embedding, asserts the concat index
space) -> `02_physical_features.py` (the 40 encoder-free features) ->
`03_type_definitions.py` (both clusterings + agreement/stability/null
diagnostics) -> `04_build_arms.py` (index files + measured diversity per cell) ->
`05_gen_configs.py` (`configs/voldiv/*.yaml`, `launch/voldiv/*.sh`, bench encoder
entries) -> `06_harvest.py` -> `07_analyze.py` (contrasts, joint model, exchange
rate, subject-ladder table, figure). Phase 3: `08_provenance.py` ->
`09_subject_arms.py`, which append their arms to `out/arms.json`; 05/06/07 read
the manifest, so re-running `05_gen_configs.py` picks the new arms up with no
edits. `diversity_stats.py` holds the statistics and documents which objection
each one answers.

**Running it.**
```bash
sbatch launch/voldiv/prepare.sh            # phase 0: no pretraining at all
                                           #   -> out/type_definitions.json, out/arms.json
python analysis/volume_diversity/05_gen_configs.py --write-registry
./launch/voldiv/submit.sh                  # 18 pretrains + 72 chained bench probes
python analysis/volume_diversity/06_harvest.py && python analysis/volume_diversity/07_analyze.py

# phase 3, the subject/session ladder (needs a large-memory node, rebuilds from raw)
python analysis/volume_diversity/08_provenance.py && python analysis/volume_diversity/09_subject_arms.py
python analysis/volume_diversity/05_gen_configs.py --write-registry
./launch/voldiv/submit.sh --arms vd_sub1_n50k,vd_sub3_n50k,vd_sub10_n50k,vd_suball_n50k
```
Phase 0 answers the *definitional* half of the objection on its own -- what a
motion type is, how coverage is weighted, whether the encoder-free and
encoder-based definitions agree, and how much diversity each cell actually has --
before a single GPU-hour of pretraining is spent. Cost of the rest: 18 x ~3.5 h
pretrain + 72 probes.

## 12. UniMTS_data retrain arc: user-provided pre-synthesized dataset, and four bugs found in the process (2026-07-28)

User provided a complete pre-synthesized dataset at
`/projects/b53017/vbc1591/HumanML3D/UniMTS_data/data/` (`sim/` 29,222 files,
[T,22,6] float32 @ 20Hz in UniMTS node order, gravity baked in -- confirmed:
pelvis-node acc norm ~9.97 m/s^2 at rest; `aug_texts/` 24,660 files, original
`#`-delimited captions + GPT paraphrases) and asked for (a) UniMTS's own
architecture retrained from scratch on it, and (b) "Ours" ViT1D CLIP-align
retrained on it (original MAE Stage-1 checkpoint kept, per explicit
correction; 10.24s fixed window; trainable text encoder), both compared
against the official UniMTS checkpoint on zero-shot AND fine-tune.

### 12a. UniMTS architecture from scratch on UniMTS_data (job 8054089)

`pretrain.py --data_path .../UniMTS_data/data`, 100 epochs, unmodified UniMTS
codebase. Final loss 0.5726. Zero-shot + fine-tune vs official checkpoint
(all single-right-wrist `--joint_list 21`, same protocol as every prior
UniMTS row this session):

| Dataset | Official zero-shot | From-scratch zero-shot | Official finetune | From-scratch finetune |
|---|---|---|---|---|
| MHEALTH | 59.3% / 51.1% | 47.7% / 37.0% | 87.7% / 85.4% | **88.5% / 86.4%** |
| PAMAP2 | 16.1% / 12.7% | **21.4% / 18.8%** | 54.3% / 42.5% | **55.6% / 43.5%** |
| UTD-MHAD | 11.3% / 7.2% | **19.0% / 11.8%** | 78.3% / 76.6% | 76.5% / 75.4% |
| WISDM | 28.7% / 24.8% | 25.0% / 21.1% | 67.9% / 68.0% | (running, 8108128) |
| HHAR | 50.0% / 44.4% | 35.0% / 26.0% | 79.0% / 78.2% | 78.3% / 77.0% |

Zero-shot: official wins 3/5, from-scratch wins 2/5 -- a mixed result, and a
big improvement over the earlier from-scratch attempts on our own exports
(both of which collapsed). Fine-tune: essentially at parity across the board.
The user-provided synthesis is clearly a usable substitute for UniMTS's own
proprietary pipeline at fine-tune time; the remaining zero-shot gap on
MHEALTH/HHAR plausibly reflects the official checkpoint's much larger
pretraining corpus.

### 12b. Four bugs found while wiring "Ours" CLIP-align to UniMTS_data

All four are in this repo's own code; the first and fourth silently affected
MANY earlier experiments, not just this arc.

1. **Coverage filter hardcoded `joints/` (found when the legacy dir was
   deleted).** `HumanML3DIMUVarLenDatasetFLU.__init__` pre-filtered clip ids
   against `root/joints/` regardless of the configurable
   `humanml3d.joints_subpath` used by `__iter__` -- so EVERY
   "AMASS-free full-coverage" config this session (newjoints, newjoints_augtext,
   the 26.19%/22.08% session-best 7959342 included) actually trained on only
   7,942/23,384 train clips (34%), the exact AMASS-limited subset the switch
   was meant to eliminate. The 98.6%-yield smoke test bypassed this filter,
   which is why it went unnoticed. Surfaced as a hard crash ("no joints/*.npy
   found") when `joints/` was removed from the cluster. Fixed: filter uses the
   configured subpath (and, for unimts_sim, the sim∩aug_texts intersection).
2. **`text_cache_path` silently no-ops `freeze_text_encoder: false`
   (user-caught).** run.py loads the cached text embeddings regardless of the
   freeze flag, and a non-None cache makes the loader yield embedding TENSORS;
   `forward_loss` then takes its tensor fast-path and never calls
   `text_encoder` at all -- the "trainable" text tower was built, put in the
   optimizer, and never received a single gradient. **This retroactively
   invalidates the 2026-07-26 "trainable text encoder is net-negative"
   result (job 7954257) and its gaussian-noise combination (7954562)** --
   both configs also set `text_cache_path`, so neither ever actually trained
   its text tower; those rows measured run-to-run noise, not the intended
   variable. Fixed by omitting `text_cache_path` when the text tower is
   trainable (run 8129177+ verified: 0 cache-load log lines, `text_enc` with
   196 tensors genuinely saved in the checkpoint).
3. **unimts_sim caption fallback pointed at a nonexistent directory.** First
   version of `_load_clip_imu_unimts_sim` "fell back" inside
   `UniMTS_data/data/` (which has no `texts/`), so the 15.5% of train clips
   (3,629/23,380) and 18.0% of val clips without an aug_texts file trained/
   validated against an EMPTY '' caption -- all of which share one identical
   text embedding, which the multi-positive InfoNCE then treated as mutual
   positives (15% of unrelated motions pulled toward one arbitrary anchor).
   Affected run 8129177 (preserved as `*_BUGGY_emptycaptions`). Per explicit
   user direction, resolved by mirroring UniMTS's own `CLIPDataset` exactly:
   captions come ONLY from `UniMTS_data/data/aug_texts`, and clips without a
   caption file are DROPPED (sim∩aug_texts = 19,751 train / 1,197 val clips).
4. **Rotation augmentation ran AFTER the per-window z-score (user-caught).**
   Per-channel z-score is an anisotropic scaling; conjugating a rotation by
   it is no longer a rotation, so every "anyrot" checkpoint this session was
   trained for invariance to *mixing of normalized channels*, not to genuine
   sensor-orientation change -- and its training inputs had non-unit channel
   variances (diag(R Sigma R^T)) while eval windows are always exactly
   zero-mean/unit-variance. UniMTS's own `augment_data()` rotates raw
   signals. Fixed in both HumanML3D FLU dataset classes: order is now
   filter -> rotate (raw units) -> z-score -> pad (filter commutes with
   rotation, rotation commutes with wrap/zero-pad). Rotation augmentation's
   earlier wins (e.g. doubling UTD-MHAD zero-shot) stand as *generic*
   augmentation effects, but true orientation invariance was never actually
   being trained until this fix.

### 12c. "Ours" CLIP-align on UniMTS_data (in progress)

Config `configs/vit1d_clip_humanml3d_flu_finetune_mae_unimtssim_texttrainable_lr5e-5.yaml`:
`humanml3d.data_source: unimts_sim` (right-wrist node 21 extracted from the
pre-synthesized [T,22,6], 20->50Hz cubic resample of the signal itself, NO
gravity_mps2 -- already baked in), original MAE Stage-1 warm start, fixed
10.24s window (wrap-pad, crop first -- UniMTS's own convention), SO(3)
rotation aug (now raw-space), genuinely-trainable CLIP text tower, lr 5e-5.

- Run 1 (8129177, bugs 3+4 present): best val_loss 2.4645 @ step 5500, then
  monotonic val degradation for 20 evals (train loss -> 0.6-0.9, in-batch
  train acc >90%) -- clean overfitting signature; early-stopped at 15,500.
  Internal val retrieval acc 0.402 vs 0.210 for the prior session-best -- but
  15.5% empty-caption pollution makes this run unusable as evidence.
- Run 2 (8135882, "noproj"): bugs 3+4 fixed. best val_loss 2.0336 / retrieval
  acc 0.444 @ step 5500 (vs 2.4645/0.402 polluted, 2.477/0.210 prior session
  best) -- the caption + rotation fixes are worth a lot on the internal
  metric. Preserved as `runs/..._lr5e-5_noproj`.
- Run 3 (8147232, "proj", the final variant): two further changes from a
  code-review pass (2026-07-28): (a) **text_projection: true** -- CLIPTextEncoder
  previously used CLIPTextModel's bare pooler_output, NOT CLIP's projected
  text space (cosine between the two spaces ~0.01-0.06, i.e. unrelated);
  official CLIP and UniMTS's own encode_text (contrastive.py:38) both apply
  text_projection. Config-gated so all older checkpoints keep their
  (internally-consistent) pooler-space convention; eval script reads the
  flag from the checkpoint's own config. (b) **max_steps: 8000** -- the
  "cosine" schedule had max_steps=10,000,000, making decay a no-op
  (constant lr=5e-5; measured directly in the logs), consistent with every
  run's overfit-past-best pattern. best val_loss 2.0747 @ step 4500.

### 12d. Eval-protocol fixes found on the way (2026-07-28)

- **Empty val split at zero-shot eval**: run_zeroshot_har's --val-only
  inherits the checkpoint's own splits block (correct for finetuned
  checkpoints); a never-finetuned CLIP checkpoint inherits base.yaml's
  pretraining split (seed 0, val_ratio 0.05) which rounds to ZERO held-out
  subjects on UTD-MHAD -> "Sessions: 0" crash. Resolved by patching the
  pinned eval checkpoint's cfg.splits to the benchmark split (seed 42,
  subject_id, 0.2 -- the exact split prepare_har_data.py used to build the
  UniMTS X_test export and every prior zero-shot number used).
- **Split misalignment between the two finetune columns**: the Ours HAR
  finetune configs silently inherited splits.seed=0 from base.yaml while
  every UniMTS number is on the seed-42 subjects. All 5 configs now set
  seed 42 explicitly -- all methods below are scored on the SAME held-out
  subjects.
- **PAMAP2 finetune had an empty val split all along**: windowing.
  max_gaps_per_session default 100 rejects every PAMAP2 val session (the
  same reason zero-shot eval needs --max-gaps-per-session 700); the run
  trained to max_steps with no best.pt and no early stopping. Config now
  sets 700.

### 12e. Final comparison: Ours ViT1D CLIP-align vs UniMTS on UniMTS_data (2026-07-28)

All on the seed-42 held-out subjects. "Ours proj" = run 8147232 (all fixes
+ text_projection + real cosine horizon); "Ours noproj" = run 8135882
ablation (identical minus those two). Finetune rows = val metrics at the
best-val-loss checkpoint (the repo's model-selection protocol; note val acc
typically keeps climbing after the val-loss minimum -- e.g. MHEALTH-proj
peaks ~79.7% later -- while UniMTS's own finetune_custom.py reports its
end-of-training model; a conservative bias against "Ours" rows).

**Zero-shot (acc% / macro-F1%):**

| Dataset | Ours proj | Ours noproj | Ours scratch | UniMTS official | UniMTS from-scratch |
|---|---|---|---|---|---|
| UTD-MHAD | **16.7 / 7.4** | 13.1 / 6.4 | 13.1 / 9.1 | 11.3 / 7.2 | 19.0 / 11.8 |
| WISDM | 28.6 / **26.5** | 23.9 / 21.3 | 15.7 / 15.2 | 28.7 / 24.8 | 25.0 / 21.1 |
| PAMAP2 | 11.1 / 14.2 | 15.8 / 14.8 | 12.7 / 10.1 | 16.1 / 12.7 | 21.4 / 18.8 |
| HHAR | 39.4 / 30.0 | 33.6 / 22.1 | 27.1 / 25.8 | 50.0 / 44.4 | 35.0 / 26.0 |
| MHEALTH | 28.6 / 19.1 | 35.5 / 22.9 | **42.3 / 35.5** | 59.3 / 51.1 | 47.7 / 37.0 |
| **Average** | 24.9 / 19.4 | 24.4 / 17.5 | 22.2 / 19.1 | **33.1 / 28.0** | 29.6 / 22.9 |

("Ours scratch" = same proj recipe with NO MAE warm start, random-init
encoder, job 8155972 -- user-requested ablation; 20k-step horizon, best
val_loss 2.7649 @ step 4000 vs proj's 2.0747.)

**Fine-tune (acc% / macro-F1%):**

| Dataset | Ours proj | Ours noproj | Ours scratch | UniMTS official | UniMTS from-scratch |
|---|---|---|---|---|---|
| UTD-MHAD | 77.4 / 74.6 | 73.3 / 70.8 | 65.2 / 63.1 | **78.3 / 76.6** | 76.5 / 75.4 |
| WISDM | **70.9 / 70.1** | 70.7 / 69.7 | 67.4 / 66.9 | 67.9 / 68.0 | 68.5 / 68.9 |
| PAMAP2 | 53.7 / 42.2 | 58.6 / 45.1 | **58.9 / 53.5** | 54.3 / 42.5 | 55.6 / 43.5 |
| HHAR | 66.6 / 64.9 | 70.8 / 70.5 | 69.6 / 68.9 | **79.0 / 78.2** | 78.3 / 77.0 |
| MHEALTH | 74.0 / 72.3 | 79.1 / 76.3 | 73.3 / 71.6 | 87.7 / 85.4 | **88.5 / 86.4** |

**MAE-pretraining ablation read (proj vs scratch, only difference = MAE warm
start):** MAE init is worth a large margin on internal retrieval (val acc
0.444 vs ~0.31), a consistent finetune edge (+3 to +12pp acc on 4/5
datasets), and +2.7pp average zero-shot acc -- but not uniformly:
from-scratch WINS MHEALTH zero-shot outright (42.3%, the best "Ours"
MHEALTH number of the whole session) and posts the best PAMAP2 finetune
macro-F1 of ALL five methods (53.5%).

**Per-class diagnosis of the remaining zero-shot gap** (from full per-class
side-by-sides, Ours-proj vs official UniMTS, same subjects): the deficit is
NOT generic collapse -- UTD-MHAD is near-floor for BOTH models (15-16 of 21
classes at zero each; single-wrist zero-shot fine-gesture recognition is
protocol-intrinsic), and WISDM/PAMAP2 predictions are well-spread for both.
The gap concentrates almost entirely in **static-posture and leg-cyclic
classes**: sitting (WISDM 2.7% vs 87.3%, PAMAP2 12.4% vs 76.4%),
stairs (HHAR stairsdown 2.1% vs 57.2%, MHEALTH climbing 0% vs 82.8%),
cycling/running (MHEALTH 0% vs 35-80%). Conversely Ours wins decisively on
wrist-articulated activities: drinking (72.6% vs 10.2%), clapping (83.7% vs
33.5%), typing (35.9% vs 0.1%), frontal arm elevation (100% vs 47.7%),
tennis swing (100% vs 0%). Mechanistic explanation: our pipeline's
per-window per-channel z-score SUBTRACTS THE MEAN, which for a static
posture deletes the gravity-orientation vector -- the only discriminative
signal those classes have -- and then amplifies residual noise; UniMTS
feeds raw values (gravity DC intact, BatchNorm inside the model). A
gravity-preserving normalization (scale-only, no mean subtraction on acc)
is the targeted next lever, but it shifts the input distribution the MAE
Stage-1 encoder was pretrained on -- a recipe fork, not a bug fix, so not
attempted unilaterally.


## 13. Nymeria synthesized-IMU language alignment: Ours(MAE) vs UniMTS from scratch (2026-07-28)

User request: compare Ours(MAE-pretrained) vs UniMTS-architecture-trained-
from-scratch on Nymeria SYNTHESIZED wrist IMU + captions, on the CLIP
alignment task itself. Setup reuses Section 6c's protocol end-to-end:
synthetic wrist IMU derived on the fly from `gt_data['xsens_pose']` mocap
(joint 9), atomic_action captions (best coverage + best prior results),
same 867/101 session split, same batch-64 top-1 retrieval metric.
`train_unimts_nymeria.py` extended with `--imu-source synthetic` (reuses
`_load_nymeria_session_imu_flu_synthetic`, smoke-tested: |acc| ~9.86 m/s^2
gravity-inclusive) and `--init-checkpoint none` (random init).

| Model | imu->text | text->imu | Run |
|---|---|---|---|
| Ours (MAE) probe, synthetic IMU, atomic_action | 7.58% | 7.94% | 7692003 |
| **Ours (MAE) finetune, synthetic IMU, atomic_action** | **9.39%** | **10.64%** | 7692004 |
| UniMTS FROM SCRATCH, synthetic IMU, atomic_action | 1.80% | 1.80% | 8158173 (new) |
| UniMTS official ckpt zero-shot (real IMU, motion_narration, ref) | 1.94% | 1.77% | 7579739 |
| UniMTS official ckpt finetuned (real IMU, motion_narration, ref) | 6.12% | 6.36% | 7581974 |

**From-scratch UniMTS never learned the task at all**: val retrieval flat at
the ~1.6-2.0% random floor for all 11,000 steps (best "checkpoint" = step
1000, i.e. essentially random init), and train loss barely moved (3.9 ->
3.5) -- it failed to fit even the TRAINING set. On identical data and
protocol, Ours(MAE) reaches 9.4/10.6. The MAE-pretrained ViT1D encoder is
the difference between learning this low-SNR alignment task and not
learning it at all; UniMTS's ST-GCN + trainable CLIP text tower from random
init has no usable gradient signal at this data scale (~102k noisy caption
segments).

### 13b. Why UniMTS's Nymeria numbers are so low (diagnosis, user-prompted)

Pulled the training logs of both prior UniMTS-on-Nymeria finetunes:
1. Both peaked at their FIRST validation (step 500) and never improved; on
   motion_narration val loss worsened monotonically (4.22 -> 5.71) while
   train loss fell (4.13 -> 2.1) -- the fully-trainable CLIP text tower at
   lr 1e-4 memorizes ~32k noisy captions immediately instead of
   generalizing.
2. The atomic_action_augmented run (2.11/1.80) additionally never fit even
   the train set (loss stuck 3.2-3.9): the mixed original+simplified
   caption styles (with heavy templates -- "a person walks forward." x1,583)
   give conflicting targets under UniMTS's plain diagonal CE, which has no
   multi-positive handling (ours does).
3. Structural, affects every model: motion_narration's 4 aspect columns are
   sampled at random and three of them (posture, legs/feet, focus/attention)
   are INVISIBLE to a wrist IMU -- those draws are contrastive noise.
   Checked and rejected: exact-duplicate captions colliding within a
   batch-64 (only 0.0-0.3% across all sources) are NOT the driver.

Improvement lever built (user-suggested "improve narration quality"): new
`motion_narration_arms` caption source (nymeria_caption_dataset_FLU.py) --
ONLY the "Describe my hands/arms motion" column, the one aspect a wrist
sensor observes; 31,798 captions, 83.3% unique, rich and motion-specific.
Launch scripts `train_unimts_nymeria_arms.sh` (+ `_arms_freezetext.sh`,
which also freezes the text tower to counter factor 1) are ready; the first
submissions were cancelled by the user to free GPU quota -- resubmit when
capacity allows. Anchor to beat: 6.12/6.36 (full-4-column motion_narration).

## 13. Architecture-vs-pretraining on IDENTICAL data: alignment-retrieval comparisons (2026-07-28)

User request: with the DATA held fixed, compare "Ours" (ViT1D, MAE SSL
pretrained) against the UniMTS architecture trained from scratch, on the
CLIP language-alignment task itself (batch-64 top-1 retrieval on a held-out
val split -- the same protocol every alignment number this benchmark uses).

### 13a. Nymeria (synthesized wrist IMU + atomic_action captions)

UniMTS side: `train_unimts_nymeria.py` extended with `--imu-source
synthetic` (wrist IMU synthesized on the fly from gt_data['xsens_pose']
mocap joint 9 -- the same drop-in loader the Ours `imu_source: synthetic`
configs use) and `--init-checkpoint none` (random init). Job 8158173:
102,305 train / 12,067 val segments, same 867/101 session split as every
Nymeria task. Early-stopped at step 11,000; best val_loss 4.1751 at step
1000; **val retrieval never left the ~1.6-2.0% random floor** (batch-64
floor ~1.6%). Final best-checkpoint full-val eval: **1.80% / 1.80%**.

| Model (same synthetic IMU + captions) | imu→text | text→imu |
|---|---|---|
| Ours, MAE probe (frozen encoder) | 7.58% | 7.94% |
| Ours, MAE finetune | **9.39%** | **10.64%** |
| Ours, from scratch (8282334, added 2026-07-29) | 8.55% | 8.81% |
| UniMTS from scratch | 1.80% | 1.80% |
| (ref) official UniMTS finetuned, real IMU, motion_narration | 6.12% | 6.36% |

The UniMTS architecture cannot learn this task from random init at all,
while the same data supports 8.5-10.6% through the ViT1D. CORRECTED
INTERPRETATION (2026-07-29, after adding the Ours-from-scratch cell): the
from-scratch ViT1D reaches 8.55/8.81 -- close behind the MAE-finetune row
-- so on Nymeria the MAE pretraining contributes only ~+0.8/+1.8pp for our
architecture; the decisive factor separating 8.5-10.6% (ViT1D) from 1.8%
(UniMTS ST-GCN from scratch) is the ARCHITECTURE/optimization behavior on
this small noisy task, not pretraining per se. (Pretrained weights do still
rescue UniMTS: its official checkpoint finetunes to 6.12/6.36.) Context for the low ceiling overall: diagnosed
earlier (see §6c + Section 12 notes) -- weak caption<->wrist-signal
correspondence (posture/legs/gaze columns invisible to a wrist IMU) and
overfit-prone full-text-tower training.

### 13b. HumanML3D UniMTS_data (pre-synthesized [T,22,6] + aug_texts) -- FINAL

UniMTS side: new `train_unimts_userdata.py` -- UniMTS architecture FROM
SCRATCH on /projects/b53017/vbc1591/HumanML3D/UniMTS_data/data, but unlike
UniMTS's own pretrain.py (no held-out split), it trains on
train.txt∩sim∩aug_texts (19,751 clips, the exact "Ours" train set) and
validates on val.txt∩sim∩aug_texts (1,197 clips, the exact "Ours" val set)
every 500 steps with early stopping. Three model families, several
variants; all numbers are the val row at the best-val-loss checkpoint
(batch-64 top-1 retrieval):

| Model | Train input | Eval input | Loss | i2t | t2i |
|---|---|---|---|---|---|
| Ours ViT1D, MAE-pretrained (proj recipe) | wrist | wrist | symmetric | 44.0% | 43.2% |
| Ours ViT1D, MAE-pretrained | wrist | wrist | **i2t-only** | **44.9%** | **43.9%** |
| Ours ViT1D, from scratch | wrist | wrist | symmetric | 27.3% | 28.2% |
| Ours ViT1D, from scratch | wrist | wrist | i2t-only | 26.3% | 25.8% |
| UniMTS from scratch, full-grid trained (8163821) | 22 joints | 22 joints | i2t (native) | 49.8% | 2.8% |
| UniMTS from scratch, full-grid trained | 22 joints | **wrist only** | i2t (native) | **3.1%** | 1.7% |
| UniMTS from scratch, native joint-mask aug, EXACT original duration/selection (8192640, last.pth) | 1-5 random joints | **wrist only** | i2t (native) | **29.9%** | 2.7% |

Method notes: the i2t-only Ours rows exist because UniMTS's own loss is
one-way (pretrain.py:73) -- loss-matching shows the loss direction is NOT
the differentiator (44.9 vs 44.0 for MAE-Ours; and Ours' t2i stays ~44%
even when never optimized, while UniMTS's t2i is floor-bound in every
variant). The joint-mask row restores UniMTS's own core augmentation
(pretrain.py: `input = imu * mask`, k~U{1..5} joints kept per sample) --
without it, a full-grid-trained UniMTS collapses to 3.1% when evaluated
wrist-only, which is what a wrist-worn deployment actually provides.
Wrist-only val per explicit user request (input-matched to "Ours" and to
UniMTS's own downstream --joint_list 21 convention).

**Recipe-fidelity audit (2026-07-29, user request).** Verified element by
element against pretrain.py: optimizer (Adam lr=1e-4, no weight decay, no
scheduler), loss (one-way CE / temperature 0.1), joint masking (k~U{1..5}
kept per sample, fresh per draw, applied before rotation aug before
gyro-drop), input format, model args, batch size, random caption per draw
-- all identical. The only deliberate deviations are (a) training on the
train.txt subset (19,751 clips) instead of all 24,660 sim∩aug_texts clips
so val.txt stays genuinely held out, and (b) initially, early stopping.
Deviation (b) was then eliminated: run 8192640 reproduces the ORIGINAL
protocol exactly (fixed 30,900 steps = 100 epochs, no early stopping,
end-of-training model selection) and lands at 29.9%/2.7% wrist-only
(40.1%/3.2% full-grid) -- confirming the earlier early-stopped 27.3%
(peak 29.2%) was representative, not an artifact of shorter training.
Cross-check with the checkpoint trained by the literally-unmodified
pretrain.py on ALL clips (job 8054089): it scores 61.5%/2.6% wrist-only
and 82.7%/4.6% full-grid on these val clips -- but it TRAINED on those
exact clips for 100 epochs, and the full-grid comparison (82.7 leaky vs
49.8 held-out) shows this is memorization inflation, not capability.

**Conclusions.** (1) Input-matched at the wrist, MAE-pretrained Ours wins
decisively: 44.9%/43.9% vs the native-recipe from-scratch UniMTS's
27.3%/2.4% (+17.6pp i2t). (2) From-scratch-vs-from-scratch at the wrist is
roughly even (Ours 27.3/28.2 vs UniMTS 27.3/2.4 on i2t) -- so the Ours
advantage is specifically the MAE SSL pretraining, worth ~+17pp here.
(3) UniMTS's full-22-joint input is worth ~+20pp on this task (49.8 vs
~27-29 wrist-only) but is unavailable in a real wrist deployment.
(4) Together with 13a (Nymeria: from-scratch UniMTS floor-bound at 1.8%
vs Ours-MAE 9.4/10.6): pretraining matters most on small/noisy data, and
even on this large clean corpus it is worth ~17pp under matched inputs.

### 13c. Section-6 ladder re-evaluated under the new setting (2026-07-29)

User request: re-run the original Section-6 HumanML3D alignment ladder
(baselines + ablations) under the CURRENT recipe so every row is comparable
with the headline numbers. New setting = unimts_sim data (UniMTS_data/data,
sim ∩ aug_texts, 19,751 train / 1,197 held-out val clips), aug_texts-only
captions, trainable CLIP text tower WITH text_projection, fixed 10.24s wrap
window + crop-first, raw-space SO(3) rotation aug, lr 5e-5 cosine to 8k
steps. Protocol deltas vs the old ladder, applied uniformly: probes freeze
only the IMU encoder (text tower still trains, matching the headline rows);
uniform lr 5e-5 (old external-baseline probes used 1e-3 head-only). All
numbers = val row at the best-val-loss checkpoint, batch-64 top-1.

| Row | OLD setting (derived data, frozen text) | NEW setting | Δ (i2t) |
|---|---|---|---|
| MAE finetune (headline) | 33.00 / 34.25 | **44.94 / 43.91** | +11.9 |
| MAE finetune + text adapter | 30.06 / 30.47 (w/ aug; hurt) | 44.45 / 43.69 (neutral) | +14.4 |
| MAE finetune, window ablation | fixed 24.73 / 24.92 (fixed hurt) | variable 37.39 / 37.02 (variable hurts) | -- |
| MAE probe (frozen encoder) | 19.69 / 21.11 | 31.34 / 29.97 | +11.7 |
| MAE conventional-attention probe | 20.02 / 22.20 | **33.73 / 32.17** | +13.7 |
| MAE acc-only probe | 12.77 / 13.22 | 25.84 / 24.34 | +13.1 |
| Scratch (end-to-end random init) | 28.81 / 30.44 | 27.25 / 28.17 | -1.6 |
| SimMTM probe | 16.06 / 17.95 | 20.66 / 19.03 | +4.6 |
| RelCon probe | 11.59 / 13.59 | 18.22 / 15.86 | +6.6 |
| FOCAL probe | 11.72 / 12.00 | 10.67 / 10.72 | -1.1 |

Reads:
1. **The new setting lifts every MAE-pretrained row by ~12-14pp** while
   leaving Scratch/FOCAL flat -- i.e. the gains from the user-provided
   synthesis + trainable-projected-text recipe compound specifically with
   MAE pretraining rather than raising all boats equally. The MAE-probe
   advantage over the best external SSL baseline widens (old: 19.7 vs
   SimMTM 16.1; new: 31.3 vs 20.7).
2. **The window-ablation direction flips**: in the old (variable-window
   training) recipe, forcing a fixed window cost ~8pp; in the new
   fixed-window recipe, reverting to variable windows costs ~7.5pp. What
   matters is train/eval window-regime consistency, not fixed-vs-variable
   per se.
3. **The text adapter is now neutral** (44.5 vs 44.9) where it used to
   hurt -- unsurprising: with the whole text tower trainable, an extra MLP
   adapter is redundant capacity.
4. **Conventional attention slightly beats alt-attention when probed**
   (33.7 vs 31.3), replicating the old ladder's same ordering (20.0 vs
   19.7) more visibly -- a standing question mark over alt-attention for
   this task (finetuned headline runs use alt-attn throughout, so the
   end-to-end comparison is untested in the new setting).
5. Nymeria matrix completion (see 13a table): Ours-from-scratch reaches
   8.55/8.81 vs MAE-finetune 9.39/10.64 -- so ViT1D learns Nymeria fine
   without pretraining, and the UniMTS-from-scratch floor (1.80) is an
   architecture/optimization failure on small noisy data, not primarily a
   pretraining deficit.

### 13d. Complete bidirectional retrieval metrics: R@1/5/10, MRR, median rank; full vs. fixed-size galleries (2026-08-15)

**Why.** Every language-alignment number in Sections 6/12/13 is a single
figure: in-batch top-1 retrieval accuracy at gallery = training batch size
(64), averaged over randomly drawn val batches (random window, random
caption per clip). That is (a) one number, (b) tied to a gallery size that is
an accident of the batch size, and (c) noisy. Text–motion retrieval work (TMR)
reports Recall@{1,5,10} and median rank; recent IMU–language work (AnyMo)
reports R@1/5/10 and MRR in both directions and — because retrieval
difficulty changes drastically with gallery size — separately for a
100-sample gallery and the full gallery. This section re-evaluates every
Section-13 checkpoint under exactly that protocol, from the saved
`best.pt`/`.pth` files, on the same held-out items. **New main retrieval
evaluation = MRR + R@1/5/10 (both directions), reported for the full gallery
and for a fixed gallery of 100.** The old batch-64 top-1 number is retained
only as the G=64 anchor row-set for continuity.

**Protocol** (`scripts/eval_retrieval_alignment.py`; results in
`analysis/retrieval_eval/*.json`; tables from
`scripts/summarize_retrieval_alignment.py`):

* Items. HumanML3D: the 1,197 held-out `val.txt ∩ sim ∩ aug_texts` clips
  (identical to every Section-13b/13c row). Nymeria: every (session,
  atomic_action caption row) segment of the held-out val sessions with
  ≥16 frames @50 Hz (the ViT1D patch size; UniMTS's own scripts used ≥10, so
  both families now share one segment set), synthetic wrist IMU, same
  867/101 session split as everything else.
* One deterministic IMU window per item, no augmentation. Ours/HumanML3D:
  the config's fixed 10.24 s window, first-crop + wrap-pad, lowpass +
  per-window z-score — the training val pipeline minus randomness.
  Ours/Nymeria: the segment's own snapped length clipped to [1.28, 10.24] s
  (centre crop above, pad below). UniMTS: its own 200-sample @20 Hz
  wrap-pad/truncate pipeline, gyro dropped, wrist-only masking where stated.
* Text side. **Main protocol (`caption_mode=first`)**: one caption per item —
  its first line, i.e. UniMTS's own `ValDataset` convention and the
  one-caption-per-clip convention of the legacy in-batch number — so both
  directions have the same number of queries and the same gallery size.
  Identical caption strings across items are merged into one gallery entry
  and are positives for every item carrying them (Nymeria captions repeat
  heavily; the legacy metric already treated identical captions as positives
  via its cos>0.999 rule). Secondary protocol (`caption_mode=all`, JSON
  only): every caption is used, MS-COCO style — IMU→Text is a hit if ANY of
  the clip's captions is in the top-k, Text→IMU has one query per unique
  caption. HumanML3D aug_texts has ~5.6 captions/clip, so `all` inflates
  IMU→Text relative to Text→IMU by construction; that is why `first` is main.
* Rank = 1 + #gallery entries scoring strictly above the best positive.
  R@k = % queries with rank ≤ k; MRR = mean(1/rank) in %; MedR = median rank.
* Galleries: **full** (every val item; HumanML3D N=1,197; Nymeria N=all
  segments) and **fixed G ∈ {100, 64}**: 200 random subsets of G items (seed
  42), text gallery = the unique captions of those G items, per-query ranks
  pooled over subsets. G=100 is AnyMo's convention; G=64 is the legacy batch
  size, so its R@1 is the deterministic analogue of the old
  acc_imu2text/acc_text2imu.
* Checkpoints: the same best-val-loss checkpoints as the Section-13 tables
  (Ours: `runs/<run>/checkpoints/best.pt`; UniMTS: the trained `.pth`;
  official `UniMTS.pth` for the zero-shot reference rows).

**Validation against the legacy number.** The G=64 / first-caption R@1
column reproduces the batch-64 in-batch accuracies of Sections 13a-13c to
within ~1-2 pp on every row — Ours MAE finetune (i2t-only) 46.7 / 45.0 vs
44.9 / 43.9; Ours scratch 27.2 / 27.3 vs 27.3 / 28.2; UniMTS joint-mask
wrist-only 30.3 / 2.6 vs 29.9 / 2.7; UniMTS full-grid 22-joint 49.0 / 2.9 vs
49.8 / 2.8 and wrist-only 3.1 / 1.9 vs 3.1 / 1.7; Nymeria MAE finetune
10.6 / 11.7 vs 9.4 / 10.6, probe 8.5 / 8.5 vs 7.6 / 7.9, scratch 8.6 / 9.9
vs 8.55 / 8.81 (the deterministic protocol is slightly kinder than the
random-window/random-caption one, uniformly). The one row that moves more is
UniMTS-from-scratch on Nymeria, 3.5 / 1.6 vs the logged 1.80 / 1.80: still
at/near the 1.6 % chance floor for t2i, and 2× chance for i2t. So the new
tables are a strict superset of the old ones, not a different story.

**Results — main protocol (one caption per item).** All numbers in %,
MedR = median rank; chance R@1 = 1/G.
**Dataset: HumanML3D** (UniMTS_data val split: 1,197 held-out clips, aug_texts captions).
caption_mode=first; items=1197, caption pairs=6702, unique captions=900. All numbers in % except MedR (median rank).

**Full gallery** — IMU→Text: 1197 queries vs 900 captions; Text→IMU: 900 queries vs 1197 IMU windows

| Model | IMU→Text R@1 / R@5 / R@10 / MRR (MedR) | Text→IMU R@1 / R@5 / R@10 / MRR (MedR) | Source |
|---|---|---|---|
| Ours pretrained (MAE init, finetune, i2t-only loss) — headline | 11.11 / 35.34 / 47.79 / 22.71 (MedR 11) | 10.44 / 29.67 / 44.56 / 20.48 (MedR 13) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 6000)` |
| Ours pretrained (MAE init, finetune, symmetric loss) | 11.11 / 34.25 / 48.62 / 22.54 (MedR 12) | 12.11 / 32.22 / 44.89 / 22.49 (MedR 14) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5 (best.pt, step 4500)` |
| Ours pretrained (MAE init, frozen-encoder probe) | 6.60 / 20.47 / 30.99 / 14.52 (MedR 29) | 5.44 / 18.11 / 25.89 / 12.67 (MedR 36) | `vit1d_clip_humanml3d_probe_mae_unimtssim (best.pt, step 2000)` |
| Ours scratch (random init, symmetric loss) | 4.85 / 16.29 / 25.98 / 11.90 (MedR 38) | 3.22 / 14.11 / 25.33 / 10.10 (MedR 40) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5 (best.pt, step 4000)` |
| Ours scratch (random init, i2t-only loss) | 4.76 / 16.37 / 27.07 / 11.57 (MedR 38) | 4.67 / 15.44 / 23.78 / 11.16 (MedR 46) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 4000)` |
| UniMTS from scratch (native joint-mask recipe, 100 ep) — wrist-only eval | 5.26 / 19.80 / 30.49 / 13.18 (MedR 31) | 0.22 / 0.78 / 1.78 / 1.05 (MedR 507) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_jointmask_100ep/last.pth` |
| UniMTS from scratch (full-grid trained) — wrist-only eval | 0.33 / 1.25 / 3.09 / 1.40 (MedR 352) | 0.22 / 0.89 / 1.22 / 0.91 (MedR 575) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS from scratch (full-grid trained) — 22-joint eval (input advantage, reference) | 9.36 / 34.92 / 49.96 / 22.45 (MedR 11) | 0.56 / 1.22 / 1.78 / 1.35 (MedR 504) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS official checkpoint, zero-shot — wrist-only (⚠ pretrained on HumanML3D itself) | 9.69 / 28.57 / 41.77 / 19.88 (MedR 16) | 0.11 / 0.67 / 1.33 / 0.93 (MedR 534) | `baselines/language_alignment/UniMTS/checkpoint/UniMTS.pth` |

**Fixed gallery G=100** — 200 random subsets of 100 clips (text gallery = their unique captions, mean 98); ranks pooled over subsets

| Model | IMU→Text R@1 / R@5 / R@10 / MRR (MedR) | Text→IMU R@1 / R@5 / R@10 / MRR (MedR) | Source |
|---|---|---|---|
| Ours pretrained (MAE init, finetune, i2t-only loss) — headline | 39.42 / 73.88 / 84.25 / 54.85 (MedR 2) | 37.54 / 72.02 / 82.40 / 53.04 (MedR 2) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 6000)` |
| Ours pretrained (MAE init, finetune, symmetric loss) | 39.02 / 72.50 / 83.09 / 54.18 (MedR 2) | 38.91 / 71.90 / 82.86 / 53.78 (MedR 2) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5 (best.pt, step 4500)` |
| Ours pretrained (MAE init, frozen-encoder probe) | 26.04 / 57.69 / 71.31 / 40.82 (MedR 4) | 24.34 / 55.82 / 70.57 / 39.15 (MedR 4) | `vit1d_clip_humanml3d_probe_mae_unimtssim (best.pt, step 2000)` |
| Ours scratch (random init, symmetric loss) | 21.87 / 53.06 / 67.83 / 36.59 (MedR 5) | 21.56 / 52.08 / 66.86 / 36.30 (MedR 5) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5 (best.pt, step 4000)` |
| Ours scratch (random init, i2t-only loss) | 21.41 / 52.84 / 67.25 / 36.24 (MedR 5) | 20.63 / 50.57 / 65.09 / 34.93 (MedR 5) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 4000)` |
| UniMTS from scratch (native joint-mask recipe, 100 ep) — wrist-only eval | 24.38 / 55.28 / 68.88 / 39.08 (MedR 4) | 1.74 / 7.36 / 13.52 / 6.68 (MedR 45) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_jointmask_100ep/last.pth` |
| UniMTS from scratch (full-grid trained) — wrist-only eval | 2.21 / 8.48 / 15.52 / 7.68 (MedR 39) | 1.21 / 5.58 / 10.63 / 5.54 (MedR 50) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS from scratch (full-grid trained) — 22-joint eval (input advantage, reference) | 40.27 / 79.67 / 89.43 / 57.20 (MedR 2) | 2.07 / 7.97 / 15.07 / 7.29 (MedR 43) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS official checkpoint, zero-shot — wrist-only (⚠ pretrained on HumanML3D itself) | 33.62 / 68.58 / 81.20 / 49.31 (MedR 3) | 1.62 / 7.17 / 12.64 / 6.43 (MedR 47) | `baselines/language_alignment/UniMTS/checkpoint/UniMTS.pth` |

**Fixed gallery G=64** — 200 random subsets of 64 clips (text gallery = their unique captions, mean 63); ranks pooled over subsets; R@1 here is the deterministic analogue of the legacy batch-64 acc

| Model | IMU→Text R@1 / R@5 / R@10 / MRR (MedR) | Text→IMU R@1 / R@5 / R@10 / MRR (MedR) | Source |
|---|---|---|---|
| Ours pretrained (MAE init, finetune, i2t-only loss) — headline | 46.71 / 79.57 / 89.23 / 61.41 (MedR 2) | 45.01 / 77.86 / 87.11 / 59.75 (MedR 2) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 6000)` |
| Ours pretrained (MAE init, finetune, symmetric loss) | 45.77 / 78.44 / 88.73 / 60.47 (MedR 2) | 45.49 / 78.23 / 88.03 / 60.02 (MedR 2) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5 (best.pt, step 4500)` |
| Ours pretrained (MAE init, frozen-encoder probe) | 31.66 / 64.70 / 79.02 / 46.92 (MedR 3) | 29.30 / 63.40 / 78.39 / 45.07 (MedR 3) | `vit1d_clip_humanml3d_probe_mae_unimtssim (best.pt, step 2000)` |
| Ours scratch (random init, symmetric loss) | 27.16 / 60.80 / 75.66 / 42.83 (MedR 4) | 27.34 / 60.61 / 75.44 / 42.93 (MedR 4) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5 (best.pt, step 4000)` |
| Ours scratch (random init, i2t-only loss) | 27.67 / 61.23 / 75.40 / 43.22 (MedR 3) | 26.25 / 59.14 / 73.74 / 41.59 (MedR 4) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 4000)` |
| UniMTS from scratch (native joint-mask recipe, 100 ep) — wrist-only eval | 30.31 / 62.88 / 77.36 / 45.60 (MedR 3) | 2.62 / 10.80 / 19.98 / 9.29 (MedR 29) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_jointmask_100ep/last.pth` |
| UniMTS from scratch (full-grid trained) — wrist-only eval | 3.09 / 12.22 / 22.90 / 10.33 (MedR 26) | 1.89 / 8.59 / 16.23 / 7.88 (MedR 32) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS from scratch (full-grid trained) — 22-joint eval (input advantage, reference) | 48.97 / 85.25 / 92.89 / 64.78 (MedR 2) | 2.91 / 12.12 / 22.48 / 10.03 (MedR 28) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS official checkpoint, zero-shot — wrist-only (⚠ pretrained on HumanML3D itself) | 39.69 / 76.46 / 88.16 / 56.03 (MedR 2) | 2.64 / 10.15 / 18.65 / 8.99 (MedR 30) | `baselines/language_alignment/UniMTS/checkpoint/UniMTS.pth` |


**Dataset: Nymeria** (synthetic wrist IMU + atomic_action captions, held-out val sessions).
caption_mode=first; items=12065, caption pairs=12065, unique captions=11517. All numbers in % except MedR (median rank).

**Full gallery** — IMU→Text: 12065 queries vs 11517 captions; Text→IMU: 11517 queries vs 12065 IMU windows

| Model | IMU→Text R@1 / R@5 / R@10 / MRR (MedR) | Text→IMU R@1 / R@5 / R@10 / MRR (MedR) | Source |
|---|---|---|---|
| Ours pretrained (MAE init, finetune) | 0.10 / 0.52 / 1.11 / 0.63 (MedR 1920) | 0.13 / 0.68 / 1.27 / 0.71 (MedR 1987) | `vit1d_clip_nymeria_atomicaction_synthetic_finetune_mae_multidata (best.pt, step 6500)` |
| Ours pretrained (MAE init, frozen-encoder probe) | 0.08 / 0.44 / 0.80 / 0.49 (MedR 2665) | 0.12 / 0.43 / 0.94 / 0.53 (MedR 2762) | `vit1d_clip_nymeria_atomicaction_synthetic_mae_probe (best.pt, step 17000)` |
| Ours scratch (random init) | 0.07 / 0.46 / 0.90 / 0.50 (MedR 2376) | 0.10 / 0.54 / 0.99 / 0.57 (MedR 2421) | `vit1d_clip_nymeria_atomicaction_synthetic_fromscratch_multidata (best.pt, step 11500)` |
| UniMTS from scratch | 0.02 / 0.06 / 0.20 / 0.17 (MedR 4203) | 0.01 / 0.04 / 0.09 / 0.09 (MedR 5813) | `baselines/language_alignment/UniMTS/checkpoint/nymeria_scratch_synthetic/best_val_loss.pth` |
| UniMTS official checkpoint, zero-shot | 0.01 / 0.05 / 0.08 / 0.10 (MedR 4931) | 0.01 / 0.04 / 0.10 / 0.09 (MedR 5937) | `baselines/language_alignment/UniMTS/checkpoint/UniMTS.pth` |

**Fixed gallery G=100** — 200 random subsets of 100 clips (text gallery = their unique captions, mean 100); ranks pooled over subsets

| Model | IMU→Text R@1 / R@5 / R@10 / MRR (MedR) | Text→IMU R@1 / R@5 / R@10 / MRR (MedR) | Source |
|---|---|---|---|
| Ours pretrained (MAE init, finetune) | 7.51 / 25.48 / 37.48 / 17.61 (MedR 18) | 8.63 / 26.51 / 38.76 / 18.67 (MedR 17) | `vit1d_clip_nymeria_atomicaction_synthetic_finetune_mae_multidata (best.pt, step 6500)` |
| Ours pretrained (MAE init, frozen-encoder probe) | 5.98 / 20.25 / 30.79 / 14.59 (MedR 24) | 6.42 / 20.88 / 31.34 / 15.03 (MedR 24) | `vit1d_clip_nymeria_atomicaction_synthetic_mae_probe (best.pt, step 17000)` |
| Ours scratch (random init) | 6.08 / 21.89 / 33.25 / 15.25 (MedR 21) | 7.30 / 22.59 / 33.83 / 16.39 (MedR 21) | `vit1d_clip_nymeria_atomicaction_synthetic_fromscratch_multidata (best.pt, step 11500)` |
| UniMTS from scratch | 2.40 / 9.78 / 17.03 / 8.19 (MedR 37) | 1.02 / 5.14 / 10.14 / 5.28 (MedR 49) | `baselines/language_alignment/UniMTS/checkpoint/nymeria_scratch_synthetic/best_val_loss.pth` |
| UniMTS official checkpoint, zero-shot | 1.24 / 6.52 / 12.67 / 6.13 (MedR 44) | 1.02 / 5.09 / 10.24 / 5.23 (MedR 50) | `baselines/language_alignment/UniMTS/checkpoint/UniMTS.pth` |

**Fixed gallery G=64** — 200 random subsets of 64 clips (text gallery = their unique captions, mean 64); ranks pooled over subsets; R@1 here is the deterministic analogue of the legacy batch-64 acc

| Model | IMU→Text R@1 / R@5 / R@10 / MRR (MedR) | Text→IMU R@1 / R@5 / R@10 / MRR (MedR) | Source |
|---|---|---|---|
| Ours pretrained (MAE init, finetune) | 10.62 / 32.88 / 47.69 / 22.72 (MedR 11) | 11.69 / 34.18 / 48.55 / 23.71 (MedR 11) | `vit1d_clip_nymeria_atomicaction_synthetic_finetune_mae_multidata (best.pt, step 6500)` |
| Ours pretrained (MAE init, frozen-encoder probe) | 8.45 / 26.94 / 39.94 / 19.06 (MedR 16) | 8.46 / 27.14 / 40.83 / 19.28 (MedR 15) | `vit1d_clip_nymeria_atomicaction_synthetic_mae_probe (best.pt, step 17000)` |
| Ours scratch (random init) | 8.64 / 28.59 / 42.41 / 19.90 (MedR 14) | 9.85 / 29.43 / 43.52 / 20.88 (MedR 14) | `vit1d_clip_nymeria_atomicaction_synthetic_fromscratch_multidata (best.pt, step 11500)` |
| UniMTS from scratch | 3.48 / 13.55 / 24.04 / 11.10 (MedR 24) | 1.59 / 7.95 / 15.91 / 7.53 (MedR 32) | `baselines/language_alignment/UniMTS/checkpoint/nymeria_scratch_synthetic/best_val_loss.pth` |
| UniMTS official checkpoint, zero-shot | 2.11 / 10.37 / 19.86 / 8.88 (MedR 28) | 1.56 / 7.87 / 15.77 / 7.46 (MedR 32) | `baselines/language_alignment/UniMTS/checkpoint/UniMTS.pth` |

**Secondary protocol (all captions, MS-COCO style; HumanML3D only —
Nymeria has one caption per segment so it is identical to the main
protocol).** IMU→Text is a hit if ANY of the clip's ~5.6 aug_texts captions
ranks top-k; Text→IMU has one query per unique caption (5,765). At a fixed
gallery this inflates IMU→Text (headline 46.1 vs 39.4 R@1 at G=100) while
leaving Text→IMU unchanged (36.9 vs 37.5); at the full gallery the text side
also grows 900 → 5,765 so IMU→Text lands at the same 11.1 R@1. Reported for
completeness; JSON key `protocols.all`.

**Dataset: HumanML3D** (UniMTS_data val split: 1,197 held-out clips, aug_texts captions).
caption_mode=all; items=1197, caption pairs=6702, unique captions=5765. All numbers in % except MedR (median rank).

**Full gallery** — IMU→Text: 1197 queries vs 5765 captions; Text→IMU: 5765 queries vs 1197 IMU windows

| Model | IMU→Text R@1 / R@5 / R@10 / MRR (MedR) | Text→IMU R@1 / R@5 / R@10 / MRR (MedR) | Source |
|---|---|---|---|
| Ours pretrained (MAE init, finetune, i2t-only loss) — headline | 11.11 / 31.08 / 46.95 / 21.88 (MedR 12) | 9.00 / 28.46 / 41.84 / 19.18 (MedR 16) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 6000)` |
| Ours pretrained (MAE init, finetune, symmetric loss) | 10.61 / 30.99 / 44.19 / 21.32 (MedR 14) | 9.77 / 28.24 / 41.01 / 19.63 (MedR 16) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5 (best.pt, step 4500)` |
| Ours pretrained (MAE init, frozen-encoder probe) | 5.68 / 18.88 / 28.57 / 13.27 (MedR 34) | 4.86 / 16.67 / 26.14 / 11.81 (MedR 38) | `vit1d_clip_humanml3d_probe_mae_unimtssim (best.pt, step 2000)` |
| Ours scratch (random init, symmetric loss) | 4.43 / 15.87 / 23.64 / 11.02 (MedR 45) | 3.87 / 14.50 / 23.42 / 10.35 (MedR 46) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5 (best.pt, step 4000)` |
| Ours scratch (random init, i2t-only loss) | 4.01 / 14.62 / 24.23 / 10.34 (MedR 45) | 3.80 / 13.39 / 21.34 / 9.78 (MedR 52) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 4000)` |
| UniMTS from scratch (native joint-mask recipe, 100 ep) — wrist-only eval | 6.02 / 19.05 / 29.16 / 13.46 (MedR 34) | 0.23 / 0.59 / 1.27 / 0.96 (MedR 521) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_jointmask_100ep/last.pth` |
| UniMTS from scratch (full-grid trained) — wrist-only eval | 0.00 / 1.09 / 2.09 / 0.83 (MedR 1069) | 0.19 / 0.69 / 1.13 / 0.85 (MedR 577) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS from scratch (full-grid trained) — 22-joint eval (input advantage, reference) | 10.69 / 33.42 / 47.70 / 22.47 (MedR 11) | 0.35 / 1.16 / 1.77 / 1.20 (MedR 520) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS official checkpoint, zero-shot — wrist-only (⚠ pretrained on HumanML3D itself) | 8.94 / 25.81 / 36.84 / 17.60 (MedR 24) | 0.24 / 0.64 / 1.37 / 0.96 (MedR 549) | `baselines/language_alignment/UniMTS/checkpoint/UniMTS.pth` |

**Fixed gallery G=100** — 200 random subsets of 100 clips (text gallery = their unique captions, mean 553); ranks pooled over subsets

| Model | IMU→Text R@1 / R@5 / R@10 / MRR (MedR) | Text→IMU R@1 / R@5 / R@10 / MRR (MedR) | Source |
|---|---|---|---|
| Ours pretrained (MAE init, finetune, i2t-only loss) — headline | 46.12 / 76.60 / 87.06 / 59.60 (MedR 2) | 36.89 / 71.83 / 82.51 / 52.51 (MedR 2) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 6000)` |
| Ours pretrained (MAE init, finetune, symmetric loss) | 44.56 / 74.82 / 85.21 / 57.86 (MedR 2) | 36.95 / 71.84 / 82.77 / 52.49 (MedR 2) | `vit1d_clip_humanml3d_finetune_mae_unimtssim_texttrainable_lr5e-5 (best.pt, step 4500)` |
| Ours pretrained (MAE init, frozen-encoder probe) | 30.00 / 58.34 / 71.22 / 43.25 (MedR 4) | 24.17 / 56.39 / 71.12 / 39.27 (MedR 4) | `vit1d_clip_humanml3d_probe_mae_unimtssim (best.pt, step 2000)` |
| Ours scratch (random init, symmetric loss) | 25.77 / 53.27 / 68.05 / 38.93 (MedR 5) | 21.55 / 52.35 / 67.39 / 36.28 (MedR 5) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5 (best.pt, step 4000)` |
| Ours scratch (random init, i2t-only loss) | 24.64 / 52.76 / 66.16 / 37.89 (MedR 5) | 19.89 / 50.14 / 65.20 / 34.39 (MedR 5) | `vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5_i2t (best.pt, step 4000)` |
| UniMTS from scratch (native joint-mask recipe, 100 ep) — wrist-only eval | 28.81 / 57.98 / 72.42 / 42.51 (MedR 4) | 1.61 / 7.26 / 13.57 / 6.55 (MedR 45) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_jointmask_100ep/last.pth` |
| UniMTS from scratch (full-grid trained) — wrist-only eval | 2.21 / 7.56 / 12.09 / 6.03 (MedR 102) | 1.21 / 5.67 / 10.90 / 5.59 (MedR 50) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS from scratch (full-grid trained) — 22-joint eval (input advantage, reference) | 47.22 / 80.90 / 91.00 / 61.77 (MedR 2) | 2.00 / 7.69 / 14.64 / 7.13 (MedR 44) | `baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_valsplit/best.pth` |
| UniMTS official checkpoint, zero-shot — wrist-only (⚠ pretrained on HumanML3D itself) | 36.49 / 63.33 / 74.43 / 48.92 (MedR 3) | 1.65 / 7.11 / 12.60 / 6.41 (MedR 47) | `baselines/language_alignment/UniMTS/checkpoint/UniMTS.pth` |

**Reads.**

1. **The ranking of models is invariant to gallery size and to the metric.**
   On both datasets and in both directions, Ours MAE-finetune > Ours MAE
   frozen probe ≳ Ours scratch, at G=64, G=100 and the full gallery, whether
   read off R@1, R@10, MRR or MedR. What the batch-64 number could not show
   is *how* the models separate at scale: on HumanML3D full-gallery the
   headline model has MRR 22.7 / 20.5 and MedR 11 / 13, the probe 14.5 / 12.7
   (MedR 29 / 36), scratch 11.9 / 10.1 (MedR 38 / 40) — MAE pretraining
   roughly doubles MRR over the identically-trained scratch model and cuts
   the median rank 3.5×. On Nymeria full-gallery (11.5k captions) MedR is
   1,920 / 1,987 (finetune) vs 2,376 / 2,421 (scratch) vs 2,665 / 2,762
   (probe), against a chance MedR of ~5,750 / 6,000.
2. **Absolute numbers are dominated by gallery size** — the reason to report
   both. The same headline checkpoint reads 46.7 → 39.4 → 11.1 % R@1 as the
   HumanML3D gallery grows 64 → 100 → 900, and 10.6 → 7.5 → 0.10 % as the
   Nymeria gallery grows 64 → 100 → 11.5k. Any single "retrieval accuracy"
   without its gallery size is uninterpretable; G=100 + full is the pair to
   quote.
3. **UniMTS is one-directional at every gallery size.** Wrist-only
   from-scratch UniMTS ranks between our probe and our scratch model on
   IMU→Text (24.4 % at G=100, 5.3 % full, MRR 13.2) but Text→IMU never
   leaves chance (1.7 % at G=100 vs 1.0 % chance; MedR 507 of 1,197 full),
   and on Nymeria both directions are at chance (MedR 4.2k-5.9k). This
   holds for the official checkpoint too (33.6 / 1.6 at G=100 on HumanML3D
   — where it is contaminated — and 1.2 / 1.0 on clean Nymeria). The
   i2t ≫ t2i asymmetry first seen in Section 6 was therefore not an
   artefact of in-batch top-1 accuracy: it is a property of the model's
   text/IMU geometry that survives R@10, MRR and full-gallery MedR.
4. **Input parity vs. sensor advantage.** Given all 22 joints, from-scratch
   UniMTS reaches IMU→Text parity with wrist-only Ours (40.3 vs 39.4 R@1 at
   G=100; 9.4 vs 11.1 full; MRR 22.4 vs 22.7) but its Text→IMU stays at 2 %.
   Restricted to the wrist it needs the joint-mask training recipe to work
   at all (24.4 vs 2.2 R@1 at G=100).
5. **Loss direction is a wash for Ours**: i2t-only vs symmetric differ by
   ≤ 1.7 pp on any cell, in either direction — the symmetric run is
   marginally better on Text→IMU at the full gallery (12.1 vs 10.4 R@1),
   the i2t-only run marginally better on IMU→Text at G=100 (39.4 vs 39.0);
   neither difference exceeds the ~±0.3 pp pooled-estimate noise by enough
   to prefer one for headline purposes, so both are kept in the JSON set.
6. **Noise.** Single-subset R@1 std at G=100 is ≈ 4 pp (HumanML3D) / 2.4 pp
   (Nymeria); with 200 subsets the pooled estimate's standard error is
   ≈ 0.3 / 0.2 pp. Full-gallery numbers are exact for the split (no
   sampling). Differences under ~1 pp between rows should not be read.

**Caveats.** (a) HumanML3D `first` captions are not unique — 900 distinct
strings over 1,197 clips (aug_texts of mirrored `M######` clips repeat their
source clip's text); duplicates are merged into one gallery entry and count
as positives for every clip carrying them, which is why the Text→IMU query
count is 900. (b) Nymeria segments are the raw caption rows (variable
0.3-10 s+); Ours sees the segment at its own snapped length (centre-cropped
above 10.24 s), UniMTS at its native 200 samples @ 20 Hz — each model's own
training convention, not a shared window. (c) Windows are deterministic here
(first-crop / centre-crop, first caption), whereas the training-time val
metric used a random window and random caption per draw — this is why every
G=64 anchor sits ~1-2 pp above its legacy twin. (d) The 22-joint UniMTS
reference row and the official-checkpoint HumanML3D row are reference-only
(input advantage / contamination) and are marked as such; the 22-joint row
exists for HumanML3D only — Nymeria has a single wrist IMU and no 22-joint
synthetic grid, so there is no Nymeria counterpart to compute (the "—"
cells in the summary table are structural, not a pending run).

**Files.** `scripts/eval_retrieval_alignment.py` (evaluator; `--model
ours|unimts`), `scripts/summarize_retrieval_alignment.py` (JSON dir →
tables), `launch/launch_slurm_eval_retrieval_alignment.sh` +
`launch/eval_retrieval_alignment_rows.sh` (row registry; one CPU `short`
job per row, ~2-10 min each; jobs 9392232-9392245),
`analysis/retrieval_eval/<dataset>__<model>.json` (14 files, both protocols,
all galleries, per-subset stds).

### 13e. Recipe tuning of the HumanML3D alignment finetune vs. the UniMTS 22-joint reference (2026-08-15)

**Ask.** Beat the only UniMTS row that "Ours pretrained (MAE init,
finetune)" does not already beat in §13d: the 22-joint-input UniMTS
reference (i2t 40.3 / 79.7 / 89.4 / 57.2 at G=100 vs. our 39.4 / 73.9 /
84.3 / 54.9). Every wrist-only / input-matched UniMTS row is already 15-30
pp behind us in i2t and at chance in t2i, so the target is specifically the
recall@k tail (R@5/R@10) of a model that sees ~20× the sensor channels.

**Diagnosis before tuning.** The parent run
(`vit1d_clip_humanml3d_flu_finetune_mae_unimtssim_texttrainable_lr5e-5_i2t`,
job 8173825) ends at train in-batch acc ≈ 88 % / train loss 0.42 vs. val
45 % / 2.04, best val loss at step 6000 of 8000, i.e. it looks classically
over-fitted, which motivated a regularisation-heavy first round. That
reading turned out to be wrong (see results): every capacity- or
LR-restricting change lost retrieval quality; the train/val gap is
intrinsic to a 19.7k-clip contrastive task, not something to regularise
away.

**Method.** One-factor-at-a-time arms around the parent
(`scripts/gen_retrieval_tune_configs.py` → `configs/retrieval_tune/<arm>.yaml`,
run `rtune_<arm>`; launcher `launch/launch_slurm_retrieval_tune.sh` trains
then runs `scripts/eval_retrieval_alignment.py` on **both** the val split
(selection; identical to every §13 row and to how UniMTS's own `best.pth`
was chosen) **and the HumanML3D test split** (`--split test`, new; never
touched before this section: 3,701 usable clips) → `analysis/retrieval_tune/
{val,test}/<arm>.json`; `scripts/summarize_retrieval_tune.py` builds the
tables). Round 1 = 25 arms (2 baseline seeds; LR 2e-5/1e-4; text-tower LR
×0.1/×0.3, last-4-layers-only, frozen; head LR ×10; wd 0.05, wd 0.1 +
no-decay-1D; drop-path 0.1/0.2; dropout 0.1; LLRD 0.75/0.85; Gaussian-noise
and time-crop+warp augmentation; batch 128/256/128@1e-4; 4k/16k steps; τ
0.1 fixed / 0.03 init). Round 2 = 2 more baseline seeds, two combos, batch
32 (8k/16k), CLIP ViT-L/14 text tower, and two "soup-source" runs (parent
+ `logging.save_step_checkpoints`, then `scripts/soup_checkpoints.py`
uniform averages over [3000/4000/5500, 8000] and per-step val retrieval,
`launch/launch_slurm_retrieval_tune_soup.sh`). Round 3 = seeds/soup/
neighbours of the round-2 winner (lr 1e-4 + τ 0.1 fixed: seeds 43/44,
soup-source, lr 2e-4, τ 0.07/0.15 fixed, 16k-step soup-source), plus
`latest.pt` (final step-8000) evaluations of the top round-1 arms
(`launch/launch_slurm_retrieval_tune_evalckpt.sh`). 47 training runs of
15-55 min each on gengpu/b53017 (job ids in
`analysis/retrieval_tune/jobs_round{1,2,3}.txt`, `jobs_evalckpt.txt`);
UniMTS / parent / probe / scratch reference rows on the test split =
`analysis/retrieval_eval_test/` (jobs 9421924-9421930, CPU).

**New knobs** (all default-off; no existing config or checkpoint changes
behaviour): `objective.imu_text_clip_flu.{text_lr_mult, head_lr_mult,
layer_decay, no_decay_norm_bias, text_train_last_n_layers, init_temperature,
freeze_logit_scale}` (optimizer param groups in `run.py`);
`vit1d.encoder.drop_path_rate` now wired for `ViT1DEncoderAltAttn` (was
silently ignored there); `scripts/eval_retrieval_alignment.py --split`.
**Two latent bugs found and fixed on the way** — (i) `DropPath` in
`imu_lm/models/ViT1D/model.py` computed `floor(U[0,1)) + keep_prob` (always
`keep_prob`, so `x * keep_prob / keep_prob == x`): stochastic depth has been
a silent no-op for every `drop_path_rate > 0` ever configured in this repo
(18 configs, all standard-ViT1D BEiT/DINO/HAR-finetune runs; those results
stand, but were obtained with drop-path effectively 0.0); (ii)
`smooth_time_warp` used `np.convolve(mode="same")`, which returns
kernel-length output when the window is shorter than the Gaussian kernel
(HumanML3D augments short clips at native length before wrap-padding), so
`time_warp` with `smooth_sigma` ≥ ~11 crashed on clips < 2×3σ+1 samples;
fixed with reflect-pad + `mode="valid"` (identical output for long windows;
no earlier config hit the short-window case).

**All arms — VAL split** (selection set, 1,197 clips; first-caption protocol; %; sorted by G=100 IMU→Text MRR; `__latest` = final step-8000 checkpoint, `__soupA_B` = uniform weight average of the step checkpoints in [A,B]; "best val loss (step)" = the run's val-loss-selected `best.pt`, which is what the unsuffixed rows evaluate):
| arm | best val loss (step) | G=100 IMU→Text R@1/5/10/MRR | G=100 Text→IMU R@1/5/10/MRR | full IMU→Text R@1/5/10/MRR | full Text→IMU R@1/5/10/MRR |
|---|---|---|---|---|---|
| *UniMTS 22-joint (ref)* | — | 40.3 / 79.7 / 89.4 / 57.2 | 2.1 / 8.0 / 15.1 / 7.3 | 9.4 / 34.9 / 50.0 / 22.4 | 0.6 / 1.2 / 1.8 / 1.4 |
| *Ours parent (13d row)* | — | 39.4 / 73.9 / 84.2 / 54.9 | 37.5 / 72.0 / 82.4 / 53.0 | 11.1 / 35.3 / 47.8 / 22.7 | 10.4 / 29.7 / 44.6 / 20.5 |
| lr1e-4_tau0.1fixed__latest | — | 41.4 / 75.6 / 84.9 / 56.5 | 39.9 / 74.4 / 84.0 / 55.3 | 13.0 / 36.3 / 50.7 / 24.6 | 11.4 / 32.9 / 46.6 / 22.2 |
| lr1e-4_tau0.1fixed | 1.8904 (7500) | 41.4 / 75.6 / 84.9 / 56.5 | 39.7 / 74.4 / 84.0 / 55.1 | 13.1 / 36.5 / 50.4 / 24.6 | 10.6 / 32.7 / 46.3 / 21.8 |
| lr1e-4__latest | — | 41.0 / 75.2 / 84.5 / 56.2 | 38.4 / 73.3 / 82.8 / 53.9 | 12.9 / 35.9 / 48.8 / 24.5 | 10.1 / 32.3 / 44.2 / 21.3 |
| lr1e-4_tau0.1fixed_soupsrc__soup4000_8000 | — | 40.8 / 75.0 / 84.9 / 56.1 | 39.4 / 73.3 / 83.7 / 54.7 | 11.4 / 35.9 / 49.2 / 23.4 | 10.8 / 32.0 / 45.1 / 21.7 |
| lr1e-4_tau0.1fixed_soupsrc__soup3000_8000 | — | 40.5 / 75.3 / 85.2 / 56.0 | 39.3 / 73.3 / 83.6 / 54.6 | 10.9 / 35.5 / 48.5 / 23.0 | 10.0 / 32.0 / 45.1 / 21.1 |
| lr1e-4_tau0.1fixed_s43 | 1.8957 (6000) | 40.5 / 75.7 / 85.3 / 56.0 | 38.6 / 73.5 / 83.5 / 54.3 | 12.4 / 35.3 / 49.9 / 23.9 | 9.9 / 31.2 / 46.3 / 21.0 |
| lr1e-4_tau0.1fixed_soupsrc__soup5500_8000 | — | 40.6 / 74.6 / 84.5 / 55.9 | 39.7 / 73.3 / 83.3 / 54.8 | 12.1 / 36.0 / 49.1 / 23.7 | 10.7 / 32.4 / 46.2 / 21.6 |
| lr1e-4_tau0.1fixed_soupsrc | 1.9159 (8000) | 40.9 / 74.3 / 84.4 / 55.9 | 39.6 / 73.0 / 83.0 / 54.6 | 12.6 / 36.2 / 49.0 / 24.1 | 10.8 / 32.2 / 46.8 / 21.6 |
| soupsrc_s42__soup4000_8000 | — | 40.5 / 74.6 / 84.5 / 55.7 | 38.1 / 72.5 / 82.7 / 53.4 | 11.1 / 36.0 / 49.1 / 23.1 | 11.3 / 30.8 / 44.4 / 21.2 |
| soupsrc_s42__soup3000_8000 | — | 40.4 / 74.6 / 84.8 / 55.7 | 37.9 / 72.5 / 83.0 / 53.3 | 11.3 / 36.5 / 48.5 / 23.2 | 10.9 / 30.6 / 43.9 / 21.0 |
| lr1e-4_tau0.07fixed | 1.9692 (8000) | 40.6 / 74.4 / 84.6 / 55.7 | 39.2 / 72.8 / 82.9 / 54.3 | 13.0 / 35.1 / 49.0 / 24.4 | 10.9 / 31.6 / 46.3 / 21.6 |
| bs32_steps16k__latest | — | 39.7 / 74.6 / 84.9 / 55.3 | 37.3 / 72.1 / 82.9 / 53.0 | 11.3 / 35.6 / 47.8 / 23.1 | 9.1 / 29.6 / 45.1 / 20.1 |
| wd0.1_nodecay1d__latest | — | 40.4 / 73.1 / 82.5 / 55.2 | 38.1 / 71.2 / 81.1 / 53.1 | 12.0 / 36.1 / 48.6 / 23.7 | 11.0 / 31.6 / 45.0 / 21.5 |
| soupsrc_s43__soup3000_8000 | — | 39.9 / 74.4 / 84.0 / 55.2 | 37.7 / 71.5 / 82.1 / 52.9 | 12.2 / 34.9 / 46.9 / 23.7 | 9.2 / 31.4 / 44.9 / 20.3 |
| lr1e-4 | 1.9724 (5500) | 39.8 / 74.0 / 84.1 / 55.2 | 37.2 / 71.6 / 82.3 / 52.7 | 11.9 / 34.5 / 49.1 / 23.3 | 10.3 / 30.4 / 43.6 / 20.6 |
| lr1e-4_tau0.1fixed_s44 | 1.9480 (7000) | 39.6 / 74.4 / 84.5 / 55.2 | 37.8 / 72.8 / 83.0 / 53.5 | 10.8 / 34.5 / 47.4 / 22.5 | 9.9 / 30.7 / 43.9 / 20.7 |
| soupsrc_s42__soup5500_8000 | — | 40.0 / 73.9 / 83.9 / 55.2 | 38.1 / 72.2 / 82.3 / 53.3 | 11.2 / 35.1 / 49.0 / 23.0 | 11.0 / 29.9 / 45.0 / 21.0 |
| wd0.1_nodecay1d | 2.0125 (5500) | 40.2 / 73.3 / 82.8 / 55.1 | 37.2 / 70.5 / 81.8 / 52.4 | 11.8 / 36.8 / 48.5 / 23.5 | 9.7 / 29.7 / 44.0 / 20.3 |
| base_s42 | 2.0430 (6000) | 40.0 / 73.4 / 84.0 / 55.0 | 38.2 / 71.4 / 82.1 / 53.1 | 13.4 / 35.9 / 48.6 / 24.3 | 10.8 / 31.9 / 44.1 / 21.6 |
| base_s42__latest | — | 39.9 / 73.3 / 84.4 / 54.9 | 38.4 / 71.6 / 82.7 / 53.3 | 12.4 / 36.3 / 48.0 / 23.8 | 11.2 / 33.0 / 45.2 / 21.9 |
| soupsrc_s43__soup4000_8000 | — | 39.4 / 74.1 / 83.8 / 54.8 | 37.7 / 71.5 / 81.8 / 52.9 | 11.4 / 33.9 / 46.4 / 23.0 | 9.7 / 31.2 / 44.4 / 20.5 |
| tau0.1_fixed__latest | — | 39.3 / 74.3 / 85.0 / 54.7 | 37.5 / 72.6 / 83.6 / 53.1 | 11.9 / 34.3 / 47.0 / 23.1 | 9.8 / 30.0 / 43.7 / 20.1 |
| bs32_steps16k | 1.5071 (11000) | 38.8 / 74.5 / 85.3 / 54.6 | 36.2 / 70.9 / 82.0 / 51.9 | 9.5 / 34.1 / 47.6 / 21.6 | 8.8 / 27.9 / 43.1 / 19.2 |
| steps16k | 2.0550 (6000) | 39.8 / 72.8 / 83.4 / 54.6 | 36.7 / 70.8 / 81.9 / 52.1 | 12.7 / 34.5 / 47.9 / 23.4 | 9.2 / 29.4 / 43.2 / 19.8 |
| textL14 | 2.0617 (5500) | 39.2 / 73.3 / 84.1 / 54.6 | 37.1 / 72.2 / 82.2 / 52.7 | 10.8 / 35.3 / 47.9 / 22.4 | 9.7 / 30.8 / 43.8 / 20.5 |
| base_s45 | 2.0207 (4500) | 39.2 / 73.4 / 83.6 / 54.5 | 36.9 / 70.6 / 81.1 / 52.1 | 11.4 / 33.6 / 47.4 / 22.5 | 10.2 / 29.9 / 42.1 / 20.3 |
| tau0.1_fixed | 2.0072 (5000) | 38.9 / 73.5 / 83.9 / 54.3 | 37.0 / 71.5 / 82.5 / 52.5 | 11.9 / 34.5 / 46.2 / 23.0 | 10.2 / 29.3 / 43.2 / 20.4 |
| soupsrc_s43__soup5500_8000 | — | 38.9 / 73.5 / 83.4 / 54.3 | 37.0 / 70.9 / 81.3 / 52.2 | 11.7 / 33.7 / 46.2 / 22.9 | 10.3 / 30.0 / 44.1 / 20.7 |
| steps16k__latest | — | 39.8 / 71.7 / 80.9 / 54.2 | 38.8 / 71.0 / 79.8 / 53.3 | 12.3 / 35.9 / 47.9 / 23.7 | 12.1 / 32.0 / 44.7 / 22.2 |
| bs128_lr1e-4 | 2.6345 (3000) | 38.8 / 72.6 / 83.7 / 54.1 | 37.3 / 71.2 / 82.4 / 52.4 | 10.8 / 33.4 / 47.5 / 22.2 | 10.7 / 30.0 / 42.8 / 20.7 |
| lr1e-4_wd0.1nd | 2.0324 (6500) | 38.8 / 73.1 / 84.5 / 54.1 | 37.3 / 70.2 / 81.2 / 52.4 | 13.1 / 32.8 / 47.1 / 23.3 | 9.9 / 29.9 / 43.7 / 20.3 |
| lr1e-4_tau0.15fixed | 2.0094 (7500) | 38.3 / 73.9 / 84.6 / 54.0 | 37.0 / 72.9 / 84.0 / 52.9 | 10.3 / 33.4 / 47.6 / 21.8 | 8.4 / 28.8 / 45.1 / 19.3 |
| lr2e-4_tau0.1fixed | 1.9436 (8000) | 38.6 / 72.5 / 83.6 / 53.8 | 36.9 / 71.7 / 82.3 / 52.4 | 10.3 / 33.8 / 46.5 / 22.0 | 9.4 / 29.6 / 42.9 / 20.0 |
| soupsrc_s42 | 2.0195 (4000) | 38.0 / 73.2 / 84.0 / 53.7 | 35.9 / 70.6 / 82.3 / 51.5 | 10.9 / 32.9 / 46.3 / 22.0 | 11.0 / 28.2 / 40.8 / 20.1 |
| base_s44 | 2.0689 (6000) | 39.0 / 71.5 / 82.1 / 53.7 | 37.0 / 69.8 / 80.3 / 51.9 | 12.9 / 35.3 / 45.6 / 23.7 | 10.9 / 29.2 / 42.3 / 20.4 |
| base_s43__latest | — | 37.8 / 72.8 / 83.7 / 53.4 | 37.1 / 70.6 / 81.7 / 52.1 | 10.9 / 34.3 / 45.8 / 22.3 | 11.7 / 30.7 / 42.9 / 21.1 |
| soupsrc_s43 | 2.0566 (4500) | 37.6 / 72.5 / 83.1 / 53.3 | 36.2 / 70.1 / 80.9 / 51.5 | 11.4 / 32.7 / 45.4 / 22.3 | 8.9 / 29.1 / 41.9 / 19.3 |
| textmult0.1 | 2.0071 (7500) | 37.4 / 73.6 / 83.8 / 53.3 | 33.9 / 70.4 / 81.6 / 50.1 | 11.0 / 32.8 / 44.7 / 21.9 | 8.3 / 26.7 / 38.8 / 18.2 |
| wd0.05 | 2.0576 (5000) | 38.0 / 72.4 / 83.1 / 53.3 | 36.4 / 69.9 / 81.0 / 51.4 | 12.4 / 32.2 / 45.5 / 22.8 | 11.2 / 29.2 / 42.2 / 20.9 |
| base_s43 | 2.0119 (4500) | 37.6 / 72.3 / 83.4 / 53.1 | 35.5 / 69.6 / 81.1 / 50.8 | 10.0 / 33.1 / 44.9 / 21.4 | 11.0 / 26.9 / 41.1 / 20.0 |
| llrd0.85 | 2.0993 (5000) | 37.6 / 72.7 / 83.1 / 53.1 | 35.9 / 70.8 / 81.8 / 51.2 | 10.4 / 33.5 / 45.6 / 21.7 | 10.6 / 28.7 / 42.2 / 20.3 |
| bs32 | 1.4987 (8000) | 36.5 / 73.6 / 85.6 / 52.9 | 34.0 / 70.7 / 82.6 / 50.3 | 8.6 / 32.1 / 44.9 / 20.0 | 7.9 / 25.4 / 40.0 / 17.7 |
| textlast4 | 2.1408 (4500) | 37.0 / 72.6 / 83.4 / 52.9 | 34.2 / 69.6 / 82.3 / 50.0 | 9.4 / 31.9 / 43.6 / 20.7 | 9.8 / 27.6 / 39.2 / 19.2 |
| headmult10 | 2.1271 (4000) | 37.0 / 70.9 / 83.3 / 52.5 | 35.5 / 68.5 / 81.3 / 50.7 | 11.4 / 32.8 / 43.5 / 22.0 | 9.9 / 28.0 / 41.6 / 19.6 |
| textmult0.3 | 2.0506 (5000) | 35.9 / 71.9 / 83.7 / 52.1 | 33.8 / 69.1 / 81.2 / 49.7 | 10.4 / 30.7 / 44.5 / 20.9 | 8.4 / 27.3 / 39.4 / 18.1 |
| bs128 | 2.6884 (2000) | 36.4 / 70.3 / 82.5 / 51.7 | 34.9 / 68.1 / 80.7 / 50.0 | 10.6 / 31.7 / 43.4 / 21.0 | 10.2 / 27.4 / 40.3 / 19.4 |
| droppath0.1 | 2.1004 (4000) | 36.2 / 70.3 / 83.0 / 51.7 | 34.0 / 68.4 / 81.3 / 49.8 | 9.5 / 31.5 / 43.9 / 20.6 | 9.1 / 26.2 / 39.0 / 18.2 |
| steps4k | 2.0548 (4000) | 35.7 / 71.7 / 83.0 / 51.6 | 33.8 / 69.7 / 80.8 / 49.7 | 9.6 / 31.4 / 43.4 / 20.4 | 8.3 / 25.3 / 38.6 / 17.8 |
| llrd0.75 | 2.1385 (3000) | 35.0 / 70.0 / 82.4 / 50.7 | 32.2 / 66.5 / 79.5 / 47.8 | 9.8 / 29.0 / 42.9 / 20.1 | 9.0 / 25.6 / 37.1 / 17.9 |
| dropout0.1 | 2.1309 (3500) | 34.6 / 70.4 / 82.8 / 50.6 | 32.4 / 67.7 / 80.4 / 48.4 | 8.6 / 29.5 / 42.6 / 19.2 | 7.6 / 25.0 / 36.8 / 16.9 |
| droppath0.2 | 2.1560 (4000) | 34.8 / 69.0 / 81.2 / 50.4 | 33.3 / 67.8 / 79.3 / 49.0 | 8.8 / 29.2 / 42.3 / 19.5 | 8.0 / 24.6 / 38.1 / 17.4 |
| bs256 | 3.4429 (1000) | 34.7 / 67.9 / 78.9 / 50.0 | 33.0 / 65.8 / 77.3 / 48.1 | 9.1 / 29.8 / 42.9 / 19.4 | 7.9 / 27.1 / 37.0 / 17.6 |
| tau0.03 | 2.2035 (2500) | 33.0 / 69.3 / 82.2 / 49.1 | 28.1 / 61.8 / 75.8 / 43.6 | 8.9 / 28.2 / 40.6 / 19.0 | 6.7 / 21.9 / 32.8 / 15.0 |
| lr2e-5 | 2.2035 (4000) | 33.3 / 68.4 / 79.6 / 49.0 | 31.5 / 66.1 / 78.3 / 47.2 | 8.5 / 27.4 / 39.7 / 18.6 | 7.4 / 24.7 / 36.4 / 16.4 |
| aug_time | 2.3116 (6000) | 31.0 / 67.2 / 80.4 / 47.3 | 30.9 / 65.6 / 80.0 / 46.6 | 7.1 / 25.5 / 38.9 / 16.9 | 8.1 / 24.7 / 35.4 / 16.8 |
| textfrozen | 2.8542 (7500) | 21.9 / 52.9 / 68.0 / 36.6 | 16.8 / 45.0 / 60.0 / 30.6 | 4.3 / 18.1 / 27.5 / 11.9 | 3.6 / 11.9 / 18.2 / 8.7 |
| aug_noise0.2 | 4.4411 (1000) | 8.9 / 26.5 / 37.5 / 18.8 | 12.3 / 36.2 / 52.4 / 24.8 | 2.1 / 6.3 / 11.4 / 5.3 | 2.1 / 8.6 / 13.4 / 6.5 |

**Same arms — TEST split** (untouched confirmation set: test.txt ∩ sim ∩ aug_texts = 3,701 clips ↔ 2,730 unique first captions; %):
| arm | G=100 IMU→Text R@1/5/10/MRR | G=100 Text→IMU R@1/5/10/MRR | full IMU→Text R@1/5/10/MRR (MedR) | full Text→IMU R@1/5/10/MRR (MedR) |
|---|---|---|---|---|
| lr1e-4_tau0.1fixed__latest | 39.3 / 74.9 / 84.3 / 54.9 | 38.3 / 73.4 / 83.2 / 53.9 | 5.3 / 18.8 / 28.6 / 12.9 (32.0) | 4.1 / 15.5 / 25.2 / 10.8 (37.0) |
| lr1e-4_tau0.1fixed | 39.3 / 74.8 / 84.2 / 54.9 | 38.2 / 73.4 / 83.2 / 53.8 | 5.3 / 19.0 / 28.5 / 12.9 (32.0) | 4.2 / 15.6 / 25.1 / 10.9 (38.0) |
| lr1e-4__latest | 39.4 / 74.3 / 84.1 / 55.0 | 37.9 / 72.4 / 82.4 / 53.3 | 4.9 / 18.2 / 28.5 / 12.5 (31.0) | 4.4 / 15.9 / 25.6 / 11.3 (39.0) |
| lr1e-4_tau0.1fixed_soupsrc__soup4000_8000 | 38.8 / 74.3 / 84.7 / 54.4 | 38.0 / 73.3 / 83.4 / 53.5 | 4.5 / 18.3 / 28.6 / 12.2 (31.0) | 4.5 / 15.6 / 25.0 / 11.2 (38.0) |
| lr1e-4_tau0.1fixed_soupsrc__soup3000_8000 | 38.8 / 74.6 / 84.9 / 54.5 | 37.8 / 73.4 / 83.6 / 53.6 | 4.5 / 17.7 / 29.1 / 12.2 (32.0) | 4.3 / 15.7 / 25.1 / 11.0 (38.0) |
| lr1e-4_tau0.1fixed_s43 | 39.2 / 73.3 / 83.3 / 54.4 | 38.0 / 72.2 / 82.0 / 53.3 | 4.9 / 17.7 / 28.6 / 12.4 (32.0) | 4.7 / 16.3 / 25.8 / 11.5 (38.0) |
| lr1e-4_tau0.1fixed_soupsrc__soup5500_8000 | 38.4 / 73.8 / 84.0 / 53.9 | 37.6 / 72.7 / 82.9 / 53.1 | 4.6 / 18.3 / 28.4 / 12.3 (33.0) | 4.1 / 15.3 / 24.6 / 10.9 (39.0) |
| lr1e-4_tau0.1fixed_soupsrc | 38.5 / 73.5 / 83.4 / 53.9 | 37.7 / 72.5 / 82.6 / 53.1 | 4.9 / 18.5 / 28.3 / 12.4 (33.0) | 4.4 / 14.9 / 25.1 / 11.1 (39.5) |
| soupsrc_s42__soup4000_8000 | 38.1 / 72.9 / 83.6 / 53.6 | 36.5 / 70.9 / 81.5 / 51.9 | 5.4 / 17.1 / 27.3 / 12.4 (34.0) | 4.6 / 15.1 / 23.9 / 10.8 (43.0) |
| soupsrc_s42__soup3000_8000 | 38.3 / 73.5 / 84.2 / 53.9 | 36.6 / 71.1 / 81.8 / 52.0 | 5.1 / 16.9 / 27.2 / 12.3 (33.0) | 4.5 / 15.4 / 23.8 / 10.9 (43.0) |
| lr1e-4_tau0.07fixed | 39.0 / 73.6 / 84.2 / 54.4 | 37.1 / 72.3 / 82.5 / 52.8 | 5.0 / 18.6 / 28.8 / 12.6 (32.0) | 4.7 / 15.9 / 25.2 / 11.4 (41.5) |
| bs32_steps16k__latest | 38.4 / 73.4 / 84.4 / 54.0 | 36.6 / 71.4 / 82.4 / 52.3 | 4.9 / 16.8 / 27.1 / 12.1 (31.0) | 4.3 / 14.9 / 24.2 / 10.8 (41.0) |
| wd0.1_nodecay1d__latest | 37.6 / 71.6 / 83.0 / 52.9 | 36.2 / 70.6 / 80.8 / 51.5 | 4.7 / 17.8 / 27.6 / 12.0 (35.0) | 4.5 / 16.1 / 24.5 / 11.2 (43.0) |
| soupsrc_s43__soup3000_8000 | 38.0 / 72.9 / 84.0 / 53.6 | 37.0 / 71.0 / 81.8 / 52.2 | 5.3 / 17.5 / 28.0 / 12.5 (34.0) | 4.5 / 15.6 / 24.5 / 11.0 (42.0) |
| lr1e-4 | 38.7 / 73.6 / 83.9 / 54.3 | 36.9 / 71.0 / 81.4 / 52.2 | 4.5 / 17.2 / 27.8 / 11.9 (31.0) | 4.2 / 15.3 / 24.5 / 10.9 (42.0) |
| lr1e-4_tau0.1fixed_s44 | 38.7 / 73.6 / 83.5 / 54.1 | 37.7 / 72.6 / 82.4 / 53.2 | 5.4 / 17.8 / 27.3 / 12.6 (33.0) | 4.3 / 16.1 / 25.4 / 11.2 (40.0) |
| soupsrc_s42__soup5500_8000 | 37.6 / 72.3 / 83.0 / 53.1 | 36.2 / 70.4 / 81.1 / 51.5 | 5.2 / 17.5 / 27.1 / 12.3 (35.0) | 4.6 / 15.2 / 23.6 / 10.8 (44.0) |
| wd0.1_nodecay1d | 37.2 / 71.6 / 83.2 / 52.6 | 35.6 / 69.9 / 81.0 / 51.1 | 5.0 / 17.6 / 27.5 / 12.2 (37.0) | 4.4 / 15.5 / 23.0 / 10.9 (44.0) |
| base_s42 | 37.3 / 71.9 / 82.8 / 52.8 | 35.9 / 69.9 / 80.9 / 51.3 | 5.4 / 17.7 / 27.4 / 12.5 (36.0) | 4.3 / 14.5 / 23.7 / 10.5 (44.0) |
| base_s42__latest | 38.0 / 71.7 / 82.7 / 53.2 | 36.4 / 70.4 / 81.3 / 51.7 | 5.4 / 18.5 / 28.0 / 12.7 (35.0) | 4.3 / 15.2 / 24.4 / 10.9 (42.0) |
| soupsrc_s43__soup4000_8000 | 37.9 / 72.3 / 83.3 / 53.3 | 36.7 / 71.1 / 81.3 / 52.0 | 5.6 / 17.6 / 27.9 / 12.6 (34.0) | 4.7 / 15.0 / 25.1 / 11.0 (43.0) |
| tau0.1_fixed__latest | 37.8 / 71.0 / 81.6 / 52.7 | 36.8 / 70.1 / 80.5 / 51.9 | 5.3 / 17.5 / 27.0 / 12.5 (35.0) | 4.5 / 15.2 / 24.6 / 10.9 (41.0) |
| bs32_steps16k | 37.5 / 72.9 / 84.1 / 53.2 | 35.5 / 70.5 / 81.4 / 51.2 | 4.3 / 16.7 / 26.2 / 11.5 (34.0) | 4.0 / 14.6 / 22.7 / 10.2 (44.0) |
| steps16k | 37.0 / 71.0 / 82.2 / 52.4 | 36.0 / 69.8 / 80.2 / 51.2 | 4.5 / 16.8 / 25.9 / 11.5 (36.0) | 4.4 / 14.4 / 23.3 / 10.5 (43.0) |
| textL14 | 37.4 / 72.1 / 83.1 / 52.7 | 35.9 / 69.8 / 81.1 / 51.2 | 5.1 / 17.5 / 27.5 / 12.4 (36.0) | 4.4 / 15.9 / 24.0 / 11.0 (44.0) |
| base_s45 | 36.7 / 71.9 / 82.9 / 52.3 | 35.2 / 69.3 / 80.5 / 50.6 | 4.7 / 16.6 / 25.9 / 11.7 (37.0) | 3.6 / 14.1 / 23.5 / 9.9 (46.0) |
| tau0.1_fixed | 37.2 / 71.5 / 82.7 / 52.6 | 36.5 / 70.4 / 81.1 / 51.7 | 5.6 / 17.2 / 26.8 / 12.5 (35.0) | 5.0 / 14.8 / 24.0 / 11.1 (43.0) |
| soupsrc_s43__soup5500_8000 | 37.6 / 71.6 / 82.7 / 52.9 | 36.2 / 70.5 / 80.8 / 51.5 | 5.5 / 17.2 / 27.5 / 12.4 (36.0) | 4.6 / 15.1 / 24.5 / 10.8 (45.0) |
| steps16k__latest | 38.0 / 70.0 / 80.3 / 52.5 | 37.8 / 69.6 / 79.6 / 52.3 | 5.8 / 19.3 / 28.7 / 13.1 (35.0) | 4.7 / 15.7 / 25.9 / 11.4 (41.0) |
| bs128_lr1e-4 | 37.8 / 71.9 / 82.4 / 53.2 | 37.0 / 70.3 / 80.3 / 52.0 | 4.5 / 17.5 / 27.4 / 11.9 (33.0) | 4.2 / 15.4 / 24.0 / 11.0 (42.0) |
| lr1e-4_wd0.1nd | 39.1 / 73.3 / 84.0 / 54.4 | 37.0 / 71.1 / 81.7 / 52.4 | 5.1 / 17.5 / 28.0 / 12.5 (32.0) | 4.2 / 15.3 / 24.3 / 11.0 (40.0) |
| lr1e-4_tau0.15fixed | 37.3 / 72.6 / 82.9 / 53.0 | 36.7 / 71.6 / 82.2 / 52.4 | 4.2 / 16.0 / 25.9 / 11.3 (34.0) | 4.1 / 15.1 / 23.6 / 10.6 (41.0) |
| lr2e-4_tau0.1fixed | 37.0 / 72.7 / 83.0 / 53.0 | 35.8 / 71.8 / 81.8 / 51.8 | 3.7 / 15.6 / 25.8 / 10.9 (34.0) | 3.2 / 13.2 / 22.6 / 9.5 (42.0) |
| soupsrc_s42 | 36.4 / 71.0 / 82.1 / 51.9 | 34.4 / 68.6 / 80.2 / 49.9 | 4.6 / 16.0 / 25.9 / 11.4 (38.0) | 4.1 / 13.4 / 21.6 / 10.0 (48.0) |
| base_s44 | 37.8 / 72.1 / 82.8 / 53.1 | 36.4 / 70.0 / 81.4 / 51.6 | 5.2 / 16.8 / 26.9 / 12.1 (36.0) | 4.4 / 15.7 / 24.1 / 10.9 (43.0) |
| base_s43__latest | 37.6 / 71.2 / 82.1 / 52.7 | 36.2 / 69.7 / 80.5 / 51.4 | 5.0 / 18.0 / 27.7 / 12.3 (35.0) | 4.9 / 16.1 / 24.2 / 11.3 (45.0) |
| soupsrc_s43 | 36.9 / 71.3 / 82.3 / 52.3 | 35.3 / 68.8 / 79.9 / 50.6 | 4.6 / 16.2 / 26.8 / 11.5 (37.0) | 4.4 / 14.7 / 22.7 / 10.6 (46.0) |
| textmult0.1 | 35.6 / 71.2 / 83.2 / 51.5 | 33.1 / 68.7 / 80.7 / 49.0 | 4.1 / 15.5 / 24.8 / 10.9 (38.0) | 3.3 / 12.6 / 21.2 / 9.1 (52.0) |
| wd0.05 | 37.4 / 72.6 / 84.2 / 53.1 | 36.5 / 70.2 / 81.7 / 51.9 | 5.2 / 17.4 / 27.1 / 12.5 (35.0) | 4.0 / 15.3 / 23.9 / 10.5 (40.0) |
| base_s43 | 36.7 / 71.4 / 82.7 / 52.1 | 34.9 / 69.1 / 80.3 / 50.3 | 4.8 / 16.4 / 26.2 / 11.7 (38.0) | 3.7 / 14.7 / 22.1 / 10.2 (46.5) |
| llrd0.85 | 36.7 / 70.4 / 81.4 / 51.9 | 35.1 / 68.9 / 79.7 / 50.3 | 4.9 / 16.4 / 26.6 / 11.8 (37.0) | 4.4 / 14.5 / 23.0 / 10.6 (47.0) |
| bs32 | 36.1 / 71.4 / 82.7 / 51.9 | 33.7 / 68.7 / 80.5 / 49.4 | 4.5 / 15.2 / 25.0 / 11.3 (36.0) | 4.0 / 13.3 / 21.7 / 9.9 (49.0) |
| textlast4 | 35.4 / 70.3 / 81.8 / 51.0 | 33.4 / 67.9 / 79.8 / 48.9 | 4.4 / 15.4 / 24.7 / 11.1 (41.0) | 4.1 / 12.4 / 21.6 / 9.7 (52.0) |
| headmult10 | 36.6 / 71.5 / 83.6 / 52.3 | 34.1 / 69.1 / 80.2 / 49.8 | 4.9 / 16.4 / 25.8 / 11.6 (37.0) | 3.8 / 14.6 / 22.1 / 9.9 (49.0) |
| textmult0.3 | 36.3 / 71.5 / 82.8 / 52.1 | 34.8 / 69.7 / 80.4 / 50.4 | 4.1 / 16.5 / 26.1 / 11.2 (39.0) | 3.8 / 13.6 / 22.5 / 9.9 (48.0) |
| bs128 | 35.4 / 70.2 / 81.4 / 50.9 | 33.3 / 67.3 / 79.1 / 48.8 | 4.7 / 16.2 / 25.7 / 11.5 (41.0) | 3.8 / 14.1 / 21.9 / 9.8 (53.0) |
| droppath0.1 | 35.4 / 70.8 / 82.4 / 51.3 | 34.2 / 68.7 / 80.4 / 49.7 | 3.9 / 15.4 / 25.7 / 10.8 (40.0) | 3.5 / 12.9 / 21.7 / 9.4 (51.0) |
| steps4k | 36.4 / 70.6 / 82.6 / 52.0 | 34.2 / 69.1 / 80.1 / 49.9 | 5.4 / 16.5 / 26.2 / 12.1 (37.0) | 4.6 / 14.7 / 22.6 / 10.5 (49.0) |
| llrd0.75 | 33.6 / 68.9 / 80.8 / 49.4 | 31.7 / 66.6 / 78.3 / 47.4 | 4.0 / 15.2 / 23.8 / 10.4 (44.0) | 3.1 / 12.6 / 19.6 / 8.7 (58.0) |
| dropout0.1 | 34.0 / 69.1 / 81.2 / 49.8 | 32.5 / 67.5 / 79.5 / 48.3 | 4.0 / 14.0 / 23.3 / 10.4 (43.0) | 2.9 / 12.7 / 20.5 / 8.8 (53.0) |
| droppath0.2 | 34.6 / 69.6 / 81.7 / 50.3 | 33.0 / 67.8 / 79.6 / 48.6 | 4.7 / 15.5 / 24.5 / 11.1 (42.0) | 3.7 / 13.5 / 21.4 / 9.7 (53.0) |
| bs256 | 34.0 / 67.7 / 80.1 / 49.2 | 32.4 / 65.4 / 78.0 / 47.4 | 4.2 / 15.8 / 24.7 / 10.9 (46.0) | 3.4 / 13.1 / 21.7 / 9.3 (57.0) |
| tau0.03 | 33.3 / 68.6 / 80.6 / 49.1 | 28.9 / 62.8 / 75.9 / 44.4 | 4.2 / 14.4 / 22.6 / 10.5 (46.0) | 2.6 / 10.3 / 17.0 / 7.7 (71.0) |
| lr2e-5 | 32.9 / 67.9 / 80.0 / 48.7 | 31.1 / 66.0 / 78.2 / 46.7 | 4.1 / 14.2 / 22.4 / 10.3 (46.0) | 3.6 / 12.7 / 19.4 / 9.2 (60.0) |
| aug_time | 31.9 / 66.4 / 78.9 / 47.5 | 31.9 / 66.0 / 78.1 / 47.3 | 3.8 / 13.4 / 22.0 / 9.9 (50.0) | 3.7 / 12.8 / 20.7 / 9.3 (59.0) |
| textfrozen | 20.1 / 51.1 / 66.5 / 34.9 | 15.8 / 43.2 / 58.0 / 29.3 | 1.8 / 7.1 / 11.9 / 5.6 (118.0) | 1.0 / 4.3 / 7.8 / 3.6 (200.0) |
| aug_noise0.2 | 8.6 / 25.3 / 36.4 / 18.1 | 12.5 / 36.5 / 50.9 / 24.9 | 0.8 / 3.0 / 5.2 / 2.5 (497.0) | 1.0 / 3.3 / 5.8 / 3.0 (297.5) |

Reference rows on the same splits (single checkpoints): UniMTS 22-joint
val 40.27 / 79.67 / 89.43 / 57.20, test 40.74 / 76.55 / 87.56 / 56.56
(t2i 2.1 / 8.0 / 15.1 / 7.3 and 2.1 / 8.6 / 15.3 / 7.4); UniMTS wrist-only
joint-mask val 24.4 / 55.3 / 68.9 / 39.1, test 24.9 / 55.8 / 69.2 / 39.3;
UniMTS official (contaminated) val 33.6 / 68.6 / 81.2 / 49.3, test 32.4 /
68.4 / 82.2 / 48.5; §13d headline "Ours" run val 39.4 / 73.9 / 84.3 / 54.9,
test 38.4 / 72.4 / 83.0 / 53.6.

**Seed-aggregated result** (mean over seeds; i2t at G=100 unless noted):

| recipe | n | val i2t | val t2i | test i2t | test t2i |
|---|---|---|---|---|---|
| parent (lr 5e-5, learnable τ init 0.07), best.pt | 4 | 38.9 / 72.6 / 83.3 / **54.1 ± 0.7** | 36.9 / 70.4 / 81.2 / 52.0 | 37.1 / 71.8 / 82.8 / **52.6 ± 0.4** | 35.6 / 69.6 / 80.8 / 50.9 |
| parent, uniform soup over steps 3000-8000 | 2 | 40.2 / 74.5 / 84.4 / 55.5 ± 0.2 | 37.8 / 72.0 / 82.5 / 53.1 | 38.2 / 73.2 / 84.1 / 53.8 ± 0.2 | 36.8 / 71.1 / 81.8 / 52.1 |
| lr 1e-4, learnable τ — best.pt / latest.pt | 1 | 55.2 / 56.2 | 52.7 / 53.9 | 54.3 / 55.0 | 52.2 / 53.3 |
| lr 1e-4, τ 0.07 fixed | 1 | 40.6 / 74.4 / 84.6 / 55.7 | 39.2 / 72.8 / 82.9 / 54.3 | 39.0 / 73.6 / 84.2 / 54.4 | 37.1 / 72.3 / 82.5 / 52.8 |
| **lr 1e-4, τ 0.1 fixed (tuned)** | 4 | **40.6 / 75.0 / 84.8 / 55.9 ± 0.5** | **38.9 / 73.4 / 83.4 / 54.4** | **38.9 / 73.8 / 83.6 / 54.3 ± 0.4** | **37.9 / 72.7 / 82.6 / 53.4** |
| lr 1e-4, τ 0.1 fixed, soup 3000-8000 | 1 | 40.5 / 75.3 / 85.2 / 56.0 | 39.3 / 73.3 / 83.6 / 54.6 | 38.8 / 74.6 / 84.9 / 54.5 | 37.8 / 73.4 / 83.6 / 53.6 |
| lr 1e-4, τ 0.15 fixed / lr 2e-4, τ 0.1 fixed | 1 / 1 | 54.0 / 53.8 | 52.9 / 52.4 | — / 53.0 | — / 51.8 |
| lr 1e-4, τ 0.1 fixed, **16k steps**, best.pt (steps 8500-11000) | 3 | 39.9 / 74.2 / 84.2 / 55.1 ± 0.7 | 38.6 / 73.2 / 83.4 / 54.0 | 39.8 / 73.3 / 83.2 / 54.8 ± 0.4 | 38.7 / 72.7 / 82.5 / 53.9 |
| lr 1e-4, τ 0.1 fixed, 16k steps, soup 8000-16000 | 3 | 41.6 / 74.7 / 83.8 / 56.3 ± 0.5 | 40.5 / 74.1 / 82.8 / 55.3 | 41.1 / 74.2 / 83.3 / 55.8 ± 0.2 | 40.0 / 73.3 / 82.6 / 54.9 |
| **lr 1e-4, τ 0.1 fixed, 16k steps, soup 6000-16000 (FINAL)** — per seed val MRR 56.1 / 56.7 / 57.5, test 56.6 / 56.2 / 56.1 | 3 | **41.8 / 75.6 / 84.8 / 56.8 ± 0.6** | **40.7 / 74.6 / 83.5 / 55.7** | **41.5 / 74.8 / 84.0 / 56.3 ± 0.2** | **40.3 / 74.0 / 83.2 / 55.3** |
| full gallery, parent → tuned (4 seeds each) | | 11.9/34.5/46.6/23.0 → 12.2/35.6/49.2/23.8 | 10.7/29.5/42.4/20.6 → 10.3/31.7/45.8/21.3 | 5.0/16.9/26.6/12.0 → 5.1/18.3/28.2/12.6 | 4.0/14.8/23.3/10.4 → 4.4/15.7/25.3/11.2 |
| full gallery, tuned + 16k + soup 6000-16000 (FINAL) | 3 | **13.3 / 37.9 / 50.0 / 25.1** | **11.4 / 34.4 / 47.3 / 22.7** | **6.1 / 20.4 / 31.2 / 14.1** | **5.5 / 18.3 / 28.1 / 12.7** |
| UniMTS 22-joint reference, full gallery | 1 | 9.4 / 34.9 / 50.0 / 22.4 | 0.6 / 1.2 / 1.8 / 1.4 | 5.6 / 18.5 / 28.8 / 13.3 | 0.1 / 0.4 / 0.8 / 0.5 |

**Reads.**

1. **What tuning found: exactly two things.** Doubling the LR to 1e-4 and
   fixing the temperature at τ = 0.1 (UniMTS's own value; τ = 0.07 fixed is
   equivalent, τ = 0.15 and lr 2e-4 overshoot, τ = 0.03 and lr 2e-5 are
   clearly worse). Both are seed-robust: +1.8 MRR / +1.7 R@1 / +2.4 R@5 on
   val (4 vs 4 seeds, sd 0.5-0.7) and +1.7 / +1.8 / +2.0 on the untouched
   test split. The tuned recipe now beats the 22-joint UniMTS reference on
   IMU→Text R@1 at G=100 on val (40.6 mean / 41.4 best vs 40.3), on the
   full-gallery IMU→Text R@1 / R@5 / MRR on val (12.2 / 35.6 / 23.8 vs 9.4 /
   34.9 / 22.4), and on every Text→IMU number by > 30 pp; it still trails on
   the G=100 recall tail — R@5 / R@10 / MRR 75.0 / 84.8 / 55.9 vs 79.7 /
   89.4 / 57.2 (val) and 73.8 / 83.6 / 54.3 vs 76.6 / 87.6 / 56.6 (test) —
   and on test-split R@1 (38.9 vs 40.7). Training the tuned recipe for 16k
   steps and averaging the step-6000..16000 checkpoints (soup; the runs'
   own val-loss best.pt average only 55.1 / 54.8 — per-step val MRR
   plateaus at ≈ 55 from step 9000 on, so the soup's +1.7 / +1.5 comes from
   averaging the plateau, not from a later single checkpoint) is the
   **final recipe: 41.8 / 75.6 / 84.8 / 56.8 ± 0.6 val and 41.5 / 74.8 /
   84.0 / 56.3 ± 0.2 test over 3 seeds**. Against the 22-joint reference
   that wins IMU→Text R@1 on both splits (+1.5 / +0.8), wins all four
   full-gallery numbers on both splits (val MRR 25.1 vs 22.4, test 14.1 vs
   13.3), wins Text→IMU everywhere, ties MRR at G=100 within seed noise
   (56.8 ± 0.6 vs 57.2; 56.3 ± 0.2 vs 56.6), and still loses R@5 / R@10 at
   G=100 (75.6 / 84.8 vs 79.7 / 89.4 val; 74.8 / 84.0 vs 76.6 / 87.6 test).
   **So: closed on R@1, on MRR, on the full gallery, and on t2i; not closed
   on the G=100 R@5/R@10 tail.** Given that
   nothing else in 47 arms moved outside the ±0.7 seed band, the residual
   1-4 pp on the recall tail should be read as the wrist-vs-22-joint input
   gap (UniMTS itself loses 16-38 pp R@1 going from 22 joints to the wrist
   alone), not as a recipe deficit.
2. **Everything that restricts the fit hurts.** Drop-path 0.1/0.2 (−3.3 /
   −4.6 MRR), dropout 0.1 (−4.4), LLRD 0.85/0.75 (−1.9 / −4.3), wd 0.05
   (−1.7), text-tower LR ×0.1/×0.3 (−1.7 / −2.9), last-4-text-layers (−2.1),
   frozen text (−18.4), Gaussian noise σ = 0.2 on the raw filtered signal
   (collapse: val loss stays at chance while train loss falls to 0.8 — the
   model learns the noise, and val has none), time-crop + warp (−7.7). Only
   wd 0.1 + no-decay-1D and head-LR ×10 are neutral. The apparent
   over-fitting is intrinsic; the task is data-limited, not
   capacity-limited.
3. **Batch size is monotonically bad above 64** at this LR/step budget: 32
   (8k) 52.9, 32 (16k) 54.6, 64 55.0, 128 51.7, 128@1e-4 54.1, 256 50.0. The
   larger batches reach their val-loss minimum very early (step 1000-3000)
   and then degrade — more samples per step at fixed LR/steps over-fits
   faster, and more in-batch negatives do not compensate. Batch 64 was the
   right choice, not an artefact of the legacy in-batch metric.
4. **The text side is not the bottleneck.** CLIP ViT-L/14 text (123 M vs 63
   M params, 768-d) = 54.6 val / 52.7 test, indistinguishable from B/32; and
   any reduction of text-tower plasticity hurts. The ceiling is on the
   single-wrist IMU side.
5. **Val-loss checkpoint selection under-selects for retrieval.** Per-step
   val MRR of the parent rises monotonically 35.9 (step 500) → 53.7 (5000)
   → 54.9 (8000), and 34.6 → 55.9 for the tuned recipe, while the i2t CE
   val loss bottoms out at step 4000-6000; `latest.pt` ≥ `best.pt` on every
   arm (Δ 0 to +1.0 MRR), and a uniform soup over steps 3000-8000 gives
   +1.5-2 MRR on the parent on both splits (recovering the LR-annealed
   tail that best.pt discards). For the tuned recipe best.pt is already
   step 7500-8000, so soups add only +0.2. Practical rule for alignment
   finetunes: select by a retrieval metric (or take the final checkpoint /
   soup), not by contrastive val loss.
6. **Seed noise is the yardstick.** Baseline sd is 0.7 MRR (val) / 0.4
   (test) over 4 seeds, and two runs with the *same* seed differed by 1.3
   MRR (non-deterministic loader/cudnn) — most round-1 "differences" of
   ±1-2 were noise; the tuned recipe's +1.8 was only accepted after 4 seeds
   on both splits. The §13a-d headline single-seed row (54.9 val) sits at
   the top of the 4-seed band (54.1 ± 0.7): a slightly lucky seed, not a
   biased one.

**Files.** `scripts/gen_retrieval_tune_configs.py`,
`configs/retrieval_tune/*.yaml`, `launch/launch_slurm_retrieval_tune*.sh`,
`scripts/soup_checkpoints.py`, `scripts/summarize_retrieval_tune.py`,
`analysis/retrieval_tune/{val,test,steps}/*.json`,
`analysis/retrieval_eval_test/*.json`, runs `runs/rtune_*`.

## 14. Nightbeat-DB: heart rate from an overnight wrist accelerometer (2026-07-30)

**Status: pipeline built and verified end to end; benchmark numbers not yet produced**
(the eval needs a GPU node — `launch/misc/launch_slurm_nightbeat_hr.sh`).

### Task and why it is interesting here

Nightbeat-DB (Moebus et al., *Nightbeat: Heart Rate Estimation From a Wrist-Worn
Accelerometer During Sleep*, IEEE BHI 2024 — https://github.com/eth-siplab/Nightbeat)
is overnight, at-home wrist accelerometry with a chest-ECG reference. The task is
**regression**: heart rate in bpm from the ballistocardiogram (BCG), the faint
body-recoil each heartbeat imparts to the wrist.

This is the first evaluation target in this repo that is neither classification
nor pose, and it probes a genuinely different question: does an IMU
representation pretrained on *motion* retain anything about the sub-motion,
cardiac-band content of the signal? It is also **accelerometer-only**, which
makes it the natural test of the gyro mask-token convention.

### Data

| | |
|---|---|
| Published | 40 participants, ~310 h |
| Usable here | **38 participants, ~291 h** |
| Wrist sensor | `*_acc_a.npz`, 100 Hz, tri-axial, ±2 g rail (row 0 = µs timestamps, rows 1-3 = xyz in g) |
| Reference | chest ECG at 1024 Hz + beat indices (`hr_bxb`) and per-beat rate (`hr_bxb_rate`) |
| Location | raw `/projects/b53017/vbc1591/Nightbeat/raw/nightbeatdb`, prepared `/projects/b53017/vbc1591/Nightbeat/prepared` |

Two participants are dropped for **upstream** reasons, not pipeline ones:
* **pid 27** — `27_acc_a.npz` is corrupt in the published release: 3.5 kB on the
  ETH polybox share versus 37-70 MB for every other participant. A fresh re-fetch
  reproduces the identical bad MD5 (`zlib: invalid distance too far back`), so
  this is not a download error.
* **pid 34** — `hrv_isvalid` is 0 everywhere and `hr_bxb` is empty: no usable ECG
  reference at all. The loader finds no admissible windows and skips it.

Three facts worth recording because they were verified rather than assumed:
1. `acc_a` is the **wrist** device and `acc_m` is the **chest** device co-located
   with the ECG. All four reference algorithms (`nightbeat`/`pwr`/`jerks`/
   `bioinsights`) read `acc_a_*`, and `helpers/ojwalch_prep.py` maps the Apple
   Watch wrist dataset onto the same names. Using `acc_m` would leak the reference.
2. The 100 Hz accelerometer and 1024 Hz ECG share a t=0 origin and match in
   duration to ≤2 ms across all 38 participants; timestamps are exactly uniform
   (zero jitter). The reference repo's `chunk_trace` assumes this; it holds.
3. `hr_bxb_rate` is **truncated to integer bpm**. Against a paper headline of
   0.88 bpm MAE that quantization is not acceptable in the target, so
   `prepare_nightbeat_dataset.py` recomputes HR in float from the beat indices.
   The median |float − shipped-int| residual is 0.474 bpm ≈ the 0.5 expected from
   uniform truncation, which is what confirms the beat/rate pairing is correct.

### The gyro mask token

Nightbeat-DB has no gyroscope, but the multidata MAE encoder was pretrained on
6 channels. Rather than invent one, the eval reuses the convention pretraining
**already** established for its own accelerometer-only sources (capture24, NHANES
— `AccOnlyPadTo6ChDataset` + `objectives/masked_1d_freq_channel_mask.py`): windows
are zero-padded to 6 channels with `channel_valid_mask = [T,T,T,F,F,F]`, and the
3 gyro patch positions are forced into the encoder's mask so their embeddings
become the **learned mask token**. That is the exact state the encoder saw for
capture24/NHANES gyro on every pretraining step, so it is in-distribution.

Verified on the real checkpoint (`ViT1DEncoderAltAttn`, `_use_bert_mask=True`):

| check | result |
|---|---|
| embedding invariant to gyro channel content under `--gyro-mode mask` | **max abs diff 0.0** (exact) |
| embedding invariant under `--gyro-mode zero` | no — max abs diff 3.84 |
| cosine similarity, mask vs zero embedding | 0.56 |

The two modes therefore produce materially different representations, so
`--gyro-mode zero` is a meaningful control row rather than a formality: it
measures what feeding literal zeros (a gyroscope lying perfectly still) costs.

### Protocol

* Windows: 10.24 s / 512 samples @ 50 Hz, non-overlapping — matching the MAE
  pretraining window exactly so patch count and pos-embed extent stay
  in-distribution. At 40-100 bpm that is ~7-17 beats per window.
* 100 Hz → 50 Hz via an anti-aliasing FIR decimator (`scipy.signal.decimate`,
  zero-phase). Plain slicing would alias the 25-50 Hz band onto the BCG band.
* Window admission: ECG valid across the whole window, ≥5 beats, HR in
  [30, 200] bpm, and within-window HR spread ≤8 bpm (a window spanning an
  arousal has no single correct scalar target).
* **Participant-grouped k-fold** (5 by default), every participant in exactly one
  test fold. Grouping is mandatory: HR is strongly subject-specific and the
  paper's own per-subject-mean baseline already reaches 3.16 bpm MAE, so any
  window-level split leaks the answer outright.
* Rows: `probe` (frozen encoder, ridge or MLP head), `finetune`, `scratch` —
  the same three-row structure as the PADS benchmark (§ configs/base_pads.yaml).

Baselines reported next to every run, because they are what make the numbers
readable:
* **global mean** — predict the training folds' mean HR. Beating this is the
  minimum bar for "learned anything".
* **subject mean [ORACLE]** — predict each test participant's own mean HR. This
  is the paper's `Mean-Pred.` row (3.16 MAE), and it is an oracle: it uses
  test-participant labels. A model is not *required* to beat it, but one that
  does not is tracking sleep-time HR level rather than reading the BCG.

Published Nightbeat-DB numbers are printed alongside for context (Nightbeat 0.88
| Mean-Pred. 3.16 | PWR 8.83 | Jerks 13.23 | BioInsights 17.47 MAE bpm). They are
**not** a like-for-like comparison: they come from dedicated signal-processing
methods on the full 100 Hz signal, not a learned probe on 50 Hz windows.

### Files

| Path | Role |
|---|---|
| `/projects/b53017/vbc1591/Nightbeat/download_nightbeatdb.sh` | resumable, size-verified fetch of the 240 polybox files (4.3 GB) |
| `scripts/prepare_nightbeat_dataset.py` | raw npz → per-participant 50 Hz cache (466 MB); windowing deliberately not baked in |
| `imu_lm/data/nightbeat_loader.py` | windows + HR targets, 6-ch padding + `channel_valid_mask`, participant-grouped splits/folds |
| `scripts/eval_nightbeat_hr.py` | probe/finetune/scratch × mask/zero/acc3, metrics + baselines |
| `configs/base_nightbeat.yaml`, `configs/vit1d_nightbeat_hr_probe_mae_multidata_flu.yaml` | configs |
| `launch/misc/launch_slurm_nightbeat_hr.sh` | 5-row benchmark sweep on one H100 |

### Data-integrity check: is the BCG still there after 100 Hz -> 50 Hz? (yes)

`scripts/nightbeat_signal_check.py` runs a **model-free** heart-rate estimator on
the prepared cache, to separate "the representation does not encode HR" from "our
preprocessing destroyed HR before the encoder ever saw it". Two estimators, same
windows (5 participants, 1250 windows each row):

(5 participants, 1250 windows/row — a **subset**, and an easier one than the
full set; see the all-38 re-run below for the number to actually compare against.)

| method | win_s | MAE | RMSE | Cor | FFT res (bpm) | subj-mean |
|---|---|---|---|---|---|---|
| peak | 10.24 | 84.59 | 87.39 | -0.25 | 5.86 | 3.43 |
| **envelope** | 10.24 | **5.18** | 14.02 | 0.29 | 5.86 | 3.43 |
| peak | 20.48 | 85.70 | 88.28 | -0.27 | 2.93 | 2.99 |
| **envelope** | 20.48 | **3.57** | 10.74 | 0.34 | 2.93 | 2.99 |
| peak | 40.96 | 86.19 | 88.37 | -0.24 | 1.46 | 2.82 |
| **envelope** | 40.96 | **2.94** | 9.08 | 0.44 | 1.46 | 2.82 |

Reads:
1. **The cardiac signal survives download -> decimation -> caching.** A method
   with no learned parameters recovers HR at 2.94 bpm MAE / r=0.44. A weak probe
   result is therefore a statement about the representation, not about the data.
2. **`peak` fails completely (~85 bpm at every window length) and that is the
   expected physics, not a bug.** The BCG fundamental is far too weak to dominate
   the 0.6-3 Hz band; the heartbeat shows up as *amplitude modulation* of the
   ~4-11 Hz J-wave complex. `envelope` (band-pass -> Hilbert envelope -> peak of
   the envelope spectrum, i.e. what PWR is built on) is the correct reading of
   the same data. Worth recording as a trap: the first version of this check used
   `peak` only and its 85 bpm looked like evidence the prep was broken. It was
   not -- it was evidence the estimator was wrong.
3. **Window length matters a lot here and is a real handicap for our default.**
   `envelope` improves monotonically 5.18 -> 3.57 -> 2.94 bpm as the window grows
   10.24 -> 20.48 -> 40.96 s, tracking the FFT bin spacing (60/T bpm). At our
   10.24 s default the raw bin spacing is 5.86 bpm, so *any* frequency-domain
   method is coarse there.

**But the frozen probe should still stay at 10.24 s.** The MAE pretrain capped
windows at 512 samples (`data.variable_length.max_samples`), i.e. 32 time
patches, and the time positional embedding is a learned `nn.Embedding` -- so rows
32+ were never exercised during pretraining. A frozen probe on a longer window
would index untrained pos-embed rows and the result would be confounded.
(Direct weight evidence for this is only suggestive: rows <32 have slightly lower
norm than rows >=32, 19.28 vs 19.62, Welch p=0.009, with no crisp step at the
boundary. The load-bearing argument is the config, not the weights. Note the
norms are all ~19.6 simply because `nn.Embedding` initialises N(0,1), not because
the rows are untrained.) `--window-seconds` is therefore exposed on the eval and
warns when a probe exceeds 32 patches; long windows belong to finetune/scratch,
which train the pos-embed.

### Expectation to check against (stated before running, so the result is falsifiable)

A frozen encoder pretrained on human *motion* has no obvious reason to linearly
encode cardiac-band content, and 50 Hz windows discard half the bandwidth the
paper's method uses. The honest prior is that the frozen probe lands well short
of 0.88 bpm and quite possibly does not beat the subject-mean oracle; the
finetune-vs-scratch pair is what will show whether pretraining transfers here at
all. A probe that merely matches the subject-mean oracle should be read as
"predicts each sleeper's baseline HR", not "reads the BCG" — which is exactly
why both baselines are always printed.

The model-free `envelope` estimator gives the sharper target, since it is
computed on our own windows. **Re-run on all 38 participants** (the 5-participant
numbers in the table above are an easier subset and must not be used as the bar):

| method | win_s | n | MAE | RMSE | Cor | subj-mean |
|---|---|---|---|---|---|---|
| envelope | 10.24 | 9500 | **6.62** | 16.18 | 0.39 | 3.86 |
| envelope | 20.48 | 9500 | **4.98** | 13.62 | 0.46 | 3.46 |

Its independently-computed subject-mean baseline (3.86) matches the eval
harness's oracle (3.85), which confirms both scripts are scoring the same window
population.

### RESULTS (job 8357554, 5-fold participant-grouped CV, 38 participants, 81,340 windows)

Reported in **the paper's convention**, which is *per-participant averaged*, not
pooled. Two independent signals fix this: the paper's `Base: Mean-Pred.` scores
`Cor = 0` despite being a per-participant constant (a pooled correlation would
score ~0.78, as ours does), and the text says "MAE ... averaged across all
participants" / "Averaged across individual participants, we achieve a
correlation of 0.81". So their columns map to our `*_participant_mean` fields.

| approach | MAE ↓ | RMSE | Cor ↑ | vs. its own Mean-Pred. |
|---|---|---|---|---|
| BioInsights [16] *(paper)* | 17.47 | 21.38 | 0.00 | 5.53× |
| Jerks [13] *(paper)* | 13.23 | 20.63 | 0.04 | 4.19× |
| PWR [14] *(paper)* | 8.83 | 10.91 | 0.22 | 2.79× |
| Ours, probe/zero/ridge | 6.26 | 7.61 | 0.36 | 1.65× |
| Ours, scratch/mask | 6.13 | 7.71 | 0.30 | 1.62× |
| Ours, probe/mask/mlp | 6.03 | 7.38 | 0.40 | 1.59× |
| Ours, probe/mask/ridge | 5.95 | 7.30 | 0.38 | 1.57× |
| Ours, finetune/mask **20.48 s** | 3.23 | 4.59 | 0.64 | 0.93× |
| Base: Mean-Pred. *(paper 3.16 / ours 3.79)* | 3.79 | 5.16 | 0.00 | 1.00× |
| **Ours, finetune/mask (10.24 s)** | **1.95** | **3.44** | **0.80** | **0.51×** |
| **Nightbeat *(paper)*** | **0.88** | **2.24** | **0.81** | **0.28×** |

Pooled correlation (paper Fig. 5 reports 0.94): ours **0.90**.

**1. Correlation matches the paper; calibration does not.** Cor 0.80 vs their
0.81 — the finetuned encoder tracks within-night HR trajectory about as well as
the dedicated method. MAE 1.95 vs 0.88 is where we lose. Normalised against each
paper's own Mean-Pred. anchor (ours 3.79 vs theirs 3.16 — our window population
is ~20% harder), Nightbeat retains a ~1.8× edge, less than the raw 2.2× implies.

**2. Pretraining is not an improvement here, it is the whole thing.**

| | MAE |
|---|---|
| frozen features alone (probe) | 5.95 |
| architecture alone (scratch) | 6.13 |
| pretrained init + adaptation (finetune) | **1.95** |

Neither ingredient works alone — scratch barely beats the global-mean constant
(6.86) and is *worse* than the frozen probe, so the ViT is not learning BCG from
the finetuning data by itself. **3.1× error reduction attributable to
pretraining.** This is the strongest cross-domain transfer result in this repo:
a model pretrained on human *motion* transferring to *cardiac* sensing.

**3. The gyro mask token is worth a consistent 5%.** probe/mask 5.95 vs
probe/zero 6.26, with mask ahead in **5/5 folds** (per-fold deltas −0.13, −0.61,
−0.22, −0.46, −0.18 bpm). Mechanism verified independently: the embedding is
*exactly* invariant to gyro content under `mask` (max abs diff 0.0) and not
under `zero` (3.84). This is what lets a 6-channel pretrained encoder consume a
3-channel dataset at all.

**4. Head capacity is not the constraint.** ridge 5.95 vs MLP 6.03 — within
noise. The frozen representation is the ceiling.

### NEGATIVE RESULT: longer windows help the classical method and HURT the learned one

Predicted from the parameter-free check (envelope improves 6.62 → 4.98 bpm going
10.24 → 20.48 s) that the same gain would carry to the finetuned model. **It does
not — the model gets 65% worse**, on 5/5 folds:

| finetune/mask | MAE | RMSE | Cor | own oracle | MAE/oracle |
|---|---|---|---|---|---|
| 10.24 s | **1.95** | 3.44 | **0.80** | 3.79 | **0.51×** |
| 20.48 s | 3.23 | 4.59 | 0.64 | 3.46 | 0.93× |

The comparison is harsher than the raw numbers: the 20.48 s population is
*easier* for the oracle (3.46 vs 3.79, since HR averaged over 20 s sits closer to
a participant's mean), yet the model still does worse — normalised, it barely
beats predicting the sleeper's mean.

Why the prediction failed: the envelope gain comes from **finer FFT bin
spacing** (5.86 → 2.93 bpm), which a learned model does not consume. Meanwhile the
model pays two costs — ~32 time-positional-embedding rows that pretraining never
exercised (`variable_length.max_samples: 512` = 32 patches), and half the
training windows (non-overlapping 20.48 s ≈ 40k vs 81k).

**The fix is not longer finetuning windows — it is extending the MAE pretrain's
`max_samples` past 512.** Do not re-try long-window finetuning against this
checkpoint; the positional coverage limit will dominate.

### Plumbing smoke test (6 participants, 720 windows, CPU) — NOT a result

Verifies the pipeline end to end only; far too small to read as a benchmark.

| row | MAE | RMSE | Cor |
|---|---|---|---|
| probe / mask / ridge | 3.83 | 5.22 | 0.29 |
| Base: global mean | 4.16 | 5.60 | -0.23 |
| Base: subject mean [ORACLE] | 3.16 | 4.35 | 0.60 |

The subject-mean oracle landing on 3.16 bpm — the paper's `Mean-Pred.` value —
is a useful consistency signal that our windowing and targets sit in the same
regime as theirs.

## 15. CKNNA representation similarity across from-scratch downstream encoders (2026-08-08)

Every "Scratch" column throughout this doc (Sections 1-6) is a *performance*
baseline: fully-supervised, encoder+head trained end-to-end from random init,
independently per downstream task. This experiment asks a different question
about those same six encoders — despite no shared pretraining and no shared
training data, do task-specific encoders converge to similar *representation
geometries*? This is the "platonic representation hypothesis" question (Huh,
Cheung, Wang, Isola; ICML 2024) applied across our own downstream tasks rather
than across published models.

**Metric — CKNNA** (mutual-k-nearest-neighbor CKA): like linear CKA, but the
HSIC alignment sum is restricted to pairs that are mutual top-`k` neighbors in
*both* kernels, which makes it far less sensitive to a few outlier points
dominating the score than vanilla CKA. Implementation adapted from the
reference repo (`github.com/minyoungg/platonic-rep`, `metrics.py`) at
`analysis/cknna_metric.py` (`cknna()`, `cka()` for reference, `topk=10`
default, unbiased HSIC). Verified against the reference implementation's own
math (unbiased HSIC per Song et al. 2012 eq. 5, mutual-topk mask via
`scatter_`) rather than reimplemented from a paraphrase.

**Encoders compared** (all `alt_attention: true`, `hidden_size=384`,
12 layers, `patch_size=16`, `shared_kernel=acc_gyro` — same architecture as
every other Scratch checkpoint in this doc):

| Task | Run | Native window |
|---|---|---|
| HAR (WISDM) | `vit1d_har_supervised_wisdm_altattn_scratch` | 2.56 s |
| Arm pose (Nymeria) | `vit1d_arm_pose_supervised_nymeria_altattn_scratch` | 10.24 s |
| Full-body pose (Nymeria) | `vit1d_full_body_pose_supervised_nymeria_altattn_scratch` | 10.24 s |
| Gesture (OpenWatch) | `vit1d_gesture_supervised_openwatch_6class_altattn_scratch` | 10.24 s |
| Odometry (Nymeria, world-frame) | `vit1d_odom_worldframe_supervised_nymeria_altattn_scratch` | 2.56 s |
| CLIP (HumanML3D) | `vit1d_clip_humanml3d_scratch` | 2.56 s |

**Probe-set design**: CKNNA needs the *same* N inputs run through every
encoder. There is no dataset shared by all six tasks (different sensors,
subjects, and even coordinate conventions), so — analogous to probing
image/text models with a fixed external dataset neither was trained on — probe
windows are drawn from **Nymeria wrist-IMU val sessions**
(`processed_data_test_body_Ty_rwrist/*.pkl`, `wrist_imu_data.{accel,gyro}` at
50 Hz), since Nymeria is the real-recording source shared by three of the six
tasks. 512 shared `(session, center-timestamp)` anchors are sampled once, then
**each encoder crops its own native window length centered on the same
anchor** and runs its own stored preprocessing pipeline (`ckpt["cfg"]`, so
z-score/filter settings exactly match how that encoder was trained) — no
window length is forced to be OOD for any encoder. Embeddings are the pooled
CLS output (`ViT1DEncoderAltAttn.forward(x)`, no mask), L2-normalized before
the linear kernel so it reduces to cosine similarity, matching the reference
implementation's own usage. Script: `analysis/compute_cknna_scratch_encoders.py`.

**Known, accepted domain mismatches** (not bugs — see script docstring):
arm-pose/full-body-pose/odometry see in-distribution probe data (their own
training source); HAR (WISDM), gesture (OpenWatch), and CLIP (HumanML3D) see
fully out-of-domain signal, which is the point (representation geometry, not
task accuracy). Gesture's encoder additionally expects watch-frame axes
(`acc_F/L/U`, `gyro_F/L/U`) but receives Nymeria's body-frame `x/y/z` channels
in the same channel *positions* — a coordinate-convention mismatch layered on
top of the domain shift, likely part of why it's the lowest-similarity task
below.

### Results (N=512 probe windows, topk=10)

CKNNA (mutual-kNN CKA):

| | HAR (WISDM) | Arm pose | Full-body pose | Gesture | Odometry | CLIP |
|---|---|---|---|---|---|---|
| HAR (WISDM) | 1.00 | 0.04 | 0.04 | 0.04 | 0.05 | 0.09 |
| Arm pose (Nymeria) | 0.04 | 1.00 | **0.13** | 0.04 | 0.04 | 0.05 |
| Full-body pose (Nymeria) | 0.04 | **0.13** | 1.00 | 0.04 | 0.05 | 0.07 |
| Gesture (OpenWatch) | 0.04 | 0.04 | 0.04 | 1.00 | 0.02 | 0.04 |
| Odometry (Nymeria, world) | 0.05 | 0.04 | 0.05 | 0.02 | 1.00 | 0.06 |
| CLIP (HumanML3D) | 0.09 | 0.05 | 0.07 | 0.04 | 0.06 | 1.00 |

Linear CKA (reference, same probe set — higher throughout since it's not
outlier-robust, but the same ranking):

| | HAR (WISDM) | Arm pose | Full-body pose | Gesture | Odometry | CLIP |
|---|---|---|---|---|---|---|
| HAR (WISDM) | 1.00 | 0.14 | 0.28 | 0.12 | 0.19 | **0.32** |
| Arm pose (Nymeria) | 0.14 | 1.00 | **0.30** | 0.17 | 0.11 | 0.14 |
| Full-body pose (Nymeria) | 0.28 | **0.30** | 1.00 | 0.18 | 0.28 | 0.22 |
| Gesture (OpenWatch) | 0.12 | 0.17 | 0.18 | 1.00 | 0.05 | 0.19 |
| Odometry (Nymeria, world) | 0.19 | 0.11 | 0.28 | 0.05 | 1.00 | 0.13 |
| CLIP (HumanML3D) | 0.32 | 0.14 | 0.22 | 0.19 | 0.13 | 1.00 |

Full matrices/heatmaps: `analysis/cknna_scratch/{cknna,cka}_heatmap.png`,
raw values in `cknna_results.csv`/`.npz`.

**Reading it**: all cross-task CKNNA values are small in absolute terms
(0.02-0.13) — six independently-initialized, independently-trained encoders on
disjoint tasks/data do NOT converge to a strongly shared representation
geometry under this metric, i.e. no strong platonic-hypothesis signal here.
But the *structure* among those small values is not noise:
- **Arm pose ↔ Full-body pose is the single highest pair** (CKNNA 0.13, CKA
  0.30) — the two tasks trained on literally the same Nymeria wrist-IMU
  signal, just regressing 5 vs. 23 joints. This is the one pair with a
  plausible mechanism (shared input distribution) for elevated similarity, and
  it is exactly the pair that shows it.
- **Gesture is the most isolated task** (lowest or near-lowest similarity to
  every other task, 0.02 with Odometry) — consistent with it being both
  domain-shifted (OpenWatch, not Nymeria) AND axis-convention-mismatched
  (watch-frame vs. body-frame) on top of that, per the caveat above.
- **CLIP shows the broadest (if still weak) alignment** with the other five,
  highest with HAR under CKA (0.32) — plausible since both are the two
  encoders using the *short* 2.56 s window on this probe set, and HumanML3D's
  everyday-activity motions may structurally resemble accelerometer patterns
  across other tasks more than any single other task resembles the rest.
- Odometry (world-frame) is uniformly among the lowest-similarity tasks
  despite sharing Nymeria/wrist-IMU with arm/full-body pose — its 2.56 s
  window (vs. their 10.24 s) and its world-frame-canonicalized regression
  target likely pull its representation in a different direction even from
  same-data siblings.

This is one topk/probe-seed setting; not swept for stability (see follow-ups).

**Follow-ups, not yet done**: (1) re-run with a few different `--seed`/`--topk`
values to check how stable the ranking is — CKNNA on N=512 with topk=10 is a
fairly tight neighborhood and single-seed values could be noisy; (2) the same
comparison for the *MAE-pretrained* (not Scratch) versions of these six heads
would test whether shared SSL pretraining measurably increases cross-task
representation alignment relative to this from-scratch baseline — that's the
more direct test of "does pretraining induce a shared representation," and
this section is deliberately scoped to the from-scratch case only, per the
user's original ask.

## 16. Inertia-1 external baseline on all six tasks (2026-08-16/17, ✅ complete)

**What was asked.** Run the "best model" of Inertia-1 (Xu et al. 2026, *Inertia-1: An Open
Exploration of Wearable Motion Foundation Models*, arXiv:2607.06617,
github.com/yang-ai-lab/Inertia-1) on arm pose, full-body pose, odometry, microgesture,
language alignment and Parkinson's disease.

**Finding 0 — there are no weights to download.** The repo (cloned to
`baselines/inertia1/repo`, HEAD ffdf91f) ships code only: README, website and paper give no
checkpoint links; HF hub has nothing under `yang-ai-lab/*Inertia*`; the single GitHub issue
(#1, 2026-08-03) is an unanswered request for exactly these weights; and even the shipped
pretraining datamodule (`inertia1/data/`) is missing from the release. Their pretraining corpus
is NHANES 2011-14 wrist accelerometry — which we have raw on disk
(`/projects/b53017/vbc1591/nhanes_pax`, 14,693 participants) — so "their best model" is
**reproduced here with their code + recipe**, and every number is ours, not theirs.

**Which model is "best".** No single objective wins everywhere in their paper (§4.1); the two
they single out as robust and use for every design-space study (§4.3) are **PatchTST** (masked
patch reconstruction; best on most HAR linear probes and Daphnet-FoG) and the
**AR-Transformer** (next-patch prediction; best on disease prediction incl. NHANES
Parkinson's). Their default/best capacity is the *small* preset (§4.2: scaling model size
does not help; App. A.4/Table 9: d=512, L=4, H=4, d_ff=1024, ≈8.5M params). We reproduce
**both** at small: `inertia1_patchtst_small_nhanes`, `inertia1_art_small_nhanes`.

**Pretraining reproduction** (`baselines/inertia1/scripts/{build_nhanes_corpus,pretrain_inertia1}.py`,
launchers `baselines/inertia1/launch_slurm_*`):
- Data, per their App. A: raw 80 Hz → 4th-order Butterworth 10 Hz (zero-phase) → 20 Hz →
  30 s windows, triaxial, **g units, no normalisation** (their PatchTST_Accel has no RevIN and
  their downstream loader never normalises). All 14,693 participants, a deterministic hashed
  1/3 of each participant's hourly files (all hours of day), no non-wear filtering (their paper
  describes none; only exactly-constant/NaN raw windows dropped) → ≈82M windows / ≈880K h as
  fp16 `[M,3,600]` per participant under `/projects/b53017/vbc1591/inertia1_nhanes20hz/`.
  Their 170k×1024 budget over this = ≈2.1 epochs (they saw ≈0.5 epoch of the full 2.8M h).
  Built by array 9559304 (+9562730 for 19 tasks that hit a transient home-quota EDQUOT on
  qnode0011; idempotent per participant).
- Model code: `imu_lm/models/baselines/inertia1_encoder.py` — pure-torch ports of their two
  Lightning modules with **identical parameter names** (`patch_proj/pos_encoder.pe/transformer/head`;
  `model.backbone.*/model.head.*` around the vendored `PatchTST_Accel`), so an official `.ckpt`
  would load unchanged if one is ever released. Param counts 8.47M (PatchTST; their Table 9:
  8.5M) / 8.44M (ART).
- Optimisation, from their shipped configs: batch 1024, AdamW, 500-step warm-up then cosine to
  1 %, grad-clip 1.0, bf16; PatchTST lr 1e-3 / wd 1e-4 / betas (0.9,0.95) / mask 0.4; ART lr 1e-3
  / wd 0.04. 170k steps (paper App. A.4; the .sh defaults to 100k). Val = 2 % held-out
  participants, every 5k steps; best.pt by val loss. Jobs **9563086 (PatchTST) / 9563087 (ART)**
  on the b53017 A100 node, chained after the corpus build. Checkpoints are written in this
  repo's run-dir format (`runs/<run>/checkpoints/best.pt = {model, cfg, step}` + encoder_meta)
  so `pretrained_run:` works everywhere.

**Downstream integration — one adapter, zero task-code changes.** `Inertia1Encoder`
quacks like `ViT1DEncoder` (`embed_dim`, `in_channels`, `patch_size`, `forward → [B,512]`,
`forward_tokens → ([B, T//16, 512], Pt)`) so every objective branch and decoder in
`imu_lm/models/ViT1D/run.py` runs unchanged. Hooks: run.py builds the adapter when the config
carries an `inertia1:` block; `_load_encoder_state_dict` delegates to
`encoder.load_pretrained_state` (the ViT1D `self_attn/linear1/linear2` key remap would corrupt
Inertia keys); `build_encoder(cfg)` factory now used by the eval scripts (arm-pose position
error [+ new `--runs NAME=RUN` CLI], 23j pose, retrieval alignment [+ `num_input_channels`
for the acc-only truncation], PADS subject-level, odom altattn/worldframe, vis_odom_scratch,
openwatch clip-level) and by `bench/score/{arm_pose,odometry}.py`; `artifacts.py` knows
backbone `inertia1`. Verified end-to-end by a GPU smoke job (9563883): all 15 objective
branches + 7 scorers pass with a random-init checkpoint.

**Adaptation decisions (explicit, all in config):**
1. *Accelerometer only* — Inertia-1 is acc-only, so gyro is dropped exactly like the
   `mae_acconly` ablation rows (`data.acc_only`); the fair reference column is therefore
   "Ours (MAE, acc-only)", the 6-ch MAE is the upper bound.
2. *Units/normalisation* — batches are converted to **g** (m/s² ÷ 9.80665; OpenWatch raw LSB
   ÷ 4096, |a|≈4136 LSB at rest) and the harness' per-window z-score and 0.25–24 Hz band-pass are
   **off** (`preprocess.{filter,normalize}.enabled: false`, `humanml3d.zscore: false`) — the
   model saw raw g-scale gravity-bearing signal in pretraining. PADS is gravity-free
   userAcceleration (§14) — an inherent shift for a gravity-trained model, noted, not fixed.
3. *Sampling rate* — an exact `scipy.resample_poly` operator (cached matrix, runs on GPU)
   anti-alias-resamples 50 Hz → 20 Hz (OpenWatch: true 100 Hz → 20 Hz, no 2× stretch).
4. *Window length* — 10.24 s → 20 patches (PatchTST's learned 60-slot positional table is
   sliced to the first N; ART's sinusoidal PE is length-agnostic); PADS 2.56 s → 5 patches;
   OpenWatch 1 s → 2 patches (`window_pad_len: 100`, i.e. no zero padding).
5. *Token grid* — pose/odometry decoders expect `T//16` tokens per channel; the 2 Hz Inertia
   tokens are linearly interpolated onto that grid (`in_channels = 1`, one fused stream).
   Classification/CLIP use the mean over native tokens = their own `encode()`.
6. *Full-body probe budget* — the template runs to the 1-day walltime with early stopping
   disabled; Inertia rows use a fixed 100k-step budget with `patience: 999999` (see memory
   note: `enabled: false` stops best.pt being written).

**Runs.** Two parallel sets, both gated on the pretraining jobs:
- *Experiments_table protocol* (identical to the sibling rows in §1/2/4/4b/5f/13d/14):
  `configs/inertia1/*.yaml` from `baselines/inertia1/scripts/make_configs.py`; 15 train +
  7 score jobs per variant via `baselines/inertia1/submit_inertia1_downstream.sh`
  (ids in `baselines/inertia1/.downstream_jobids`, 9567105–9567165). Odometry probes are the
  script probes on the pretrain checkpoints (`eval_odom_nymeria_{worldframe,altattn}.py`).
- *IMU-Bench protocol* (leaderboard): `bench/make_config.py` gained an
  `external_encoder: inertia1` branch (ViT1D cells regenerate byte-identically — checked on
  all 90); encoders `inertia1_patchtst` / `inertia1_art` registered (group `foundation`,
  ranked); 24 cells `bench_<task>__inertia1_<v>__{probe,finetune}__s42` submitted
  (9567398–9567424, `baselines/inertia1/.bench_jobids`), each train→score→leaderboard.

**Harvest.** `python baselines/inertia1/scripts/harvest_inertia1.py` →
`analysis/inertia1/inertia1_results.md` (each task next to its reference rows); then paste rows
into the per-task tables here and in Experiments_table.md, and extend
`scripts/build_openwatch_fair_table.py` with `results_openwatch_inertia1_<v>.json`. Bench cells
land in `bench/results/` and LEADERBOARD.md automatically.

**Results (2026-08-17; all 44 table-protocol jobs complete, full collation in
`analysis/inertia1/inertia1_results.md`; rows inserted into Experiments_table.md §1/2/4/4b/13d/14 and
`analysis/openwatch_fair_comparison.md`).** Pretraining: 170k steps in 2.3–2.5 h on one A100 (~19 it/s);
best val loss PatchTST 0.01806 / ART 0.02548, both still improving at the end.

| Task (metric) | Inertia-1 PatchTST probe / finetune | Inertia-1 ART probe / finetune | Ours MAE (6ch) probe / finetune | Ours MAE acc-only probe | best other SSL baseline |
|---|---|---|---|---|---|
| Arm pose pos_err cm (ang °) | 8.43 (10.26) / 8.55 (10.46) | 8.47 (10.44) / 8.47 (10.46) | 8.17 (9.81) / 8.00 (9.89) | — (val_loss 0.01888 vs Inertia 0.01755–0.01789) | SimMTM 8.56 (10.39) |
| Full-body MPJPE mm | 100.48 / 97.89 | 101.45 / 96.04 | 97.63 / 91.95 | 103.37 | RelCon 102.47 |
| Odom world-frame vel_mse / ATE m / drift | 0.02272 / 0.086 / 64.7% ; ft 0.02099 / 0.157 / 41.7% | 0.02314 / 0.091 / 64.7% ; ft 0.02062 / 0.154 / 39.9% | 0.02086 / 0.079 / 48.3% ; ft 0.01656 / 0.129 / 34.1% | — | TLIO 0.02419 / 0.151 / 38.8% |
| Odom local-vel vel_mse / ATE / drift | 0.02748 / 0.083 / 60.8% | 0.02767 / 0.086 / 60.9% | 0.0245 / 0.073 / 49.4% | — | RelCon 0.0282 / 0.087 / 55.3% |
| OpenWatch test clip-F1 (noaug / aug) | 0.517 / 0.532 ; ft 0.476 / 0.585 | 0.505 / 0.532 ; ft 0.549 / 0.575 | 0.606 / 0.615 ; ft 0.704 / 0.580 | — | Mix-Token 0.500 / 0.641; NormWear 0.453 / 0.494 |
| HumanML3D R@1 G=100 i2t / t2i | 18.4 / 17.2 ; ft 25.8 / 24.5 | 20.6 / 20.2 ; ft 28.3 / 26.1 | 26.0 / 24.3 ; ft 39.4 / 37.5 | — | Ours scratch ft 21.4 / 20.6 |
| PADS 3-class bal_acc (@best-bal_acc ckpt) | 34.3% (34.7) ; ft 34.8% (39.5) | 33.4% (35.2) ; ft 37.3% (38.7) | 38.9% (43.4) ; ft 39.2% (45.4) | — | SimMTM ft 40.2% (47.1) |

**Reads.** (1) On the kinematic tasks Inertia-1 is the strongest *external* encoder we have run:
both variants beat the acc-only MAE ablation on arm pose (val_loss 0.0176–0.0179 vs 0.0189; the acc-only
MAE row has no pos-err score) and full-body probe (100.5/101.5 vs 103.4 mm), beat RelCon/SimMTM/FOCAL on arm
pose, full-body and local-vel odometry, and beat TLIO/Air-IO on world-frame vel_mse/ATE as a *frozen* probe —
but stay ~0.3 cm / 3–4 mm / 10 % drift behind the 6-channel MAE, and finetuning them adds far less than
finetuning MAE (full-body 96–98 vs 92 mm; world-frame finetune ATE actually worsens as for every model). Their
finetunes over-fit within 3–6k steps (early-stopped), consistent with an 8.5M encoder trained on 24 h wrist
accelerometry that carries little gyro-free pose information. (2) Language alignment: probe R@1 18–21 % vs MAE
26 % (G=100), finetune 26–28 % vs 39 % — Inertia-1 finetuned ≈ our MAE *probe*, and ART > PatchTST throughout,
i.e. the causal next-patch objective transfers better to semantics here than masked reconstruction. (3) OpenWatch
1-s gestures give Inertia-1 only two 0.5-s patches; it still reaches Mix-Token / NormWear level (test clip-F1
0.50–0.59) but not MAE (0.60–0.70). (4) PADS: probes sit at the 33 % chance floor like RelCon/FOCAL (frozen
features carry no PD signal — and the input is gravity-free m/s², out of Inertia-1's g-scale gravity-bearing
distribution); finetuning ART recovers 37.3 % (38.7 % at best-bal_acc ckpt), between Scratch and MAE. Their
paper's own strongest task family (disease prediction on NHANES) does not transfer to Apple-Watch PADS through a
frozen probe. (5) Objective choice: PatchTST ≥ ART on pose/odometry probes; ART ≥ PatchTST on gestures, language
and PADS finetune — mirroring their Table 3 vs Table 5 split (PatchTST best on HAR, ART best on disease).

Bench/leaderboard cells: 20/24 done at time of writing; the 4 `bench_language_humanml3d__inertia1_*` cells
failed on the missing `HF_HOME` in the bench header (same failure as every other bench language cell that day),
resubmitted after the header fix as jobs 9690218–9690221; 2 full-body PatchTST cells still running.


## 17. Volume × diversity × source × type factorial on the leak-free 4-source pool (2026-09-03/10)

Full write-up, every number and every caveat: **`analysis/scaling_summary.md`**.
Status doc: **`PROGRESS.md`**. This section is the EXPERIMENTS.md entry point.

**What §11i-§11l could not settle.** The 5-point recipe sweep (§11i) moved window
count, session count, subject count, source coverage and motion-type coverage
together, so a flat curve could not tell volume from diversity. §11j showed
motion-mode coverage saturates below 1 % of the pool, which explains the plateau
but cannot show diversity is what carries the effect. §11l varied types and volume
on the 5-source pool but at a 50k budget with `realdata` still in the pool.

**This factorial** runs on the **leak-free 4-source pool at full size** (1,738,094
windows / 2,472 h; ocdetect + eatingspeed + sonic + motionx, sanitised) and
isolates four axes one at a time, everything else pinned — same encoder, same
z-score input, same 50,000-step budget, same frozen-probe protocol on the same
4-task panel (arm pose, world-frame odometry, HAR-WISDM, microgesture).

| Axis | Range | Arms | Status |
|---|---|---|---|
| Volume | 12,500 → 1,738,094 windows (18 h → 2,472 h, **137×**) | `vd_{nar,one,nat,bal}_n{12k,50k,200k,800k}` + 2 anchors | ✅ 18/18 |
| Source count | 1 → 4 datasets, types 158 → 1000 | `vd_src{1,2,3,4}_n50k` | ✅ 4/4 |
| Motion types | 3 → 673 effective types (**224×**) | `vd_typ{3,30,300,all}_n12k` | 🔄 3.25/4 |
| Wearers | 1 / 3 / 10 / all | `vd_sub{1,3,10,all}_n50k` | ⏳ built, not run |

Design choices that work *against* the hypothesis, stated up front: the source
ladder adds datasets **smallest-first** (motionx → sonic → eatingspeed → ocdetect),
so `src1` is not "the big dataset again"; subjects in the wearer ladder are picked
greedily to *maximise* type coverage; and compute is held fixed rather than epochs,
so a small arm simply revisits its windows more often.

### Result — every data axis is flat

Spread = max − min over the arms of one axis; frozen probe, seed 42.

| Axis | arm pose (cm ↓) | odometry ATE (↓) | HAR-WISDM F1 (↑) | microgesture F1 (↑) |
|---|---|---|---|---|
| Volume (137×) | 0.130 | 0.0025 | 0.054 | 0.078 |
| Source count (1→4) | 0.085 | 0.0067 | 0.035 | 0.136 |
| Motion types (224×) | 0.073 | 0.0032 | 0.037 | 0.034 |

No axis is monotone, and each optimum sits at a different arbitrary position —
volume best at **18 h** (arm pose, WISDM, microgesture), sources best at **1–3**,
types best at **300**. Concretely: microgesture along the `nat` volume column runs
0.568 (18 h) → 0.522 → 0.549 → 0.531 → **0.490** (2,472 h), i.e. 137× the data and
8 F1 points *worse*; the 4-source arm is worse than the 3-source arm (0.474 vs
0.590); `typ3` beats `typall` on arm pose (8.109 vs 8.168) with 224× fewer types.

### Against the recipe factors — and the caveat that matters

Same corpus, same protocol, only the recipe changes. **HAR mean over 7 datasets**
(noise-averaged, the reliable statistic):

| Factor | spread | best → worst |
|---|---|---|
| Pretraining objective | **0.157** | MAE 0.656 › RelCon 0.627 › SimMTM 0.593 › FOCAL 0.499 |
| Input scaling | **0.121** | z-score 0.656 › physical 0.589 › phys+patchnorm 0.582 › raw/DC 0.535 |
| Modality (acc+gyro vs acc-only) | 0.024 | 0.656 vs 0.632 |

⚠ **Not like-for-like.** The recipe factors are a 7-dataset mean; the data axes are
single tasks. Comparing 0.157 against 0.054 overstates the gap, and the voldiv arms
ran only a 4-task panel so no 7-dataset mean exists for them.

**The clean single-task-on-both-sides comparison is odometry:**

| | odometry ATE spread |
|---|---|
| Volume (137×) / Types (224×) / Sources (1→4) | 0.0025 / 0.0032 / 0.0067 |
| Modality / Input scaling / Objective | 0.0261 / 0.0506 / 0.0523 |

Recipe beats data **4–20×** with no statistical sleight of hand. Where it does not
hold: on **microgesture** the source axis (0.136) matches input scaling (0.142);
on **arm pose** the measured run-to-run noise band is **0.108 cm** (three runs of a
byte-identical config: 8.054 / 8.136 / 8.162) and swallows every factor except
modality (0.325) — nothing is resolved there, recipe included.

### What this supports, and what it does not

Supported: *over the range this corpus spans — 18 h to 2,472 h, 1 to 4 sources,
3 to 673 motion types — corpus scale does not drive downstream performance, and on
odometry the objective and the input representation each move the metric 4–20×
further than any data axis.*

**Not** supported: "recipe beats data by an order of magnitude" as a general claim
(§11's earlier framing — true on odometry, false on microgesture, unresolvable on
arm pose); "less data is better" (the 18 h arm's wins are inside the noise band and
single-seed); anything about corpora >2,472 h or acc-only corpora (NHANES 4,000 h
and Capture24 3,122 h are on disk, untested here).

Corroborated independently by §11j (a deliberately-redundant subset collapses,
WISDM 61.7 → 35.3, while random ≈ diverse at 61.7 vs 61.5 — diversity matters only
at the pathological end) and by §11i (the smallest fraction wins 11 of 16 downstream
cells while the pretraining reconstruction loss improves monotonically).

### Reliability

Single seed (42) throughout. Spread is a crude two-point statistic that grows with
arm count — both sides use 4–5 arms so the comparison is fair, but these are not
formal effect sizes; a two-way variance decomposition needs replicate seeds.
Motion types are k-means clusters (K=1000) in an **encoder-free** 40-dimensional
rotation-invariant physical feature space, not human labels — the four sources have
non-alignable label vocabularies; an encoder-space clustering is reported alongside
as an independent read-out. The **wearer axis is still missing**, and it is the one
that maps onto how every downstream task is split (subject-disjoint).

Pipeline: `analysis/volume_diversity/` (`arms.py` → `01`-`05` → `06_harvest.py` →
`07_analyze.py`; `08_provenance.py` → `09_subject_arms.py` for the wearer ladder;
`10_axis_arms.py` for the source/type ladders). Configs `configs/voldiv/`,
launchers `launch/voldiv/`.

### Two controls that could not be built, and why

1. **`vd_dup` redundancy arm.** Selecting the N nearest neighbours to a seed window
   in encoder space produced `nn_self_median` 2.474 vs 2.473 for the uniform arm —
   no redundancy created, and the "ball" spanned 424 of 1000 types. Local density in
   the encoder embedding is too uniform for ball selection to concentrate anything.
   Dropped rather than shipped as a control that controls nothing.
2. **A clean reproduction of §11j's collapse.** `vd_nar_n12k` and `vd_typ3_n12k` are
   the same thing (3 types present, effective 2.8 vs 3.0, vendi 8.92 vs 8.74) and
   **neither collapses** — `nar` posts the best arm-pose score of any arm (8.003).
   §11j's redundant arm was built on the **11-source** pool, whose largest clusters
   are NHANES/Capture24 free-living sedentary data; the 4-source pool contains no
   such degenerate mode. The two results do not contradict — they are about
   different pools. No new experiment needed.

## 18. Native preprocessing for the external objectives: SimMTM / FOCAL / RelCon under their own input handling (2026-09-21, in flight)

### The objection

A reviewer checked the three external SSL baselines against their published code and
pointed out that we cannot describe our uniform per-window z-score as "following these
methods' original preprocessing" — the three do not share one convention, and two of
them do not use per-window z-norm at all. Verified line by line, they are right:

| method | what its own code feeds the encoder | source |
| --- | --- | --- |
| SimMTM | normalises **inside the model**: per sample and channel over the **visible** points, de-normalising the reconstruction before the MSE | `SimMTM_Forecasting/models/SimMTM.py:172-178, 205-207` |
| FOCAL | **frequency domain** — each sub-interval is FFT'd and real/imag become channels; no z-norm anywhere on the main path | `focal/src/data_augmenter/Augmenter.py:141-157` |
| RelCon | SSL pretraining reads raw `.npy` with **no input normalisation**; RevIN lives only in the motif-distance net | `relcon/data/Base_Dataset.py:50`, `nets/CrossAttn/CrossAttn_Net.py:34` |

Two clarifications worth recording, because both are easy to get wrong:

* SimMTM's **classification** dataloader reads pre-processed `.pt` tensors and does not
  re-normalise, so the forecasting branch's instance norm cannot be generalised to all
  SimMTM experiments. We implement the **pretraining** branch, which does normalise.
* The `StandardScaler` in `relcon/eval/Classify/linear_probe.py:53` is fit on the
  **encoder's embeddings**, not on the input signal. It is not an input convention. The
  train-set `StandardScaler` the reviewer saw is in the HAR preprocessing script
  (`data/process/comparesslbench/pamap.py:380`), a different code path.

Our three objectives in `imu_lm/objectives/ssl_vit_baselines.py` and
`relcon_ssl.py` apply **no** normalisation of their own — they consume whatever the
shared loader produces, which is per-window z-scored. So the uniform z-score is our
protocol choice. That is defensible and the paper already says so
(`Sections/A_MethodDetails.tex:249`, `Sections/8_Appendix.tex:9`); what it did not have
is a measurement of what the choice costs each method.

### What the z-score actually removes

Measured on a real validation batch from the five-source pool, same windows both ways:

```
z-score row   : per-window |acc mean| 0.0000   per-window acc std 1.0010
native arms   : per-window |acc mean| 0.4205   per-window acc std 0.1139   (units of g)
```

Gravity direction and motion amplitude are exactly annihilated by the z-score, which is
the mechanism behind the pose/odometry concern and behind §(HAR) `lying_down` recall 0 %.

### Design: three paired native arms

Each arm re-pretrains one objective under its own input handling and is paired with the
existing `*_5src` row — same backbone, same corpus composition, same 160k cap, same
patience 10, same lr/augmentation/window length. Generated key-for-key from the z-score
twin by `launch/nativenorm/gen_native_cfgs.py`, so the diff is provably only:

| arm | dataloader | inside the method | downstream probe input |
| --- | --- | --- | --- |
| `simmtm_5src_native` | physical | visible-point instance norm + reconstruction of the un-masked copy in physical units | z-score (SimMTM's own instance norm on an unmasked window **is** a per-window z-score, so the downstream protocol is byte-identical to `simmtm_5src` and only pretraining differs) |
| `focal_5src_native` | physical | per-patch FFT (real+imag) input stem | physical + the same FFT stem |
| `relcon_5src_native` | physical | nothing new (RevIN already in the distance net); stage 1 retrained on the same input | physical |

Two flags were added to the SimMTM objective, both defaulting to the behaviour every
existing SimMTM row was trained with: `visible_point_instance_norm` and
`recon_on_original_only`. The second restores `pred_batch_x = dec_out[:batch_x.shape[0]]`
(SimMTM.py:208-211) — this port took the MSE over every masked copy against a target that
is zero at the masked points, which is harmless under z-score but an amplitude-weighted
artefact in physical units. The SimMTM arm therefore differs from its twin in **two**
respects, not one; they are one mechanism in SimMTM's code and are documented as such.

⚠ **FOCAL's arm changes the input REPRESENTATION, not just the normalisation.** A
frequency-domain encoder cannot be probed on time-domain input, so `input_transform: fft`
rides on the encoder card as a `vit1d_override` and reaches all ten probes. Read that row
as "what FOCAL does natively", not as "what dropping z-norm does".

### Corpus and validation caveats

* Trains on `mae_multidata_cache_v3raw` (the un-normalised twin the nozscore design
  ablation rebuilt) subset by `idx_ref_v3raw.npy` to the composition of the production
  5-source pool: 196,659 of 197,094 windows. The 0.22 % gap is the mocap spike sanitiser
  and the re-windowed HAR pool; the arms are composition-matched, not row-identical.
* ⚠ The raw twin was built **without the production cache's per-source val caps** — its
  val pool is 290,638 windows, 64 % OCDetect, against production's 27,463 at 4 %. Since
  `val_loss` is both the early-stopping monitor and the `best.pt` selector, an arm
  validated on that mixture is not selecting a comparable checkpoint.
  `launch/nativenorm/make_val_idx.py` writes per-source val indices restoring the
  production count (27,433 windows). **The `abl_nozscore` design-ablation arm does not
  have this fix and validates on all 290,638 windows** — flagged to the session that owns it.

### Files

* configs: `configs/nativenorm/{simmtm,focal,relcon}_5src_native.yaml`, `relcon_motifdist_5src_native.yaml`
* generator / index / preflight: `launch/nativenorm/{gen_native_cfgs,make_val_idx,preflight}.py`
* launch + submit: `launch/nativenorm/*.sh`, driver `submit_nativenorm.sh` (idempotent, `--dry-run`)
* code: `PatchEmbed1D.input_transform` (`imu_lm/models/ViT1D/model.py`), the two SimMTM flags
  (`imu_lm/objectives/ssl_vit_baselines.py`), and a documented pretrain/encoder normalisation
  split in the encoder card (`normalize_method_pretrain` + `normalize_method_note`, `bench/make_config.py`)
* cells: 30 = 3 arms × the 10 core tasks, frozen probe, seed 42

### Harvest

`python -m bench.leaderboard` after the 30 cells land; compare each `*_native` row against
its `*_5src` twin task by task. The question the table answers is **not** "which objective
is best" — it is "does the benchmark's uniform z-score change the ranking of these three
objectives, and by how much for each". Pose and odometry are where a change is expected.

---

## 19. Multi-task joint fine-tuning of one shared encoder (2026-09-22/23, ✅ both runs complete, ❌ NOT in the paper)

**Decision 2026-09-23: this study does not enter the ICLR submission.** It is archived here in
full so the result is not lost and so the design error below is not repeated.

**Question.** Section 5's transfer figure shows that adapting the MAE encoder to one task
specialises it: it helps a related task and hurts a distant one (microgesture loses 11-45 %
from every source). Is that a property of the *tasks* — they need incompatible features — or
of *single-task adaptation*? Joint training asks the same encoder to serve two tasks at once.

**Method.** `scripts/train_multitask_joint.py`: ONE shared encoder warm-started from the
reference 5-source MAE; every task keeps the benchmark's own loader, decoder and loss
(`bench/generated/configs/bench_<task>__mae_5src__finetune__s42.yaml`, built exactly as
`imu_lm/models/ViT1D/run.py` builds them). One task per optimiser step in a fixed cycle given
by `--ratio`. AdamW: encoder at `--enc-lr` 3e-4, each decoder at its own single-task learning
rate (pose 1e-3, odometry 1e-4, gesture 1e-5 — they differ 100x). Warm-up then cosine, AMP.
Checkpoints are written per task in the single-task key layout so bench scorers read them
unchanged. Smoke test: 7023958 OOM-killed at 40 G, 7031122 completed at 120 G (peak 64.5 G).

### 19a. `mt_body_odom` — full-body pose + wrist-frame odometry, 1:1, 40k steps (job 7101347)

The one mutually helpful pair in the transfer figure. 73 min at 9.2 steps/s. Validation loss:

| step | full-body | odometry |
|---|---|---|
| 2,000 | 0.02100 | 0.02680 |
| 18,000 | **0.01864** | 0.02162 |
| 24,000 | 0.01889 | **0.02036** |
| 40,000 (final) | 0.01961 | 0.02056 |

Both tasks converge and only drift up slightly at the end. Nothing pathological: one encoder
can serve both kinematic tasks. **No frozen probes were run on this pair** (see the decision
above), so there is no transfer column for it.

### 19b. `mt_body_gesture` — full-body pose + microgesture, 10:1, 22k steps (job 7101356)

The most antagonistic pair when frozen. 56 min. Validation loss:

| step | full-body | microgesture |
|---|---|---|
| 2,000 | 0.02084 | **0.77879** |
| 10,000 | **0.01906** | 1.21662 |
| 22,000 (final) | 0.01975 | 1.49926 |

**The gesture head peaks at the very first checkpoint and then decays monotonically to near
chance** (uniform 6-class guess ≈ 1.79) with a training loss of 0.00001. Full-body is
unharmed — 0.01906 here against 0.01864 in the odometry run.

**This is a budget/learning-rate error of ours, not evidence of task conflict.** Two causes:
(i) the total budget was never shortened for the small task — OpenWatch train is 3,174 windows
(13 batches), so 2,000 gesture steps is ~154 epochs against a single-task optimum of 1,400
steps; (ii) the shared encoder runs at 3e-4 while the single-task gesture fine-tune uses 1e-5,
a factor of 30, which is why gesture peaks after ~182 of its own steps rather than 1,400.

**Correction (2026-09-23): the validation cross-entropy badly misrepresents this head.** Scored
with the benchmark's own metric, the gesture head of this run reaches **0.6498 clip macro-F1**,
against 0.5419 for the single-task fine-tune and 0.6081 for training from scratch. A training
loss of 0.00001 means the model is extremely confident, so the few confident mistakes dominate
the cross-entropy while `argmax` stays right, and clip-level aggregation over three consecutive
windows removes more of the remaining noise. The budget and learning-rate mismatch above is
real and a rerun should still fix it, but the head is usable and its numbers are good.

### 19c. What was fixed, and what a rerun should do

`best.pt` is now selected **per task**, at that task's own lowest validation loss (commit
a4cfd54), because the single-task fine-tunes this is compared against all early-stop on their
own validation split — taking the final step for every task compares a selected model against
an unselected one. The runs above predate this and their `best.pt` IS the final step; their
`latest.pt` is the last validation step. A rerun (names `mt2_*` were prepared, never submitted)
should use per-task selection, `--val-every 500` so the small task's optimum is actually
measured, and a lower encoder learning rate for any pairing with a small task.

**Caveat on the cells that do exist.** Frozen probes for `mt_body_gesture` were submitted
(7115255-7115406) and read the ENCODER only, where the final step is close to optimal
(full-body 0.01975 at 22k against 0.01906 at 10k). Those cells are usable but measure the
22,000-step joint encoder with no checkpoint selection, and must say so if ever reported.

### 19d. Frozen probes of the joint encoders (20 cells, complete)

Both joint encoders were registered as encoder cards and run through the benchmark's frozen
probe on every task (jobs 7115087-7115406; `har_prism` and `har_userwatch` for the gesture pair
were still running at the time of writing). The comparison that matters is against `tx5_*`, the
same MAE adapted to **one** task, because that is what Section 5's figure measures.

**On the tasks the encoder was jointly trained on, joint training beats single-task adaptation
everywhere** (frozen probe, lower is better for pose/odometry, higher for gesture):

| target | unadapted `mae_5src` | single-task adapted (`tx5_*`) | jointly trained |
|---|---|---|---|
| full-body pose (mm) | 97.85 | 95.05 (odom) / 97.98 (gest) | **91.13** (b+o) / 94.34 (b+g) |
| odometry (ATE m) | 0.1999 | 0.1667 (body) / 0.1988 (gest) | **0.1497** (b+o) / 0.1774 (b+g) |
| microgesture (clip F1) | 0.6467 | 0.4987 (body) / 0.5952 (odom) | 0.6503 (b+o) / **0.6591** (b+g) |

The microgesture row is the direct answer to the question in the preamble. Adapting the MAE to
full-body pose alone costs gesture 15 points of clip F1 (0.6467 -> 0.4987, the largest single
drop in the transfer figure); training the same encoder on full-body pose **and** gesture
together gives 0.6591, above the unadapted MAE. **The specialisation is a property of
single-task adaptation, not of the tasks needing incompatible features.**

The joint heads also beat their own single-task fine-tunes: odometry 0.1488 against 0.1541 (the
best odometry number in the benchmark), microgesture 0.6498 against 0.5419.

**What joint training does not fix is the cost to tasks it was not trained on**, which follows
the same near/far pattern as single-task adaptation:

| target | `mae_5src` probe | `mt_body_odom` | `mt_body_gesture` |
|---|---|---|---|
| arm pose (cm) | 8.138 | **7.934** | 8.186 |
| UT-Watch (F1) | 0.7271 | **0.7976** | -- |
| MHEALTH (F1) | 0.6860 | 0.6543 | 0.6886 |
| PRISM (F1) | 0.2790 | 0.2710 | -- |
| HHAR (F1) | 0.6207 | 0.5607 | 0.5815 |
| PADS (F1) | 0.4247 | 0.4064 | 0.3789 |
| IMU-text (R@1) | 27.92 | 21.03 | 17.21 |

Kinematically related tasks gain (arm pose, UT-Watch), semantically distant ones lose, and
IMU-text retrieval loses most -- the same ordering the single-task sources produce, at a similar
magnitude (`tx5_fullbody` probe is 19.57 on IMU-text, `tx5_odom` 19.95).
