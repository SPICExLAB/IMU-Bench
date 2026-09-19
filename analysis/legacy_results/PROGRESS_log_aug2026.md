# Experiment Progress — 2026-08-09 19:35 CDT

## Currently running jobs (4 total)

| Job ID | Name | Partition | Elapsed | What it's doing |
|---|---|---|---|---|
| 8818231 | `vit1d_ssl_mae_stage1_acconly_mr50` | b53017 | 1d 21h | Stage-1 mask-ratio ablation, mr=0.50 |
| 8818232 | `vit1d_ssl_mae_stage1_acconly_mr90` | b53017 | 1d 21h | Stage-1 mask-ratio ablation, mr=0.90 |
| 8836201 | `vit1d_full_body_pose_probe_nymeria_mae_recipe1pct` | gengpu | 5h 27m | 1% recipe downstream (last of 16) |
| 8836216 | `vit1d_full_body_pose_finetune_nymeria_mae_recipe10pct` | gengpu | 25m | 10% recipe downstream |

Plus **13 more 10%-recipe downstream jobs queued PENDING** (priority, not blocked/dependency — will run as slots free up): 8836217–8836229.

---

## A. 1% / 10% data-scaling recipe sweep

Extends the fixed-step (300K), `session_fraction` scaling sweep to 5 points (1/5/10/25/100%).

**Pretrain — both done:**
| fraction | job | status | elapsed |
|---|---|---|---|
| 1% | 8836196 | ✅ COMPLETED | 23h10m |
| 10% | 8836197 | ✅ COMPLETED | 23h18m |

**Downstream (16 jobs per fraction: arm-pose, full-body-pose, gesture, HAR×4, odom — each probe+finetune):**

| fraction | done | running | pending |
|---|---|---|---|
| 1% | 15/16 | 1 (8836201, full-body-pose probe, step 295,000) | 0 |
| 10% | 2/16 | 1 (8836216, full-body-pose finetune, step 14,000) | 13 |

**Caveat found while checking:** both full-body-pose *probe* runs (1% and the 5-source one below) have `best_val_loss = inf` on disk — `best.pt` never got created, only `best_mpjpe.pt` (1%: mpjpe 97.29mm @ step 29,000). Val-loss-based early stopping/checkpoint selection isn't working for this probe config family; harvest from `best_mpjpe.pt`, not `best.pt`, for these two. Same class of issue as the `best.pt`-vs-`best_mpjpe.pt` checkpoint-selection mismatch found earlier in the LR sweep, but here `best.pt` is missing entirely rather than just picking a suboptimal step — worth a real look before trusting any probe number from this family.

**Next step when finished:** harvest via `analysis/mae_scaling_diagnostics/12_recipe_val_rescore.py`, extend `analysis/plot_data_scaling.py` FRACTIONS to `[1, 5, 10, 25, 100]`. Full harvesting checklist in memory (`recipe-sweep-1pct-10pct-inflight`).

---

## B. New MAE on fewer datasets (5-source vs 11-source)

Evaluates a new SSL MAE checkpoint pretrained on only 5 sources (`realdata`, `ocdetect`, `eatingspeed`, `motionx`, `sonic`) against the original 11-source baseline, on full-body-pose, odometry, and HAR (WHARF excluded — different baseline architecture).

**17 of 18 done:**

| task | done | running |
|---|---|---|
| Full-body-pose (Nymeria + TotalCapture, probe+finetune) | 3/4 | 1 (8892234, Nymeria probe, step 214,000) |
| Odometry (Nymeria, probe+finetune) | 2/2 | — |
| HAR (WISDM/UTD-MHAD/PAMAP2/RecoFit/HHAR/MHEALTH, probe+finetune) | 12/12 | — |

Same `best_val_loss=inf` / missing-`best.pt` issue on the running job (8892234) as above — `best_mpjpe.pt` has mpjpe=98.11mm @ step 56,000 in the meantime.

**Caveat to keep in the final write-up:** full-body-pose/odometry are genuinely held-out for the 5-source checkpoint but were *not* held-out for the 11-source baseline (~4.4% of its pretrain pool). HAR is transductive-overlap for both equally. Full checklist + job→run-name table in memory (`5src-vs-11src-mae-eval-inflight`).

---

## C. Stage-1 + Stage-2 curriculum ablations

**Baseline two-stage — done:** Stage-1 val 0.704533 → Stage-2 val 0.749114.

**Mask-ratio ablation:**
| arm | status |
|---|---|
| Stage-1, mr=0.50 (8818231) | 🔄 RUNNING, step 280,000/300,000 (93%), val 0.6395 |
| Stage-1, mr=0.90 (8818232) | 🔄 RUNNING, step 280,000/300,000 (93%), val 0.8103 |
| Stage-2, mr=0.50 / mr=0.90 | ⏸ not launched — waiting on their own Stage-1 arm |
| Fixed-eval-ratio rescore (`rescore_mask_ratio.py`) | ⏸ written, not run — raw val losses across arms aren't comparable until this runs |

**Data ablations — both done:**
| arm | val loss | vs baseline (0.749114) |
|---|---|---|
| A: real-only (no synthetic) | 0.744823 | better |
| B: no Stage-1 (from scratch) | 0.759673 | worse — confirms Stage-1 is earning its cost |

**LR sweep (full-body-pose finetune) — done:** plateau 5e-4→2e-3 (~87.5mm), breaks at 3e-3 (93.5mm).

**Two-stage odometry finetune (8870064) — done**, visualized (single-session + 9-session grids, best/worst-case sessions identified).

**Immediate next action once the mr50/mr90 Stage-1 arms finish (~7% left, should be soon):** launch the two Stage-2 mask-ratio arms, then run the fixed-ratio rescore for a fair comparison.

---

# IMU-Bench (controlled leaderboard) — launched 2026-08-16 ~17:30 CDT

New `bench/` harness (see `bench/README.md`): task registry with unified input/output
cards for the six headline tasks, encoder registry, one generated config per
(task × encoder × protocol × seed), one scorer per task writing
`bench/results/<task>/<enc>__<proto>__s<seed>.json`, and `LEADERBOARD.md` /
`bench/leaderboard.html` generated only from those JSONs.

**Grid:** {mae_11src, mae_5src, mae_noaltattn, mae_acconly, focal, simmtm, relcon} ×
{probe, finetune} + scratch, on arm_pose_nymeria, full_body_pose_nymeria,
odometry_nymeria_worldframe, microgesture_openwatch (paper protocol, test split),
language_humanml3d (§13e tuned recipe, test split), pd_pads (select on val bal_acc,
subject-level metric) = 90 cells.

**Submitted:** ~90 SLURM jobs (`bench/generated/submissions.tsv`): 22 score-only jobs on
recipe-equivalent legacy runs (arm pose, gesture, odometry finetune/scratch, language
MAE finetune), 11 legacy PD (val_loss-selected, shown as *partial*), and ~60
train+score jobs `bench_<task>__<enc>__<proto>__s42` (all 6 PD × 15, language × 14,
odometry × 12, gesture finetunes × 7, arm/full-body finetunes × 5 each). Full-body legacy
cells were scored locally (10 done). All GPU jobs were PENDING (Priority) at launch;
gengpu had ~1100 pending jobs.

**Harvest:** `python -m bench.launch --fill-missing --dry-run` lists what is still missing
(it also re-scores any trained-but-unscored run); `python -m bench.leaderboard &&
python -m bench.leaderboard_html` regenerates. Every job regenerates the markdown itself.

**Infra:** home fileset hit 99.9% of the 80 GB quota during launch (`sed` failed with
"Disk quota exceeded"); moved `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct`
(15 GB) to `/projects/b53017/vbc1591/hf_cache/hub/` (+symlink) and purged the pip cache
(3.3 GB) → 78%. Remaining big consumer in home: `baselines/` (45 GB) — candidate for the same
move+symlink treatment as `runs/`.

## IMU-Bench — check-in 2026-08-17 08:30 CDT

* **~72/90 grid cells scored overnight** (arm pose, odometry, microgesture, PD complete or nearly; full-body probes done). Board: `LEADERBOARD.md`, artifact redeployed.
* **Language: all 15 jobs failed at start** (CLIP text tower not found: my job header lacked
  `HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1`, which the old launchers export). Fixed in
  `bench/launch.py`; resubmitted 15 + the legacy MAE score job (9690128-9690144). Pending.
* **PD race:** the 11 legacy val-loss "partial" score jobs finished AFTER 7 bench PD runs and overwrote
  their results. Moved legacy rows to protocol tag `legacy_valloss` (`<enc>__legacy_valloss_<proto>__s42.json`,
  never collides), patched the harvester, and re-submitted 7 score-only jobs (9690224-9690230).
* **Full-body finetune LR:** the controlled grid had used the template's untuned 1e-4 (MAE 92.4 mm) — the
  user's expected ~87.8 mm comes from the §2a LR sweep at 1e-3. Changed the task recipe to
  `finetune lr 1e-3` for EVERY encoder, `checkpoint: best_mpjpe` (min-val-MPJPE for legacy and bench runs
  alike), mapped legacy `..._finetune_nymeria_mae_multidata_lr1e-3` as mae_11src/finetune, archived the
  1e-4 results (`bench/results/_archive/full_body_pose_nymeria_finetune_lr1e-4/`), moved the 1e-4 bench
  run dirs to `*__lr1e-4`, and resubmitted 6 finetunes + 1 score (9690296-9690303).
  ⚠ The Inertia-1 full-body finetunes (other session) are still at lr 1e-4 (`inertia1_art` done,
  `inertia1_patchtst` running); they need a relaunch under the new recipe to stay comparable.
* Still running from day 1: `bench_odometry_nymeria_worldframe__relcon__probe__s42` (~4 h).

## IMU-Bench — check-in 2026-08-17 09:05 CDT

* Moved all open cells to the **b53017 buy-in node** (`--buyin`; jobs 9690498-9690545): 4 running
  (full-body finetunes lr 1e-3: mae_5src / noaltattn / acconly / focal at 11-15 evals, best-so-far
  89.0 / 89.6 / 96.2 / 100.5 mm), 22 queued behind them (relcon+simmtm fb finetune, 15 language, 6 PD scores).
* Legacy MAE full-body finetune at lr 1e-3 scored: **87.46 mm** (min-val-MPJPE ckpt); language legacy MAE
  finetune (rtune lr1e-4 τ0.1, latest.pt, TEST split): **R@1 G=100 = 38.8** (i2t 39.3 / t2i 38.3, MRR 54.9).
* Fixed: score-only jobs on harness runs were mis-tagged † (legacy) — `--run` no longer implies legacy;
  1 result repaired. Archived the Inertia-1 PatchTST full-body finetune (trained at lr 1e-4 before the recipe
  change) next to the other 1e-4 results; both Inertia-1 fb finetunes need a relaunch at 1e-3.
* Still on gengpu: RelCon odometry probe (4.6 h, walltime 8 h) and 4 pending Inertia-1 language jobs (other session).

## IMU-Bench — controlled grid COMPLETE 2026-08-17 15:30 CDT

All 90 cells of the controlled grid (7 encoders × probe/finetune + scratch × 6 tasks) are scored;
`python -m bench.launch --fill-missing --dry-run` → 0 jobs. Buy-in node drained the last 27 jobs in ~6 h.
Board: `LEADERBOARD.md` / `bench/leaderboard.html`. Only open cells: Inertia-1 full-body finetunes
(other session; must be relaunched at the new lr 1e-3 recipe).

Headline (finetune, mean rank): MAE-11src 1.83 (#1 arm 8.00 cm, full-body 87.46 mm, odom 0.1276 m,
gesture 0.704 F1) > MAE-5src 2.17 > MAE-noAA 3.00 > SimMTM 4.67 > RelCon 5.17 > MAE-acc 6.83 > FOCAL 7.67.
Probe: MAE-noAA 2.00 ≈ MAE-11src 2.17 ≈ MAE-5src 2.50 ≫ Inertia-1 PT 5.0 > MAE-acc 5.8 > Inertia-1 ART 6.0
> SimMTM 6.3 > RelCon 6.7 > FOCAL 8.5. Scratch (same arch, no pretraining) is the reference row: it beats
every frozen probe on odometry (0.140) and full-body (92.4 mm) — frozen features are not enough there —
but loses to MAE finetune everywhere. Language finetune (test R@1 G=100): noAA 39.4 ≈ MAE 38.8 = 5src 38.8
≫ SimMTM/ART 31.3, RelCon 30.7, acc 29.7, PT 27.7, FOCAL 23.5, scratch 23.2. PD subject bal-acc is noisy
(66 test subjects, 3 classes): RelCon 58.7 > 5src 56.1 > acc 55.8 > MAE 54.8 finetune; probes 33-42%.

## IMU-Bench — 2026-08-17 17:00 CDT: MobilePoser odometry row completed; Inertia-1 large launched

* **MobilePoser odometry (all metrics, bench protocol):** ATE **0.2355 m**, drift **59.42 %**, vel MSE **0.02963**
  (73,515 val windows, 2.56 s context, job 9719402; scorer `baselines/full_body_pose/SimplePoser/eval_nymeria_worldframe_odom.py`,
  result JSON is the source of truth — `external.mobileposer: {scored: true}` in tasks.yaml stops `--harvest-external`
  from overwriting it). The old "0.150" was MobilePoser's own 1-s translation error, not an ATE. Gotchas: its `test.pt`
  is a SHUFFLED file list (RandomState(42) after the train shuffle) → sessions matched by length + wrist-acc content;
  its imu vector stores acc/30 (acc_scale).
* **Inertia-1 full-body finetunes** relaunched at lr 1e-3 (9719215 ART, 9719217 PatchTST; running).
* **Inertia-1 large preset** (d1024/8L/16H/ff4096, 101M params; their presets/large.yaml) — same NHANES corpus + recipe
  as the small repro: pretrain jobs **9721392 (patchtst) / 9721393 (ar_transformer)** on b53017 (3-day limit, `--resume`
  on requeue), 24 bench cells chained `afterok` (ids in `baselines/inertia1/.bench_large_jobids`), encoders
  `inertia1_patchtst_large` / `inertia1_art_large` in encoders.yaml (`preset: large` now flows through
  `inertia1_downstream_config(preset=)`). ⚠ buy-in node currently full (2 GPUs xtu2338 + 2 mine); pretrains queue behind
  xtu2338's 2 pending jobs. Expect ~10× the small pretrain's 2.4 h → ~1 day each once started.
* **17:30 — large pretrains twinned across partitions** (user: "use gengpu if available"): each variant now has a
  canonical b53017 job (9722174 patchtst / 9722175 ART, 3-day) AND a gengpu twin (9722176 / 9722177, 2-day, account
  p32139); whichever schedules first trains (`--resume`, atomic ckpts); the twin exits if the canonical is running,
  the canonical cancels a running twin and resumes from its latest.pt. 1-CPU sentinels (9722204 / 9722206) exit when
  `runs/<run>/logs/metrics.txt` hits step 170000; the 24 bench cells (`baselines/inertia1/.bench_large_jobids`)
  depend `afterok` on the sentinels, so they start as soon as either partition finishes.
* **17:45** — both large pretrains RUNNING on b53017 (9722174 patchtst 3.2 it/s → ~15 h; 9722175 ART just started);
  gengpu twins stay pending and will no-op. Inertia-1 small full-body finetunes at lr 1e-3 done: ART **94.12 mm**
  (was 95.66 @1e-4), PatchTST **93.61 mm** (was 98.73) — full-body finetune column now complete for all 9 ranked rows.
* **17:35 — large pretrains RESTARTED at lr 2e-4 / warmup 2000** (jobs 9723699 patchtst / 9723701 ART on b53017,
  twins 9723702/9723703 on gengpu): at the small recipe (1e-3/500) PatchTST-L diverged (val 0.108 @5k vs 0.023 small,
  train loss 0.031→0.13) and ART-L spiked to loss 22 @2.5k. First attempts kept in `runs/*_large_nhanes__lr1e-3_diverged`.
  Sentinels (9722204/9722206) and the 24 chained bench cells unchanged (they key on the run name). ~3.3 it/s → ~15 h.

## IMU-Bench — HAR added 2026-08-17 18:20 CDT

* Six HAR tasks (`har_wisdm/utd_mhad/pamap2/recofit/hhar/mhealth`) + virtual aggregate column **`har_mean`**
  (mean macro-F1 over the six; leaderboard synthesises it in `inject_aggregates`, members hidden from the main
  tables). Recipe = Experiments_table §3 configs; selection min val loss; scorer `bench.score.har` RE-EVALUATES
  the checkpoint with pooled predictions (pre-2026-07-18 logs carry the per-batch F1 bug: WISDM MAE probe logs
  0.206, true 0.639 — reproduced exactly by the scorer). DeepConvLSTM = external reference (summary.json).
* Submitted: 60 legacy score-only cells + 6 DeepConvLSTM (b53017, ~1 min each) and **30 new finetunes**
  (noAA/acc/FOCAL/SimMTM/RelCon × 6 datasets) on gengpu. `bench/generated/har_submit.log` has the ids.
* **19:15** — all 66 HAR legacy/DeepConvLSTM cells scored (re-evaluated, pooled metrics; WISDM/UTD reproduce §3
  exactly). `HAR (6 datasets)` mean macro-F1 — probe: noAA 0.637 ≈ 5src 0.637 ≈ MAE 0.635 > acc 0.567 > scratch
  0.559 > SimMTM 0.544 > DeepConvLSTM 0.539 > FOCAL 0.457 > RelCon 0.355; finetune (MAE/5src only so far): 0.644 /
  0.641. Adding HAR moves probe #1 to MAE-noAA (mean rank 1.86 vs MAE 2.29); finetune #1 stays MAE-11src (1.71).
  30 baseline HAR finetunes pending on gengpu; large pretrains at 20k steps (PT-L val 0.0199, ART-L 0.0264).
* **20:05** — 30 HAR baseline finetunes were idle in gengpu's queue while 2 buy-in GPUs sat free → added a
  **twin guard** to every bench train script (exit if the cell's result JSON exists or a same-named job is RUNNING),
  cancelled the unguarded gengpu copies and resubmitted each cell on BOTH partitions (60 jobs; ids in
  `bench/generated/har_twin_submit.log`). Two already running on b53017. Large pretrains at ~30k/170k, healthy.
* **23:45 — HAR grid COMPLETE** (90 HAR cells + 6 DeepConvLSTM; gengpu twins cancelled). Finetune `HAR (6 datasets)`
  mean macro-F1: MAE 0.644 > 5src 0.641 > noAA 0.637 > SimMTM 0.595 ≈ acc 0.594 ≈ FOCAL 0.593 > RelCon 0.560
  (scratch 0.559). Finetuning barely moves the MAE variants (+0.00-0.01) but lifts the SSL baselines a lot
  (FOCAL +0.14, RelCon +0.21, SimMTM +0.05); PAMAP2 and HHAR are worse under finetune for every MAE row (dataset detail
  tables). Standings with 7 columns: probe #1 MAE-noAA 1.86 / MAE 2.29 / 5src 2.43; finetune #1 MAE 1.71 / 5src 2.14 /
  noAA 3.00 (MAE-11src #1 on 5 of 7 finetune columns). Board redeployed. Only Inertia-1 large still in flight
  (PT-L 73k, ART-L 78k of 170k; ~8 h).
* **2026-08-18 03:00 — Inertia-1 small HAR cells done (24; twins on gengpu cancelled).** HAR mean macro-F1:
  PatchTST-small probe 0.656 / finetune **0.713**, ART-small 0.688 / 0.686 — ABOVE every ViT1D row (MAE 0.635/0.644),
  driven mostly by MHEALTH (0.85-0.92 vs MAE 0.63-0.69) and HHAR finetune (0.785 vs 0.628). Likely protocol asymmetry,
  not a bug: our pipeline z-scores each window (removes the gravity/DC component) while the Inertia-1 adapter feeds
  raw g-scale acc — for posture-heavy label sets (MHEALTH standing/sitting/lying) the gravity direction is the
  discriminant. Worth an ablation (ViT1D probe with normalize.enabled=false) before drawing conclusions. Both small
  rows now 7/7; probe mean rank PT-S 4.57 / ART-S 5.29 (MAE variants 2.14-2.71); finetune ART-S 5.14 / PT-S 6.43.
  Board redeployed. Large pretrains ~110-115k/170k, ~5 h left.
* **08:55 — Inertia-1 large pretrains DONE**: PatchTST-L best val 0.01672 (small 0.01806, −7 %), ART-L 0.02460
  (small 0.02548), 14.3 h / 13.4 h at lr 2e-4. Sentinels fired; 24 chained cells running on b53017 (4 at a time,
  ART-L first). First cell: arm-pose ART-L probe.
* **2026-08-18 11:00 — PD primary metric switched to subject macro-F1** (bal-acc kept as secondary, new
  `n_classes_predicted` sanity flag; 27 result JSONs re-derived from their saved per-subject confusions, no re-scoring).
  Trigger: ART-large PD probe hit bal-acc 0.543 by NEVER predicting PD (0/58) — subject acc 31 %, macro-F1 0.30.
  Under macro-F1: finetune RelCon 0.619 > acc 0.606 > 5src 0.597 > MAE 0.542 > SimMTM 0.515 > noAA 0.458 > Inertia-1
  0.386 > FOCAL 0.331; probes 0.25-0.40 (all at/near chance; several predict only 1-2 classes). Selection is still on
  val window bal-acc (v1); switching selection to macro-F1 = next protocol change (needs re-runs, ~30 min each).
* **2026-08-18 11:40 — HAR normalisation diagnostic done** (`analysis/har_normalization_diagnostic.md`): Inertia-1's
  HAR edge vanishes when its input is per-window z-scored (MHEALTH 0.89→0.55, HHAR 0.71→0.50, WISDM 0.61→0.41);
  6 raw window stats alone = 0.69 on MHEALTH (= MAE probe); raw-input scratch ViT1D 0.73 on WISDM (> every MAE row).
  Verdict: mostly our per-window z-score deleting gravity/intensity, plus genuine NHANES pretraining value on raw
  posture structure (raw scratch 0.61 vs Inertia-1 0.89 on MHEALTH). Remedies: DC-preserving re-pretrain, or
  probe heads with raw window stats for all encoders.
* **2026-08-18 15:20 — Inertia-1 pretraining-vs-pipeline controls done** (12 cells, `inertia1_patchtst_scratch`
  / `inertia1_patchtst_randinit`, group=reference so they are shown but not ranked; leaderboard now maps any
  run-less encoder to the `scratch` protocol). HAR mean macro-F1: NHANES finetune 0.713 > NHANES probe 0.656 >
  **scratch same-pipeline 0.612** > our MAE probe 0.635/finetune 0.644 (z-scored) > our ViT1D scratch 0.559 >
  random-frozen 0.440. Split: +0.053 from the raw-g pipeline, +0.101 from NHANES pretraining, +0.216 pretraining
  value in linear separability. Also: all 24 Inertia-1-large cells scored (see below); 18 MAE+12-stats diagnostic
  runs finished, scoring job submitted.
* **2026-08-18 15:40 — step-1 fix works: MAE + 12 raw window stats.** `StatsConcatEncoder` (no re-pretraining;
  `stats_concat.enabled` + normalize off; wrapper z-scores internally so the pretrained encoder sees its usual
  input). HAR mean macro-F1: probe 0.635 -> **0.679** (+0.044, positive on all 6 sets, MHEALTH +0.104) which
  BEATS Inertia-1 PatchTST-S probe 0.656 and ties ART-S 0.688; finetune 0.644 -> 0.670 (+0.026, still −0.043 vs
  their 0.713). Scratch+stats −0.011 → the gain is specific to pretrained representations missing the DC term.
  Full table in analysis/har_normalization_diagnostic.md.

## NHANES-scale × normalisation 2×2 — launched 2026-08-18 16:45 CDT

Motivation: our whole 11-source corpus is **890 h** (NHANES slice 200 h) — hours = windows x 5.12 s because
hop = half of the 10.24 s window; the Inertia-1 repro used **650,000 h** of NHANES = **730x our corpus**. So the HAR
comparison carries a large data-scale asymmetry, and the diagnostic showed the missing DC/gravity term
is exactly what NHANES is rich in.

| | 400 h NHANES | 20,000 h NHANES |
|---|---|---|
| per-window z-score | **A** | **B** |
| fixed physical scaling | **C** | **D** |

* All arms: same 11-source pool, `sampling_mode: balanced` with `source_weights {nhanes: 3.67}` →
  NHANES = **25 % of sampled mass** in every arm (acc-only total ≈32 %, well under the 50 % cap), so
  A→B changes only how much *unique* NHANES data exists, not the mixture. `epoch_windows: 200000`
  (new option; balanced epochs were otherwise sized by the 137-window imuposer source).
* New code: `preprocess.normalize.method: physical` (DC-preserving: acc/9.81 g, gyro/5 rad·s⁻¹, ±8 clip);
  `objective.modality_loss_weights {acc: 1.0, gyro: 2.0}` + `modality_loss_lambda` with **acc/gyro MSE
  always logged separately** (`acc_mse`, `gyro_mse`, `acc_frac`, `gyro_frac`).
* Budget 100k steps/arm; jobs 9804647-9804663 twinned on gengpu+b53017 (A/C now, B/D `afterok` on the
  cache build **9804419** → `/projects/b53017/vbc1591/mae_multidata_cache_nhanes20k`, ~43 GB, non-NHANES
  arrays hard-linked from the existing cache).
* Readout: B−A = data scaling, C−A = normalisation, D−(B+C−A) = interaction. Evaluate all 7 task
  families; acceptance ΔHAR > 0 with Δpose ≈ Δodom ≈ 0, Δgesture ≥ 0.

* **18:30 — LEAKAGE FIX (user-flagged).** The `realdata` MAE source is the master parquet filtered to 7
  six-axis HAR sets — and **4 of them (pamap2, recofit, utd_mhad, wisdm) are also HAR *evaluation* sets**
  (EXPERIMENTS.md:5104). Every historical MAE row therefore pretrained on 4 of the 6 HAR eval datasets;
  Inertia-1 never saw any of them, so the HAR comparison was tilted the other way. Fix: rebuilt the source
  from the 3 non-eval datasets only (samosa, shoaib, ut_watch) into `realdataclean_*.npy` (job 9812944, both
  caches), added a `cache_name` option so a source can read a differently-named cached slice, and pointed all
  four 2x2 arms at it. HHAR and MHEALTH were always clean. ⚠ The board's HAR column for every ViT1D row is
  still from leaked pretraining — must be stated, and ideally re-run once a clean encoder exists.
* NHANES cache build OOM-killed at 180 GB (it collected the full 20,000 h = 14.06M windows = 86.4 GB buffer,
  then died saving); resubmitted with 420 GB / 32 cpus (job 9813212). Arms A/C chained on the clean-realdata
  build, B/D on both.
* **19:05 — 12-hour plan.** Pretrain budget cut 100k -> **50k steps** (~3.2 h/arm at 4.3 steps/s; early stopping
  patience 999999 so the budget is fixed but best.pt is still written). Clean-realdata cache DONE (job 9812944:
  38,024 windows = **54 h**, down from 133 h — that is the leakage removed). NHANES 20k-h cache rebuilding with
  420 GB (job 9813212, ~1 h in). Wave-1 evaluation is pre-chained per arm (jobs 9818050/9818175/9818177/9818185):
  6 HAR probes + arm-pose / full-body / gesture probes = 9 cells x 4 arms = 36 cells, each twinned on both
  partitions, downstream preprocessing matched to the arm's own normalisation. Odometry / language / PD and all
  finetunes are WAVE 2 (odometry probes alone are ~5 h and do not fit the 12 h window).
* **21:30 — LEAK-FREE POOL (user decision).** All four 2x2 arms restarted on **6 sources only**: OCDetect,
  EatingSpeed, MotionX, SONIC, Capture24, NHANES. Disabled: `nymeria` (= arm-pose/full-body/odometry eval,
  was 27.7 % of the pool), `openwatch` (= microgesture eval), `humanml3d` (= language eval), `realdata`
  (HAR eval sets, already trimmed this morning), plus totalcapture/imuposer. **No evaluation dataset now
  appears anywhere in pretraining** — the first pool in this project for which that is true, and the only
  version whose comparison against Inertia-1 (NHANES-only) is apples-to-apples.
  Pool: 369,090 windows total; balanced sampling with `source_weights {nhanes: 1.667}` -> NHANES 25 % of mass,
  capture24 15 % (acc-only total 40 %, under the 50 % cap). Arms A/C draw from the 200 h NHANES cache, B/D from
  the 20,000 h cache. Jobs 9828427-9828434 (twinned), evals 9828447-9828451.
  Wave 1 = **all 6 HAR datasets** (wisdm, utd_mhad, pamap2, recofit, hhar, mhealth) + arm-pose/full-body/gesture
  probes, which are now genuinely out-of-domain regression checks. Expect pose/odom/gesture/language to drop
  vs the board rows -- that drop IS the measurement of how much of our lead was in-domain pretraining data.

## Full-size pretraining cache + diversity-balanced mixture — launched 2026-08-19 15:10 CDT

**Why:** the existing cache was truncated for the two largest real acc+gyro sources — the stage-2 build
log recorded OCDetect 1,253,077 windows (1,782 h) and EatingSpeed 575,356 (818 h), but the .npy files hold
8,703 (12 h) and 57,536 (82 h). SONIC/MotionX matched exactly, so this is a truncated/partial write, not a
configured cap. Every MAE trained in this project (including the board's mae_11src) therefore saw ~0.7 % of
OCDetect.

**Build (job 9919355, `mae_multidata_cache_full`):** all seven sources with **every cap removed**
(max_hours / max_files / sample_fraction deleted): OCDetect 1,782 h, EatingSpeed 818 h, Capture24 2,038 h,
SONIC 46 h, MotionX 7 h, TotalCapture 1 h — ~4,700 h of non-NHANES data (~3.3M windows, ~41 GB), vs 325 h
before. NHANES capped at **60,000 h**: the full 185,585 h is ~130M windows (~1.6 TB fp32, ~19 h build) and a
50k-step run draws only 6.4M windows, so past ~9,000 h nothing is even seen once; 60,000 h leaves headroom
for a longer schedule. 32 cpus / 420 GB / 2-day limit.

**Balancing (job chained afterok, `analysis/diversity_saturation/source_coverage.py`):** the saturation study
found the pool's intrinsic motion diversity is fully covered by ~0.5 % of windows (coverage 0.937 at 0.1 %,
0.999 at 0.5 %), so hours are the wrong currency for mixture weights. The new script k-means the pooled
descriptor space (K=1000) and scores each source by clusters covered and by clusters it covers *exclusively*,
then emits `suggested_source_weights` (weight = coverage + 2x exclusivity, normalised so 1.0 = equal share)
to drop straight into `objective.*.source_weights`.
* **19:30 — full cache: the truncation is fixed.** `mae_multidata_cache_full` now holds every hour of the
  non-NHANES sources: OCDetect **1,782 h** (was 12), Capture24 **2,038 h** (was 178), EatingSpeed **818 h**
  (was 82), SONIC 46 h, MotionX 7 h, TotalCapture 1 h = **4,692 h** vs 325 h before (14x). NHANES is still
  streaming (545/13,224 participants, ETA ~32 h for the 60,000 h cap) — but the 2x2 measured only +0.017 from a
  20x NHANES increase, so rather than wait I assembled `mae_multidata_cache_full_now` = the six finished sources
  hard-linked + the existing 4,000 h NHANES slice = **8,692 h**, usable immediately. Diversity/coverage analysis
  submitted on it to set the mixture weights.

## MAE recipe ablation — launched 2026-08-19 22:45 CDT

Three defects in the current recipe, fixed one at a time on the 8,692 h diversity-balanced cache
(`mae_multidata_cache_full_now`, `source_weights` from the cluster-coverage analysis), 50k steps each:

| arm | change |
|---|---|
| **R0** | baseline = today's recipe (octahedral rotation, `full_mse`, lr 1e-4) |
| **R1** | + **yaw-only rotation** (new `rotation_set: yaw`, rotates about the gravity axis only) + **physical scaling** (DC preserved) + **lr 3e-4** |
| **R2** | R1 + **band-weighted loss** (`freq_weighted`, bands 0-1/1-3/3-8/8-25 Hz weighted 4/2/1/0.5 — low-frequency emphasis) |
| **R3** | R2 + **modality masking** (`modality_mask_prob 0.3`: whole acc or whole gyro hidden, forcing cross-modal prediction) + `modality_loss_mode: replace` so the acc/gyro weights are the objective rather than a second MSE on top |

New code: `rotation_set: yaw|gravity_preserving` in `augmentations/transform.py` (verified: octahedral gives
gravity-Y std 5.80 across draws, yaw gives 0.00); `objective.modality_mask_prob` and
`objective.modality_loss_mode` in `masked_1d_freq_channel_mask.py`.

Rationale: octahedral rotation randomises gravity direction, i.e. it explicitly trains posture-agnosticism —
the cue the 2x2 showed HAR needs most (+0.10 from probe-time DC preservation). `full_mse` on z-scored input
spends its budget on high-frequency detail while the DC term is not even in the input. lr 1e-4 at batch 128 is
below the linear-scaling convention and left arms A and C at *identical* val loss (1.164819 vs 1.164633) after
50k steps, i.e. undertrained. Jobs 9948123-9948131 (twinned).
* **2026-08-20 01:45 — unit bug found and fixed before the recipe arms could run on it.** The pool mixes
  accelerometer units: **Capture24 and NHANES store g** (per-window |mean acc| ~1.0) while OCDetect,
  EatingSpeed, SONIC, MotionX and TotalCapture store **m/s^2** (~8.7-9.7). Per-window z-scoring had hidden this
  for the entire project; `normalize.method: physical` divides by a fixed 9.81, so 48 % of the pool (245 h of
  345 h) would have entered the model 9.81x too small. `CachedWindowDataset` now detects the unit from the data
  (or `sources.<name>.acc_units: g|m/s2`) and converts to m/s^2 before normalising. Verified: after the fix all
  sources agree at |mean acc| 0.78-1.00 g. Recipe arms restarted.
  Ablation pool (raw cache): ocdetect 100 h, eatingspeed 147 h, capture24 145 h, nhanes 100 h, sonic 14.2 h,
  motionx 6.7 h, totalcapture 1 h = **514 h**, 30k steps, ~11 passes.

## Mask-ratio ablation — launched 2026-08-20 10:20 CDT

**Question:** what `objective.mask_ratio` should the MAE pretrain use? Never answered so far: the
only prior evidence is two 300k-step Stage-1 arms (mr 0.50 / 0.90, jobs 8818231/8818232, both
COMPLETED 2026-08-09) whose logged val losses are *not comparable* — the MAE loss is taken over
masked patches only, so a 0.90 arm is scored on a strictly harder problem than a 0.50 one.

**Design.** Six arms on the leak-free 514 h raw cache, 30k steps, seed 42, everything except
`objective.mask_ratio` byte-identical to `configs/nhanes_scale/mae_recipe_R0.yaml` (the recipe
ablation's baseline arm), so the 0.75 point is **runs/mae_recipe_R0 itself** — not a retrain.

| ratio | 0.40 | 0.50 | 0.60 | 0.75 | 0.85 | 0.90 |
|---|---|---|---|---|---|---|
| run | mae_mr40 | mae_mr50 | mae_mr60 | mae_recipe_R0 | mae_mr85 | mae_mr90 |
| job | 9972195/6 | 9972197/8 | 9972199/200 | (done) | 9972201/2 | 9972204/5 |

Equal steps really is equal compute here: `ViT1DEncoderAltAttn._use_bert_mask = True`, so every
token goes through the encoder (masked ones replaced by a learned mask token) and per-step FLOPs
do not depend on the mask ratio — unlike token-dropping MAEs, where a 0.9 arm would get 6x more
steps for the same budget.

**Evaluation** = frozen-encoder probes through the bench harness: the 6 leaderboard HAR sets +
arm-pose (Nymeria) + microgesture (OpenWatch), 8 cells per arm, 4 cells per GPU
(`bench/generated/pack_maskratio_cells.py`, jobs `mrpk*` 9972277-9972306, chained afterany on
each arm's pretrain). Encoders `mae_mr40/50/60/85/90` are registered in `bench/registry/encoders.yaml`;
the 0.75 row is read from the `mae_recipe_r0` cells.

**Second, independent evidence stream:** `rescore_mask_ratio.py` (job 9972314/7) rescores the three
300k-step Stage-1 arms (0.50/0.75/0.90) at ONE fixed eval mask ratio — run at 0.50, 0.75 and 0.90 so
the ranking cannot be an artifact of the ratio we score at. This is the "⏸ written, not run" item
from section C above.

**Harvest:** `python analysis/harvest_mask_ratio.py` -> `analysis/mask_ratio_ablation.md`
(per-task table, HAR mean, best arm, plus the fixed-ratio rescore table; lists missing cells).

**Result (2026-08-20 18:15) — six of seven arms scored.** Frozen-probe HAR mean (6 sets):

| ratio | 0.40 | 0.50 | 0.60 | 0.75 (production) | 0.85 | 0.90 |
|---|---|---|---|---|---|---|
| HAR mean | 0.539 | 0.532 | **0.548** | 0.518 | 0.505 | 0.497 |
| gesture clip-F1 | 0.536 | 0.501 | 0.550 | 0.503 | 0.541 | 0.490 |
| arm pose (cm) | 8.46 | 8.41 | 8.53 | 8.45 | 8.47 | 8.53 |

Shape: a **plateau over 0.40-0.60** (0.532-0.548) and a monotone decline from 0.75 on
(0.518 -> 0.505 -> 0.497). The plateau's internal ordering is not resolvable at one seed
(0.50 dips below both neighbours => cell noise ~±0.01, the size of the within-plateau gaps),
but plateau-vs-0.75+ is ~4x that. Arm pose is flat across every arm (8.41-8.53 cm) and
carries no signal. **Recommendation: move the production mask ratio from 0.75 to 0.5-0.6
(~+0.02-0.03 HAR macro-F1); 0.90 is the worst setting tested.** A 0.30 arm (job 3621841)
is queued to locate the plateau's lower edge; a seed replicate is what would separate
0.40/0.50/0.60 from each other.

**The reconstruction signal points the other way and must not be used to choose.** Rescoring
all six arms at fixed evaluation ratios (0.5/0.75/0.9) puts 0.85 and 0.90 *best* at every
evaluation ratio (e.g. at 0.75: 0.382/0.401 vs 0.485-0.525 for the 0.40-0.75 arms) — exactly
the arms that transfer worst. MAE loss is anti-correlated with probe accuracy here even after
the fixed-ratio correction, so probes are the only valid selector.

## 2026-08-22 — diversity-selected raw pool + 60 % mask DC-preserving pretrain
- Raw (unnormalised) full-size cache rebuilt: Capture24 `all_windows` (+1,084 h), EatingSpeed right wrist only, NHANES 4,000 h spread over 4,629 participants (was the first ~30), Motion-X v1 (4 non-overlapping subsets) added, short mocap clips recovered.
- Found & fixed synthetic-IMU spikes (MotionX 27 % of windows > 10 g, max 97,808 m/s²): `imu_lm/data/synthetic_imu_sanitize.py`, both synthetic loaders rebuilt.
- Diversity analysis (`analysis/pool_diversity.py`, `analysis/nhanes_diversity.py`, reports under `analysis/pool_diversity/out_*`, `analysis/subset_selection/`): breadth from free-living, exclusive motion types from SONIC/EatingSpeed/MotionX; balanced selection keeps 100 % of motion types at 1-2 % of hours.
- Pretrain `mae_div60_dc` (job 3846391): 853 h train / 94 h val — nhanes 54, capture24 60, ocdetect 300, eatingspeed 150 (balanced subsets), sonic 216, motionx 72 full. HAR probes chained (d60pk0-2).
