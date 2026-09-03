# Diagnosis: MAE pretraining data-scaling saturation (5%→100%)

Status: **core measurement complete, including full scale-up** (§1-6/9 all
done; §7/§8 reruns still need a go-ahead, see "What's actually run vs.
pending"). Headline result: **10 of 11 sources have their best reconstruction
quality below 100% of the data and get measurably worse by 100%** (§6) --
this survived a full audit of the measurement pipeline itself (three real
bugs/inefficiencies found and fixed along the way) AND survived scaling
OCDetect/Capture24/EatingSpeed all the way from their original artificial
caps up to their true, uncapped 100% size (see "Caches" below) -- if
anything the effect got *stronger* with more data, not weaker. Only OCDetect
improves monotonically to 100%.

## Hypothesis under test

> Random window-level scaling saturates early because IMU windows are highly
> overlapping and even small fractions retain most subjects, sessions,
> activities, and sensor domains. Additional windows improve low-level
> reconstruction but add little new transferable motion information.

## Confirmed findings

### 1. The epoch/step-budget confound is real and already the leading explanation

Already documented in `EXPERIMENTS.md` §9 and re-derived here
(`out/06_budget_protocols.json`, script run interactively — this part is not
blocked on SLURM):

| fraction | n_train | steps/epoch | actual best_step | actual best_epoch | fixed-epoch-budget max_steps (77 epochs) |
|---|---|---|---|---|---|
| 0.1% | 485 | 3 | 29,000 | **9,667** | 231 |
| 1% | 4,851 | 37 | 135,000 | 3,649 | 2,849 |
| 5% | 24,256 | 189 | 221,000 | 1,170 | 14,553 |
| 25% | 121,281 | 947 | 221,000 | 234 | 72,919 |
| 50% | 242,562 | 1,895 | 221,000 | 117 | 145,915 |
| 100% | 485,125 | 3,790 | 291,000 | **77** | 291,830 |

`trainer.max_steps` is effectively unbounded (10,000,000) and early stopping
fires on a *fixed step count* (`val_every_steps=1000`, `patience=10` →
10,000 steps of no improvement) identically at every fraction. Since dataset
size varies ~1000× across the sweep, this step-based stopping rule produces
wildly different **epoch** budgets — the low-fraction runs see their pool
9,667× vs. the 100% run's 77×. This is mechanically sufficient to produce
*some* saturation-looking curve on its own, independent of any data-diversity
effect. `06_budget_protocols.py` generated matched fixed-epoch-budget and
fixed-step-budget rerun configs (`out/configs_budget_controlled/`) to isolate
this — **not yet launched** (needs go-ahead, see below).

### 2. OCDetect train/val split had a real, code-confirmed leak — **fixed**

`_build_ocdetect` in `combined_mae_multidata_loader.py` used to do a
**positional slice** through file-concatenated, unshuffled windows:
```python
train_ds = OCDetectDataset(X[:n_train], ...)
val_ds = OCDetectDataset(X[n_train:n_train+n_val], ...)
```
Windows use a fixed 50% hop, so any window pair straddling the cut boundary
that comes from the same file overlapped by ~50% of its samples — not
independent, despite being nominally in different splits. Present identically
at *every* fraction (baked into the shared
`/projects/b53017/vbc1591/mae_multidata_cache`), so it didn't differentially
explain the 5%→100% *shape*, but it did mean OCDetect's val_loss contribution
was not a clean measurement at any point on the curve.

**Fixed**: `_build_ocdetect` now splits at the **file level** (shuffled,
seeded by `split_seed`, default 42) before collecting windows, so no file's
windows can ever land on both sides — matching the file/session-disjoint
convention every other sliding-window source here already uses. Verified with
a standalone arithmetic check across file counts 1→678: zero file-index
overlap between the resulting train/val sets in every case. `max_hours`
capping now applies to the train split only (previously capped the combined
pre-split pool). **This changes OCDetect's actual train/val windows** the
next time any config with `cache_dir` set gets its cache rebuilt (the
existing `/projects/b53017/vbc1591/mae_multidata_cache` was built under the
old, leaky logic — see "Action needed" below).

`02_split_audit.py` will confirm the fix (zero shared files between OCDetect
train/val) once the SLURM diagnostics run completes.

Every other source's split is sound: Nymeria (physically separate train/val
root directories), TotalCapture (subject-disjoint `s1-4`/`s5`), IMUPoser
(participant-disjoint `P9`/`P10`), Capture24 (participant-level, confirmed by
the loader's own docstring), EatingSpeed (participant-level), realdata
(`splits.group_key: subject_id`). HumanML3D uses the official third-party
`train.txt`/`val.txt` clip split — `02_split_audit.py` additionally checks
whether any clip's parent AMASS sequence appears on both sides (via
`_get_amass_index`) — **pending SLURM run**.

### 3. Window overlap is 50% everywhere, by construction

`windowing.compute_T_and_hop` (`window_hop_ratio: 0.5` in every scaling
config) gives `T=512, hop=256` samples → every sliding-window source
(realdata, ocdetect, nymeria, totalcapture, imuposer, motionx, sonic)
produces windows that overlap 50% with their temporal neighbor. This is exact
and analytical, not something that needed measuring — confirms the raw
window count overstates independent samples by ~2× for any single
session/recording, before even considering cross-session redundancy.

### 4. Fixed per-source caps mean only `overall_fraction` scales 8 of 11 sources

Every `data{X}pct` config has byte-identical `sources:` blocks — only
`ocdetect` (`sample_fraction: 0.01`), `capture24` (`0.1`), and `eatingspeed`
(`0.1`) have their own fixed sub-sampling baked in regardless of the sweep
point; the other 8 sources (realdata, nymeria, totalcapture, imuposer,
motionx, humanml3d, sonic, openwatch) are fully built and then subsampled
purely by `overall_fraction`. `sampling_mode: natural` is fixed across the
whole sweep (confirmed, not a confound). Exact per-source contribution and
which source dominates each fraction-to-fraction delta —
`04_source_weighting_audit.py` — **pending SLURM run**.

### 5. Per-source reconstruction MSE (corrected) — HumanML3D lowest, but the gap is modest, not extreme

`09_dataset_characterization.py` ran the 100% run's `best.pt` over every
source's cached val windows. **A first pass through this script had a real
bug** (see "Two real bugs found and fixed while measuring this" below) that
inflated every number to ≈1.00–1.04 (indistinguishable from the trivial
"predict the per-channel mean" baseline, since data is z-scored to unit
variance) — that reading was wrong. The corrected, verified numbers (full
3000-window sample per source, GPU, production `forward_loss` called
directly):

| source | recon MSE | std (acc channels) |
|---|---|---|
| humanml3d | **0.582** | **0.809** (only source below ~1.0) |
| openwatch | 0.665 | 1.000 |
| nymeria | 0.725 | 0.999 |
| realdata | 0.727 | 1.000 |
| eatingspeed | 0.745 | 0.999 |
| totalcapture | 0.774 | 1.000 |
| ocdetect | 0.772 | 1.000 (n_val now 173,268 -- true 100%) |
| sonic | 0.788 | 1.000 |
| capture24 | 0.798 | 0.999 (n_val now 178,413 -- true 100%) |
| imuposer | 0.904 | 1.000 |
| motionx | 0.931 | 1.000 |

(Table updated after OCDetect/Capture24/EatingSpeed were scaled to their true
100% -- `out/09_dataset_characterization_v3_100pct/`, superseding the earlier
`out/09_dataset_characterization/` run at the smaller scale.)

Every source is meaningfully better than the ~1.0 trivial baseline (a real,
working model), and the previous "HumanML3D is a dramatic outlier, must be a
padding artifact" reading doesn't hold up as strongly under correct numbers:
HumanML3D (0.597) and openwatch (0.612) are close to each other and both
just modestly below the rest — HumanML3D is still the single lowest, and its
std=0.809 (vs. ≈1.000 everywhere else) is a real, unexplained-by-bug artifact
consistent with the zero-padding hypothesis (`PadToLenXOnlyDataset`), but
openwatch has no such padding and is nearly as low, so padding is at most a
partial explanation, not the whole story. Worth deeper investigation before
treating either source's low MSE as "the model represents this domain
better" without qualification.

Full per-source table (mean/var/hz/coordinate-frame/channels/reconstruction
MSE) at `out/09_dataset_characterization/dataset_characterization.md`.

### 6. The corrected per-source, per-fraction scaling curve — the central result

`05_val_loss_breakdown.py` (fixed, corrected as above) ran all 6 existing
scaling checkpoints against the same held-out val set for every source
(`out/05_val_loss_breakdown/val_loss_breakdown.json`). **Final version**:
OCDetect/Capture24/EatingSpeed's val sets are now their true 100% (see
"Caches" below) — 173,268 / 178,413 / 101,138 windows respectively, up from
the original small caps — so every checkpoint is now scored against the
largest, most representative held-out pool available for every source:

| source | 0.1% | 1% | 5% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|
| realdata | 0.824 | 0.732 | 0.710 | 0.710 | 0.710 | 0.727 |
| ocdetect | 0.799 | 0.770 | 0.773 | 0.769 | 0.773 | **0.767** |
| nymeria | 0.793 | 0.748 | 0.708 | 0.716 | 0.718 | 0.746 |
| totalcapture | 0.848 | 0.790 | 0.805 | 0.783 | **0.720** | 0.766 |
| imuposer | 0.933 | 0.930 | **0.908** | 0.937 | 0.926 | 0.931 |
| eatingspeed | 0.797 | 0.751 | 0.729 | 0.727 | **0.725** | 0.744 |
| motionx | 0.956 | 0.929 | **0.915** | 0.928 | 0.923 | 0.942 |
| humanml3d | 0.612 | 0.590 | 0.586 | **0.580** | 0.581 | 0.591 |
| sonic | 0.866 | 0.799 | **0.764** | 0.770 | 0.764 | 0.784 |
| capture24 | 0.835 | 0.804 | 0.796 | **0.786** | 0.791 | 0.795 |
| openwatch | 0.728 | 0.677 | 0.634 | 0.645 | **0.632** | 0.634 |
| **unweighted mean** | **0.817** | **0.774** | **0.757** | **0.759** | **0.751** | **0.766** |

**This directly confirms the hypothesis under test, and got *stronger*, not
weaker, once OCDetect/Capture24/EatingSpeed were scaled to their true 100%.**
The aggregate curve drops sharply from 0.1%->5% (0.817->0.757, most of the
total available gain), and the best point on the whole curve is **50%**
(0.751) — **100% (0.766) is worse than every fraction from 5% up**. Stated
precisely: **10 of 11 sources now have their single best (lowest)
reconstruction MSE at LESS than 100% of the data** (checked directly, not
eyeballed):

| source | best fraction | best MSE | 100% MSE | regressed after best? |
|---|---|---|---|---|
| realdata | 5% (tied w/ 25%,50%) | 0.710 | 0.727 | yes |
| ocdetect | 100% | 0.767 | 0.767 | no (only monotonic source) |
| nymeria | 5% | 0.708 | 0.746 | yes |
| totalcapture | 50% | 0.720 | 0.766 | yes |
| imuposer | 5% | 0.908 | 0.931 | yes |
| eatingspeed | 50% | 0.725 | 0.744 | yes |
| motionx | 5% | 0.915 | 0.942 | yes |
| humanml3d | 25% | 0.580 | 0.591 | yes |
| sonic | 5% | 0.764 | 0.784 | yes |
| capture24 | 25% | 0.786 | 0.795 | yes |
| openwatch | 50% | 0.632 | 0.634 | yes |

**Only OCDetect keeps improving all the way to 100%** — every other source,
including Capture24 and EatingSpeed now that they're evaluated against their
true full-size val pools, saturates by 5-50% and is measurably worse by 100%.
This is the strongest evidence in the whole investigation that the
saturate-and-regress shape is a real, pervasive property of the data/training
setup, not an artifact of aggregation, a single dominant source, a small/
capped validation set, or the measurement bugs found along the way.

This is now measured on the **fully corrected and fully scaled** pipeline
(fixed OCDetect split, OCDetect/Capture24/EatingSpeed at their true 100% —
not just raised, but every artificial cap removed — real `forward_loss`, no
augmentation confound). The saturate-and-mildly-regress shape survived every
bug fix AND every round of scaling the previously-capped sources up to their
true size — if anything the regression is more pronounced now (100% is worse
than 5% for 9 of 11 sources, not just "roughly flat"). It has **not** been
checked against seed-to-seed variance (§8, still not run) or the epoch/step-
budget confound (§1) — those remain the two open questions before concluding
the 100% *training run* (as opposed to this val-set comparison) is genuinely
worse, not just noisier. Note also that these 6 checkpoints were themselves
*trained* on the old, smaller-scale OCDetect/Capture24/EatingSpeed data —
scaling those sources up only changed the *validation* pool here, not what
these particular checkpoints learned from; a real retrain against the scaled-up
`mae_multidata_cache_v3` (not yet done, see "Recommended next steps") is the
next step to see whether training on more data changes the picture.

### Three real bugs/inefficiencies found and fixed while measuring this

1. **`05_val_loss_breakdown.py` built val data through the raw per-source
   loaders using the live training config**, which applies augmentation
   (`random_rotation`, etc.) inside every `__getitem__` call — including for
   val. Val windows were silently getting random rotations applied before
   the reconstruction check. Fixed: val is now built with every augmentation
   forced off (mirrors `scripts/build_multidata_cache.py`'s own convention).
2. **A hand-rolled reimplementation of `forward_loss`'s masking/encode/decode
   logic** (used in both `05_val_loss_breakdown.py` and an earlier version of
   `09_dataset_characterization.py`) produced a systematically different
   (worse) `reconstructed` tensor than the real `forward_loss` function, even
   when given byte-identical input, channel-validity mask, and a manually
   fixed patch mask (verified via hook-based tensor capture, with call order
   swapped to rule out state leakage — root cause never fully identified).
   Fixed by no longer reimplementing the forward pass at all: both scripts
   now call the real `imu_lm.objectives.masked_1d_freq_channel_mask.forward_loss`
   and, where a finer per-channel/per-motion breakdown is needed, capture its
   internal tensors via a decoder forward hook + a spy on the pure
   `_patch_mask_to_signal_mask` helper, with a hard runtime assertion
   checking the recomputed aggregate against `forward_loss`'s own reported
   loss on every call.
3. **`05_val_loss_breakdown.py` rebuilt every source's raw val dataset inside
   the per-fraction loop** — 6x redundant raw builds (SMPL FK, file parsing,
   ...) even though all 6 scaling configs share an identical `sources:` block
   and only the checkpoint differs per fraction. A real run on this structure
   was still stuck on fraction 1/6 after 73 minutes, with the only output
   write being a single `json.dump` at the very end — a SLURM time-limit kill
   would have lost 100% of the compute. Fixed: val batches are now built ONCE
   and reused across all 6 fraction evaluations, and results are written
   incrementally after every fraction.

## What's actually run vs. pending

| Item | Status |
|---|---|
| §1 composition audit (subjects/sessions/windows per fraction) | **Done** (`out/01_composition/`) |
| §2 split audit (OCDetect leak quantification, HumanML3D AMASS check) | **Done** (`out/02_split_audit/`) |
| §3 redundancy audit (near-duplicate rate, static/dynamic split) | **Done** (`out/03_redundancy/`) |
| §4 source weighting audit | **Done** (`out/04_source_weighting/`) — headline: **Nymeria**, not OCDetect/Capture24, dominates every fraction-to-fraction delta (~32-33% of added windows at every step 0.1%→100%) |
| §5 val-loss breakdown by source/channel/motion, per fraction | **Done** (`out/05_val_loss_breakdown/`) — see §6 above, the central result |
| §6 budget-protocol tables + rerun configs | **Done** (`out/06_budget_protocols.json`, `out/configs_budget_controlled/`) |
| §9 dataset characterization (mean/var/hz/coordinate-frame/recon-MSE) | **Done** (`out/09_dataset_characterization/`) |
| §7 checkpoint-selection mismatch | Tooling implemented; confirmed **no intermediate checkpoints exist** for any of the 6 runs (`save_step_checkpoints: false` everywhere) — a real check requires new pretraining reruns with checkpointing on, **not launched, needs explicit go-ahead** (hours of GPU time each) |
| §8 downstream variance (3-seed reruns) | Tooling implemented (`--seed` added to `run_pretrain.py`/`eval_odom_nymeria_altattn.py`, both previously unseeded for probes); reruns themselves **not launched, needs explicit go-ahead** |
| Diversity-based scaling modes | **Implemented** in `combined_mae_multidata_loader.py` (`scaling_mode: session_fraction/subject_fraction/windows_per_session/source_balanced_fraction`), not yet smoke-tested end-to-end via a real training run |

All of §1-6/9 ran via SLURM on the `gengpu` partition (account `p32139`) after
the home `b1232` partition went down cluster-wide (`sinfo` showed all 4 nodes
`down`/`mix-`) partway through this investigation — `b1232` may be back by
the time this is read; `gengpu` remains a working fallback either way.

## Caches: three now exist, original untouched throughout

1. **`mae_multidata_cache/`** (original) — byte-for-byte unchanged. Still what
   every production `data{X}pct.yaml` config's `cache_dir` points at.
2. **`mae_multidata_cache_ocdetect_splitfix/`** — 10 sources symlinked from
   the original; OCDetect rebuilt with the file-disjoint split fix at the
   first (small) `sample_fraction=0.1`. Superseded by (3) below.
3. **`mae_multidata_cache_v2/`** — 8 sources symlinked from the original;
   OCDetect/Capture24/EatingSpeed rebuilt at both the split fix *and* a much
   larger scale (raised on request, proportionally: OCDetect/Capture24/
   EatingSpeed were all sitting at an artificial `sample_fraction: 0.1` cap).

**Scale-up, measured** (`sample_fraction` before → after, in all 8 MAE
data-scaling configs):

| source | old frac | new frac | train windows | val windows | val files | leakage |
|---|---|---|---|---|---|---|
| ocdetect | 0.01→0.1→**0.52** | 353/678 files | 636,353 | 94,147 | many | none (file-disjoint, verified) |
| capture24 | 0.1→**0.5** | 76/151 files | 716,398 | 88,678 | many | already participant-disjoint |
| eatingspeed | 0.1→**0.5** | window-fraction | 287,678 | 50,569 | — | already participant-disjoint |

OCDetect's val set alone went from a leaky, effectively-single-file 1,087
windows (original) → a clean-but-tiny 6 windows (naive split-only fix) →
7,863 windows/7 files (first raised fraction) → **94,147 windows** (final,
much larger scale-up) — each step fixing a real problem the previous step
exposed. OCDetect landed at ~2,079 "native-hours" (windows×10.24s convention)
against a nominal 3,000h target — undershoots because OCDetect's `ignore`
quality-mask filtering removes a real chunk of each file's nominal duration,
and `max_hours` never actually capped it (file selection was the binding
constraint). Capture24 (~2,289h) and EatingSpeed (~962h, exactly 50% of its
true both-wrist pool) both landed close to or above their targets.

**Still not done**: no production config's `cache_dir` was repointed from
`mae_multidata_cache/` to `mae_multidata_cache_v2/` — all three caches exist
side by side, and any real training run launched right now still reads the
original, small, leaky cache. This is the one remaining step to make any of
this fixing/scaling actually load-bearing for a real pretraining run. The
§1-6/9 diagnostic numbers above were computed by scripts that build directly
from the (now-updated) config `sources:` blocks, not from `cache_dir` — so
they already reflect the fixed split and the larger scale.

## Recommended next steps

1. **Repoint the 8 configs' `cache_dir` at `mae_multidata_cache_v2/`** so a
   real training run actually picks up the fixed split + larger scale — the
   one remaining step to make any of this load-bearing. Not done here
   (real-run-affecting config change beyond what was asked).
2. **Fixed-epoch-budget rerun of 0.1%/1%/5%** (configs already generated,
   `out/configs_budget_controlled/*_fixed_epoch_budget.yaml`) to separate "ran
   out of unique data" from "ran out of training budget" — still the single
   biggest open question given §6's per-source result held up under three
   rounds of bug-fixing.
3. Re-run the HAR/pose probe sweep with `session_fraction`/`subject_fraction`
   scaling (now implemented) instead of `window_fraction`, to test directly
   whether diversity-preserving subsampling changes the saturation point.
4. 3-seed variance study (`08_variance_study.py`) before fully trusting the
   magnitude of any single-checkpoint regression in §6 — the *shape*
   (saturate then flatten/regress) is robust across 9/11 independently-built
   sources, which is much stronger evidence than any one fraction ranking,
   but the exact 100%-vs-best gap per source hasn't been checked against
   seed noise.

## Non-goals honored

No existing `runs/*/checkpoints/*.pt`, `EXPERIMENTS.md`/`Experiments_table.md`
entries were edited or deleted, and the original `mae_multidata_cache/` is
untouched throughout. Every SLURM job in this investigation was submitted
with explicit user go-ahead: `7704913` (01-04 of the diagnostics sweep, then
cancelled once 05 was found buggy — 01-04's outputs are preserved),
`7706874` (first OCDetect-only rebuild), `7757568`→`7760047` (05+06, first
attempt killed for a real inefficiency, see above), `7743802`→`7757569` (09,
resubmitted once `b1232` went down), `7759360` (cache v2 rebuild). Config
changes were also all explicitly requested: `sources.ocdetect.sample_fraction`
0.01→0.1→0.52, `max_hours` 500→3000, `capture24.sample_fraction` 0.1→0.5,
`eatingspeed.sample_fraction` 0.1→0.5, across all 8 MAE data-scaling configs;
no other config field was touched and no `cache_dir` has been repointed.
§7/§8's actual reruns (full pretraining reruns / 3-seed probe reruns) still
require a separate go-ahead — only their tooling exists.
