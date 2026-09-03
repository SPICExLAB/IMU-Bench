# MAE pretraining data-scaling diagnostics

Investigates why increasing MAE pretraining data from 5%→100% barely moves
reconstruction loss or downstream probe accuracy (`EXPERIMENTS.md` §9). See
`DIAGNOSIS.md` for the write-up; this file is just "how to run things."

Nothing here modifies `runs/`, `configs/`, or any canonical checkpoint. All
outputs land under `out/`.

## Order of operations

**Run via SLURM, not interactively** — `01_composition_audit.py` builds every
pretraining source from raw at full resolution once (SMPL FK for
nymeria/totalcapture/motionx/sonic, HumanML3D clip loading, OCDetect/capture24
parsing) and this OOMs under the interactive session's ~4GB memory cgroup
(confirmed: `oom-kill` on the realdata+OCDetect build during a smoke test).
Use `launch_slurm_mae_scaling_diagnostics.sh` (mirrors the existing
`launch_slurm_MAE_scaling_*.sh` pattern: `--mem=300G`, `b1232` partition):

```bash
sbatch launch_slurm_mae_scaling_diagnostics.sh
```

This runs, in order (later steps reuse earlier ones' cached output where
possible instead of rebuilding from raw):

1. `01_composition_audit.py` — §1, the expensive one-time raw build + per-fraction composition
2. `02_split_audit.py` — §2, OCDetect leak quantification + HumanML3D AMASS-parent check
3. `03_redundancy_audit.py` — §3, analytical + cache-based redundancy
4. `04_source_weighting_audit.py` — §4, reuses `01`'s build
5. `05_val_loss_breakdown.py` — §5, needs a GPU + existing `best.pt`s (same job already has one)
6. `06_budget_protocols.py` — §6, pure math, cheap — already run once interactively, safe to rerun

Each script is also runnable standalone; `01`/`02`/`03`(partially)/`05` are
the ones that need real memory/compute, everything else is cheap.

## Gated, not run automatically

- `07_checkpoint_sweep.py` — first checks whether intermediate step
  checkpoints exist (they don't, for any of the 6 existing scaling runs —
  `save_step_checkpoints: false` in every `data{X}pct.yaml`). Without
  `--screen` it only inventories + generates rerun configs (full pretraining
  reruns, hours of GPU each) — does not submit anything.
- `08_variance_study.py` — prints (does not launch, unless `--launch` is
  passed) the commands for a 3-seed rerun of one HAR probe and one pose probe
  across fractions, using the new `--seed` flag added to
  `scripts/run_pretrain.py`. `--summarize` aggregates results once runs exist.

Both are real GPU jobs (probe runs are cheap/short; pretraining reruns are
not) — confirm compute budget before launching either.

## Production-code changes this relies on

- `imu_lm/data/combined_mae_multidata_loader.py`: opt-in `collect_provenance`
  param (default `False`, zero effect on real training runs) that exposes
  per-window subject/session identity; new `scaling_mode` config key
  (`session_fraction` / `subject_fraction` / `windows_per_session` /
  `source_balanced_fraction`, default `window_fraction` = unchanged prior
  behavior) for diversity-based scaling experiments (§9).
- `imu_lm/data/OCDetect_dataloader.py::_collect_windows`: opt-in
  `return_file_ids` param (default `False`) — OCDetect had no per-window
  provenance mechanism at all before.
- `nymeria_real_wrist_full_body_loader_FLU.py` / `totalcapture_wrist_loader_FLU.py`
  / `motionx_wrist_loader_FLU.py` / `bones_seed_wrist_full_body_loader_FLU.py`
  / `imuposer_wrist_loader_FLU.py`: each dataset class now also stores
  `self._session_files` (source file/clip-id per session, parallel to the
  existing `self._sessions`) — free (a list of strings), always on.
- `scripts/run_pretrain.py` / `scripts/eval_odom_nymeria_altattn.py`: new
  `--seed` CLI override (previously: pretraining honored `trainer.seed` from
  the config only; the odom probe script had no seed control of any kind).
