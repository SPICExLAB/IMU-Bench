# launch/ — SLURM launch scripts, grouped by task family

| dir | contents |
|---|---|
| `arm_pose/` | Nymeria arm-pose probes/finetunes (incl. recipe/volume sweeps) |
| `full_body_pose/` | 23-joint full-body pose |
| `odometry/` | wrist→pelvis odometry (local + world-frame) |
| `har/` | HAR probes/finetunes/scratch, zero-shot HAR |
| `gesture/` | OpenWatch microgesture (+ eval scripts) |
| `language/` | HumanML3D CLIP alignment + retrieval tuning |
| `pd/` | PADS Parkinson's (pd / pdhc / pddd) |
| `pretrain/` | MAE / SimMTM / FOCAL / RelCon / Inertia-1-style SSL pretrains (incl. mae_4src_* arms) |
| `cache/` | multidata cache + NHANES shard/corpus builds |
| `unimts/` | UniMTS baseline comparisons |
| `data_misc/` | per-dataset one-offs (nymeria, imuposer, totalcapture, realdata, …) |
| `misc/` | everything else (vis, one-off evals) |
| `rawfull/`, `framematch/`, `grid5src/`, `har_hhar_mhealth/`, `probe_dec/`, `probe_lr/` | pre-existing experiment batches (unchanged) |

Full move record from the 2026-08-27 reorganisation: `REORG_MANIFEST.tsv`
(root `launch_slurm_*.sh` → here; root `results_*.json` → `analysis/legacy_results/`;
download manifests → `scripts/download/`; `catboost_info/` → `logs/`).
Bench-generated cell/pack scripts intentionally stay in `bench/generated/slurm/` —
queued jobs reference those paths.
