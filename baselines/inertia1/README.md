# Inertia-1 external baseline

Inertia-1 (Xu et al. 2026, arXiv:2607.06617, github.com/yang-ai-lab/Inertia-1) — a
wearable-motion foundation model pretrained on NHANES wrist accelerometry.
**The release ships no pretrained weights** (issue #1 there is an unanswered request), so
everything here reproduces their two robust objectives with their code + recipe on the raw
NHANES archives we already have, then runs them through this repo's six task pipelines via
an adapter. Full write-up: `EXPERIMENTS.md §16`.

```
repo/                       upstream clone (ffdf91f); PatchTST_Accel + patch utils are imported from here
scripts/build_nhanes_corpus.py   NHANES raw 80 Hz -> Butterworth-4 10 Hz -> 20 Hz -> 30 s fp16 windows (SLURM array)
scripts/pretrain_inertia1.py     their pretraining recipe, pure torch -> runs/inertia1_{patchtst,art}_small_nhanes
scripts/make_configs.py          configs/inertia1/*.yaml = sibling MAE configs with only the encoder side changed
scripts/smoke_downstream.py      GPU smoke test: every objective branch + scorer with a random-init ckpt
scripts/harvest_inertia1.py      -> analysis/inertia1/inertia1_results.md
submit_inertia1_downstream.sh    submits 15 train + 7 score jobs per variant, gated on the pretrain job
launch_slurm_*.sh                SLURM templates (corpus array, pretrain, generic downstream/score job, smoke)
```

Adapter: `imu_lm/models/baselines/inertia1_encoder.py` (`Inertia1Encoder`, `build_encoder`,
`inertia1_downstream_config`). A config with an `inertia1:` block makes `run.py` build the
adapter instead of `ViT1DEncoder`; the bench harness has the same switch
(`external_encoder: inertia1` in `bench/registry/encoders.yaml`).

Reproduce:
```bash
sbatch baselines/inertia1/launch_slurm_build_nhanes_corpus.sh                    # ~2.5 h, 96 CPU tasks
VARIANT=patchtst       sbatch --dependency=afterany:<build> baselines/inertia1/launch_slurm_pretrain_inertia1.sh
VARIANT=ar_transformer sbatch --dependency=afterany:<build> baselines/inertia1/launch_slurm_pretrain_inertia1.sh
python baselines/inertia1/scripts/make_configs.py
bash baselines/inertia1/submit_inertia1_downstream.sh <pt_job_patchtst> <pt_job_art>
python baselines/inertia1/scripts/harvest_inertia1.py
```
