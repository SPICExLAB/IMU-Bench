#!/bin/bash
# Submits the LR sweep for "Ours (MAE) finetune" on full-body pose (Nymeria),
# Section 2 of EXPERIMENTS.md: baseline lr=0.0001 gets 92.39mm mpjpe, 0.51mm
# behind MobilePoser's 91.88mm. Sweeps 6 additional LRs (3 below, 3 above the
# baseline) to see if a different LR closes that gap, holding every other
# hyperparameter (weight_decay, warmup, schedule, batch_size, architecture)
# fixed at the baseline config's values.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL

for tag in 3e-5 5e-5 7e-5 1.5e-4 2e-4 3e-4; do
    echo "Submitting launch_slurm_full_body_pose_finetune_nymeria_mae_lr${tag}.sh"
    sbatch "launch_slurm_full_body_pose_finetune_nymeria_mae_lr${tag}.sh"
done
