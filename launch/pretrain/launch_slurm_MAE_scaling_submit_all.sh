#!/bin/bash
# Submits the full data-scaling-law sweep for "Ours" (MAE, alt-attn, multidata):
# pretrains on 0.1%/1%/5%/25%/50% of the combined 11-source corpus. The 100%
# point is NOT resubmitted here -- it's the existing baseline checkpoint at
# runs/vit1d_ssl_mae_varlen_altattn_multidata/checkpoints/best.pt.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL

for key in 0.1pct 1pct 5pct 25pct 50pct; do
    echo "Submitting launch_slurm_MAE_scaling_${key}.sh"
    sbatch "launch_slurm_MAE_scaling_${key}.sh"
done
