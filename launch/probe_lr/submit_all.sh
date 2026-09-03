#!/bin/bash
# Submit the full 4-LR x 2-schedule frozen-probe sweep (8 cells, 60K steps each).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
for f in launch/probe_lr/launch_slurm_vit1d_full_body_pose_probe_nymeria_mae_multidata_lr*.sh; do
    echo -n "$(basename "$f"): "
    sbatch "$f"
done
