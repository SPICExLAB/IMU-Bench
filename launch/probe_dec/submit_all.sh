#!/bin/bash
# Submit the 7 decoder-capacity cells (control = existing lr3e-4_cos run).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
for f in launch/probe_dec/launch_slurm_*.sh; do echo -n "$(basename "$f"): "; sbatch "$f"; done
