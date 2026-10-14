#!/bin/bash
# Random physical-type x volume grid: pretrain each cell, chain the 8-task probe panel.
#   ./launch/voldiv/submit_pk.sh [--probes-only] [--arms a,b] [--buyin]
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
PROBES_ONLY=0; ONLY=""; BUYIN=""
while [ $# -gt 0 ]; do case "$1" in
  --probes-only) PROBES_ONLY=1;; --buyin) BUYIN=--buyin;;
  --arms) ONLY="$2"; shift;; *) echo "unknown arg $1"; exit 1;; esac; shift; done
PANEL="arm_pose_nymeria odometry_nymeria_worldframe har_wisdm microgesture_openwatch har_utd_mhad har_pamap2 har_recofit full_body_pose_nymeria"
ARMS="vd_pk1_n0p1 vd_pk1_n1 vd_pk10_n0p1 vd_pk10_n1 vd_pk10_n10 vd_pk50_n0p1 vd_pk50_n1 vd_pk50_n10 vd_pk100_n0p1 vd_pk100_n1 vd_pk100_n10"
for ARM in $ARMS; do
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$ARM,"* ]]; then continue; fi
  DEP=""
  if [ "$PROBES_ONLY" -eq 0 ]; then
    if squeue -u "$USER" -h -o '%j' | grep -qx "$ARM"; then echo "$ARM already queued"; continue; fi
    JID=$(sbatch --parsable launch/voldiv/$ARM.sh)
    echo "pretrain $ARM -> $JID"
    DEP="--dependency afterok:$JID"
  fi
  for T in $PANEL; do
    python -m bench.launch --cell $T $ARM probe $DEP $BUYIN
  done
done
