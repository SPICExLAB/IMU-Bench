#!/bin/bash
# Submit the volume x diversity factorial (EXPERIMENTS.md 11m).
#   ./launch/voldiv/submit.sh [--probes-only] [--arms a,b,c] [--buyin]
# Each cell's four probes are chained behind its pretrain with afterok,
# and run through the standard bench recipe (bench/registry/tasks.yaml).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
PROBES_ONLY=0; ONLY=""; BUYIN=""
while [ $# -gt 0 ]; do case "$1" in
  --probes-only) PROBES_ONLY=1;; --buyin) BUYIN=--buyin;;
  --arms) ONLY="$2"; shift;; *) echo "unknown arg $1"; exit 1;; esac; shift; done
PANEL="arm_pose_nymeria odometry_nymeria_worldframe har_wisdm microgesture_openwatch har_utd_mhad har_pamap2 har_recofit full_body_pose_nymeria"   # 8 tasks, matching the printed grid
ARMS="vd_nar_n12k vd_nar_n50k vd_nar_n200k vd_nar_n800k vd_one_n12k vd_one_n50k vd_one_n200k vd_one_n800k vd_nat_n12k vd_nat_n50k vd_nat_n200k vd_nat_n800k vd_bal_n12k vd_bal_n50k vd_bal_n200k vd_bal_n800k vd_nat_nfull vd_bal_nfull vd_src1_n50k vd_src2_n50k vd_src3_n50k vd_src4_n50k vd_typ3_n12k vd_typ30_n12k vd_typ300_n12k vd_typall_n12k vd_sub1_n50k vd_sub3_n50k vd_sub10_n50k vd_suball_n50k vd_etyp3_n12k vd_etyp30_n12k vd_etyp300_n12k vd_etypall_n12k vd_typ30_n50k vd_typ300_n50k vd_typ300_n200k vd_typ300_n800k vd_typall_n50k vd_typall_n200k vd_typall_n800k vd_ek1_f0p01 vd_ek1_f0p1 vd_ek5_f0p01 vd_ek5_f0p1 vd_ek5_f1 vd_ek10_f0p01 vd_ek10_f0p1 vd_ek10_f1 vd_ek25_f0p01 vd_ek25_f0p1 vd_ek25_f1 vd_ek50_f0p01 vd_ek50_f0p1 vd_ek50_f1 vd_ek50_f10 vd_ek100_f0p01 vd_ek100_f0p1 vd_ek100_f1 vd_ek100_f10 vd_ek1000_f0p01 vd_ek1000_f0p1 vd_ek1000_f1 vd_ek1000_f10 vd_ek1000_f100"
for ARM in $ARMS; do
  if [ -n "$ONLY" ] && [[ ",$ONLY," != *",$ARM,"* ]]; then continue; fi
  DEP=""
  if [ "$PROBES_ONLY" -eq 0 ]; then
    JID=$(sbatch --parsable launch/voldiv/$ARM.sh)
    echo "pretrain $ARM -> $JID"
    DEP="--dependency afterok:$JID"
  fi
  for T in $PANEL; do
    python -m bench.launch --cell $T $ARM probe $DEP $BUYIN
  done
done
