#!/bin/bash
# Submit the three new shared-backbone SSL baselines (SSL-Wearables / DINO / I-JEPA) and their 60 IMU-Bench cells.
# Pretrains on b53017; probes and fine-tunes on gengpu (bench.launch default), each chained afterok its pretrain.
# Idempotent: a job whose name is already queued or running is not submitted again (twin-collision guard).
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
LOCK=bench/generated/tmp/ssl3_submit.lock
if ! mkdir "$LOCK" 2>/dev/null; then echo "another submission holds $LOCK -- abort"; exit 1; fi
echo "$(date -Is) pid $$ host $(hostname)" > "$LOCK/owner"
TASKS="arm_pose_nymeria full_body_pose_nymeria odometry_nymeria_local microgesture_openwatch language_humanml3d pd_pads har_hhar har_mhealth har_prism har_userwatch"
queued() { [ -n "$(squeue -u vbc1591 -h --name="$1" -o %i)" ]; }
for m in sslwear dino jepa; do
  PN=vit1d_ssl_${m}_5src
  if queued "$PN"; then PJ=$(squeue -u vbc1591 -h --name="$PN" -o %i | head -1); echo "$m pretrain already queued as $PJ"; 
  else PJ=$(sbatch --parsable launch/pretrain5src/${PN}.sh) || { echo "sbatch failed for $m"; continue; }; echo "$m pretrain $PJ"; fi
  echo "$PJ" > bench/generated/tmp/${m}_5src_pretrain_job.txt
  for P in probe finetune; do
    for T in $TASKS; do
      JN=bench_${T}__${m}_5src__${P}__s42
      if queued "$JN"; then echo "  [have] $JN"; continue; fi
      out=$(python -m bench.launch --cell $T ${m}_5src $P --dependency afterok:$PJ 2>&1 | tail -1); echo "  $out"
      if [ "$P" = finetune ]; then jid=$(echo "$out" | grep -o "\[sbatch\] [0-9]*" | grep -o "[0-9]*"); [ -n "$jid" ] && scontrol update jobid=$jid nice=300; fi
    done
  done
done
echo "$(date -Is) done" >> "$LOCK/owner"
