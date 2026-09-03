#!/bin/bash
# Harvest best val mpjpe for every cell of the frozen-probe LR/schedule sweep.
# Reads the min mpjpe_mm over split=val rows. NOTE: metrics.txt is append-only,
# so this is only safe while each run dir holds exactly one run.
cd /gpfs/home/vbc1591/IMU-SSL
printf '%-10s %-8s %-12s %-10s %s\n' LR SCHED BEST_MPJPE AT_STEP STATE
for sch in const cos; do
  for lr in 3e-4 1e-3 3e-3 1e-2; do
    d="runs/vit1d_full_body_pose_probe_nymeria_mae_multidata_lr${lr}_${sch}"
    m="$d/logs/metrics.txt"
    if [[ ! -f "$m" ]]; then printf '%-10s %-8s %-12s %-10s %s\n' "$lr" "$sch" - - "no metrics yet"; continue; fi
    read -r best step < <(grep "split=val" "$m" | awk '{for(i=1;i<=NF;i++){split($i,a,"=");v[a[1]]=a[2]}; print v["mpjpe_mm"], v["step"]}' | sort -g | head -1)
    last=$(grep -c "split=val" "$m")
    printf '%-10s %-8s %-12s %-10s %s\n' "$lr" "$sch" "${best:--}" "${step:--}" "${last} evals"
  done
done
echo
echo "incumbent baseline (untuned lr=3e-4, constant LR, 130K+ steps): 97.63 mm"
