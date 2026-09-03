#!/bin/bash
# Harvest the HHAR/MHEALTH probe cells.
#
# Convention (verified against the reported MAE probe numbers -- HHAR job
# 7745122 reproduces 66.79 / 68.24 / 0.679 exactly): the reported cell is the
# `split=val` line with the MINIMUM val loss, i.e. the row corresponding to
# best.pt, NOT the last row.
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL

printf '%-46s %-9s %8s %8s %8s\n' RUN STATE ACC BAL_ACC MACRO_F1
for ds in hhar mhealth; do
  for m in focal simmtm relcon mae_noaltattn mae_acconly; do
    run="vit1d_har_probe_${ds}_${m}_multidata"
    # A queued/running job for this cell supersedes any older log on disk --
    # otherwise a stale FAILED log from a cancelled attempt reads as the result.
    live=$(squeue -u "$USER" -h -n "$run" -o "%T" 2>/dev/null | head -1)
    if [ -n "$live" ]; then
      printf '%-46s %-9s %8s %8s %8s\n' "$run" "$live" - - -
      continue
    fi
    log=$(ls -t logs/${run}_*.out 2>/dev/null | head -1)
    if [ -z "${log}" ]; then
      printf '%-46s %-9s %8s %8s %8s\n' "$run" NOLOG - - -
      continue
    fi
    jid=$(basename "$log" .out | sed "s/^${run}_//")
    state=$(sacct -j "$jid" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')
    line=$(grep 'split=val' "$log" 2>/dev/null \
           | sed 's/.*split=val loss=\([0-9.]*\).*/\1 &/' | sort -n | head -1)
    if [ -z "$line" ]; then
      printf '%-46s %-9s %8s %8s %8s\n' "$run" "${state:-?}" - - -
      continue
    fi
    acc=$(echo   "$line" | sed 's/.* acc=\([0-9.]*\).*/\1/')
    bal=$(echo   "$line" | sed 's/.* bal_acc=\([0-9.]*\).*/\1/')
    f1=$(echo    "$line" | sed 's/.* macro_f1=\([0-9.]*\).*/\1/')
    printf '%-46s %-9s %8.2f %8.2f %8.3f\n' \
      "$run" "${state:-?}" \
      "$(echo "$acc*100" | bc -l)" "$(echo "$bal*100" | bc -l)" "$f1"
  done
done
