#!/bin/bash
# Keep at most CAP of this user's jobs eligible on the b53017 buy-in partition.
#
# Why this exists: b53017 is a shared 4-GPU buy-in node and we hold ourselves to
# one job at a time on it. Slurm has no per-user concurrency knob for an already
# submitted set, so the throttle is external: everything is `scontrol hold`-ed and
# exactly one is released at a time.
#
# Run it DETACHED so it survives the shell that started it -- a throttle tied to
# an interactive session dies with the session and silently strands every held
# job (this happened once already):
#
#   setsid nohup bash analysis/diversity_saturation/b53017_throttle.sh \
#          > logs/b53017_throttle.log 2>&1 &
#
# Stop it with:  pkill -f b53017_throttle.sh
# It also exits on its own once the b53017 pool is drained.

set -uo pipefail
CAP="${CAP:-1}"
USER_NAME="${USER:-vbc1591}"
SLEEP="${SLEEP:-120}"

echo "[$(date '+%F %T')] throttle start (CAP=$CAP, user=$USER_NAME)"

while true; do
  # eligible = my b53017 jobs in any non-held state (running, completing, released)
  elig=$(squeue -u "$USER_NAME" -h --format="%i|%P|%r" 2>/dev/null \
         | awk -F'|' '($2=="b53017") && ($3!="JobHeldUser")' | wc -l)
  cand=$(squeue -u "$USER_NAME" -h --format="%i|%P|%r" 2>/dev/null \
         | awk -F'|' '($2=="b53017") && ($3=="JobHeldUser"){print $1}' | sort -n)
  ncand=$(echo "$cand" | grep -c .)

  if [ "$elig" -lt "$CAP" ] && [ "$ncand" -gt 0 ]; then
    j=$(echo "$cand" | head -1)
    if scontrol release "$j" >/dev/null 2>&1; then
      echo "[$(date '+%F %T')] released $j ($((ncand-1)) still held)"
    fi
  fi

  if [ "$ncand" -eq 0 ] && [ "$elig" -eq 0 ]; then
    echo "[$(date '+%F %T')] b53017 pool drained -- exiting"
    break
  fi
  sleep "$SLEEP"
done
