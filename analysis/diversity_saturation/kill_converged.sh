#!/bin/bash
# Kill full-body-pose probes that have demonstrably converged, to stop them
# burning their 12h walltime for nothing.
#
# Why: those configs ship with `early_stopping: false` (inherited from 11i, and
# deliberately left alone so the "min over training" harvest stays comparable),
# so they never stop on their own. EXPERIMENTS.md 2 documents the pattern -- one
# ran 639,000 steps with its best at step 32,000. Observed here: sub0p01pct hit
# its best 102.69mm at step 50,000 and was still running 517 evals later at
# 104.61mm.
#
# Killing is SAFE and loses nothing: `best_mpjpe.pt` is written on every
# improvement, and the harvest rule for pose is min-over-training, which is
# already fixed once the minimum stops moving.
#
# Threshold is deliberately conservative -- 150 evals with zero improvement AND
# at least 3h elapsed. The real cases overshoot by 500+ evals, so this will not
# fire on a run that is still descending.
#
# Run detached:
#   setsid nohup bash analysis/diversity_saturation/kill_converged.sh \
#          > logs/kill_converged.log 2>&1 &
# Stop:  pkill -f kill_converged.sh

set -uo pipefail
MIN_EVALS_SINCE_BEST="${MIN_EVALS_SINCE_BEST:-150}"
MIN_SECONDS="${MIN_SECONDS:-10800}"
USER_NAME="${USER:-vbc1591}"
cd /gpfs/home/vbc1591/IMU-SSL

echo "[$(date '+%F %T')] converged-killer start (>=${MIN_EVALS_SINCE_BEST} flat evals, >=${MIN_SECONDS}s)"

secs() {  # D-HH:MM:SS | HH:MM:SS | MM:SS -> seconds
  awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4;
                  else if (NF==3) print $1*3600+$2*60+$3;
                  else if (NF==2) print $1*60+$2; else print 0 }' <<<"$1"
}

while true; do
  while IFS='|' read -r jid name elapsed; do
    [ -z "${jid:-}" ] && continue
    case "$name" in *full_body_pose_probe*) ;; *) continue ;; esac
    [ "$(secs "$elapsed")" -lt "$MIN_SECONDS" ] && continue
    f="runs/${name}/logs/metrics.txt"
    [ -f "$f" ] || continue
    read -r flat best step <<<"$(awk '
      /split=val/ { m=""; s="";
        for (i=1;i<=NF;i++) { split($i,kv,"="); if (kv[1]=="mpjpe_mm") m=kv[2]; if (kv[1]=="step") s=kv[2] }
        if (m!="") { n++; v[n]=m+0; st[n]=s+0 } }
      END { if (n==0) { print 0,0,0; exit }
            bi=1; for (i=2;i<=n;i++) if (v[i]<v[bi]) bi=i
            print n-bi, v[bi], st[bi] }' "$f")"
    if [ "${flat:-0}" -ge "$MIN_EVALS_SINCE_BEST" ]; then
      if scancel "$jid" 2>/dev/null; then
        echo "[$(date '+%F %T')] killed $jid $name -- best ${best}mm @ step ${step}, ${flat} flat evals since"
      fi
    fi
  done < <(squeue -u "$USER_NAME" -h -t RUNNING --format="%i|%j|%M" 2>/dev/null)
  sleep 600
done
