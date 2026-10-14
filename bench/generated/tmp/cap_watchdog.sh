#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=20:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G
#SBATCH --job-name=cap_watchdog
#SBATCH --output=bench/generated/tmp/%x_%j.out
#SBATCH --error=bench/generated/tmp/%x_%j.err
# Enforce the 25,000-step cap on the two NormWear kinematic probes that were already RUNNING when the cap was set
# (their configs cannot be changed mid-run, and a running job's walltime cannot be extended or shortened without
# operator rights). Poll their metrics.txt; when a run passes the cap, cancel it and submit the scoring job that
# reads its best.pt. Runs on a CPU node so it survives any interactive session ending.
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
CAP=15000
declare -A DONE
watch_one() {                       # $1 = task, $2 = run name
  local task="$1" run="$2"
  [ "${DONE[$task]:-0}" = "1" ] && return
  local m="runs/$run/logs/metrics.txt"
  [ -f "$m" ] || return
  local st; st=$(grep -o 'step=[0-9]*' "$m" | tail -1 | cut -d= -f2)
  [ -z "$st" ] && return
  local jid; jid=$(squeue -u "$USER" -h -n "$run" -o "%i" | head -1)
  if [ -z "$jid" ]; then            # already finished on its own
    echo "$(date -Is) $task: job gone at step $st -> scoring"
    sbatch bench/generated/tmp/score_normwear_capped.sh "$task probe"; DONE[$task]=1; return
  fi
  if [ "$st" -ge "$CAP" ]; then
    echo "$(date -Is) $task: step $st >= $CAP -> cancel $jid and score"
    scancel "$jid"; sleep 30
    sbatch bench/generated/tmp/score_normwear_capped.sh "$task probe"; DONE[$task]=1
  fi
}
while true; do
  watch_one arm_pose_nymeria       bench_arm_pose_nymeria__normwear_base__probe__s42
  watch_one full_body_pose_nymeria bench_full_body_pose_nymeria__normwear_base__probe__s42
  [ "${DONE[arm_pose_nymeria]:-0}" = "1" ] && [ "${DONE[full_body_pose_nymeria]:-0}" = "1" ] && break
  sleep 300
done
echo "[cap_watchdog] both probes handled $(date -Is)"
