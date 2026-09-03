#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=b4src_eval
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
# submit each baseline's 30 downstream cells as soon as ITS pretrain finishes
declare -A done_arm
for i in $(seq 1 5600); do
  all=1
  for a in simmtm focal relcon acconly; do
    if [ -f runs/${a}_4src/.done ] && [ -z "${done_arm[$a]:-}" ]; then
      echo "[b4src] ${a}_4src pretrain complete -> submitting downstream cells"
      python bench/generated/fill_4src.py ${a}_4src && done_arm[$a]=1
    fi
    [ -z "${done_arm[$a]:-}" ] && all=0
  done
  [ "$all" = "1" ] && { echo "[b4src] all four baselines evaluated"; break; }
  sleep 30
done
