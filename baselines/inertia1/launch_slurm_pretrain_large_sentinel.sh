#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Sentinel: exits 0 once runs/<RUN>/logs/metrics.txt records the final val step (170000),
# i.e. once EITHER the b53017 canonical or the gengpu twin pretrain job has finished.
# Bench cells depend afterok on this job, so they start as soon as the weights exist.
set -u
cd /gpfs/home/vbc1591/IMU-SSL
RUN="${RUN:?}"; STEPS="${MAX_STEPS:-170000}"
until grep -q "\"step\": ${STEPS}[,}]" "runs/${RUN}/logs/metrics.txt" 2>/dev/null; do sleep 300; done
echo "pretrain ${RUN} finished at step ${STEPS} ($(date))"; exit 0
