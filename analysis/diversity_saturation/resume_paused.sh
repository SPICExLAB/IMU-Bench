#!/bin/bash
# Resume the full-body probes that were paused to free GPUs on b53017.
#
# Slurm's `scontrol suspend` is operator-only, so "pause" here means: cancel the
# job, leave its checkpoints on disk, and later resubmit with `--resume latest`.
# The run picks up from `latest.pt` rather than starting over.
#
# Safe because:
#   * `best_mpjpe.pt` is written on every improvement, so the best-so-far
#     survives the cancel regardless;
#   * `metrics.txt` is append-only and the pose harvest rule is min-over-training,
#     so rows from before and after the pause are read as one run -- which is
#     exactly right here (it IS one continuing run, unlike the 11k case where two
#     *separate* runs would have interleaved).
#
# Usage:  bash analysis/diversity_saturation/resume_paused.sh [runs.txt]
# Default list: analysis/diversity_saturation/paused_runs.txt

set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
LIST="${1:-analysis/diversity_saturation/paused_runs.txt}"
[ -f "$LIST" ] || { echo "no such list: $LIST"; exit 1; }

while read -r run; do
  [ -z "$run" ] && continue
  ck="runs/$run/checkpoints/latest.pt"
  if [ ! -f "$ck" ]; then
    echo "SKIP $run -- no latest.pt (it had not checkpointed; resubmit fresh instead)"
    continue
  fi

  # locate the matching config: grid cells live under configs/grid5src/
  cfg=""
  for c in "configs/grid5src/${run}_multidata_flu.yaml" \
           "configs/${run}_multidata_flu.yaml"; do
    [ -f "$c" ] && { cfg="$c"; break; }
  done
  if [ -z "$cfg" ]; then
    echo "SKIP $run -- config not found"; continue
  fi

  jid=$(sbatch --parsable \
        --account=b53017 --partition=b53017 --qos=buyin --gres=gpu:1 \
        --time=12:00:00 --mem=64G --cpus-per-task=8 \
        --job-name="$run" --output="logs/%x_%j.out" --error="logs/%x_%j.err" \
        --wrap="cd /gpfs/home/vbc1591/IMU-SSL && source TransfHAR_env/bin/activate && \
                CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
                  --config configs/base.yaml --model-config $cfg \
                  --run-name $run --resume latest")
  echo "resumed $run -> job $jid (from $(basename "$ck"))"
done < "$LIST"
