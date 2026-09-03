#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=pads_subject_level_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Aggregate every PADS run's window-level predictions up to one prediction per
# subject, which is the unit the original PADS paper reports. See
# scripts/eval_pads_subject_level.py.

set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs analysis/pads_subject_level

run_eval () {  # $1=config-prefix $2=run-name
  local cfg="configs/$1_multidata_flu.yaml"
  local run="$2"
  if [ ! -f "$cfg" ]; then cfg="configs/$1.yaml"; fi
  if [ ! -d "runs/$run/checkpoints" ]; then
    echo "SKIP $run (no checkpoints)"; return
  fi
  echo "=================== $run ==================="
  python scripts/eval_pads_subject_level.py \
      --config configs/base.yaml \
      --model-config "$cfg" \
      --run "$run" \
      --out "analysis/pads_subject_level/${run}.json" 2>&1 | grep -vE "^INFO|preload"
}

# --- 3-class differential diagnosis ---
for m in mae simmtm relcon focal; do
  run_eval "vit1d_pd_probe_pads_${m}"    "vit1d_pd_probe_pads_${m}"
  run_eval "vit1d_pd_finetune_pads_${m}" "vit1d_pd_finetune_pads_${m}"
done
run_eval "vit1d_pd_supervised_pads_altattn_scratch" "vit1d_pd_supervised_pads_scratch"

# --- binary arms matching the original paper ---
for task in pdhc pddd; do
  for m in mae simmtm relcon focal; do
    run_eval "vit1d_${task}_probe_pads_${m}"    "vit1d_${task}_probe_pads_${m}"
    run_eval "vit1d_${task}_finetune_pads_${m}" "vit1d_${task}_finetune_pads_${m}"
  done
  run_eval "vit1d_${task}_supervised_pads_scratch" "vit1d_${task}_supervised_pads_scratch"
done

echo "=== all subject-level evals done ==="
