#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=pads_pddd_subject_level_rescore
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Re-score the 9 PD-vs-DD (pddd) runs with the extended
# scripts/eval_pads_subject_level.py: adds AUROC + oracle-threshold balanced
# accuracy (window & subject level) and dumps per-window scores, so the PADS
# table can carry the paper's method next to ours on identical subjects.

set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs analysis/pads_subject_level

run_eval () {  # $1=config-prefix $2=run-name
  local cfg="configs/$1_multidata_flu.yaml"
  local run="$2"
  if [ ! -f "$cfg" ]; then cfg="configs/$1.yaml"; fi
  if [ ! -d "runs/$run/checkpoints" ]; then echo "SKIP $run"; return; fi
  echo "=================== $run ==================="
  python scripts/eval_pads_subject_level.py \
      --config configs/base.yaml \
      --model-config "$cfg" \
      --run "$run" \
      --out "analysis/pads_subject_level/${run}.json" \
      --dump-npz "analysis/pads_subject_level/${run}.npz" 2>&1 | grep -vE "^INFO|preload"
}

for m in mae simmtm relcon focal; do
  run_eval "vit1d_pddd_probe_pads_${m}"    "vit1d_pddd_probe_pads_${m}"
  run_eval "vit1d_pddd_finetune_pads_${m}" "vit1d_pddd_finetune_pads_${m}"
done
run_eval "vit1d_pddd_supervised_pads_scratch" "vit1d_pddd_supervised_pads_scratch"
echo "=== done ==="
