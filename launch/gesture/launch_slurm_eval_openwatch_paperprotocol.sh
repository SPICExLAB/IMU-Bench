#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=eval_openwatch_paperprotocol
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

# Window-level + clip-level(k=3) metrics for the 4 paper-protocol runs
# (windowed training, posture-stratified contamination-safe split), on BOTH
# val and test — test is the paper-comparable number (model selection used
# val). See EXPERIMENTS.md §5f.
python scripts/eval_openwatch_windowed_protocol.py \
    --runs \
      "Ours (MAE) probe win/noaug:vit1d_gesture_probe_openwatch_6class_mae_windowed_strat:openwatch_gesture_probe" \
      "Ours (MAE) probe win/aug:vit1d_gesture_probe_openwatch_6class_mae_windowed_strat_aug:openwatch_gesture_probe" \
      "Ours (MAE) finetune win/noaug:vit1d_gesture_finetune_openwatch_6class_mae_windowed_strat:openwatch_gesture_finetune" \
      "Ours (MAE) finetune win/aug:vit1d_gesture_finetune_openwatch_6class_mae_windowed_strat_aug:openwatch_gesture_finetune" \
    --splits val test \
    --out-json results_openwatch_paperprotocol.json
