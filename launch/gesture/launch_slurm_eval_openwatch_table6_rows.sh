#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=eval_openwatch_table6_rows
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
# Table-6 baseline/ablation rows under the §5f protocol (window+clip, val+test)
python scripts/eval_openwatch_windowed_protocol.py \
    --runs \
      "FOCAL win/noaug:vit1d_gesture_probe_openwatch_6class_focal_windowed_strat:openwatch_gesture_probe" \
      "FOCAL win/aug:vit1d_gesture_probe_openwatch_6class_focal_windowed_strat_aug:openwatch_gesture_probe" \
      "SimMTM win/noaug:vit1d_gesture_probe_openwatch_6class_simmtm_windowed_strat:openwatch_gesture_probe" \
      "SimMTM win/aug:vit1d_gesture_probe_openwatch_6class_simmtm_windowed_strat_aug:openwatch_gesture_probe" \
      "RelCon win/noaug:vit1d_gesture_probe_openwatch_6class_relcon_windowed_strat:openwatch_gesture_probe" \
      "RelCon win/aug:vit1d_gesture_probe_openwatch_6class_relcon_windowed_strat_aug:openwatch_gesture_probe" \
      "Ours (MAE), no alt-attn win/noaug:vit1d_gesture_probe_openwatch_6class_mae_noaltattn_windowed_strat:openwatch_gesture_probe" \
      "Ours (MAE), no alt-attn win/aug:vit1d_gesture_probe_openwatch_6class_mae_noaltattn_windowed_strat_aug:openwatch_gesture_probe" \
      "Ours (MAE), acc-only win/noaug:vit1d_gesture_probe_openwatch_6class_mae_acconly_windowed_strat:openwatch_gesture_probe" \
      "Ours (MAE), acc-only win/aug:vit1d_gesture_probe_openwatch_6class_mae_acconly_windowed_strat_aug:openwatch_gesture_probe" \
      "Scratch win/noaug:vit1d_gesture_supervised_openwatch_6class_altattn_scratch_windowed_strat:openwatch_gesture_supervised" \
      "Scratch win/aug:vit1d_gesture_supervised_openwatch_6class_altattn_scratch_windowed_strat_aug:openwatch_gesture_supervised" \
    --splits val test \
    --out-json results_openwatch_table6_rows.json
