#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_gesture_probe_openwatch_6class_mae_noaltattn_windowed_strat_aug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

# OpenWatch paper-protocol Table 6 row (windowed 1s/stride-20, 6 win/clip,
# posture-stratified contamination-safe split, weighted CE + val macro-F1). §5f.
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_gesture_probe_openwatch_6class_mae_noaltattn_multidata_flu_windowed_stratified_aug.yaml \
    --run-name vit1d_gesture_probe_openwatch_6class_mae_noaltattn_windowed_strat_aug
