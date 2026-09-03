#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_clip_nymeria_finetune_capmerged
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

# Same config as the original finetune run (lr:1e-4, matching the original
# baseline recipe exactly, so this isolates the data-cleaning effect from
# the separately-documented LR-tuning effect) -- the only change is that
# imu_lm/data/nymeria_caption_dataset_FLU.py::_parse_caption_csv now merges
# consecutive identical-caption rows by default (see EXPERIMENTS.md §6c).
BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/vit1d_clip_nymeria_flu_finetune_mae_multidata.yaml"

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "$BASE_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --run-name vit1d_clip_nymeria_finetune_mae_capmerged
