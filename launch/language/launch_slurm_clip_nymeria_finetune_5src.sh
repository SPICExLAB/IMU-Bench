#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=vit1d_clip_nymeria_finetune_mae_realdata_ocdetect_eatingspee
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

TEXT_CACHE="/projects/b53017/vbc1591/nymeria_captions/clip_text_emb_cache.pt"
if [[ ! -f "$TEXT_CACHE" ]]; then
    echo "ERROR: text embedding cache not found: $TEXT_CACHE" >&2
    exit 1
fi
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_clip_nymeria_flu_finetune_mae_multidata_realdata_ocdetect_eatingspeed_motionx_sonic.yaml \
    --run-name vit1d_clip_nymeria_finetune_mae_realdata_ocdetect_eatingspeed_motionx_sonic
