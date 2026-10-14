#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --qos=buyin
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_corpus_temp03
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Corpus-construction ablation arm 'temp03': source temperature sampling p_s ~ n_s^0.3, uniform within source (N=195,929).
# Same recipe/budget as grid cell k1000_f10 (50,000 steps). --resume latest lets a TIMEOUT resubmit as-is.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/corpus_ablation/vit1d_ssl_mae_corpus_temp03.yaml \
    --run-name vit1d_ssl_mae_corpus_temp03 \
    --resume latest
