#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_div1pct_diverse
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Diversity-saturation causal run: diverse-1% subset (N=46,116), fixed-step 300K
# budget, identical to the recipe sweep. Loads from v3 cache via
# scaling_mode=index_file -- no raw rebuild, so 24h covers 300K steps at the
# observed ~3.86 steps/s (~21.6h) with margin. --resume latest is a no-op on
# first run and lets a TIMEOUT resubmit as-is.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_div1pct_diverse.yaml \
    --run-name vit1d_ssl_mae_div1pct_diverse \
    --resume latest
