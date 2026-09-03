#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=26:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_recipe_10pct
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# 10% point extending the Section-11 recipe sweep (5%/25%/100%) to 5 fractions.
# Same approved fixed-step-budget protocol: 300,000 steps identical across all
# fractions, session_fraction scaling, natural sampling, NHANES excluded.
# Rebuilds all 11 sources from raw with provenance at job start
# (scaling_mode requires it, bypasses cache_dir) -- ~1.5-2h overhead, same as
# 5pct/25pct.
#
# Walltime: fresh-run budget, unlike the 5pct/25pct scripts whose 14h is a
# RESUME budget. True observed throughput ~3.86 steps/s (job 7919072):
# 300,000 steps ~= 21.6h + ~2h raw rebuild + margin -> 26h. --resume latest
# is a no-op on the first submission (missing checkpoint warns and starts
# from scratch) and lets a TIMEOUT be resubmitted as-is.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/vit1d_ssl_mae_varlen_altattn_multidata_flu_recipe10pct.yaml"

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "$BASE_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --run-name vit1d_ssl_mae_recipe_10pct \
    --resume latest
