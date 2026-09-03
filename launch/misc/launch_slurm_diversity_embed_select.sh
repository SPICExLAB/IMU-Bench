#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --job-name=diversity_embed_select
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Diversity-saturation experiment stage 1: embed the full v3-cache train pool
# with the frozen recipe_100pct encoder, k-means the embedding space, write
# the equal-size random/diverse/redundant 1% index files, and compute the
# coverage/novelty diagnostics. The three vit1d_ssl_mae_div1pct_* pretrains
# are chained afterok on this job (they read the index files).
# ~3.55M windows: forward pass well under 1h on H100; k-means + NN ~30-60 min.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

CUDA_VISIBLE_DEVICES=0 python analysis/diversity_saturation/01_embed_select_analyze.py
