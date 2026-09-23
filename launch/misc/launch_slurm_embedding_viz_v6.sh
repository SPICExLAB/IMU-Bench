#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name=embedding_viz_v6
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# t-SNE with fine-grained labels (scripts/plot_ssl_embeddings_v6.py).
# Bigger than v5: +TransfHAR, +OpenWatch, +OCDetect, +EatingSpeed and the
# un-bucketed PRISM / UTD-MHAD vocabularies, so ~13.5k windows instead of 9.4k
# and an O(N^2) similarity matrix of ~0.7 GB. 96G because reading the master
# parquet per source is what OOM-killed the login node.
set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python scripts/plot_ssl_embeddings_v6.py
