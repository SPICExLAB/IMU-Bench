#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=embedding_viz_v5
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# t-SNE of the reference 5-source MAE encoder (scripts/plot_ssl_embeddings_v5.py).
# CPU-only: 9,368 windows through a 21M-parameter encoder plus t-SNE is a few
# minutes on 16 cores. Do not run this on a login node -- an earlier attempt sat
# at zero accumulated CPU time for 22 minutes behind a load average of 92.
set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python scripts/plot_ssl_embeddings_v5.py
