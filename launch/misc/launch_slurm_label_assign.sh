#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --qos=buyin
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=170G
#SBATCH --job-name=label_assign
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Names motion types by embedding labelled realdata windows and assigning them
# to the nearest of the 1000 centroids. Needs a GPU for the encoder and a large
# --mem because WindowDataset preloads every session from the 13 GB parquet.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python -u analysis/diversity_saturation/20_label_by_assignment.py
