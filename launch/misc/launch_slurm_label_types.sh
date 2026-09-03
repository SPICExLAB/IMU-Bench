#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=170G
#SBATCH --job-name=label_motion_types
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Joins realdata's HAR activity labels onto the K=1000 motion types.
# Needs a big --mem: build_session_index streams the 13 GB master parquet and
# WindowDataset then preloads every train session into RAM. This OOMs on a
# login node (~80 GB available there), hence SLURM.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

python -u analysis/diversity_saturation/17_label_types.py
