#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=verify_coverage
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# CPU-only verification of the coverage-saturation claim (EXPERIMENTS.md 11j
# diagnostics): multi-draw empirical coverage, a second k-means seed, three
# mode granularities (K=100/1000/4000), and a closed-form cross-check.
# Thread caps are REQUIRED: an unpinned OpenBLAS aborts (SIGABRT/exit 134) on
# this cluster's high-core nodes -- that is what killed the first attempt.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

python analysis/diversity_saturation/04_verify_coverage.py
