#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=128G
#SBATCH --job-name=pads_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# The PADS authors' smartwatch pipeline under the IMU-Bench Parkinson's protocol (CPU only).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export WANDB_MODE=disabled OMP_NUM_THREADS=32
python baselines/pd/pads_pipeline.py --n-jobs 32
echo "[pads] finished $(date -Is)"
