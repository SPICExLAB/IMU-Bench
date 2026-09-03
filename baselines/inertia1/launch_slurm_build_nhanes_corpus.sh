#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=03:55:00
#SBATCH --array=0-95
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=inertia1_nhanes20hz_build
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Inertia-1 (Xu et al. 2026) pretraining corpus: raw NHANES PAX (80 Hz, g)
# -> Butterworth-4 10 Hz -> 20 Hz -> 30 s fp16 windows, per participant.
# Idempotent per participant; re-submit failed array indices to resume.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
OUT=/projects/b53017/vbc1591/inertia1_nhanes20hz
python baselines/inertia1/scripts/build_nhanes_corpus.py \
    --out "$OUT" --shard-idx "$SLURM_ARRAY_TASK_ID" --num-shards 96 --keep-frac 0.3333333
