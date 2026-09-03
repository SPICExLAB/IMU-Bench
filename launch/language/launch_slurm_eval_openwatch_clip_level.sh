#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=eval_openwatch_clip_level
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

python scripts/eval_openwatch_clip_level.py --out-json results_openwatch_clip_level.json
