#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --job-name=i1_5src_pre
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
VARIANT=$1   # patchtst | ar_transformer
RUN=$2
python -u baselines/inertia1/scripts/pretrain_inertia1.py --variant "$VARIANT" --preset small \
  --corpus /projects/b53017/vbc1591/inertia1_5src20hz --run-name "$RUN" --max-steps 170000 --resume
