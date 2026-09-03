#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --exclude=qgpu3006,qgpu3013
#SBATCH --gres=gpu:a100:1
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name=focal_4src
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
RUN=focal_4src
if [ -f runs/$RUN/.done ]; then echo "$RUN already complete"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml --model-config configs/nhanes_scale/focal_4src.yaml --run-name "$RUN" --resume latest && touch runs/$RUN/.done
