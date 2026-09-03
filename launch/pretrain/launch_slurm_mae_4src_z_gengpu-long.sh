#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu-long
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name=mae_4src_z
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
RUN=mae_4src_z
if [ -f runs/$RUN/.done ]; then echo "$RUN already complete"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \
  --model-config configs/nhanes_scale/mae_4src_z.yaml --run-name "$RUN" --resume latest && touch runs/$RUN/.done
