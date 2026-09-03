#!/bin/bash
# usage: sbatch launch/launch_slurm_mae_4src.sh dc|phys
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --job-name=mae_4src
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
set -euo pipefail
ARM=$1                      # dc = raw/DC-preserving | phys = physical rescale
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
RUN=mae_4src_${ARM}
if [ -f runs/$RUN/.done ]; then echo "$RUN already complete"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \
  --model-config configs/nhanes_scale/mae_4src_${ARM}.yaml \
  --run-name "$RUN" --resume latest && touch runs/$RUN/.done
