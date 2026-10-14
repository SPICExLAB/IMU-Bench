#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name=vd_one_n800k
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# volume x diversity factorial, cell vd_one_n800k (EXPERIMENTS.md 11m)
# single-source, uniform random, N=800,000 windows (1,138 h), equal 50,000-step budget.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
RUN=vd_one_n800k
if [ -f runs/$RUN/.done ]; then echo "$RUN already complete"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \
  --model-config configs/voldiv/vd_one_n800k.yaml --run-name "$RUN" --resume latest \
  && touch runs/$RUN/.done
