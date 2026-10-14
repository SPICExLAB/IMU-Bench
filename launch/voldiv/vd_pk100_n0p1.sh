#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --exclude=qgpu3006,qgpu3013,qgpu2014,qgpu2006
#SBATCH --job-name=vd_pk100_n0p1
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# volume x diversity factorial, cell vd_pk100_n0p1 (EXPERIMENTS.md 11m)
# 4-source pool, 100% of T-phys types (random, source-stratified, gyro-unit fixed), N=1,738 windows (2 h), equal 50,000-step budget.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
RUN=vd_pk100_n0p1
if [ -f runs/$RUN/.done ]; then echo "$RUN already complete"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \
  --model-config configs/voldiv/vd_pk100_n0p1.yaml --run-name "$RUN" --resume latest \
  && touch runs/$RUN/.done
