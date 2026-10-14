#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --exclude=qgpu3006,qgpu3013,qgpu2014,qgpu2006
#SBATCH --gres=gpu:a100:1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name=mae_4src_nocls
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# CLS ablation. Byte-identical to mae_4src_z except vit1d.encoder.use_cls=false,
# so the only difference is the token: same pool, mask 0.60, z-score, octahedral
# rotation, lr 1e-4, 100k steps, seed 42.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
RUN=mae_4src_nocls
if [ -f runs/$RUN/.done ]; then echo "$RUN already complete"; exit 0; fi
python -u scripts/run_pretrain.py --config configs/base.yaml \
  --model-config configs/nhanes_scale/mae_4src_nocls.yaml --run-name "$RUN" --resume latest && touch runs/$RUN/.done
