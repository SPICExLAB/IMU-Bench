#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --exclude=qgpu3006,qgpu3013,qgpu2014
#SBATCH --gres=gpu:a100:1
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --job-name=mae_4src_noAA
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# The alternating frame/global attention ablation, on the LEAK-FREE 4-source pool.
# Config is byte-identical to mae_4src_z.yaml except vit1d.encoder.alt_attention.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
python -u scripts/run_pretrain.py --config configs/base.yaml \
  --model-config configs/nhanes_scale/mae_4src_noaltattn.yaml \
  --run-name mae_4src_noaltattn --resume latest
