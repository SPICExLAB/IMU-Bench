#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_full_body_pose_probe_nymeria_mae_k50_f1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Full-body-pose (23-joint) Nymeria linear-probe scaling check for the
# 3-point recipe experiment (session_fraction scaling, fixed-step 300K
# budget). Same methodology as EXPERIMENTS.md section 9d. This config has
# early_stopping.enabled=false + max_steps effectively unbounded (same
# template gap noted throughout EXPERIMENTS.md) -- ALWAYS runs the full
# 12h walltime by design; best_mpjpe.pt is checkpointed independently so
# the actual best result survives the walltime kill regardless (confirmed
# pattern from jobs 7491654/7491746, both TIMEOUT'd at 12h/COMPLETED-equivalent).

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/grid5src/vit1d_full_body_pose_probe_nymeria_mae_k50_f1_multidata_flu.yaml \
    --run-name vit1d_full_body_pose_probe_nymeria_mae_k50_f1
