#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_full_body_pose_probe_nymeria_mae_multidata_lr3e-4_cos
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Full-body-pose (23-joint) Nymeria frozen probe -- LR/schedule sweep cell
# lr=3e-4, sched=cos. Identical to the baseline probe
# (vit1d_full_body_pose_probe_nymeria_mae_multidata, 97.63mm) in every other
# hyperparameter. Equal 60K-step budget for all 8 cells; early_stopping is
# enabled with patience=999999 so it cannot fire (guarantees the full budget
# AND writes best.pt -- see EXPERIMENTS.md). Harvest best_mpjpe.pt / the
# min mpjpe_mm over split=val rows in logs/metrics.txt.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

PRETRAINED_CKPT="runs/vit1d_ssl_mae_varlen_altattn_multidata/checkpoints/best.pt"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/probe_lr/vit1d_full_body_pose_probe_nymeria_mae_multidata_lr3e-4_cos_flu.yaml \
    --run-name vit1d_full_body_pose_probe_nymeria_mae_multidata_lr3e-4_cos
