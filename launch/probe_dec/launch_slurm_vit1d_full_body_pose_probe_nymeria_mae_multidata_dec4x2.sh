#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_full_body_pose_probe_nymeria_mae_multidata_dec4x2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Decoder-capacity sweep cell: 4 cross-channel x 2 temporal layers.
# Control is the existing lr3e-4_cos run (decoder 1x0). Optimizer held at
# lr=3e-4 + cosine, equal 60K budget. NOTE generic --gres=gpu:1 (only 14 of
# gengpu's 38 nodes are H100; pinning h100 strands the job) and a 6h walltime
# (60K steps is ~2.3h on an A100 at the baseline depth, plus headroom for the
# deeper cells) -- see EXPERIMENTS.md section 2 scheduling notes.

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
    --model-config configs/probe_dec/vit1d_full_body_pose_probe_nymeria_mae_multidata_dec4x2_flu.yaml \
    --run-name vit1d_full_body_pose_probe_nymeria_mae_multidata_dec4x2
