#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_ssl_relcon_5src_native
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# RELCON under its OWN input preprocessing instead of the benchmark's per-window z-score.
# Paired with the vit1d_ssl_relcon_5src row: same backbone, corpus composition, 160k cap and
# patience. See configs/nativenorm/relcon_5src_native.yaml for exactly what differs and why.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

python launch/nativenorm/preflight.py configs/nativenorm/relcon_5src_native.yaml

# resume after a preemption/timeout; a fresh run starts from scratch
RESUME=()
if [ -f "runs/vit1d_ssl_relcon_5src_native/checkpoints/latest.pt" ]; then RESUME=(--resume latest); fi
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/nativenorm/relcon_5src_native.yaml \
    --run-name vit1d_ssl_relcon_5src_native "${RESUME[@]}"
