#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_transfer_fullbodypose_from_pd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

# twin guard: this cell is submitted to both b53017 and gengpu; whichever lands
# first runs it, the other exits (same pattern as bench/launch.py).
DONE_MARKER="/gpfs/home/vbc1591/IMU-SSL/runs/vit1d_transfer_fullbodypose_from_pd/.transfer_done"
if [ -f "$DONE_MARKER" ]; then echo "[pd-transfer] already done -> skip"; exit 0; fi
if squeue -u "$USER" -h -n "$SLURM_JOB_NAME" -t RUNNING -o "%i" | grep -qv "^$SLURM_JOB_ID$"; then
    echo "[pd-transfer] twin RUNNING elsewhere -> skip"; exit 0
fi
echo "[pd-transfer] job=$SLURM_JOB_ID node=$(hostname) start=$(date -Is)"

PRETRAINED_CKPT="runs/vit1d_pd_finetune_pads_mae/checkpoints/best.pt"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_transfer_fullbodypose_from_pd.yaml \
    --run-name vit1d_transfer_fullbodypose_from_pd

mkdir -p "/gpfs/home/vbc1591/IMU-SSL/runs/vit1d_transfer_fullbodypose_from_pd" && touch "$DONE_MARKER"
echo "[pd-transfer] done $(date -Is)"
