#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_transfer_odomworldframe_from_fullbodypose
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL

if [ ! -f TransfHAR_env/bin/activate ]; then
    python -m venv TransfHAR_env
    source TransfHAR_env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m pip install -e .
else
    source TransfHAR_env/bin/activate
fi

PRETRAINED_CKPT="runs/vit1d_full_body_pose_finetune_nymeria_mae_multidata/checkpoints/best.pt"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

# eval_odom_nymeria_worldframe.py: frozen-encoder probe on the canonical
# world-frame velocity target (heading canonicalized per-window from movement
# direction). Reconstruction is a plain cumulative sum (no gyro integration
# needed, unlike the wrist-FLU pipeline) since the frame is fixed per window.
python scripts/eval_odom_nymeria_worldframe.py \
    --run vit1d_full_body_pose_finetune_nymeria_mae_multidata \
    --runs-root runs \
    --train-dir /projects/b53017/vbc1591/processed_data_body_Ty_rwrist \
    --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \
    --out-dir data_vis/nymeria_odom_worldframe_transfer_from_fullbodypose
