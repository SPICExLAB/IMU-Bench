#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_odom_probe_nymeria_5src
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL

# Set up the venv once; reused (not rebuilt) on every subsequent job submission.
# Delete TransfHAR_env/ to force a full reinstall (e.g. after changing requirements.txt).
if [ ! -f TransfHAR_env/bin/activate ]; then
    python -m venv TransfHAR_env
    source TransfHAR_env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m pip install -e .
else
    source TransfHAR_env/bin/activate
fi

PRETRAINED_CKPT="runs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic/checkpoints/best.pt"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT (run launch_slurm_MAE.sh first)" >&2
    exit 1
fi

# eval_odom_nymeria_altattn.py is a standalone script (no --model-config): it
# reads the encoder architecture straight out of the checkpoint's saved "cfg",
# builds its own Nymeria local-velocity training windows, trains a small frozen-
# encoder odometry probe head, and reports vel_mse / ATE (m) / drift (%).
python scripts/eval_odom_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic \
    --runs-root runs \
    --train-dir /projects/b53017/vbc1591/processed_data_body_Ty_rwrist \
    --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \
    --out-dir data_vis/nymeria_odom_altattn_probe_5src
