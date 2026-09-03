#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=vit1d_odom_worldframe_probe_nymeria_5src
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

PRETRAINED_CKPT="runs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic/checkpoints/best.pt"
if [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "ERROR: pretrained checkpoint not found: $PRETRAINED_CKPT" >&2
    exit 1
fi

python scripts/eval_odom_nymeria_worldframe.py \
    --run vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic \
    --runs-root runs \
    --train-dir /projects/b53017/vbc1591/processed_data_body_Ty_rwrist \
    --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \
    --out-dir data_vis/nymeria_odom_worldframe_probe_5src
