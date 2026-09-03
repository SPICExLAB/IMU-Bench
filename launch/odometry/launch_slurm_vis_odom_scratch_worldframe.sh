#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=buyin
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vis_odom_scratch_worldframe
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

# Pure inference (no training): loads the already-finished, already-saved
# encoder + ViT1DOdomDecoder joint checkpoint from the world-frame odometry
# objective and scores ATE/drift via cumulative-sum reconstruction (no gyro
# integration -- see vis_odom_scratch_worldframe.py docstring).
python scripts/vis_odom_scratch_worldframe.py \
    --run vit1d_odom_worldframe_supervised_nymeria_altattn_scratch \
    --runs-root runs \
    --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \
    --out-dir data_vis/nymeria_odom_worldframe_probe_multidata \
    --label "Scratch"
