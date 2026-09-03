#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=buyin
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=mobileposer_nymeria_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python -m pip install --quiet opencv-python-headless

cd baselines/full_body_pose/SimplePoser

# --window-seconds 10.24 matches the ViT1D benchmark's windowing.window_seconds
# so MobilePoser doesn't get an unfair temporal-context advantage from being
# evaluated on whole (unbounded) sessions -- see evaluate_nymeria.py's
# evaluate() docstring.
echo "== evaluating (reusing existing checkpoints/nymeria_rw/best.pt, window-matched to Ours) =="
python evaluate_nymeria.py \
    --weights checkpoints/nymeria_rw/best.pt \
    --config train_nymeria.yaml \
    --window-seconds 10.24 \
    --out-json results/nymeria_mobileposer_results.json

echo "== done =="
