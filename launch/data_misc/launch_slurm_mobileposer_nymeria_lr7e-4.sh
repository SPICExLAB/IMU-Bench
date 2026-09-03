#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=buyin
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=mobileposer_nymeria_lr7e-4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python -m pip install --quiet opencv-python-headless

cd baselines/full_body_pose/SimplePoser

# LR-sweep variant of the MobilePoser Nymeria baseline (train_nymeria.yaml,
# lr=0.001 -> 91.88mm/9.19cm mpjpe). Reuses the already-preprocessed
# data_processed/Nymeria/{train,test}.pt (867/101 sessions, built by the
# original launch_slurm_mobileposer_nymeria.sh run) -- no need to
# re-preprocess. Only learning_rate differs from train_nymeria.yaml; same
# 60 epochs, same architecture, same data.
echo "== training MobilePoser on Nymeria (lr=7e-4) =="
python train.py --config train_nymeria_lr7e-4.yaml

echo "== evaluating (window-matched to Ours, 10.24s) =="
python evaluate_nymeria.py \
    --weights checkpoints/nymeria_rw_lr7e-4/best.pt \
    --config train_nymeria_lr7e-4.yaml \
    --window-seconds 10.24 \
    --out-json results/nymeria_mobileposer_results_lr7e-4.json

echo "== done =="
