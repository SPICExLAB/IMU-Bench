#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=buyin
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=mobileposer_nymeria
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
# opencv-python-headless: articulate/math/angular.py's rotation_matrix_to_axis_angle
# uses cv2.Rodrigues. Listed in requirements.txt too, but that's only installed on
# first venv creation above, so also install it here for already-existing venvs.
python -m pip install --quiet pyyaml tqdm scipy opencv-python-headless

cd baselines/full_body_pose/SimplePoser

# 1) Preprocess Nymeria real-wrist sessions into MobilePoser's .pt format.
# Full dataset: 867 train / 101 val pkl sessions (all available files in each dir).
echo "== preprocessing Nymeria (full dataset) =="
python preprocess_nymeria.py \
    --train-dir /projects/b53017/vbc1591/processed_data_body_Ty_rwrist \
    --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \
    --max-train-files 867 \
    --max-val-files 101

# 2) Train MobilePoser from scratch on the real right-wrist IMU combo ('rw').
echo "== training MobilePoser on Nymeria =="
python train.py --config train_nymeria.yaml

# 3) Evaluate: odometry + full-body pose + arm-pose numbers in one pass.
echo "== evaluating =="
python evaluate_nymeria.py \
    --weights checkpoints/nymeria_rw/best.pt \
    --config train_nymeria.yaml \
    --out-json results/nymeria_mobileposer_results.json

echo "== done =="
