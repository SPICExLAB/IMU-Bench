#!/usr/bin/env bash
set -euo pipefail

USERS=(
  Mingyuan Timmy Taeyoung Jerry Xiaan
  Jesse Trenton Anwesha Ge Brian
  Ziyi Vasco Jhan Pouya Tony
)

for USER in "${USERS[@]}"; do
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/train_surfacexr_imu_cnn_50hz.py \
      --pkl-path /mnt/nas2/vasco/SurfaceXR/dataset_raw_imu.pkl \
      --split lopo \
      --test-user "$USER" \
      --window-seconds 1.0 \
      --hop-seconds 0.5 \
      --batch-size 256 \
      --epochs 50 \
      --run-name "surfacexr_imu_cnn_50hz_lopo_${USER}"
done
