#!/usr/bin/env bash
# Nymeria SYNTHETIC wrist FLU downstream tasks: full-body pose + odometry
#
# Data: synthesized right-wrist IMU from xsens_pose (joint 9) via finite
#       differences, rotated into FLU local frame using the wrist joint
#       orientation. Same coordinate convention as real-wrist FLU data.
#
# Pretrained backbone: vit1d_mae_nymeria_syn_wrist_flu
#   (MAE on synthetic FLU data — train this first before probe/finetune)
#
# Modes per task:
#   supervised — encoder + decoder both trained from random init
#   probe      — frozen MAE encoder + trainable decoder
#   finetune   — MAE encoder warm-started (trainable) + decoder
set -euo pipefail

NYMERIA_TRAIN=/mnt/nas2/processed_data_body_Ty_rwrist
NYMERIA_TEST=/mnt/nas2/processed_data_test_body_Ty_rwrist

############################# MAE PRETRAINING #############################
# Run this first — probe and finetune configs depend on this checkpoint.

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_mae_nymeria_syn_wrist_flu.yaml \
    --run-name     vit1d_mae_nymeria_syn_wrist_flu

# Evaluate MAE reconstruction quality
python scripts/eval_mae_nymeria_syn.py \
    --ckpt       runs/vit1d_mae_nymeria_syn_wrist_flu/checkpoints/best.pt \
    --train-root "$NYMERIA_TRAIN" \
    --val-root   "$NYMERIA_TEST"


############################# FULL-BODY POSE #############################

# Pose — fully supervised (from scratch)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_syn_wrist_flu.yaml \
    --run-name     vit1d_full_body_pose_supervised_nymeria_syn_wrist_flu

python scripts/run_pose_eval_24j_tpose.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_syn_wrist_flu.yaml \
    --run          vit1d_full_body_pose_supervised_nymeria_syn_wrist_flu \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_syn_flu_supervised

# Pose — probe (frozen MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_syn_wrist_flu.yaml \
    --run-name     vit1d_full_body_pose_probe_nymeria_syn_wrist_flu

python scripts/run_pose_eval_24j_tpose.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_syn_wrist_flu.yaml \
    --run          vit1d_full_body_pose_probe_nymeria_syn_wrist_flu \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_syn_flu_probe

# Pose — finetune (MAE encoder warm-started, trainable)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_syn_wrist_flu.yaml \
    --run-name     vit1d_full_body_pose_finetune_nymeria_syn_wrist_flu

python scripts/run_pose_eval_24j_tpose.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_syn_wrist_flu.yaml \
    --run          vit1d_full_body_pose_finetune_nymeria_syn_wrist_flu \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_syn_flu_finetune


############################# ODOMETRY #############################

# Odom — fully supervised (from scratch)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_odom_supervised_nymeria_syn_wrist_flu.yaml \
    --run-name     vit1d_odom_supervised_nymeria_syn_wrist_flu

python scripts/run_odom_eval_dataset.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_odom_supervised_nymeria_syn_wrist_flu.yaml \
    --run          vit1d_odom_supervised_nymeria_syn_wrist_flu \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/odom_syn_flu_supervised

# Odom — probe (frozen MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_odom_probe_nymeria_syn_wrist_flu.yaml \
    --run-name     vit1d_odom_probe_nymeria_syn_wrist_flu

python scripts/run_odom_eval_dataset.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_odom_probe_nymeria_syn_wrist_flu.yaml \
    --run          vit1d_odom_probe_nymeria_syn_wrist_flu \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/odom_syn_flu_probe

# Odom — finetune (MAE encoder warm-started, trainable)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_odom_finetune_nymeria_syn_wrist_flu.yaml \
    --run-name     vit1d_odom_finetune_nymeria_syn_wrist_flu

python scripts/run_odom_eval_dataset.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_odom_finetune_nymeria_syn_wrist_flu.yaml \
    --run          vit1d_odom_finetune_nymeria_syn_wrist_flu \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/odom_syn_flu_finetune
