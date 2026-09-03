#!/usr/bin/env bash
# TotalCapture FLU downstream tasks: arm pose, odometry, full-body pose
# Modes: supervised (from scratch) | probe (frozen encoder) | finetune (trainable encoder)
# Encoders: OCDetect MAE | TotalCapture MAE (in-domain) | in-domain odom/fullbody supervised
# Train: s1-s4  |  Val/Test: s5
set -euo pipefail

TC_TEST_DIR=/mnt/nas2/DIP_recalculate   # all pkls in one dir; filter to s5_* for eval

############################# ARM POSE #############################

# Arm pose - fully supervised (real wrist IMU from DIP pkl)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_supervised_totalcapture_flu.yaml \
	--run-name     vit1d_arm_pose_supervised_totalcapture_flu

# Arm pose - fully supervised (synthetic wrist IMU from GT SMPL FK)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_supervised_totalcapture_syn_flu.yaml \
	--run-name     vit1d_arm_pose_supervised_totalcapture_syn_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_supervised_totalcapture_syn_flu.yaml \
	--run          vit1d_arm_pose_supervised_totalcapture_syn_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_supervised_totalcapture_syn_flu_eval_dataset

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_supervised_totalcapture_flu.yaml \
	--run          vit1d_arm_pose_supervised_totalcapture_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_supervised_totalcapture_flu_eval_dataset

# Arm pose - probe (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_probe_totalcapture_flu.yaml \
	--run-name     vit1d_arm_pose_probe_totalcapture_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_probe_totalcapture_flu.yaml \
	--run          vit1d_arm_pose_probe_totalcapture_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_probe_totalcapture_flu_eval_dataset

# Arm pose - finetune (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_totalcapture_flu.yaml \
	--run-name     vit1d_arm_pose_finetune_totalcapture_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_totalcapture_flu.yaml \
	--run          vit1d_arm_pose_finetune_totalcapture_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_finetune_totalcapture_flu_eval_dataset

############################# ODOMETRY #############################

# Odom - fully supervised
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_supervised_totalcapture_flu.yaml \
	--run-name     vit1d_odom_supervised_totalcapture_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_supervised_totalcapture_flu.yaml \
	--run          vit1d_odom_supervised_totalcapture_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--out-dir      data_vis/odom_supervised_totalcapture_flu_eval_dataset

# Odom - probe (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_totalcapture_flu.yaml \
	--run-name     vit1d_odom_probe_totalcapture_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_totalcapture_flu.yaml \
	--run          vit1d_odom_probe_totalcapture_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--out-dir      data_vis/odom_probe_totalcapture_flu_eval_dataset

# Odom - finetune (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_totalcapture_flu.yaml \
	--run-name     vit1d_odom_finetune_totalcapture_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_totalcapture_flu.yaml \
	--run          vit1d_odom_finetune_totalcapture_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--out-dir      data_vis/odom_finetune_totalcapture_flu_eval_dataset


############################# FULL-BODY POSE #############################

# Full-body pose - fully supervised
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_supervised_totalcapture_flu.yaml \
	--run-name     vit1d_full_body_pose_supervised_totalcapture_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_supervised_totalcapture_flu.yaml \
	--run          vit1d_full_body_pose_supervised_totalcapture_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   24 \
	--out-dir      data_vis/full_body_pose_supervised_totalcapture_flu_eval_dataset

# Full-body pose - probe (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_totalcapture_flu.yaml \
	--run-name     vit1d_full_body_pose_probe_totalcapture_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_totalcapture_flu.yaml \
	--run          vit1d_full_body_pose_probe_totalcapture_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   24 \
	--out-dir      data_vis/full_body_pose_probe_totalcapture_flu_eval_dataset

# Full-body pose - finetune (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_totalcapture_flu.yaml \
	--run-name     vit1d_full_body_pose_finetune_totalcapture_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_totalcapture_flu.yaml \
	--run          vit1d_full_body_pose_finetune_totalcapture_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   24 \
	--out-dir      data_vis/full_body_pose_finetune_totalcapture_flu_eval_dataset

############################# TotalCapture MAE ENCODER — PROBE/FINETUNE #############################

# Arm pose - probe (TotalCapture MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_probe_totalcapture_mae_flu.yaml \
	--run-name     vit1d_arm_pose_probe_totalcapture_mae_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_probe_totalcapture_mae_flu.yaml \
	--run          vit1d_arm_pose_probe_totalcapture_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_probe_totalcapture_mae_flu_eval_dataset

# Arm pose - finetune (TotalCapture MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_totalcapture_mae_flu.yaml \
	--run-name     vit1d_arm_pose_finetune_totalcapture_mae_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_totalcapture_mae_flu.yaml \
	--run          vit1d_arm_pose_finetune_totalcapture_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_finetune_totalcapture_mae_flu_eval_dataset

# Odom - probe (TotalCapture MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_totalcapture_mae_flu.yaml \
	--run-name     vit1d_odom_probe_totalcapture_mae_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_totalcapture_mae_flu.yaml \
	--run          vit1d_odom_probe_totalcapture_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--out-dir      data_vis/odom_probe_totalcapture_mae_flu_eval_dataset

# Odom - finetune (TotalCapture MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_totalcapture_mae_flu.yaml \
	--run-name     vit1d_odom_finetune_totalcapture_mae_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_totalcapture_mae_flu.yaml \
	--run          vit1d_odom_finetune_totalcapture_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--out-dir      data_vis/odom_finetune_totalcapture_mae_flu_eval_dataset

# Full-body pose - probe (TotalCapture MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_totalcapture_mae_flu.yaml \
	--run-name     vit1d_full_body_pose_probe_totalcapture_mae_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_totalcapture_mae_flu.yaml \
	--run          vit1d_full_body_pose_probe_totalcapture_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   24 \
	--out-dir      data_vis/full_body_pose_probe_totalcapture_mae_flu_eval_dataset

# Full-body pose - finetune (TotalCapture MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_totalcapture_mae_flu.yaml \
	--run-name     vit1d_full_body_pose_finetune_totalcapture_mae_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_totalcapture_mae_flu.yaml \
	--run          vit1d_full_body_pose_finetune_totalcapture_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   24 \
	--out-dir      data_vis/full_body_pose_finetune_totalcapture_mae_flu_eval_dataset

############################# Nymeria Real-Wrist MAE ENCODER — PROBE/FINETUNE #############################

# Arm pose - probe (Nymeria real-wrist MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_probe_totalcapture_nymeria_mae_flu.yaml \
	--run-name     vit1d_arm_pose_probe_totalcapture_nymeria_mae_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_probe_totalcapture_nymeria_mae_flu.yaml \
	--run          vit1d_arm_pose_probe_totalcapture_nymeria_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_probe_totalcapture_nymeria_mae_flu_eval_dataset

# Arm pose - finetune (Nymeria real-wrist MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_totalcapture_nymeria_mae_flu.yaml \
	--run-name     vit1d_arm_pose_finetune_totalcapture_nymeria_mae_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_totalcapture_nymeria_mae_flu.yaml \
	--run          vit1d_arm_pose_finetune_totalcapture_nymeria_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_finetune_totalcapture_nymeria_mae_flu_eval_dataset

# Odom - probe (Nymeria real-wrist MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_totalcapture_nymeria_mae_flu.yaml \
	--run-name     vit1d_odom_probe_totalcapture_nymeria_mae_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_totalcapture_nymeria_mae_flu.yaml \
	--run          vit1d_odom_probe_totalcapture_nymeria_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--out-dir      data_vis/odom_probe_totalcapture_nymeria_mae_flu_eval_dataset

# Odom - finetune (Nymeria real-wrist MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_totalcapture_nymeria_mae_flu.yaml \
	--run-name     vit1d_odom_finetune_totalcapture_nymeria_mae_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_totalcapture_nymeria_mae_flu.yaml \
	--run          vit1d_odom_finetune_totalcapture_nymeria_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--out-dir      data_vis/odom_finetune_totalcapture_nymeria_mae_flu_eval_dataset

# Full-body pose - probe (Nymeria real-wrist MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_totalcapture_nymeria_mae_flu.yaml \
	--run-name     vit1d_full_body_pose_probe_totalcapture_nymeria_mae_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_totalcapture_nymeria_mae_flu.yaml \
	--run          vit1d_full_body_pose_probe_totalcapture_nymeria_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   24 \
	--out-dir      data_vis/full_body_pose_probe_totalcapture_nymeria_mae_flu_eval_dataset

# Full-body pose - finetune (Nymeria real-wrist MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_totalcapture_nymeria_mae_flu.yaml \
	--run-name     vit1d_full_body_pose_finetune_totalcapture_nymeria_mae_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_totalcapture_nymeria_mae_flu.yaml \
	--run          vit1d_full_body_pose_finetune_totalcapture_nymeria_mae_flu \
	--test-dir     "$TC_TEST_DIR" \
	--ckpt         best \
	--num-joints   24 \
	--out-dir      data_vis/full_body_pose_finetune_totalcapture_nymeria_mae_flu_eval_dataset