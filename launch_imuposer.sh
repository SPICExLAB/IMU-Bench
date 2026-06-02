#!/usr/bin/env bash
# IMUPoser FLU downstream tasks: arm pose, odometry, full-body pose
# Modes: supervised (from scratch) | probe (frozen OCDetect MAE) | finetune (trainable OCDetect MAE)
set -euo pipefail

IMUPOSER_TEST=/mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test

############################# ARM POSE #############################

# Arm pose - fully supervised
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_supervised_imuposer_flu.yaml \
	--run-name     vit1d_pose_supervised_imuposer_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_supervised_imuposer_flu.yaml \
	--run          vit1d_pose_supervised_imuposer_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/pose_supervised_imuposer_flu_eval_dataset

# Arm pose - probe (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_probe_imuposer_ocdetect_flu.yaml \
	--run-name     vit1d_pose_probe_imuposer_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_probe_imuposer_ocdetect_flu.yaml \
	--run          vit1d_pose_probe_imuposer_ocdetect_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/pose_probe_imuposer_ocdetect_flu_eval_dataset

python scripts/vis_imuposer_arm_pose.py \
	--pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00009.pkl \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_probe_imuposer_ocdetect_flu.yaml \
	--run          vit1d_pose_probe_imuposer_ocdetect_flu \
	--ckpt         best \
	--play-fps     10

# Arm pose - finetune (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_finetune_imuposer_ocdetect_flu.yaml \
	--run-name     vit1d_pose_finetune_imuposer_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_finetune_imuposer_ocdetect_flu.yaml \
	--run          vit1d_pose_finetune_imuposer_ocdetect_flu \
	--test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
	--ckpt         best \
	--out-dir      data_vis/pose_finetune_imuposer_ocdetect_flu_eval_dataset


python scripts/vis_imuposer_arm_pose.py \
	--pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00001.pkl \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_finetune_imuposer_ocdetect_flu.yaml \
	--run          vit1d_pose_finetune_imuposer_ocdetect_flu \
	--ckpt         best \
	--play-fps     10

# Arm pose - probe (odom pretrained encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_probe_imuposer_odom_flu.yaml \
	--run-name     vit1d_pose_probe_imuposer_odom_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_probe_imuposer_odom_flu.yaml \
	--run          vit1d_pose_probe_imuposer_odom_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/pose_probe_imuposer_odom_flu_eval_dataset

# Arm pose - finetune (odom pretrained encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_finetune_imuposer_odom_flu.yaml \
	--run-name     vit1d_pose_finetune_imuposer_odom_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_finetune_imuposer_odom_flu.yaml \
	--run          vit1d_pose_finetune_imuposer_odom_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/pose_finetune_imuposer_odom_flu_eval_dataset

# Arm pose - probe (full-body pretrained encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_probe_imuposer_fullbody_flu.yaml \
	--run-name     vit1d_pose_probe_imuposer_fullbody_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_pose_probe_imuposer_fullbody_flu.yaml \
	--run          vit1d_pose_probe_imuposer_fullbody_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/pose_probe_imuposer_fullbody_flu_eval_dataset

############################# ODOMETRY #############################

# Odom - fully supervised (world-frame XYZ displacement)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_imuposer_flu.yaml \
	--run-name     vit1d_odom_imuposer_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_imuposer_flu.yaml \
	--run          vit1d_odom_imuposer_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/odom_imuposer_flu_eval_dataset

# Local displacement + delta yaw - fully supervised
# y = [dx, dy, dz, dψ] in pelvis local frame (Y-up SMPL) + world-up yaw change
# Avoids global-yaw-unobservable problem; reconstruct global traj via: ψ[p+1] = ψ[p] + dψ̂, Δp_w = Rz(ψ[p]) @ d̂
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_local_disp_imuposer_flu.yaml \
	--run-name     vit1d_local_disp_imuposer_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_imuposer_flu.yaml \
	--run          vit1d_odom_imuposer_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/odom_imuposer_flu_eval_dataset

# Odom - probe (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_imuposer_ocdetect_flu.yaml \
	--run-name     vit1d_odom_probe_imuposer_ocdetect_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_imuposer_ocdetect_flu.yaml \
	--run          vit1d_odom_probe_imuposer_ocdetect_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/odom_probe_imuposer_ocdetect_flu_eval_dataset

# Odom - finetune (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_imuposer_ocdetect_flu.yaml \
	--run-name     vit1d_odom_finetune_imuposer_ocdetect_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_imuposer_ocdetect_flu.yaml \
	--run          vit1d_odom_finetune_imuposer_ocdetect_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/odom_finetune_imuposer_ocdetect_flu_eval_dataset


############################# FULL-BODY POSE #############################

# Full-body pose - fully supervised
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_supervised_imuposer_flu.yaml \
	--run-name     vit1d_full_body_pose_supervised_imuposer_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_supervised_imuposer_flu.yaml \
	--run          vit1d_full_body_pose_supervised_imuposer_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/full_body_pose_supervised_imuposer_flu_eval_dataset

# Full-body pose - probe (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_imuposer_ocdetect_flu.yaml \
	--run-name     vit1d_full_body_pose_probe_imuposer_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_imuposer_ocdetect_flu.yaml \
	--run          vit1d_full_body_pose_probe_imuposer_ocdetect_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/full_body_pose_probe_imuposer_ocdetect_flu_eval_dataset

# Full-body pose - finetune (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_imuposer_ocdetect_flu.yaml \
	--run-name     vit1d_full_body_pose_finetune_imuposer_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_imuposer_ocdetect_flu.yaml \
	--run          vit1d_full_body_pose_finetune_imuposer_ocdetect_flu \
	--test-dir     "$IMUPOSER_TEST" \
	--ckpt         best \
	--out-dir      data_vis/full_body_pose_finetune_imuposer_ocdetect_flu_eval_dataset
	
############################# IMUPoser FLU: local-velocity odom — fully supervised from scratch ####
# Encoder + odom decoder trained end-to-end from random init.
# Input : real right-wrist MEMS IMU in FLU body frame [B, 6, T]
# Output: pelvis velocity in wrist FLU frame, 4 sub-patches per patch [B, N_patches, 4, 3]
#   y[b, p, k] = R_X_from_world[t_mid] @ v_world   (m/s)
#   Decoder outputs [B, N_patches, 12] (flattened), MSE loss.
# Data: /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/{train,test}/
# Loader: imu_lm/data/imuposer_wrist_local_odom_loader_FLU.py

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
        --config       configs/base.yaml \
        --model-config configs/vit1d_local_vel_imuposer_flu.yaml \
        --run-name     vit1d_local_vel_imuposer_flu

python scripts/run_local_vel_eval_dataset.py \
        --config       configs/base.yaml \
        --model-config configs/vit1d_local_vel_imuposer_flu.yaml \
        --run          vit1d_local_vel_imuposer_flu \
        --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
        --ckpt         best \
        --out-dir      data_vis/local_vel_imuposer_flu_eval

# Local-velocity odom - probe (Nymeria real-wrist MAE encoder, encoder frozen)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
        --config       configs/base.yaml \
        --model-config configs/vit1d_local_vel_probe_imuposer_nymeria_flu.yaml \
        --run-name     vit1d_local_vel_probe_imuposer_nymeria_flu

python scripts/run_local_vel_eval_dataset.py \
        --config       configs/base.yaml \
        --model-config configs/vit1d_local_vel_probe_imuposer_nymeria_flu.yaml \
        --run          vit1d_local_vel_probe_imuposer_nymeria_flu \
        --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
        --ckpt         best \
        --out-dir      data_vis/local_vel_probe_imuposer_nymeria_flu_eval

# Local-velocity odom - finetune (Nymeria real-wrist MAE encoder, encoder trainable)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
        --config       configs/base.yaml \
        --model-config configs/vit1d_local_vel_finetune_imuposer_nymeria_flu.yaml \
        --run-name     vit1d_local_vel_finetune_imuposer_nymeria_flu

python scripts/run_local_vel_eval_dataset.py \
        --config       configs/base.yaml \
        --model-config configs/vit1d_local_vel_finetune_imuposer_nymeria_flu.yaml \
        --run          vit1d_local_vel_finetune_imuposer_nymeria_flu \
        --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
        --ckpt         best \
        --out-dir      data_vis/local_vel_finetune_imuposer_nymeria_flu_eval