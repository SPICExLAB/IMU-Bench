#!/usr/bin/env bash
# Nymeria real-wrist FLU downstream tasks: arm pose, odometry, full-body pose
# Encoder: OCDetect MAE pretrained (vit1d_mae_ocdetect_flu)
# Modes: supervised (from scratch) | probe (frozen encoder) | finetune (trainable encoder)
set -euo pipefail

NYMERIA_TRAIN=/mnt/nas2/processed_data_body_Ty_rwrist
NYMERIA_TEST=/mnt/nas2/processed_data_test_body_Ty_rwrist

############################# ARM POSE #############################

# Arm pose - fully supervised
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_supervised_nymeria_real_wrist_flu.yaml \
	--run-name     vit1d_arm_pose_supervised_nymeria_real_wrist_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_supervised_nymeria_real_wrist_flu.yaml \
	--run          vit1d_arm_pose_supervised_nymeria_real_wrist_flu \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_supervised_nymeria_real_wrist_flu_eval_dataset

# Arm pose - probe (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
	--run-name     vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
	--run          vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_probe_nymeria_real_wrist_ocdetect_flu_eval_dataset

# Arm pose - finetune (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
	--run-name     vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
	--run          vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_finetune_nymeria_real_wrist_ocdetect_flu_eval_dataset


############################# ODOMETRY #############################

# Odom - fully supervised
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_supervised_nymeria_real_wrist_flu.yaml \
	--run-name     vit1d_odom_supervised_nymeria_real_wrist_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_supervised_nymeria_real_wrist_flu.yaml \
	--run          vit1d_odom_supervised_nymeria_real_wrist_flu \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--out-dir      data_vis/odom_supervised_nymeria_real_wrist_flu_eval_dataset

# Odom - probe (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_nymeria_real_wrist_ocdetect_flu.yaml \
	--run-name     vit1d_odom_probe_nymeria_real_wrist_ocdetect_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_probe_nymeria_real_wrist_ocdetect_flu.yaml \
	--run          vit1d_odom_probe_nymeria_real_wrist_ocdetect_flu \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--out-dir      data_vis/odom_probe_nymeria_real_wrist_ocdetect_flu_eval_dataset

# Odom - finetune (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
	--run-name     vit1d_odom_finetune_nymeria_real_wrist_ocdetect_flu

python scripts/run_odom_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
	--run          vit1d_odom_finetune_nymeria_real_wrist_ocdetect_flu \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--out-dir      data_vis/odom_finetune_nymeria_real_wrist_ocdetect_flu_eval_dataset


############################# FULL-BODY POSE #############################

# Full-body pose - fully supervised
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu.yaml \
	--run-name     vit1d_full_body_pose_supervised_nymeria_real_wrist_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu.yaml \
	--run          vit1d_full_body_pose_supervised_nymeria_real_wrist_flu \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--num-joints   23 \
	--out-dir      data_vis/full_body_pose_supervised_nymeria_real_wrist_flu_eval_dataset

# Full-body pose - probe (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
	--run-name     vit1d_full_body_pose_probe_nymeria_real_wrist_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
	--run          vit1d_full_body_pose_probe_nymeria_real_wrist_ocdetect_flu \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--num-joints   23 \
	--out-dir      data_vis/full_body_pose_probe_nymeria_real_wrist_ocdetect_flu_eval_dataset

# Full-body pose - finetune (OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
	--run-name     vit1d_full_body_pose_finetune_nymeria_real_wrist_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_full_body_pose_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
	--run          vit1d_full_body_pose_finetune_nymeria_real_wrist_ocdetect_flu \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--num-joints   23 \
	--out-dir      data_vis/full_body_pose_finetune_nymeria_real_wrist_ocdetect_flu_eval_dataset


python scripts/vis_nymeria_arm_pose.py \
    --pkl      /mnt/nas2/processed_data_test_body_Ty_rwrist/20231222_s0_denise_carter_act2_d8mdi5_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config   configs/base.yaml \
    --model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
    --run      vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu \
    --ckpt     best


python scripts/vis_nymeria_full_body_pose.py \
    --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20231222_s0_denise_carter_act2_d8mdi5_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
    --run          vit1d_full_body_pose_probe_nymeria_real_wrist_ocdetect_flu \
    --ckpt         best

python scripts/vis_nymeria_full_body_pose.py \
    --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20231222_s0_denise_carter_act2_d8mdi5_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu.yaml \
    --run          vit1d_full_body_pose_supervised_nymeria_real_wrist_flu \
    --ckpt         best

