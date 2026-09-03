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
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
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

# Arm pose - finetune (OCDetect MAE encoder v2)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
	--run-name     vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu_v2


# Arm pose - finetune (Realdata MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_realdata.yaml \
	--run-name     vit1d_arm_pose_finetune_nymeria_real_wrist_realdata

python scripts/run_pose_eval_dataset.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_realdata.yaml \
	--run          vit1d_arm_pose_finetune_nymeria_real_wrist_realdata \
	--test-dir     "$NYMERIA_TEST" \
	--ckpt         best \
	--num-joints   5 \
	--out-dir      data_vis/arm_pose_finetune_nymeria_real_wrist_realdata_eval_dataset


############################# ODOMETRY #############################

# Odom - fully supervised
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_supervised_nymeria_real_wrist_flu.yaml \
	--run-name     vit1d_odom_supervised_nymeria_real_wrist_flu

# Odom - fully supervised - simple (fewer layers, no temporal attention)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
	--config       configs/base.yaml \
	--model-config configs/vit1d_odom_supervised_simple_nymeria_real_wrist_flu.yaml \
	--run-name     vit1d_odom_supervised_simple_nymeria_real_wrist_flu

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
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
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

# Full-body pose - 24 joints supervised (from scratch)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose.yaml \
    --run-name     vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose

python scripts/run_pose_eval_24j_tpose.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose.yaml \
    --run          vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_eval_24j_tpose

# Full-body pose - 24 joints probe (frozen OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_flu_24j_tpose.yaml \
    --run-name     vit1d_full_body_pose_probe_nymeria_real_wrist_flu_24j_tpose

python scripts/run_pose_eval_24j_tpose.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_flu_24j_tpose.yaml \
    --run          vit1d_full_body_pose_probe_nymeria_real_wrist_flu_24j_tpose \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_eval_24j_tpose_probe

# Full-body pose - 24 joints finetune (MAE encoder warm-started, trainable)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_real_wrist_flu_24j_tpose.yaml \
    --run-name     vit1d_full_body_pose_finetune_nymeria_real_wrist_flu_24j_tpose

python scripts/run_pose_eval_24j_tpose.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_real_wrist_flu_24j_tpose.yaml \
    --run          vit1d_full_body_pose_finetune_nymeria_real_wrist_flu_24j_tpose \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_eval_24j_tpose_finetune

######################## FULL-BODY POSE 23J (K=8, jitter) ########################

# Full-body pose - 23 joints supervised (from scratch)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_23j.yaml \
    --run-name     vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_23j_simple

CUDA_VISIBLE_DEVICES=1 python scripts/run_pose_eval_23j.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_23j.yaml \
    --run          vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_23j \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         latest \
    --out-dir      data_vis/pose_eval_23j_supervised

# Full-body pose - 23 joints supervised simple (from scratch)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_simple_nymeria_real_wrist_flu_23j.yaml \
    --run-name     vit1d_full_body_pose_supervised_simple_nymeria_real_wrist_flu_23j

CUDA_VISIBLE_DEVICES=1 python scripts/run_pose_eval_23j.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_simple_nymeria_real_wrist_flu_23j.yaml \
    --run          vit1d_full_body_pose_supervised_simple_nymeria_real_wrist_flu_23j \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         latest \
    --out-dir      data_vis/pose_eval_23j_supervised

# Full-body pose - 23 joints linear probe (frozen OCDetect MAE encoder, no transformer decoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_linear_probe_nymeria_real_wrist_ocdetect_flu_23j.yaml \
    --run-name     vit1d_full_body_pose_linear_probe_nymeria_real_wrist_ocdetect_flu_23j

# Full-body pose - 23 joints linear probe (frozen OCDetect DINO encoder, any orientation)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_linear_probe_nymeria_real_wrist_dino_anyrot_23j.yaml \
    --run-name     vit1d_full_body_pose_linear_probe_nymeria_real_wrist_dino_anyrot_23j

python scripts/run_pose_eval_23j.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_linear_probe_nymeria_real_wrist_ocdetect_flu_23j.yaml \
    --run          vit1d_full_body_pose_linear_probe_nymeria_real_wrist_ocdetect_flu_23j \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_eval_23j_linear_probe

# Full-body pose - 23 joints probe (frozen OCDetect MAE encoder)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_flu_23j.yaml \
    --run-name     vit1d_full_body_pose_probe_nymeria_real_wrist_flu_23j

# Full-body pose - 23 joints probe (frozen OCDetect MAE encoder, no norm augmentation)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_no_norm_aug_23j.yaml \
    --run-name     vit1d_full_body_pose_probe_nymeria_real_wrist_no_norm_aug_23j

# Full-body pose - 23 joints probe (frozen OCDetect DINO encoder, any orientation)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_dino_anyrot_23j.yaml \
    --run-name     vit1d_full_body_pose_probe_nymeria_real_wrist_dino_anyrot_23j

# Full-body pose - 23 joints probe (frozen OCDetect DINO v4 encoder, any orientation)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_dino_anyrot_v4_23j.yaml \
    --run-name     vit1d_full_body_pose_probe_nymeria_real_wrist_dino_anyrot_v4_23j

# Full-body pose - 23 joints linear probe (frozen OCDetect DINO v4 encoder, any orientation)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_linear_probe_nymeria_real_wrist_dino_anyrot_v4_23j.yaml \
    --run-name     vit1d_full_body_pose_linear_probe_nymeria_real_wrist_dino_anyrot_v4_23j

# Full-body pose - 23 joints linear probe (frozen OCDetect DINO MAE encoder, any orientation)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_linear_probe_nymeria_real_wrist_mae_dino_anyrot_23j.yaml \
    --run-name     vit1d_full_body_pose_linear_probe_nymeria_real_wrist_mae_dino_anyrot_23j

python scripts/run_pose_eval_23j.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_flu_23j.yaml \
    --run          vit1d_full_body_pose_probe_nymeria_real_wrist_flu_23j \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_eval_23j_probe

# Full-body pose - 23 joints finetune (MAE encoder warm-started, trainable)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_real_wrist_flu_23j.yaml \
    --run-name     vit1d_full_body_pose_finetune_nymeria_real_wrist_flu_23j

CUDA_VISIBLE_DEVICES=1 python scripts/run_pose_eval_23j.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_real_wrist_flu_23j.yaml \
    --run          vit1d_full_body_pose_finetune_nymeria_real_wrist_flu_23j \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_eval_23j_finetune


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


