#!/bin/bash
# 1% Nymeria data experiments — 24-joint T-pose sub-patch model.
# All three paradigms: supervised (from scratch), probe (frozen MAE), finetune (MAE warm-start).
# Eval uses run_pose_eval_24j_tpose.py for consistent 24j/23j MPJPE + MPVPE reporting.

NYMERIA_TEST=${NYMERIA_TEST:-/mnt/nas2/processed_data_test_body_Ty_rwrist}

########## Supervised (from scratch, 1% data) ##################################

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_1pct_flu_24j_tpose.yaml \
    --run-name     vit1d_full_body_pose_supervised_nymeria_1pct_flu_24j_tpose

python scripts/run_pose_eval_24j_tpose.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_1pct_flu_24j_tpose.yaml \
    --run          vit1d_full_body_pose_supervised_nymeria_1pct_flu_24j_tpose \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_eval_24j_tpose_supervised_1pct

########## Probe (frozen OCDetect MAE encoder, 1% data) ########################

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_1pct_ocdetect_flu_24j_tpose.yaml \
    --run-name     vit1d_full_body_pose_probe_nymeria_1pct_ocdetect_flu_24j_tpose

python scripts/run_pose_eval_24j_tpose.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_1pct_ocdetect_flu_24j_tpose.yaml \
    --run          vit1d_full_body_pose_probe_nymeria_1pct_ocdetect_flu_24j_tpose \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_eval_24j_tpose_probe_1pct

########## Finetune (MAE encoder warm-started + trainable, 1% data) ############

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_1pct_ocdetect_flu_24j_tpose.yaml \
    --run-name     vit1d_full_body_pose_finetune_nymeria_1pct_ocdetect_flu_24j_tpose

python scripts/run_pose_eval_24j_tpose.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_finetune_nymeria_1pct_ocdetect_flu_24j_tpose.yaml \
    --run          vit1d_full_body_pose_finetune_nymeria_1pct_ocdetect_flu_24j_tpose \
    --test-dir     "$NYMERIA_TEST" \
    --ckpt         best \
    --out-dir      data_vis/pose_eval_24j_tpose_finetune_1pct
