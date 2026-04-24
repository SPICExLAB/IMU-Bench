#!/usr/bin/env bash
# launch_watchHand_data.sh
# Fully-supervised gesture classification on WatchHand IMU data (Study-1, right wrist, 50 Hz FLU).
#
# Three independent runs — one per gesture group:
#   1. Simple gestures     (5 classes)
#   2. ASL gestures        (10 classes)
#   3. Wrist rotations     (3 classes)
#
# Each run trains ViT1D encoder + LinearHead from scratch (or warm-start by
# un-commenting pretrained_run in the config).
#
# Usage:
#   bash launch_watchHand_data.sh            # run all three sequentially
#   CUDA_VISIBLE_DEVICES=1 bash launch_watchHand_data.sh
#
set -euo pipefail

cd "$(dirname "$0")"
source TransfHAR_env/bin/activate

# --------------------------------------------------------------------------
# 1. Simple gestures (5 classes: Thumb/Index/Middle/Ring/Little-bent)
# --------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_gesture_supervised_watchhand_simple.yaml \
  --run-name     vit1d_gesture_supervised_watchhand_simple

# --------------------------------------------------------------------------
# 2. ASL gestures (10 classes: ASL-0 through ASL-9)
# --------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_gesture_supervised_watchhand_asl.yaml \
  --run-name     vit1d_gesture_supervised_watchhand_asl

# --------------------------------------------------------------------------
# 3. Wrist rotations (3 classes: Flexion / Extension / Ulnar_deviation)
# --------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_gesture_supervised_watchhand_wrist.yaml \
  --run-name     vit1d_gesture_supervised_watchhand_wrist

# ==========================================================================
# WITHIN-SUBJECT splits (train=blocks 0–7, val=blocks 8–9, same 24 subjects)
# ==========================================================================

# --------------------------------------------------------------------------
# 4. Simple gestures — within subject (5 classes)
# --------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_gesture_supervised_watchhand_simple_within.yaml \
  --run-name     vit1d_gesture_supervised_watchhand_simple_within

# --------------------------------------------------------------------------
# 5. ASL gestures — within subject (10 classes)
# --------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_gesture_supervised_watchhand_asl_within.yaml \
  --run-name     vit1d_gesture_supervised_watchhand_asl_within

# --------------------------------------------------------------------------
# 6. Wrist rotations — within subject (3 classes)
# --------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_gesture_supervised_watchhand_wrist_within.yaml \
  --run-name     vit1d_gesture_supervised_watchhand_wrist_within
