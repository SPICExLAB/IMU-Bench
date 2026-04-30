source TransfHAR_env/bin/activate

############################# MAE pretrain on real 6D data (~154 h) ##################
# train_on_datasets: ["pamap2", "recofit", "samosa", "shoaib", "ut_watch", "utd_mhad", "wisdm"]
# Encoder saved to: runs/vit1d_realdata_pretrain/checkpoints/best.pt
# (set train_on_datasets in configs/base.yaml before running)

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d.yaml \
  --run-name     vit1d_realdata_pretrain

############################# MAE pretrain on real 6D data with 50→16Hz→50Hz degradation
# Same datasets as vit1d_realdata_pretrain, but each session is downsampled to 16 Hz
# then band-limited interpolated back to 50 Hz before windowing and MAE training.
# Useful for studying whether degraded-signal pretraining transfers to low-rate devices.
# Encoder saved to: runs/vit1d_mae_realdata_16hz_to_50hz/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_realdata_16hz_to_50hz.yaml \
  --run-name     vit1d_mae_realdata_16hz_to_50hz

############################# MAE pretrain on real 6D data with 50→30Hz→50Hz degradation
# Same datasets as vit1d_realdata_pretrain, but each session is downsampled to 30 Hz
# then band-limited interpolated back to 50 Hz before windowing and MAE training.
# Simulates 30 Hz device quality (e.g. IMUPoser) — encoder may transfer better
# to 30 Hz downstream tasks compared to the full-bandwidth 50 Hz encoder.
# Encoder saved to: runs/vit1d_mae_realdata_30hz_to_50hz/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_realdata_30hz_to_50hz.yaml \
  --run-name     vit1d_mae_realdata_30hz_to_50hz

############################# IMUPoser FLU: pose probe — 16Hz→50Hz realdata encoder ###
# Frozen vit1d_mae_realdata_16hz_to_50hz encoder; only pose decoder trains.
# Data: IMUPoser right-wrist FLU IMU → right-arm chain labels (5 × 6D = 30 dims)
# Requires: runs/vit1d_mae_realdata_16hz_to_50hz/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_16hz_to_50hz.yaml \
  --run-name     vit1d_pose_probe_imuposer_16hz_to_50hz

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_16hz_to_50hz.yaml \
  --run          vit1d_pose_probe_imuposer_16hz_to_50hz \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_imuposer_16hz_to_50hz_eval_dataset

############################# IMUPoser FLU: pose finetune — 16Hz→50Hz realdata encoder #
# Encoder warm-started from vit1d_mae_realdata_16hz_to_50hz and kept trainable (lower LR).
# Data: IMUPoser right-wrist FLU IMU → right-arm chain labels (5 × 6D = 30 dims)
# Requires: runs/vit1d_mae_realdata_16hz_to_50hz/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_16hz_to_50hz.yaml \
  --run-name     vit1d_pose_finetune_imuposer_16hz_to_50hz

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_16hz_to_50hz.yaml \
  --run          vit1d_pose_finetune_imuposer_16hz_to_50hz \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_imuposer_16hz_to_50hz_eval_dataset

############################# IMUPoser FLU 16→50Hz: fully supervised from scratch ####
# Encoder + pose decoder trained end-to-end from random init.
# IMU data is 50→16→50 Hz degraded; labels are still 50 Hz SMPL FK ground truth.
# Baseline: how well can the model learn right-arm pose from degraded-only IMU?

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_supervised_imuposer_flu_16hz_to_50hz.yaml \
  --run-name     vit1d_pose_supervised_imuposer_flu_16hz_to_50hz

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_supervised_imuposer_flu_16hz_to_50hz.yaml \
  --run          vit1d_pose_supervised_imuposer_flu_16hz_to_50hz \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_supervised_imuposer_flu_16hz_to_50hz_eval_dataset

############################# IMUPoser FLU 16→50Hz: pose probe — 16Hz→50Hz realdata encoder
# Frozen vit1d_mae_realdata_16hz_to_50hz encoder; only pose decoder trains.
# Both pretraining data and downstream IMU are 50→16→50 Hz degraded (domain-matched).
# Requires: runs/vit1d_mae_realdata_16hz_to_50hz/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_flu_16hz_to_50hz.yaml \
  --run-name     vit1d_pose_probe_imuposer_flu_16hz_to_50hz

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_flu_16hz_to_50hz.yaml \
  --run          vit1d_pose_probe_imuposer_flu_16hz_to_50hz \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_imuposer_flu_16hz_to_50hz_eval_dataset

############################# IMUPoser FLU 16→50Hz: pose finetune — 16Hz→50Hz realdata encoder
# Encoder warm-started from vit1d_mae_realdata_16hz_to_50hz and kept trainable (lower LR).
# Both pretraining data and downstream IMU are 50→16→50 Hz degraded (domain-matched).
# Requires: runs/vit1d_mae_realdata_16hz_to_50hz/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_flu_16hz_to_50hz.yaml \
  --run-name     vit1d_pose_finetune_imuposer_flu_16hz_to_50hz

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_flu_16hz_to_50hz.yaml \
  --run          vit1d_pose_finetune_imuposer_flu_16hz_to_50hz \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_imuposer_flu_16hz_to_50hz_eval_dataset

############################# IMUPoser FLU 16→50Hz: pose probe — original 50Hz realdata encoder
# Frozen vit1d_realdata_pretrain encoder (pretrained on full-bandwidth 50 Hz data); only pose decoder trains.
# Downstream IMU is 50→16→50 Hz degraded (cross-domain: encoder sees high-BW, data is low-BW).
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_flu_16hz_to_50hz_realdata.yaml \
  --run-name     vit1d_pose_probe_imuposer_flu_16hz_to_50hz_realdata

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_flu_16hz_to_50hz_realdata.yaml \
  --run          vit1d_pose_probe_imuposer_flu_16hz_to_50hz_realdata \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_imuposer_flu_16hz_to_50hz_realdata_eval_dataset

############################# IMUPoser FLU 16→50Hz: pose finetune — original 50Hz realdata encoder
# Encoder warm-started from vit1d_realdata_pretrain and kept trainable (lower LR).
# Downstream IMU is 50→16→50 Hz degraded (cross-domain: encoder sees high-BW, data is low-BW).
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_flu_16hz_to_50hz_realdata.yaml \
  --run-name     vit1d_pose_finetune_imuposer_flu_16hz_to_50hz_realdata

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_flu_16hz_to_50hz_realdata.yaml \
  --run          vit1d_pose_finetune_imuposer_flu_16hz_to_50hz_realdata \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_imuposer_flu_16hz_to_50hz_realdata_eval_dataset

############################# MAE pretrain on Nymeria FLU data ######################
# Pretrain the ViT1D encoder on Nymeria FLU synthetic right-wrist IMU
# (linear acc, FLU body frame — no gravity component).
# Encoder saved to: runs/vit1d_mae_nymeria_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_nymeria_flu.yaml \
  --run-name     vit1d_mae_nymeria_flu

############################# MAE pretrain on Nymeria REAL wrist IMU (FLU) ############
# Pretrain the ViT1D encoder on Nymeria real wrist IMU data (FLU frame).
# Pipeline: wrist IMU local CPF → global CPF (world) → local FLU.
# Requires pkl files with has_wrist_imu=True.
# Encoder saved to: runs/vit1d_mae_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_mae_nymeria_real_wrist_flu

############################# MAE pretrain on realdata + Nymeria REAL wrist IMU (FLU) #
# Pretrain the ViT1D encoder on the union of:
#   (a) Nymeria REAL wrist IMU (FLU frame, has_wrist_imu=True pkl files)
#   (b) Real 6D parquet datasets (pamap2, recofit, samosa, shoaib, ut_watch, utd_mhad, wisdm)
# Encoder saved to: runs/vit1d_mae_realdata_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_realdata_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_mae_realdata_nymeria_real_wrist_flu

############################# MAE pretrain on realdata + Nymeria REAL wrist IMU (FLU) — 10.24 s window
# Same data as vit1d_mae_realdata_nymeria_real_wrist_flu but with a fixed 10.24 s window
# (max_patches_per_channel=32, batch_size=64 to keep patch-count budget constant).
# Encoder saved to: runs/vit1d_mae_realdata_nymeria_real_wrist_flu_10s/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_realdata_nymeria_real_wrist_flu_10s.yaml \
  --run-name     vit1d_mae_realdata_nymeria_real_wrist_flu_10s

############################# GRU: MAE pretrain on realdata + Nymeria REAL wrist IMU (FLU)
# Same combined dataset as vit1d_mae_realdata_nymeria_real_wrist_flu but with a
# GRU encoder (hidden_size=384, 4 layers, bidirectional=false, mean pooling).
# Encoder saved to: runs/rnn1d_mae_realdata_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/rnn1d_mae_realdata_nymeria_real_wrist_flu.yaml \
  --run-name     rnn1d_mae_realdata_nymeria_real_wrist_flu

############################# TCN (CNN1D): MAE pretrain on realdata + Nymeria REAL wrist IMU (FLU)
# Same combined dataset as vit1d_mae_realdata_nymeria_real_wrist_flu but with a
# SAMoSA-style CNN1D encoder (4 conv, 3 FC, embed_dim=384).
# Encoder saved to: runs/cnn1d_mae_realdata_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/cnn1d_mae_realdata_nymeria_real_wrist_flu.yaml \
  --run-name     cnn1d_mae_realdata_nymeria_real_wrist_flu

############################# IMUPoser FLU: pose probe — Nymeria real wrist MAE encoder
# Frozen vit1d_mae_nymeria_real_wrist_flu encoder; only pose decoder trains.
# Data: IMUPoser right-wrist FLU IMU → right-arm chain labels (5 × 6D = 30 dims)
# Requires: runs/vit1d_mae_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_nymeria_real_wrist.yaml \
  --run-name     vit1d_pose_probe_imuposer_nymeria_real_wrist

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_nymeria_real_wrist.yaml \
  --run          vit1d_pose_probe_imuposer_nymeria_real_wrist \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_imuposer_nymeria_real_wrist_eval_dataset

############################# IMUPoser FLU: pose finetune — Nymeria real wrist MAE encoder
# Encoder warm-started from vit1d_mae_nymeria_real_wrist_flu and kept trainable (lower LR).
# Requires: runs/vit1d_mae_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_nymeria_real_wrist.yaml \
  --run-name     vit1d_pose_finetune_imuposer_nymeria_real_wrist

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_nymeria_real_wrist.yaml \
  --run          vit1d_pose_finetune_imuposer_nymeria_real_wrist \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_imuposer_nymeria_real_wrist_eval_dataset

############################# Pose probe: Nymeria FLU MAE encoder + FLU Nymeria data #
# Frozen vit1d_mae_nymeria_flu encoder; only pose decoder trains.
# Data loader: nymeria_wrist_loader_FLU  (linear acc, FLU body frame, right-arm chain)
# Requires: runs/vit1d_mae_nymeria_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_nymeria_flu.yaml \
  --run-name     vit1d_pose_probe_nymeria_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_nymeria_flu.yaml \
  --run          vit1d_pose_probe_nymeria_flu \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_nymeria_flu_eval_dataset

############################# Pose finetune: Nymeria FLU MAE encoder + FLU Nymeria data
# Encoder warm-started from vit1d_mae_nymeria_flu and kept trainable (lower LR).
# Requires: runs/vit1d_mae_nymeria_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_nymeria_flu.yaml \
  --run-name     vit1d_pose_finetune_nymeria_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_nymeria_flu.yaml \
  --run          vit1d_pose_finetune_nymeria_flu \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_nymeria_flu_eval_dataset

############################# MAE pretrain on combined Nymeria FLU + real 6D data ####
# Train on the union of:
#   (a) Nymeria FLU synthetic IMU (nymeria_wrist_loader_FLU, linear acc, FLU body frame)
#   (b) Real 6D parquet: pamap2, recofit, samosa, shoaib, ut_watch, utd_mhad, wisdm (~154 h)
# Encoder saved to: runs/vit1d_mae_combined_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_combined_flu.yaml \
  --run-name     vit1d_mae_combined_flu

############################# Pose probe: realdata MAE encoder + FLU Nymeria data ####
# Frozen vit1d_realdata_pretrain encoder; only pose decoder trains.
# Data loader: nymeria_wrist_loader_FLU  (linear acc, FLU body frame, right-arm chain)
# Step 1: ensure vit1d_realdata_pretrain/checkpoints/best.pt exists

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_FLU.yaml \
  --run-name     vit1d_pose_probe_FLU

# Single-sequence eval (visual sanity check)
# NOTE: run_pose_eval.py extracts IMU inline (world-frame, not FLU).
# To eval correctly in FLU frame, run_pose_eval.py would need FLU support.
python scripts/run_pose_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_FLU.yaml \
  --run          vit1d_pose_probe_FLU \
  --pkl          /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
  --ckpt         best \
  --max-frames   5000 \
  --out-dir      data_vis/pose_probe_FLU_eval

# Dataset-level eval: geodesic + MPJPE across all test sequences
python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_FLU.yaml \
  --run          vit1d_pose_probe_FLU \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_FLU_eval_dataset

############################# Pose finetune: realdata MAE encoder (trainable) + FLU Nymeria data ####
# Encoder warm-started from vit1d_realdata_pretrain and kept trainable (lower LR).
# Data loader: nymeria_wrist_loader_FLU  (linear acc, FLU body frame, right-arm chain)
# Step 1: ensure vit1d_realdata_pretrain/checkpoints/best.pt exists

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_FLU.yaml \
  --run-name     vit1d_pose_finetune_FLU

# Single-sequence eval
python scripts/run_pose_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_FLU.yaml \
  --run          vit1d_pose_finetune_FLU \
  --pkl          /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
  --ckpt         best \
  --max-frames   5000 \
  --out-dir      data_vis/pose_finetune_FLU_eval

# Dataset-level eval
python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_FLU.yaml \
  --run          vit1d_pose_finetune_FLU \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_FLU_eval_dataset

############################# Pose probe: combined MAE encoder + FLU Nymeria data ####
# Frozen vit1d_mae_combined_flu encoder; only pose decoder trains.
# Step 1: ensure vit1d_mae_combined_flu/checkpoints/best.pt exists

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_combined_FLU.yaml \
  --run-name     vit1d_pose_probe_combined_FLU

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_combined_FLU.yaml \
  --run          vit1d_pose_probe_combined_FLU \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_combined_FLU_eval_dataset

############################# Pose finetune: combined MAE encoder (trainable) + FLU Nymeria data ##
# Encoder warm-started from vit1d_mae_combined_flu and kept trainable (lower LR).
# Step 1: ensure vit1d_mae_combined_flu/checkpoints/best.pt exists

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_combined_FLU.yaml \
  --run-name     vit1d_pose_finetune_combined_FLU

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_combined_FLU.yaml \
  --run          vit1d_pose_finetune_combined_FLU \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_combined_FLU_eval_dataset

############################# IMUPoser FLU: fully supervised from scratch ###########
# Encoder + pose decoder trained end-to-end from random init on IMUPoser FLU data.
# No pretrained encoder is used (baseline for comparing transfer-learning benefit).

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_supervised_imuposer_flu.yaml \
  --run-name     vit1d_pose_supervised_imuposer_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_supervised_imuposer_flu.yaml \
  --run          vit1d_pose_supervised_imuposer_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_supervised_imuposer_flu_eval_dataset

############################# Nymeria FLU: fully supervised from scratch #############
# Encoder + pose decoder trained end-to-end from random init on Nymeria FLU data.
# Synthetic right-wrist IMU (linear acc, FLU body frame) → right-arm chain labels.
# No pretrained encoder is used (baseline for comparing transfer-learning benefit).

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_supervised_nymeria_flu.yaml \
  --run-name     vit1d_pose_supervised_nymeria_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_supervised_nymeria_flu.yaml \
  --run          vit1d_pose_supervised_nymeria_flu \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_supervised_nymeria_flu_eval_dataset

############################# IMUPoser FLU: MAE pretrain ##############################
# Pretrain the ViT1D encoder on IMUPoser right-wrist FLU IMU (30 Hz → 50 Hz resample).
# Data: /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/{train,val}/
# Encoder saved to: runs/vit1d_mae_imuposer_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_imuposer_flu.yaml \
  --run-name     vit1d_mae_imuposer_flu

############################# IMUPoser FLU: pose probe — realdata encoder #############
# Frozen vit1d_realdata_pretrain encoder; only pose decoder trains.
# Data: IMUPoser right-wrist FLU IMU → right-arm chain labels
# Requires: vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_realdata.yaml \
  --run-name     vit1d_pose_probe_imuposer_realdata

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_realdata.yaml \
  --run          vit1d_pose_probe_imuposer_realdata \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_imuposer_realdata_eval_dataset

############################# IMUPoser FLU: pose probe — Nymeria MAE encoder ##########
# Frozen vit1d_mae_nymeria encoder; only pose decoder trains.
# Requires: vit1d_mae_nymeria/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_nymeria.yaml \
  --run-name     vit1d_pose_probe_imuposer_nymeria

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_nymeria.yaml \
  --run          vit1d_pose_probe_imuposer_nymeria \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_imuposer_nymeria_eval_dataset

############################# IMUPoser FLU: full finetune — realdata encoder ##########
# Encoder warm-started from vit1d_realdata_pretrain and kept trainable (lower LR).
# Requires: vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_realdata.yaml \
  --run-name     vit1d_pose_finetune_imuposer_realdata

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_realdata.yaml \
  --run          vit1d_pose_finetune_imuposer_realdata \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_imuposer_realdata_eval_dataset

############################# IMUPoser FLU: full finetune — Nymeria MAE encoder #######
# Encoder warm-started from vit1d_mae_nymeria and kept trainable (lower LR).
# Requires: vit1d_mae_nymeria/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_nymeria.yaml \
  --run-name     vit1d_pose_finetune_imuposer_nymeria

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_nymeria.yaml \
  --run          vit1d_pose_finetune_imuposer_nymeria \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_imuposer_nymeria_eval_dataset

############################# IMUPoser FLU: pose probe — OCDetect FLU MAE encoder ####
# Frozen vit1d_mae_ocdetect_flu encoder; only pose decoder trains.
# Requires: runs/vit1d_mae_ocdetect_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_ocdetect_flu.yaml \
  --run-name     vit1d_pose_probe_imuposer_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_ocdetect_flu.yaml \
  --run          vit1d_pose_probe_imuposer_ocdetect_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_imuposer_ocdetect_flu_eval_dataset

############################# IMUPoser FLU: pose finetune — OCDetect FLU MAE encoder #
# Encoder warm-started from vit1d_mae_ocdetect_flu and kept trainable (lower LR).
# Requires: runs/vit1d_mae_ocdetect_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
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


# Visualize FLU sequence (sanity check)
python scripts/vis_wrist_flu.py \
    --pkl /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
    --fps 50 \
    --axis-len 0.25

# Analyze IMU stats (sanity check)
python scripts/analyze_imu_stats.py \
    --nymeria-dir /mnt/nas2/processed_data_test_body_Ty \
    --parquet     /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
    --nymeria-n   50 \
    --parquet-n   100000 \
    --fps 50 \
    --out-dir data_vis/imu_stats

############################# IMU-text CLIP probe: realdata encoder + HumanML3D FLU ##
# Frozen vit1d_realdata_pretrain encoder; only CLIP projection head trains.
# IMU derived in FLU body frame via humanml3d_dataset_FLU.py.
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt
# Text embeddings cached at: /home/yiquan/HumanML3D/clip_text_emb_cache.pt
#   (generate with: python scripts/cache_clip_text_embeddings.py \
#       --humanml3d-root /home/yiquan/HumanML3D \
#       --out /home/yiquan/HumanML3D/clip_text_emb_cache.pt)

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_clip_humanml3d_flu_probe.yaml \
  --run-name     vit1d_clip_humanml3d_flu_probe

############################# IMU-text CLIP finetune: realdata encoder + HumanML3D FLU
# Encoder warm-started from vit1d_realdata_pretrain and kept TRAINABLE (lower LR).
# IMU derived in FLU body frame via humanml3d_dataset_FLU.py.
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_clip_humanml3d_flu_finetune.yaml \
  --run-name     vit1d_clip_humanml3d_flu_finetune

############################# IMU-text CLIP from scratch: HumanML3D FLU — fixed 10.24 s window
# Train IMU encoder + CLIP projection head end-to-end from random init.
# Fixed 10.24 s window (32 patches per channel, batch_size=64).
# IMU derived in FLU body frame via humanml3d_dataset_FLU.py.
# Text embeddings from cached CLIP embeddings at:
#   /home/yiquan/HumanML3D/clip_text_emb_cache.pt
#   (generate with: python scripts/cache_clip_text_embeddings.py \
#       --humanml3d-root /home/yiquan/HumanML3D \
#       --out /home/yiquan/HumanML3D/clip_text_emb_cache.pt)

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_clip_humanml3d_flu_10s.yaml \
  --run-name     vit1d_clip_humanml3d_flu_10s

############################# IMU-text CLIP probe: 10s MAE encoder + HumanML3D FLU — fixed 10.24 s window
# Frozen vit1d_mae_realdata_nymeria_real_wrist_flu_10s encoder; only CLIP projection trains.
# Fixed 10.24 s window (32 patches per channel).
# Requires: runs/vit1d_mae_realdata_nymeria_real_wrist_flu_10s/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_clip_humanml3d_flu_probe_10s.yaml \
  --run-name     vit1d_clip_humanml3d_flu_probe_10s

############################# IMU-text CLIP finetune: 10s MAE encoder + HumanML3D FLU — fixed 10.24 s window
# Encoder warm-started from vit1d_mae_realdata_nymeria_real_wrist_flu_10s and kept TRAINABLE.
# Fixed 10.24 s window (32 patches per channel).
# Requires: runs/vit1d_mae_realdata_nymeria_real_wrist_flu_10s/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_clip_humanml3d_flu_finetune_10s.yaml \
  --run-name     vit1d_clip_humanml3d_flu_finetune_10s

############################# IMUPoser FLU: fully supervised odometry from scratch ####
# Encoder + odom decoder trained end-to-end from random init.
# Real right-wrist MEMS IMU in FLU body frame → per-patch pelvis root displacement [3].
# Data: /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/{train,test}/
# Encoder saved to: runs/vit1d_odom_imuposer_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_imuposer_flu.yaml \
  --run-name     vit1d_odom_imuposer_flu

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_imuposer_flu.yaml \
  --run          vit1d_odom_imuposer_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/odom_imuposer_flu_eval_dataset

############################# IMUPoser FLU: odom probe — realdata encoder #############
# Frozen vit1d_realdata_pretrain encoder; only odom decoder trains.
# Real right-wrist MEMS IMU in FLU body frame → per-patch pelvis root displacement [3].
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_imuposer_realdata.yaml \
  --run-name     vit1d_odom_probe_imuposer_realdata

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_imuposer_realdata.yaml \
  --run          vit1d_odom_probe_imuposer_realdata \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/odom_probe_imuposer_realdata_eval_dataset

############################# IMUPoser FLU: odom probe — Nymeria MAE encoder ##########
# Frozen vit1d_mae_nymeria_flu encoder; only odom decoder trains.
# Requires: runs/vit1d_mae_nymeria_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_imuposer_nymeria.yaml \
  --run-name     vit1d_odom_probe_imuposer_nymeria

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_imuposer_nymeria.yaml \
  --run          vit1d_odom_probe_imuposer_nymeria \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/odom_probe_imuposer_nymeria_eval_dataset

############################# IMUPoser FLU: odom finetune — realdata encoder ##########
# Encoder warm-started from vit1d_realdata_pretrain and kept trainable (lower LR).
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_imuposer_realdata.yaml \
  --run-name     vit1d_odom_finetune_imuposer_realdata

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_imuposer_realdata.yaml \
  --run          vit1d_odom_finetune_imuposer_realdata \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/odom_finetune_imuposer_realdata_eval_dataset

############################# IMUPoser FLU: odom finetune — Nymeria MAE encoder #######
# Encoder warm-started from vit1d_mae_nymeria_flu and kept trainable (lower LR).
# Requires: runs/vit1d_mae_nymeria_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_imuposer_nymeria.yaml \
  --run-name     vit1d_odom_finetune_imuposer_nymeria

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_imuposer_nymeria.yaml \
  --run          vit1d_odom_finetune_imuposer_nymeria \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/odom_finetune_imuposer_nymeria_eval_dataset

############################# IMUPoser FLU: odom probe — combined FLU MAE encoder ######
# Frozen vit1d_mae_combined_flu encoder (Nymeria FLU + real 6D); only odom decoder trains.
# Real right-wrist MEMS IMU in FLU body frame → per-patch pelvis root displacement [3].
# Requires: runs/vit1d_mae_combined_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_imuposer_combined_flu.yaml \
  --run-name     vit1d_odom_probe_imuposer_combined_flu

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_imuposer_combined_flu.yaml \
  --run          vit1d_odom_probe_imuposer_combined_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/odom_probe_imuposer_combined_flu_eval_dataset

############################# IMUPoser FLU: odom finetune — combined FLU MAE encoder ##
# Encoder warm-started from vit1d_mae_combined_flu and kept trainable (lower LR).
# Requires: runs/vit1d_mae_combined_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_imuposer_combined_flu.yaml \
  --run-name     vit1d_odom_finetune_imuposer_combined_flu

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_imuposer_combined_flu.yaml \
  --run          vit1d_odom_finetune_imuposer_combined_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/odom_finetune_imuposer_combined_flu_eval_dataset

############################# IMUPoser FLU: odom probe — realdata+Nymeria real-wrist FLU encoder
# Frozen vit1d_mae_realdata_nymeria_real_wrist_flu encoder; only odom decoder trains.
# Real right-wrist MEMS IMU in FLU body frame → per-patch pelvis root displacement [3].
# Requires: runs/vit1d_mae_realdata_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_odom_probe_imuposer_realdata_nymeria_real_wrist_flu

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run          vit1d_odom_probe_imuposer_realdata_nymeria_real_wrist_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/odom_probe_imuposer_realdata_nymeria_real_wrist_flu_eval_dataset

############################# IMUPoser FLU: odom finetune — realdata+Nymeria real-wrist FLU encoder
# Encoder warm-started from vit1d_mae_realdata_nymeria_real_wrist_flu and kept trainable (lower LR).
# Real right-wrist MEMS IMU in FLU body frame → per-patch pelvis root displacement [3].
# Requires: runs/vit1d_mae_realdata_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_odom_finetune_imuposer_realdata_nymeria_real_wrist_flu

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run          vit1d_odom_finetune_imuposer_realdata_nymeria_real_wrist_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/odom_finetune_imuposer_realdata_nymeria_real_wrist_flu_eval_dataset

############################# IMUPoser FLU: pose probe — combined FLU MAE encoder #####
# Frozen vit1d_mae_combined_flu encoder (Nymeria FLU + real 6D); only pose decoder trains.
# Data: IMUPoser right-wrist FLU IMU → right-arm chain labels (5 joints × 6D = 30 dims)
# Requires: runs/vit1d_mae_combined_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_combined_flu.yaml \
  --run-name     vit1d_pose_probe_imuposer_combined_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_combined_flu.yaml \
  --run          vit1d_pose_probe_imuposer_combined_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_imuposer_combined_flu_eval_dataset

############################# IMUPoser FLU: pose finetune — combined FLU MAE encoder ##
# Encoder warm-started from vit1d_mae_combined_flu and kept trainable (lower LR).
# Data: IMUPoser right-wrist FLU IMU → right-arm chain labels
# Requires: runs/vit1d_mae_combined_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_combined_flu.yaml \
  --run-name     vit1d_pose_finetune_imuposer_combined_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_combined_flu.yaml \
  --run          vit1d_pose_finetune_imuposer_combined_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_imuposer_combined_flu_eval_dataset

############################# IMUPoser FLU: arm pose probe — realdata+Nymeria real-wrist FLU encoder
# Frozen vit1d_mae_realdata_nymeria_real_wrist_flu encoder; only arm pose decoder trains.
# Data: IMUPoser right-wrist FLU IMU → right-arm chain labels (5 joints × 6D = 30 dims)
# Requires: runs/vit1d_mae_realdata_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_pose_probe_imuposer_realdata_nymeria_real_wrist_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run          vit1d_pose_probe_imuposer_realdata_nymeria_real_wrist_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_imuposer_realdata_nymeria_real_wrist_flu_eval_dataset

############################# IMUPoser FLU: arm pose finetune — realdata+Nymeria real-wrist FLU encoder
# Encoder warm-started from vit1d_mae_realdata_nymeria_real_wrist_flu and kept trainable (lower LR).
# Data: IMUPoser right-wrist FLU IMU → right-arm chain labels (5 joints × 6D = 30 dims)
# Requires: runs/vit1d_mae_realdata_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu

# Single-sequence arm pose visualization (pred vs GT)
# Replace --pkl with one file from:
#   /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test
python scripts/run_pose_eval_imuposer.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_ocdetect_flu.yaml \
  --run          vit1d_pose_finetune_imuposer_ocdetect_flu \
  --pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00000.pkl  \
  --ckpt         best \
  --max-frames   5000 \
  --fps          50 \
  --out-dir      data_vis/pose_finetune_imuposer_ocdetect_flu_single_eval

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run          vit1d_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_imuposer_realdata_nymeria_real_wrist_flu_eval_dataset

############################# IMUPoser FLU: full-body pose supervised from scratch ####
# Encoder + full-body pose decoder (24 joints × 6D = 144 dims) trained end-to-end
# from random init on IMUPoser FLU right-wrist IMU data.
# No pretrained encoder is used (baseline for comparing transfer-learning benefit).

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_supervised_imuposer_flu.yaml \
  --run-name     vit1d_full_body_pose_supervised_imuposer_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_supervised_imuposer_flu.yaml \
  --run          vit1d_full_body_pose_supervised_imuposer_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_supervised_imuposer_flu_eval_dataset

############################# IMUPoser FLU: full-body pose probe — Nymeria FLU MAE encoder
# Frozen vit1d_mae_nymeria_flu encoder; only full-body pose decoder trains.
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_nymeria_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_nymeria.yaml \
  --run-name     vit1d_full_body_pose_probe_imuposer_nymeria

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_nymeria.yaml \
  --run          vit1d_full_body_pose_probe_imuposer_nymeria \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_probe_imuposer_nymeria_eval_dataset

############################# IMUPoser FLU: full-body pose probe — realdata MAE encoder
# Frozen vit1d_realdata_pretrain encoder; only full-body pose decoder trains.
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_realdata.yaml \
  --run-name     vit1d_full_body_pose_probe_imuposer_realdata

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_realdata.yaml \
  --run          vit1d_full_body_pose_probe_imuposer_realdata \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_probe_imuposer_realdata_eval_dataset

############################# IMUPoser FLU: full-body pose probe — combined FLU MAE encoder
# Frozen vit1d_mae_combined_flu encoder; only full-body pose decoder trains.
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_combined_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_combined_flu.yaml \
  --run-name     vit1d_full_body_pose_probe_imuposer_combined_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_combined_flu.yaml \
  --run          vit1d_full_body_pose_probe_imuposer_combined_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_probe_imuposer_combined_flu_eval_dataset

############################# IMUPoser FLU: full-body pose finetune — Nymeria FLU MAE encoder
# Encoder warm-started from vit1d_mae_nymeria_flu and kept trainable (lower LR).
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_nymeria_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_nymeria.yaml \
  --run-name     vit1d_full_body_pose_finetune_imuposer_nymeria

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_nymeria.yaml \
  --run          vit1d_full_body_pose_finetune_imuposer_nymeria \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_finetune_imuposer_nymeria_eval_dataset

############################# IMUPoser FLU: full-body pose finetune — realdata MAE encoder
# Encoder warm-started from vit1d_realdata_pretrain and kept trainable (lower LR).
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_realdata.yaml \
  --run-name     vit1d_full_body_pose_finetune_imuposer_realdata

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_realdata.yaml \
  --run          vit1d_full_body_pose_finetune_imuposer_realdata \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_finetune_imuposer_realdata_eval_dataset

############################# IMUPoser FLU: full-body pose finetune — combined FLU MAE encoder
# Encoder warm-started from vit1d_mae_combined_flu and kept trainable (lower LR).
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_combined_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_combined_flu.yaml \
  --run-name     vit1d_full_body_pose_finetune_imuposer_combined_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_combined_flu.yaml \
  --run          vit1d_full_body_pose_finetune_imuposer_combined_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_finetune_imuposer_combined_flu_eval_dataset

############################# IMUPoser FLU: full-body pose probe — realdata+Nymeria real-wrist FLU encoder
# Frozen vit1d_mae_realdata_nymeria_real_wrist_flu encoder; only full-body pose decoder trains.
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_realdata_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_full_body_pose_probe_imuposer_realdata_nymeria_real_wrist_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run          vit1d_full_body_pose_probe_imuposer_realdata_nymeria_real_wrist_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_probe_imuposer_realdata_nymeria_real_wrist_flu_eval_dataset

############################# IMUPoser FLU: full-body pose finetune — realdata+Nymeria real-wrist FLU encoder
# Encoder warm-started from vit1d_mae_realdata_nymeria_real_wrist_flu and kept trainable (lower LR).
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_realdata_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_full_body_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu.yaml \
  --run          vit1d_full_body_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_finetune_imuposer_realdata_nymeria_real_wrist_flu_eval_dataset


############################# Visualize realdata 16Hz degradation
python scripts/vis_realdata_16hz_degradation.py \
    --parquet  /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
    --dataset  pamap2 \
    --n-seconds 50 \
    --out-dir  data_vis/realdata_16hz_comparison


python scripts/vis_realdata_16hz_degradation.py \
    --parquet  /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
    --dataset  utd_mhad \
    --n-seconds 1 \
    --out-dir  data_vis/realdata_16hz_comparison

python scripts/vis_humanml3d_flu.py \
    --npy /home/yiquan/HumanML3D/joints/000021.npy \
    --fps 10 \
    --max-frames 500000 \
    --axis-len 0.25 


############################# MAE: combined FLU, fixed 10.24 s window ####################
# Pretrain the ViT1D encoder on the union of Nymeria FLU synthetic IMU + real 6D parquet
# using a fixed 10.24 s (512-sample @ 50 Hz) window — no padding or variable length.
# patch_size=16 → 32 patches/channel, max_patches_per_channel=32.
# Identical data mix as vit1d_mae_combined_flu but with 4× longer context.
# Encoder saved to: runs/vit1d_mae_combined_flu_10s/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_combined_flu_10s.yaml \
  --run-name     vit1d_mae_combined_flu_10s

############################# IMUPoser FLU: full-body pose fully supervised from scratch — 10.24 s window
# Encoder + full-body pose decoder (24 joints × 6D = 144 dims) trained end-to-end
# from random init on IMUPoser FLU right-wrist IMU data with 10.24 s windows.
# Baseline for comparing transfer-learning benefit at the longer context length.

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_supervised_imuposer_flu_10s.yaml \
  --run-name     vit1d_full_body_pose_supervised_imuposer_flu_10s

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_supervised_imuposer_flu_10s.yaml \
  --run          vit1d_full_body_pose_supervised_imuposer_flu_10s \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_supervised_imuposer_flu_10s_eval_dataset

############################# IMUPoser FLU: full-body pose probe — combined FLU 10s MAE encoder
# Frozen vit1d_mae_combined_flu_10s encoder; only full-body pose decoder trains.
# Downstream windows match pretraining: 10.24 s (512 samples @ 50 Hz → 32 patches/channel).
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_combined_flu_10s/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_combined_flu_10s.yaml \
  --run-name     vit1d_full_body_pose_probe_imuposer_combined_flu_10s

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_combined_flu_10s.yaml \
  --run          vit1d_full_body_pose_probe_imuposer_combined_flu_10s \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_probe_imuposer_combined_flu_10s_eval_dataset

############################# IMUPoser FLU: full-body pose finetune — combined FLU 10s MAE encoder
# Encoder warm-started from vit1d_mae_combined_flu_10s and kept trainable (lower LR).
# Downstream windows match pretraining: 10.24 s (512 samples @ 50 Hz → 32 patches/channel).
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_combined_flu_10s/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_combined_flu_10s.yaml \
  --run-name     vit1d_full_body_pose_finetune_imuposer_combined_flu_10s

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_combined_flu_10s.yaml \
  --run          vit1d_full_body_pose_finetune_imuposer_combined_flu_10s \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_finetune_imuposer_combined_flu_10s_eval_dataset

############################# MAE: combined FLU, variable length
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_mae_combined_flu_var_length.yaml \
  --run-name vit1d_mae_combined_flu_var_length

############################# IMUPoser FLU: full-body pose probe — combined FLU var-length MAE encoder
# Frozen vit1d_mae_combined_flu_var_length encoder; only full-body pose decoder trains.
# Fixed 2.56 s windows (128 samples @ 50 Hz → 8 patches/channel, within the 32-patch max).
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_combined_flu_var_length/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_combined_flu_var_length.yaml \
  --run-name     vit1d_full_body_pose_probe_imuposer_combined_flu_var_length

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe_imuposer_combined_flu_var_length.yaml \
  --run          vit1d_full_body_pose_probe_imuposer_combined_flu_var_length \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_probe_imuposer_combined_flu_var_length_eval_dataset

############################# IMUPoser FLU: full-body pose finetune — combined FLU var-length MAE encoder
# Encoder warm-started from vit1d_mae_combined_flu_var_length and kept trainable (lower LR).
# Fixed 2.56 s windows (128 samples @ 50 Hz → 8 patches/channel, within the 32-patch max).
# Data: IMUPoser right-wrist FLU IMU → full SMPL body labels (24 × 6D = 144 dims)
# Requires: runs/vit1d_mae_combined_flu_var_length/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_combined_flu_var_length.yaml \
  --run-name     vit1d_full_body_pose_finetune_imuposer_combined_flu_var_length

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune_imuposer_combined_flu_var_length.yaml \
  --run          vit1d_full_body_pose_finetune_imuposer_combined_flu_var_length \
  --test-dir     /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_finetune_imuposer_combined_flu_var_length_eval_dataset

############################# Nymeria real wrist FLU: fully supervised baselines ################
# All three tasks trained from scratch (no pretrained encoder) using REAL wrist IMU
# from Nymeria _rwrist pkl files. Data paths: processed_data_body_Ty_rwrist (train/val).

# 1. Arm pose (5-joint right-arm chain, 30 dims)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_supervised_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_arm_pose_supervised_nymeria_real_wrist_flu

# 2. Full body pose (24 SMPL joints, 144 dims)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_full_body_pose_supervised_nymeria_real_wrist_flu

# 3. Odometry (root displacement, 3 dims per patch)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_supervised_nymeria_real_wrist_flu.yaml \
  --run-name     vit1d_odom_supervised_nymeria_real_wrist_flu

############################# Nymeria real wrist FLU: arm pose probe — OCDetect FLU MAE encoder ####
# Frozen vit1d_mae_ocdetect_flu encoder; only arm pose decoder trains.
# Data: Nymeria real right-wrist IMU → right-arm chain labels (5 joints × 6D = 30 dims)
# Requires: runs/vit1d_mae_ocdetect_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
  --run-name     vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
  --run          vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \
  --ckpt         best \
  --out-dir      data_vis/arm_pose_probe_nymeria_real_wrist_ocdetect_flu_eval_dataset

python scripts/run_pose_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
  --run          vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu \
  --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20230607_s0_james_johnson_act3_ifj2gc_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
  --ckpt         best \
  --max-frames   5000 \
  --fps          50 \
  --out-dir      data_vis/arm_pose_probe_nymeria_real_wrist_ocdetect_flu_single_eval

############################# Nymeria real wrist FLU: arm pose finetune — OCDetect FLU MAE encoder ####
# Encoder warm-started from vit1d_mae_ocdetect_flu and kept trainable (lower LR).
# Data: Nymeria real right-wrist IMU → right-arm chain labels (5 joints × 6D = 30 dims)
# Requires: runs/vit1d_mae_ocdetect_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
  --run-name     vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
  --run          vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \
  --ckpt         best \
  --out-dir      data_vis/arm_pose_finetune_nymeria_real_wrist_ocdetect_flu_eval_dataset

python scripts/run_pose_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
  --run          vit1d_arm_pose_finetune_nymeria_real_wrist_ocdetect_flu \
  --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20230607_s0_james_johnson_act3_ifj2gc_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
  --ckpt         best \
  --max-frames   5000 \
  --fps          50 \
  --out-dir      data_vis/arm_pose_finetune_nymeria_real_wrist_ocdetect_flu_single_eval

############################# Nymeria real wrist FLU: arm pose probe — realdata MAE encoder ####
# Frozen vit1d_realdata_pretrain encoder; only arm pose decoder trains.
# Data: Nymeria real right-wrist IMU → right-arm chain labels (5 joints × 6D = 30 dims)
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_realdata.yaml \
  --run-name     vit1d_arm_pose_probe_nymeria_real_wrist_realdata

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_realdata.yaml \
  --run          vit1d_arm_pose_probe_nymeria_real_wrist_realdata \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \
  --ckpt         best \
  --out-dir      data_vis/arm_pose_probe_nymeria_real_wrist_realdata_eval_dataset

python scripts/run_pose_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_realdata.yaml \
  --run          vit1d_arm_pose_probe_nymeria_real_wrist_realdata \
  --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20230607_s0_james_johnson_act3_ifj2gc_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
  --ckpt         best \
  --max-frames   5000 \
  --fps          50 \
  --out-dir      data_vis/arm_pose_probe_nymeria_real_wrist_realdata_single_eval

############################# Nymeria real wrist FLU: arm pose finetune — realdata MAE encoder ####
# Encoder warm-started from vit1d_realdata_pretrain and kept trainable (lower LR).
# Data: Nymeria real right-wrist IMU → right-arm chain labels (5 joints × 6D = 30 dims)
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_realdata.yaml \
  --run-name     vit1d_arm_pose_finetune_nymeria_real_wrist_realdata

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_realdata.yaml \
  --run          vit1d_arm_pose_finetune_nymeria_real_wrist_realdata \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \
  --ckpt         best \
  --out-dir      data_vis/arm_pose_finetune_nymeria_real_wrist_realdata_eval_dataset

python scripts/run_pose_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_arm_pose_finetune_nymeria_real_wrist_realdata.yaml \
  --run          vit1d_arm_pose_finetune_nymeria_real_wrist_realdata \
  --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20230607_s0_james_johnson_act3_ifj2gc_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
  --ckpt         best \
  --max-frames   5000 \
  --fps          50 \
  --out-dir      data_vis/arm_pose_finetune_nymeria_real_wrist_realdata_single_eval

############################# Nymeria real wrist FLU: odom probe — OCDetect FLU MAE encoder ####
# Frozen vit1d_mae_ocdetect_flu encoder; only odom decoder trains.
# Data: Nymeria real right-wrist IMU → per-patch pelvis root displacement [3]
# Requires: runs/vit1d_mae_ocdetect_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_nymeria_real_wrist_ocdetect_flu.yaml \
  --run-name     vit1d_odom_probe_nymeria_real_wrist_ocdetect_flu

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_nymeria_real_wrist_ocdetect_flu.yaml \
  --run          vit1d_odom_probe_nymeria_real_wrist_ocdetect_flu \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \
  --ckpt         best \
  --out-dir      data_vis/odom_probe_nymeria_real_wrist_ocdetect_flu_eval_dataset

############################# Nymeria real wrist FLU: odom finetune — OCDetect FLU MAE encoder ####
# Encoder warm-started from vit1d_mae_ocdetect_flu and kept trainable (lower LR).
# Data: Nymeria real right-wrist IMU → per-patch pelvis root displacement [3]
# Requires: runs/vit1d_mae_ocdetect_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
  --run-name     vit1d_odom_finetune_nymeria_real_wrist_ocdetect_flu

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_nymeria_real_wrist_ocdetect_flu.yaml \
  --run          vit1d_odom_finetune_nymeria_real_wrist_ocdetect_flu \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \
  --ckpt         best \
  --out-dir      data_vis/odom_finetune_nymeria_real_wrist_ocdetect_flu_eval_dataset

############################# Nymeria real wrist FLU: odom probe — realdata MAE encoder ####
# Frozen vit1d_realdata_pretrain encoder; only odom decoder trains.
# Data: Nymeria real right-wrist IMU → per-patch pelvis root displacement [3]
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_nymeria_real_wrist_realdata.yaml \
  --run-name     vit1d_odom_probe_nymeria_real_wrist_realdata

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_nymeria_real_wrist_realdata.yaml \
  --run          vit1d_odom_probe_nymeria_real_wrist_realdata \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \
  --ckpt         best \
  --out-dir      data_vis/odom_probe_nymeria_real_wrist_realdata_eval_dataset

############################# Nymeria real wrist FLU: odom finetune — realdata MAE encoder ####
# Encoder warm-started from vit1d_realdata_pretrain and kept trainable (lower LR).
# Data: Nymeria real right-wrist IMU → per-patch pelvis root displacement [3]
# Requires: runs/vit1d_realdata_pretrain/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_nymeria_real_wrist_realdata.yaml \
  --run-name     vit1d_odom_finetune_nymeria_real_wrist_realdata

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_nymeria_real_wrist_realdata.yaml \
  --run          vit1d_odom_finetune_nymeria_real_wrist_realdata \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty_rwrist \
  --ckpt         best \
  --out-dir      data_vis/odom_finetune_nymeria_real_wrist_realdata_eval_dataset

############################# Odom probe: encoder from IMUPoser full-body pose supervised ####
# Frozen encoder (from vit1d_full_body_pose_supervised_imuposer_flu) + trainable odom decoder.
# Tests whether a pose-supervised encoder transfers to odometry on IMUPoser FLU data.
# Requires: runs/vit1d_full_body_pose_supervised_imuposer_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_imuposer_flu_from_pose_supervised.yaml \
  --run-name     vit1d_odom_probe_imuposer_flu_from_pose_supervised

############################# Odom finetune: encoder from IMUPoser full-body pose supervised ####
# Encoder warm-started from vit1d_full_body_pose_supervised_imuposer_flu (stays trainable) + odom decoder.
# Tests whether a pose-supervised encoder can be adapted to odometry on IMUPoser FLU data.
# Requires: runs/vit1d_full_body_pose_supervised_imuposer_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_imuposer_flu_from_pose_supervised.yaml \
  --run-name     vit1d_odom_finetune_imuposer_flu_from_pose_supervised

############################# Nymeria real wrist FLU: odom probe — encoder from full-body pose supervised ####
# Frozen encoder (from vit1d_full_body_pose_supervised_nymeria_real_wrist_flu) + trainable odom decoder.
# Tests whether a pose-supervised Nymeria real-wrist encoder transfers to odometry on same dataset.
# Requires: runs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe_nymeria_real_wrist_flu_from_pose_supervised.yaml \
  --run-name     vit1d_odom_probe_nymeria_real_wrist_flu_from_pose_supervised

############################# Nymeria real wrist FLU: odom finetune — encoder from full-body pose supervised ####
# Encoder warm-started from vit1d_full_body_pose_supervised_nymeria_real_wrist_flu (stays trainable) + odom decoder.
# Tests whether a pose-supervised Nymeria real-wrist encoder can be adapted to odometry on same dataset.
# Requires: runs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune_nymeria_real_wrist_flu_from_pose_supervised.yaml \
  --run-name     vit1d_odom_finetune_nymeria_real_wrist_flu_from_pose_supervised

############################# SurfaceXR: 9-class gesture — fully supervised (LOPO) ####
# ViT1D encoder (in_channels=24, patch_size=25) + LinearHead trained end-to-end.
# Dataset: Google Pixel Watch 3, 200 Hz, 24 Butterworth channels, FLU frame.
# LOPO split: run once per held-out user; update --run-name and test_user in config accordingly.
# Change CUDA_VISIBLE_DEVICES and test_user as needed for each user.

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_gesture_supervised_surfaceXR.yaml \
  --run-name     vit1d_gesture_supervised_surfaceXR_lopoTimmy

############################# SurfaceXR: 9-class gesture — linear probe (LOPO) ####
# Frozen encoder loaded from a pretrained run + trainable LinearHead.
# Update pretrained_run in the config to point to the desired pretrained encoder.
# Requires: runs/<pretrained_run>/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_gesture_probe_surfaceXR.yaml \
  --run-name     vit1d_gesture_probe_surfaceXR_lopoTimmy







CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_gesture_supervised_surfaceXR.yaml \
  --run-name     vit1d_gesture_supervised_surfaceXR


