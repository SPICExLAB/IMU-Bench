source TransfHAR_env/bin/activate

############################# MAE pretrain on real 6D data (~154 h) ##################
# train_on_datasets: ["pamap2", "recofit", "samosa", "shoaib", "ut_watch", "utd_mhad", "wisdm"]
# Encoder saved to: runs/vit1d_realdata_pretrain/checkpoints/best.pt
# (set train_on_datasets in configs/base.yaml before running)

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d.yaml \
  --run-name     vit1d_realdata_pretrain

############################# MAE pretrain on Nymeria FLU data ######################
# Pretrain the ViT1D encoder on Nymeria FLU synthetic right-wrist IMU
# (linear acc, FLU body frame — no gravity component).
# Encoder saved to: runs/vit1d_mae_nymeria_flu/checkpoints/best.pt

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_mae_nymeria_flu.yaml \
  --run-name     vit1d_mae_nymeria_flu

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

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
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

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
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

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
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