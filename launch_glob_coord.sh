
source TransfHAR_env/bin/activate

############################# Nymeria wrist to right arm ##############################
python scripts/run_pretrain.py --config configs/base.yaml --model-config configs/vit1d_pose.yaml --run-name vit1d_nymeria_pose


python scripts/run_pose_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose.yaml \
  --run          vit1d_nymeria_pose \
  --pkl          /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl\
  --ckpt         best \
  --max-frames   5000 \
  --out-dir      data_vis/pose_eval

# Dataset-level eval: geodesic + MPJPE across ALL test sequences
CUDA_VISIBLE_DEVICES=1 python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose.yaml \
  --run          vit1d_nymeria_pose \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_eval_dataset


############################# Nymeria wrist to full body ##############################

python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_full_body_pose.yaml \
  --run-name vit1d_nymeria_full_body

python scripts/run_pose_eval.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_full_body_pose.yaml \
  --run vit1d_nymeria_full_body \
  --pkl /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
  --ckpt best --max-frames 5000 --out-dir data_vis/pose_eval_full_body

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose.yaml \
  --run          vit1d_nymeria_full_body \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_supervised_eval_dataset



############################# Nymeria wrist MAE pretrain ##############################

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_mae_nymeria.yaml \
  --run-name vit1d_mae_nymeria

############################# Nymeria wrist MAE pretrain (var length) ##############################

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_mae_nymeria_var_length.yaml \
  --run-name vit1d_mae_nymeria_var_length

############################# Nymeria wrist MAE pretrain (variable length) ############
# Sinusoidal pos-embed + window T sampled uniformly from [1.28 s, 10.24 s] each batch.
# Encoder and decoder weights are NOT compatible with vit1d_mae_nymeria (different
# pos-embed type); use vit1d_mae_nymeria_var_length as the pretrained_run for
# downstream tasks that need variable-length generalisation.

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_mae_nymeria_var_length.yaml \
  --run-name vit1d_mae_nymeria_var_length

############################# IMU-text CLIP pretrain (UniMTS-style) ##################
# Pretrain: ViT1D encoder (trainable, from scratch) + linear projection head aligned
# against frozen CLIP text embeddings from HumanML3D (InfoNCE, batch 64).
# Prerequisites:
#   1. objective.imu_text_clip.humanml3d_root is set correctly in the config

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_clip_humanml3d.yaml \
  --run-name vit1d_clip_humanml3d

############################# IMU-text CLIP probe (frozen MAE encoder) ###############
# Probe: MAE var-length encoder is FROZEN; only the linear projection head trains.
# Useful for evaluating representation quality without altering the encoder weights.
# Warm-starts encoder from vit1d_mae_nymeria_var_length.
# Prerequisites:
#   1. vit1d_mae_nymeria_var_length has a checkpoints/best.pt
#   2. objective.imu_text_clip.humanml3d_root is set correctly in the config

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_clip_humanml3d_probe.yaml \
  --run-name vit1d_clip_humanml3d_probe

############################# IMU-text CLIP finetune (MAE encoder trainable) ##########
# Finetune: ViT1D encoder warm-started from vit1d_mae_nymeria_var_length and TRAINABLE.
# Both encoder and projection head are updated (CLIP text encoder stays frozen).
# Use this after vit1d_mae_nymeria_var_length has converged for best results.
# Prerequisites:
#   1. vit1d_mae_nymeria_var_length has a checkpoints/best.pt
#   2. objective.imu_text_clip.humanml3d_root is set correctly in the config

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_clip_humanml3d_finetune.yaml \
  --run-name vit1d_clip_humanml3d_finetune

############################# Pose probe: frozen MAE encoder + right-arm decoder ######

# Step 1: make sure vit1d_mae_nymeria has a best.pt before running this
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_pose_probe.yaml \
  --run-name vit1d_pose_probe

python scripts/run_pose_eval.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_pose_probe.yaml \
  --run vit1d_pose_probe \
  --pkl /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
  --ckpt best --max-frames 5000 --out-dir data_vis/pose_probe_eval

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_probe.yaml \
  --run          vit1d_pose_probe \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_probe_eval_dataset

############################# Full-body pose probe: frozen MAE encoder + full-body decoder ##

# Step 1: make sure vit1d_mae_nymeria has a best.pt before running this
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe.yaml \
  --run-name vit1d_full_body_pose_probe

python scripts/run_pose_eval.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe.yaml \
  --run vit1d_full_body_pose_probe \
  --pkl /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
  --ckpt best --max-frames 5000 --out-dir data_vis/full_body_pose_probe_eval

CUDA_VISIBLE_DEVICES=1 python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_probe.yaml \
  --run          vit1d_full_body_pose_probe \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_probe_eval_dataset

############################# Full-body pose finetune: MAE encoder (trainable) + full-body decoder ##

# Step 1: make sure vit1d_mae_nymeria has a best.pt before running this
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune.yaml \
  --run-name vit1d_full_body_pose_finetune

python scripts/run_pose_eval.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune.yaml \
  --run vit1d_full_body_pose_finetune \
  --pkl /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
  --ckpt best --max-frames 5000 --out-dir data_vis/full_body_pose_finetune_eval

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_full_body_pose_finetune.yaml \
  --run          vit1d_full_body_pose_finetune \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/full_body_pose_finetune_eval_dataset


############################# Odometry: global-frame wrist IMU → root displacement ####

# Train end-to-end (encoder + ViT1DOdomDecoder, MSE on per-patch XYZ displacement)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_odom_nymeria.yaml \
  --run-name vit1d_odom_nymeria

# Evaluate: compare predicted vs GT root trajectory, save PNG plots
python scripts/run_odom_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_nymeria.yaml \
  --run          vit1d_odom_nymeria \
  --pkl          /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl\
  --ckpt         best \
  --max-frames   5000 \
  --out-dir      data_vis/odom_eval

# Dataset-level eval: ATE + RTE-1s across ALL test sequences
python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_nymeria.yaml \
  --run          vit1d_odom_nymeria \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --rte-window   1.0 2.56 \
  --out-dir      data_vis/odom_eval_dataset

############################# Linear probe on Nymeria-pretrained encoders #############

# Frozen encoder + linear head trained on parquet probe split (e.g. PRISM activities)
# Works for any run that saved artifacts/encoder_meta.json + checkpoints/best.pt

# -- MAE-pretrained encoder --
CUDA_VISIBLE_DEVICES=1 python scripts/run_probe_train.py \
  --config       configs/base.yaml \
  --probe-config configs/nymeria_probe.yaml \
  --run          vit1d_mae_nymeria

python scripts/run_probe_eval.py \
  --config       configs/base.yaml \
  --probe-config configs/nymeria_probe.yaml \
  --run          vit1d_mae_nymeria

# -- Wrist-pose-pretrained encoder --
CUDA_VISIBLE_DEVICES=1 python scripts/run_probe_train.py \
  --config       configs/base.yaml \
  --probe-config configs/nymeria_probe.yaml \
  --run          vit1d_nymeria_pose

python scripts/run_probe_eval.py \
  --config       configs/base.yaml \
  --probe-config configs/nymeria_probe.yaml \
  --run          vit1d_nymeria_pose

################################### Wrist pose finetune ####################################
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_pose_finetune.yaml \
  --run-name vit1d_pose_finetune

python scripts/run_pose_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune.yaml \
  --run          vit1d_pose_finetune \
  --pkl          /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
  --ckpt         best \
  --max-frames   5000 \
  --out-dir      data_vis/pose_finetune_eval

python scripts/run_pose_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_pose_finetune.yaml \
  --run          vit1d_pose_finetune \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --out-dir      data_vis/pose_finetune_eval_dataset

############################# Odom probe: frozen MAE encoder + odom decoder ####

# Step 1: make sure vit1d_mae_nymeria has a best.pt before running this
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_odom_probe.yaml \
  --run-name vit1d_odom_probe

python scripts/run_odom_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe.yaml \
  --run          vit1d_odom_probe \
  --pkl          /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
  --ckpt         best \
  --max-frames   5000 \
  --out-dir      data_vis/odom_probe_eval

python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_probe.yaml \
  --run          vit1d_odom_probe \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --rte-window   1.0 2.56 \
  --out-dir      data_vis/odom_probe_eval_dataset

############################# Odom finetune: MAE encoder (trainable) + odom decoder ###

# Step 1: make sure vit1d_mae_nymeria has a best.pt before running this
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
  --config configs/base.yaml \
  --model-config configs/vit1d_odom_finetune.yaml \
  --run-name vit1d_odom_finetune

python scripts/run_odom_eval.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune.yaml \
  --run          vit1d_odom_finetune \
  --pkl          /mnt/nas2/processed_data_test_body_Ty/20231222_s0_denise_carter_act6_26wfkz_cpfbody_imu50hz_pose50hz.pkl \
  --ckpt         best \
  --max-frames   5000 \
  --out-dir      data_vis/odom_finetune_eval

CUDA_VISIBLE_DEVICES=1 python scripts/run_odom_eval_dataset.py \
  --config       configs/base.yaml \
  --model-config configs/vit1d_odom_finetune.yaml \
  --run          vit1d_odom_finetune \
  --test-dir     /mnt/nas2/processed_data_test_body_Ty \
  --ckpt         best \
  --rte-window   1.0 2.56 \
  --out-dir      data_vis/odom_finetune_eval_dataset


############################# ViT1D realdata pretrain ###
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py --config configs/base.yaml --model-config configs/vit1d.yaml --run-name vit1d_realdata_pretrain