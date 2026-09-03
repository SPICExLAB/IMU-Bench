########## ARM POSE #######################################

## Total Capture
python scripts/vis_totalcapture_arm_pose.py \
    --pkl          /mnt/nas2/DIP_recalculate/s1_acting1.pkl\
    --config       configs/base.yaml \
    --model-config configs/vit1d_arm_pose_supervised_totalcapture_flu.yaml \
    --run          vit1d_arm_pose_supervised_totalcapture_flu \
    --ckpt         best

## IMUPoser
python scripts/vis_imuposer_arm_pose.py \
    --pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00011.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_pose_supervised_imuposer_flu.yaml \
    --run          vit1d_pose_supervised_imuposer_flu \
    --ckpt         best

python scripts/vis_imuposer_arm_pose.py \
    --pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00009.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_pose_probe_imuposer_ocdetect_flu.yaml \
    --run          vit1d_pose_probe_imuposer_ocdetect_flu \
    --ckpt         best

python scripts/vis_imuposer_arm_pose.py \
    --pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00011.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_pose_finetune_imuposer_ocdetect_flu.yaml \
    --run          vit1d_pose_finetune_imuposer_ocdetect_flu \
    --ckpt         best

## Nymeria_full
python scripts/vis_nymeria_arm_pose.py \
    --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20231222_s0_denise_carter_act2_d8mdi5_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_arm_pose_supervised_nymeria_real_wrist_flu.yaml \
    --run          vit1d_arm_pose_supervised_nymeria_real_wrist_flu \
    --ckpt         best

## Nymeria_full - probe (OCDetect MAE encoder)
python scripts/vis_nymeria_arm_pose.py \
    --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20230607_s0_james_johnson_act3_ifj2gc_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
    --run          vit1d_arm_pose_probe_nymeria_real_wrist_ocdetect_flu \
    --ckpt         best

########## Odometry #######################################

## IMUPoser
python scripts/vis_imuposer_odom.py \
    --pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00032.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_odom_imuposer_flu.yaml \
    --run          vit1d_odom_imuposer_flu \
    --ckpt         best

## Nymeria_full
python scripts/vis_nymeria_odom.py \
    --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20231108_s0_nicholas_hicks_act0_g5hz4p_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_odom_supervised_nymeria_real_wrist_flu.yaml \
    --run          vit1d_odom_supervised_nymeria_real_wrist_flu \
    --ckpt         best

########## Full Body Pose #######################################

## IMUPoser
python scripts/vis_imuposer_full_body_pose.py \
    --pkl          /mnt/nas2/yiquan_data/IMUPoser/ioposer_with_hip_30hz_2/test/imuposer_seq_00019.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_imuposer_flu.yaml \
    --run          vit1d_full_body_pose_supervised_imuposer_flu \
    --ckpt         best

## TotalCapture
python scripts/vis_totalcapture_full_body_pose.py \
    --pkl          /mnt/nas2/DIP_recalculate/s1_acting1.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_totalcapture_flu.yaml \
    --run          vit1d_full_body_pose_supervised_totalcapture_flu \
    --ckpt         best

## Nymeria_full

python scripts/vis_nymeria_full_body_pose.py \
    --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20231222_s0_denise_carter_act2_d8mdi5_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_syn_wrist_global.yaml \
    --run          vit1d_full_body_pose_supervised_nymeria_syn_wrist_global \
    --ckpt         best

## Nymeria - 24-joint T-pose sub-patch model
python scripts/vis_nymeria_full_body_pose_24j_tpose.py \
    --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20231222_s0_denise_carter_act2_d8mdi5_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose.yaml \
    --run          vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_24j_tpose \
    --ckpt         best \
    --max-frames   5000

## Nymeria - 23-joint T-pose sub-patch model
python scripts/vis_nymeria_full_body_pose_23j.py \
    --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20231222_s0_denise_carter_act6_26wfkz_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_23j.yaml \
    --run          vit1d_full_body_pose_supervised_nymeria_real_wrist_flu_23j \
    --ckpt         latest \
    --max-frames   5000





python scripts/vis_nymeria_full_body_pose.py \
    --pkl          /mnt/nas2/processed_data_test_body_Ty_rwrist/20231222_s0_denise_carter_act2_d8mdi5_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --config       configs/base.yaml \
    --model-config configs/vit1d_full_body_pose_probe_nymeria_real_wrist_ocdetect_flu.yaml \
    --run          vit1d_full_body_pose_probe_nymeria_real_wrist_ocdetect_flu \
    --ckpt         best


python /home/yiquan/TransfHAR/scripts/plot_ocdetect_sequence.py \
  /mnt/nas2/yiquan_data/preprocessed/OCDetect_01_recording_00_55ef074f-147f-424f-b9c6-58c0cf9dfee0.csv

python /home/yiquan/TransfHAR/scripts/plot_ocdetect_sequence.py \
  /mnt/nas2/yiquan_data/preprocessed/OCDetect_25_recording_12_5c20634e-03c2-4a30-8275-6a66d7268bb4.csv