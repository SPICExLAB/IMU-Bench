python scripts/eval_mae_nymeria.py \
    --ckpt     runs/vit1d_mae_nymeria_real_wrist_flu/checkpoints/best.pt \
    --val-root /mnt/nas2/processed_data_test_body_Ty_rwrist \
    --batch-size 256 \
    --device cuda

python scripts/vis_mae_reconstruction.py \
    --ckpt runs/vit1d_mae_nymeria_real_wrist_flu/checkpoints/best.pt \
    --pkl  /mnt/nas2/processed_data_body_Ty_rwrist/20230607_s0_james_johnson_act1_7xwm28_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --out  data_vis/mae_reconstruction_nymeria.png \
    --window-idx 20

python scripts/eval_mae_pretrain.py \
    --ckpt  runs/vit1d_realdata_pretrain/checkpoints/best.pt \
    --parquet /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
    --batch-size 256 \
    --device cuda

python scripts/vis_mae_reconstruction.py \
    --ckpt      runs/vit1d_realdata_pretrain/checkpoints/best.pt \
    --parquet   /mnt/nas2/yiquan_data/IMULM_master_dataset_v2.parquet \
    --dataset   utd_mhad \
    --out       data_vis/mae_reconstruction_realdata_utd_mhad.png \
    --window-idx 2 \
    --seed      0


python scripts/eval_mae_ocdetect.py \
    --ckpt     runs/vit1d_mae_ocdetect_flu_v2/checkpoints/best.pt \
    --data-dir /mnt/nas2/yiquan_data/preprocessed \
    --batch-size 256 \
    --device cuda
    
python scripts/vis_mae_reconstruction.py \
    --ckpt runs/vit1d_mae_ocdetect_flu_v2/checkpoints/best.pt \
    --csv  /mnt/nas2/yiquan_data/preprocessed/OCDetect_30_recording_20_4d342e93-613e-47e4-be04-97d6112bc4b5.csv \
    --out  data_vis/mae_reconstruction_ocdetect.png \
    --window-idx 20

python scripts/vis_mae_ocdetect_no_norm.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_mae_ocdetect_flu_no_norm_aug.yaml \
    --ckpt         runs/vit1d_mae_ocdetect_flu_no_norm_aug/checkpoints/latest.pt \
    --csv          /mnt/nas2/yiquan_data/preprocessed/OCDetect_01_recording_00_55ef074f-147f-424f-b9c6-58c0cf9dfee0.csv \
    --window-idx   20 \
    --out          data_vis/mae_recon_no_norm_aug.png