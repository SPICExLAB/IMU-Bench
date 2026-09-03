# MAE (variable length + alt-attn)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn

# MAE (variable length + alt-attn), all sources combined: realdata + OCDetect (1/100) +
# Nymeria real wrist + TotalCapture/DIP-IMU + IMUPoser + EatingSpeed (1/10) +
# Motion-X++ (synth) + AMASS/HumanML3D (synth) + SONIC/bones-seed (synth) +
# Capture24 (1/10, acc-only -- gyro channels masked out of encoder input + loss) +
# OpenWatch (real wrist gesture, acc+gyro)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_multidata_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_multidata

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_fullocd_balanced_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_new_trained

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_fullocd_balanced_realdata_ocdetect_flu_win128.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_win128_trained

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_fullocd_balanced_spectral_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_spectral

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_fullocd_balanced_bandweighted_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_bandweighted

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_fullocd_balanced_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen
###
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_fullocd_balanced_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_resume100k \
    --resume /home/yiquan/TransfHAR/runs/vit1d_ssl_mae_varlen_altattn_new_trained/checkpoints/step_0100000.pt 

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_fullocd_balanced_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_resume270k \
    --resume /home/yiquan/TransfHAR/runs/vit1d_ssl_mae_varlen_altattn_resume100k/checkpoints/step_0270000.pt

CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_fullocd_balanced_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_resume100k \
    --resume /home/yiquan/TransfHAR/runs/vit1d_ssl_mae_varlen/checkpoints/step_0100000.pt

###

# MAE (variable length + alt-attn) + full occlusion detection
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_register_fullocd_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn_register_fullocd


# SIMMTM, size-matched, full OCDetect + Realdata, natural ratio, 2.56s window
CUDA_VISIBLE_DEVICES=0 python scripts/run_ssl_baseline.py \
    --model simmtm \
    --config configs/ssl_simmtm_fullocd_realdata_ocdetect.yaml \
    --run-name ssl_simmtm_fullocd_realdata_ocdetect_sizematched
# LIMU-BERT, size-matched, full OCDetect + Realdata, natural ratio, 2.56s window
CUDA_VISIBLE_DEVICES=1 python scripts/run_ssl_baseline.py \
    --model limu_bert \
    --config configs/ssl_limubert_fullocd_realdata_ocdetect.yaml \
    --run-name ssl_limubert_fullocd_realdata_ocdetect_sizematched
# FOCAL, size-matched, full OCDetect + Realdata, natural ratio, 2.56s window
CUDA_VISIBLE_DEVICES=1 python scripts/run_ssl_baseline.py \
    --model focal \
    --config configs/ssl_focal_fullocd_realdata_ocdetect.yaml \
    --run-name ssl_focal_fullocd_realdata_ocdetect_sizematched
# MOMENT, size-matched, full OCDetect + Realdata, natural ratio, 2.56s window
CUDA_VISIBLE_DEVICES=0 python scripts/run_ssl_baseline.py \
    --model moment \
    --config configs/ssl_moment_fullocd_realdata_ocdetect.yaml \
    --run-name ssl_moment_fullocd_realdata_ocdetect
