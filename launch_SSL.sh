# MAE (variable length + alt-attn)
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_flu.yaml \
    --run-name vit1d_ssl_mae_varlen_altattn

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
