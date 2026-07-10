# Linear probe (frozen encoder, only head trains)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_har_probe_utd_mhad_ocdetect_flu.yaml \
    --run-name     vit1d_har_probe_utd_mhad_ocdetect_flu

# Finetune (warm-start encoder, train everything end-to-end)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_har_finetune_utd_mhad_ocdetect_flu.yaml \
    --run-name     vit1d_har_finetune_utd_mhad_ocdetect_flu

# Fully supervised (random init, no pretrained weights)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_har_supervised_utd_mhad.yaml \
    --run-name     vit1d_har_supervised_utd_mhad

#########################
python scripts/eval_dino_ckpts_utd_mhad.py \
    --config       configs/base.yaml \
    --probe-config configs/vit1d_har_probe_utd_mhad_dino_anyrot.yaml \
    --dino-run     vit1d_dino_ocdetect_anyrot_v4

python scripts/eval_dino_ckpts_utd_mhad.py \
    --config       configs/base.yaml \
    --probe-config configs/vit1d_har_probe_utd_mhad_dino_anyrot.yaml \
    --dino-run     vit1d_dino_ocdetect_anyrot_v4 \
    --steps 200000

python scripts/eval_dino_ckpts_utd_mhad.py \
    --config    configs/base.yaml \
    --dino-run  vit1d_dino_ocdetect_anyrot_v4 \
    --steps     200000 \
    --datasets  utd_mhad samosa prism

CUDA_VISIBLE_DEVICES=1 python scripts/eval_beit_ckpts.py \
    --config   configs/base.yaml \
    --beit-run vit1d_beit_ocdetect_anyrot_nopooling_8lvl \
    --steps    45000 \
    --datasets utd_mhad \
    --force

python scripts/eval_dino_ckpts_utd_mhad.py \
    --config       configs/base.yaml \
    --probe-config configs/vit1d_har_probe_utd_mhad_dino_anyrot.yaml \
    --dino-run     vit1d_dino_ocdetect_anyrot_v3 \
    --ckpt         /home/yiquan/TransfHAR/runs/vit1d_dino_ocdetect_anyrot_v3/checkpoints/latest.pt


CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_har_probe_utd_mhad_mae_flu.yaml \
    --run-name     vit1d_har_probe_utd_mhad_mae_flu

# Samosa
CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_har_probe_samosa_mae_flu.yaml \
    --run-name     vit1d_har_probe_samosa_mae_flu

# Prism
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_har_probe_prism_mae_flu.yaml \
    --run-name     vit1d_har_probe_prism_mae_flu
    
CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_har_finetune_utd_mhad_mae_flu.yaml \
    --run-name     vit1d_har_finetune_utd_mhad_mae_flu


CUDA_VISIBLE_DEVICES=1 python scripts/run_pretrain.py \
    --config       configs/base.yaml \
    --model-config configs/vit1d_har_supervised_utd_mhad.yaml \
    --run-name     vit1d_har_supervised_utd_mhad

CUDA_VISIBLE_DEVICES=1 python scripts/eval_mae_ckpts.py \
    --config   configs/base.yaml \
    --mae-run  vit1d_mae_ocdetect_flu_no_norm_aug \
    --datasets prism \
    --force


CUDA_VISIBLE_DEVICES=1 python scripts/eval_mae_ckpts.py \
    --config   configs/base.yaml \
    --mae-run  vit1d_mae_ocdetect_anyrot \
    --datasets samosa \
    --force

CUDA_VISIBLE_DEVICES=1 python scripts/eval_mae_ckpts.py \
    --config        configs/base.yaml \
    --mae-run       vit1d_realdata_pretrain \
    --ckpt          runs/vit1d_realdata_pretrain/checkpoints/best.pt \
    --datasets      utd_mhad \
    --probe-variant flu