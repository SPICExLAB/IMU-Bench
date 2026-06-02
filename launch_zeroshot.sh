CUDA_VISIBLE_DEVICES=1 python scripts/run_zeroshot_har.py \
  --run  runs/vit1d_clip_humanml3d_flu_finetune_10s \
  --config configs/base.yaml \
  --dataset wisdm

CUDA_VISIBLE_DEVICES=1 python scripts/run_zeroshot_har.py \
  --run  runs/vit1d_clip_humanml3d_flu_finetune_10s \
  --config configs/base.yaml \
  --dataset wisdm

CUDA_VISIBLE_DEVICES=1 python scripts/run_zeroshot_har.py \
  --run runs/vit1d_clip_humanml3d_flu_finetune_10s \
  --config configs/base.yaml \
  --dataset utd_mhad \
  --window-seconds 2.56

CUDA_VISIBLE_DEVICES=1 python scripts/run_zeroshot_har.py \
  --run runs/vit1d_clip_humanml3d_flu_finetune_10s \
  --config configs/base.yaml \
  --dataset pamap2