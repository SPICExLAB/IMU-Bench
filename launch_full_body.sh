# Pose probe + finetune with W&B MPJPE logging

# Ours: MAE (variable length + alt-attn)

CUDA_VISIBLE_DEVICES=0 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn \
    --mode probe \
    --out-dir data_vis/nymeria_pose_altattn_probe/ \
    --max-train-files 867
    

CUDA_VISIBLE_DEVICES=1 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn \
    --mode finetune \
    --out-dir data_vis/nymeria_pose_altattn_finetune/ \
    --max-train-files 867

CUDA_VISIBLE_DEVICES=0 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn_register_fullocd \
    --mode probe \
    --out-dir data_vis/nymeria_pose_altattn_register_fullocd_probe/ \
    --max-train-files 867
    
# Baselines

# MOMENT Nymeria real-wrist pose probe
CUDA_VISIBLE_DEVICES=0 python scripts/eval_pose_nymeria_ssl_baseline.py \
    --model moment \
    --run ssl_moment_fullocd_realdata_ocdetect \
    --mode probe \
    --out-dir data_vis/nymeria_pose_baseline_probe/moment \
    --max-train-files 867

# FOCAL Nymeria real-wrist pose probe
CUDA_VISIBLE_DEVICES=1 python scripts/eval_pose_nymeria_ssl_baseline.py \
    --model focal \
    --run ssl_focal_fullocd_realdata_ocdetect_sizematched \
    --mode probe \
    --out-dir data_vis/nymeria_pose_baseline_probe/focal \
    --max-train-files 867

# LIMU-BERT Nymeria real-wrist pose probe
CUDA_VISIBLE_DEVICES=1 python scripts/eval_pose_nymeria_ssl_baseline.py \
    --model limu_bert \
    --run ssl_limubert_fullocd_realdata_ocdetect_sizematched \
    --mode probe \
    --out-dir data_vis/nymeria_pose_baseline_probe/limubert \
    --max-train-files 867

# SimMTM Nymeria real-wrist pose probe
CUDA_VISIBLE_DEVICES=0 python scripts/eval_pose_nymeria_ssl_baseline.py \
    --model simmtm \
    --run ssl_simmtm_fullocd_realdata_ocdetect_sizematched \
    --mode probe \
    --out-dir data_vis/nymeria_pose_baseline_probe/simmtm \
    --max-train-files 867

