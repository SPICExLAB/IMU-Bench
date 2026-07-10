# Pose probe + finetune with W&B MPJPE logging



CUDA_VISIBLE_DEVICES=1 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn_resume100k --step 270000 --mode probe \
    --out-dir data_vis/nymeria_pose_altattn_resume100k_probe_step270k/ \
    --max-train-files 867 --epochs 150 --patience 150

CUDA_VISIBLE_DEVICES=0 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn_spectral --step 100000 --mode probe \
    --out-dir data_vis/nymeria_pose_altattn_spectral_probe/ \
    --max-train-files 867 --epochs 150 --patience 150

CUDA_VISIBLE_DEVICES=0 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn_bandweighted --step 100000 --mode probe \
    --out-dir data_vis/nymeria_pose_altattn_bandweighted_probe/ \
    --max-train-files 867 --epochs 150 --patience 150

CUDA_VISIBLE_DEVICES=1 python scripts/eval_pose_nymeria_ssl.py \
    --runs vit1d_ssl_mae_varlen --step 100000 --mode probe \
    --out-dir data_vis/nymeria_pose_ssl_varlen_probe_step100k/ \
    --max-train-files 867 --epochs 150 --patience 150

# Ours: MAE (variable length + alt-attn)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn \
    --mode probe \
    --out-dir data_vis/nymeria_pose_altattn_probe/ \
    --max-train-files 867
    
CUDA_VISIBLE_DEVICES=0 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn \
    --mode probe \
    --out-dir data_vis/nymeria_pose_altattn_probe/ \
    --max-train-files 867 \
    --epochs 150 \
    --patience 150

CUDA_VISIBLE_DEVICES=1 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn \
    --mode finetune \
    --out-dir data_vis/nymeria_pose_altattn_finetune/ \
    --max-train-files 867

CUDA_VISIBLE_DEVICES=1 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn_register_fullocd \
    --mode probe \
    --out-dir data_vis/nymeria_pose_altattn_register_fullocd_probe/ \
    --max-train-files 867

CUDA_VISIBLE_DEVICES=0 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn \
    --mode probe \
    --head indep \
    --out-dir data_vis/nymeria_pose_altattn_indepreg_probe/ \
    --max-train-files 867

# No encoder registers: independent per-section register tokens cross-attend
# (one local MHA layer) to each section's channel tokens, then a temporal head.
# (vit1d_ssl_mae_varlen_altattn is the trained no-register encoder; the data-matched
#  _fullocd no-register encoder has no checkpoint yet.)
CUDA_VISIBLE_DEVICES=1 python scripts/eval_pose_nymeria_altattn.py \
    --run vit1d_ssl_mae_varlen_altattn \
    --mode probe \
    --head indep \
    --out-dir data_vis/nymeria_pose_altattn_indepreg_probe/ \
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

