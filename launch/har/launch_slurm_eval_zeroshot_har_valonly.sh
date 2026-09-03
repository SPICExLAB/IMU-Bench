#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=eval_zeroshot_har_valonly
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

# Apples-to-apples zero-shot: same held-out val split (--val-only) and same
# accuracy/macro-F1 metric as UniMTS's zero-shot eval (evaluate_custom.py
# reads only X_test/y_test, the same val split prepare_har_data.py exported)
# -- the ORIGINAL, un-finetuned HumanML3D-CLIP checkpoint (no target-dataset
# training at all), unlike launch_slurm_eval_finetuned_har_classification.sh
# (which evaluates the FINETUNED checkpoints).

echo "=== UTD-MHAD ==="
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_humanml3d_finetune_mae_multidata \
    --config configs/base.yaml \
    --dataset utd_mhad \
    --val-only \
    --window-seconds 2.56 \
    --local-files-only

echo "=== WISDM ==="
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_humanml3d_finetune_mae_multidata \
    --config configs/base.yaml \
    --dataset wisdm \
    --val-only \
    --local-files-only

echo "=== PAMAP2 ==="
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_humanml3d_finetune_mae_multidata \
    --config configs/base.yaml \
    --dataset pamap2 \
    --val-only \
    --max-gaps-per-session 700 \
    --local-files-only
