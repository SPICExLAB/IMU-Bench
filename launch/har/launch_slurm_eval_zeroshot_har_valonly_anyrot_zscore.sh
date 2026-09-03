#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=eval_zeroshot_har_valonly_anyrot_zscore
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

# Debugging test (2026-07-25): same rotation-aug-only checkpoint as
# launch_slurm_eval_zeroshot_har_valonly_anyrot.sh, but with --zscore-input
# added -- tests whether forcing z-score normalization on the real target
# dataset windows (matching what HumanML3D training windows were always
# z-scored to, unconditionally, via HumanML3DIMUVarLenDatasetFLU's
# hardcoded per-window z-score) closes the zero-shot gap, without any
# retraining.

echo "=== UTD-MHAD ==="
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_humanml3d_finetune_mae_multidata_anyrot \
    --config configs/base.yaml \
    --dataset utd_mhad \
    --val-only \
    --window-seconds 2.56 \
    --zscore-input \
    --local-files-only

echo "=== WISDM ==="
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_humanml3d_finetune_mae_multidata_anyrot \
    --config configs/base.yaml \
    --dataset wisdm \
    --val-only \
    --zscore-input \
    --local-files-only

echo "=== PAMAP2 ==="
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_humanml3d_finetune_mae_multidata_anyrot \
    --config configs/base.yaml \
    --dataset pamap2 \
    --val-only \
    --max-gaps-per-session 700 \
    --zscore-input \
    --local-files-only
