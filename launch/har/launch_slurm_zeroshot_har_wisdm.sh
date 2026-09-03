#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=zeroshot_har_wisdm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

export HF_HOME=/gpfs/projects/p32139/.cache

# Zero-shot HAR classification using the HumanML3D-CLIP-finetuned model
# (33.00%/34.25% imu<->text retrieval on HumanML3D itself) -- no training on
# wisdm at all, classification is purely nearest-CLIP-text-embedding lookup
# over "wisdm"'s own activity label set.
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_humanml3d_finetune_mae_multidata \
    --config configs/base.yaml \
    --dataset wisdm \
    --local-files-only
