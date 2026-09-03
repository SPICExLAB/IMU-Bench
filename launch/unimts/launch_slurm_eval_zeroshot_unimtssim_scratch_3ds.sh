#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=eval_zeroshot_unimtssim_scratch_3ds
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

RUN=runs/vit1d_clip_humanml3d_fromscratch_unimtssim_texttrainable_lr5e-5

echo "=== UTD-MHAD ==="
python -u scripts/run_zeroshot_har.py \
    --run "$RUN" --ckpt best_pinned --config configs/base.yaml --dataset utd_mhad \
    --val-only --full-session --window-seconds 10.24 --local-files-only

echo "=== WISDM ==="
python -u scripts/run_zeroshot_har.py \
    --run "$RUN" --ckpt best_pinned --config configs/base.yaml --dataset wisdm \
    --val-only --window-seconds 10.24 --local-files-only

echo "=== PAMAP2 ==="
python -u scripts/run_zeroshot_har.py \
    --run "$RUN" --ckpt best_pinned --config configs/base.yaml --dataset pamap2 \
    --val-only --window-seconds 10.24 --max-gaps-per-session 700 --local-files-only
