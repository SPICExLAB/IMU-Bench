#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=eval_zeroshot_har_valonly_lr5e-5_fixedwin1024
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

RUN=runs/vit1d_clip_humanml3d_finetune_mae_multidata_anyrot_smoothresample_gravity_lr5e-5_fixedwin1024

# UTD-MHAD: median session duration ~3.5s, far shorter than 10.24s -- a
# standard sliding window at 10.24s would produce ZERO valid windows.
# --full-session wrap-pads each full (short) session up to T_model=10.24s
# instead, matching the training window size in spirit but NOT a like-for-
# like windowing comparison against WISDM/PAMAP2 below (see report caveat).
echo "=== UTD-MHAD (full-session, wrap-padded to 10.24s) ==="
python scripts/run_zeroshot_har.py \
    --run "$RUN" \
    --config configs/base.yaml \
    --dataset utd_mhad \
    --val-only \
    --full-session \
    --window-seconds 10.24 \
    --local-files-only

echo "=== WISDM ==="
python scripts/run_zeroshot_har.py \
    --run "$RUN" \
    --config configs/base.yaml \
    --dataset wisdm \
    --val-only \
    --window-seconds 10.24 \
    --local-files-only

echo "=== PAMAP2 ==="
python scripts/run_zeroshot_har.py \
    --run "$RUN" \
    --config configs/base.yaml \
    --dataset pamap2 \
    --val-only \
    --window-seconds 10.24 \
    --max-gaps-per-session 700 \
    --local-files-only
