#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=eval_zeroshot_hhar_mhealth_bareword
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

RUN=runs/vit1d_clip_humanml3d_finetune_mae_multidata_anyrot_smoothresample_gravity_lr5e-5_unimtscrop

echo "=== HHAR (bare-word template, matching UniMTS exactly) ==="
python scripts/run_zeroshot_har.py \
    --run "$RUN" \
    --config configs/base_hhar.yaml \
    --dataset hhar \
    --val-only \
    --window-seconds 10.24 \
    --template "{}" \
    --local-files-only

echo "=== MHEALTH (bare-word template, matching UniMTS exactly) ==="
python scripts/run_zeroshot_har.py \
    --run "$RUN" \
    --config configs/base_mhealth.yaml \
    --dataset mhealth \
    --val-only \
    --window-seconds 10.24 \
    --template "{}" \
    --local-files-only
