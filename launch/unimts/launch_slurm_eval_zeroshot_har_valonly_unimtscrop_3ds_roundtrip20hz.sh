#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=eval_zeroshot_har_valonly_unimtscrop_3ds_roundtrip20hz
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Same checkpoint/data/splits as launch_slurm_eval_zeroshot_har_valonly_unimtscrop_3ds.sh,
# with --roundtrip-20hz added: downsample each target-dataset window to 20Hz
# then upsample back to 50Hz before encoding, matching the effective
# bandwidth cap HumanML3D-derived training data has (real content capped by
# AMASS mocap's native 20Hz frame rate, even though presented at 50Hz).

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

RUN=runs/vit1d_clip_humanml3d_finetune_mae_multidata_anyrot_smoothresample_gravity_lr5e-5_unimtscrop

echo "=== UTD-MHAD (full-session, wrap-padded to 10.24s) ==="
python scripts/run_zeroshot_har.py \
    --run "$RUN" \
    --config configs/base.yaml \
    --dataset utd_mhad \
    --val-only \
    --full-session \
    --roundtrip-20hz \
    --local-files-only

echo "=== WISDM ==="
python scripts/run_zeroshot_har.py \
    --run "$RUN" \
    --config configs/base.yaml \
    --dataset wisdm \
    --val-only \
    --window-seconds 10.24 \
    --roundtrip-20hz \
    --local-files-only

echo "=== PAMAP2 ==="
python scripts/run_zeroshot_har.py \
    --run "$RUN" \
    --config configs/base.yaml \
    --dataset pamap2 \
    --val-only \
    --window-seconds 10.24 \
    --max-gaps-per-session 700 \
    --roundtrip-20hz \
    --local-files-only
