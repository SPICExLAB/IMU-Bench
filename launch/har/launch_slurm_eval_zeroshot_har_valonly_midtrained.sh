#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=eval_zeroshot_har_valonly_midtrained
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache

# Same protocol as launch_slurm_eval_zeroshot_har_valonly.sh, but against
# the MIDTRAINED checkpoint (vit1d_clip_realdata_midtrain_from_humanml3d)
# instead of the plain HumanML3D-CLIP checkpoint.

echo "=== UTD-MHAD ==="
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_realdata_midtrain_from_humanml3d \
    --config configs/base.yaml \
    --dataset utd_mhad \
    --val-only \
    --window-seconds 2.56 \
    --local-files-only

echo "=== WISDM ==="
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_realdata_midtrain_from_humanml3d \
    --config configs/base.yaml \
    --dataset wisdm \
    --val-only \
    --local-files-only

echo "=== PAMAP2 ==="
python scripts/run_zeroshot_har.py \
    --run runs/vit1d_clip_realdata_midtrain_from_humanml3d \
    --config configs/base.yaml \
    --dataset pamap2 \
    --val-only \
    --max-gaps-per-session 700 \
    --local-files-only
