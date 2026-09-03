#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=prepare_humanml3d_alljoint_newjoints
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

cd baselines/language_alignment/UniMTS

python prepare_humanml3d_alljoint.py --split train \
    --joints-subpath HumanML3D/new_joints --rotation-source new_joint_vecs --captions-subpath aug_texts \
    --out-dir ./data_ours_alljoint_humanml3d_newjoints_train --progress-every 1000
python prepare_humanml3d_alljoint.py --split val \
    --joints-subpath HumanML3D/new_joints --rotation-source new_joint_vecs --captions-subpath aug_texts \
    --out-dir ./data_ours_alljoint_humanml3d_newjoints_val --progress-every 1000
