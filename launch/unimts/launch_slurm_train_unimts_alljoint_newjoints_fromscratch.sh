#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=train_unimts_alljoint_newjoints_fromscratch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# UniMTS's own ST-GCN architecture, trained from scratch (as before), but
# now on the AMASS-free all-joint export (new_joints + new_joint_vecs/
# root_rotations_3d + aug_texts) instead of the AMASS-index-limited one
# (job 7953827, only 8,246/23,384 train clips, 35%). Tests directly whether
# ~3x more training data closes the gap found by that earlier run
# (10.26%/3.37% avg zero-shot, far below both "Ours" and official UniMTS),
# isolating corpus scale from the architecture question already answered.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL/baselines/language_alignment/UniMTS
source /gpfs/home/vbc1591/IMU-SSL/TransfHAR_env/bin/activate

python pretrain.py \
    --data_path ./data_ours_alljoint_humanml3d_newjoints_train \
    --run_tag ours_alljoint_newjoints_fromscratch \
    --num_epochs 100 \
    --batch_size 64 \
    --log 10
