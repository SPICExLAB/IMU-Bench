#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=train_unimts_alljoint_fromscratch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# UniMTS's own ST-GCN architecture, trained FROM SCRATCH (random IMU-encoder
# init, per ContrastiveModule -- no checkpoint load) on OUR OWN synthesized
# HumanML3D IMU data, but -- unlike every other "Ours" checkpoint this
# session -- covering ALL 22 SMPL joints (via prepare_humanml3d_alljoint.py),
# not just the single right-wrist joint used everywhere else. This isolates
# whether UniMTS's remaining zero-shot advantage (esp. HHAR/MHEALTH postural
# classes) comes from architecture + all-joint data richness (testable here)
# vs. IMUSim-specific synthesis fidelity / proprietary additional
# pretraining data (not directly testable without IMUSim).
#
# Uses UniMTS's own unmodified pretrain.py end to end: CLIPDataset reads
# {data_path}/sim/*.npy + {data_path}/aug_texts/*.txt (produced by
# prepare_humanml3d_alljoint.py), their own joint-masking augmentation
# (mask 1-21 of 22 joints per sample), their own rotation augmentation
# (--aug 1, default), accel-only (--gyro 0, default, matches how the
# officially released checkpoint was itself pretrained), Adam lr=1e-4
# (hardcoded in pretrain.py), batch_size 64, temperature 0.1.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL/baselines/language_alignment/UniMTS
source /gpfs/home/vbc1591/IMU-SSL/TransfHAR_env/bin/activate

python pretrain.py \
    --data_path ./data_ours_alljoint_humanml3d_train \
    --run_tag ours_alljoint_fromscratch \
    --num_epochs 100 \
    --batch_size 64 \
    --log 10
