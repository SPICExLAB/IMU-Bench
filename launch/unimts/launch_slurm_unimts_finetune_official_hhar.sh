#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=unimts_finetune_official_hhar
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL/baselines/language_alignment/UniMTS
source /gpfs/home/vbc1591/IMU-SSL/TransfHAR_env/bin/activate

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

python finetune_custom.py \
    --mode full \
    --X_train_path UniMTS_data/hhar/X_train.npy \
    --y_train_path UniMTS_data/hhar/y_train.npy \
    --X_test_path UniMTS_data/hhar/X_test.npy \
    --y_test_path UniMTS_data/hhar/y_test.npy \
    --config_path UniMTS_data/hhar/hhar.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --num_class 6 \
    --checkpoint checkpoint/UniMTS.pth \
    --run_tag hhar_finetune_official \
    --num_epochs 200 \
    --batch_size 64
