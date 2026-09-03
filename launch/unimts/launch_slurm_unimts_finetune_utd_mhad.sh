#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=unimts_finetune_utd_mhad
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
cd baselines/language_alignment/UniMTS

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

python finetune_custom.py \
    --mode full \
    --num_epochs 200 \
    --batch_size 64 \
    --checkpoint checkpoint/UniMTS.pth \
    --X_train_path UniMTS_data/utd_mhad/X_train.npy \
    --y_train_path UniMTS_data/utd_mhad/y_train.npy \
    --X_test_path UniMTS_data/utd_mhad/X_test.npy \
    --y_test_path UniMTS_data/utd_mhad/y_test.npy \
    --config_path UniMTS_data/utd_mhad/utd_mhad.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --num_class 21 \
    --run_tag utd_mhad_finetune
