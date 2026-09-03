#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=unimts_finetune_pamap2
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
    --X_train_path UniMTS_data/pamap2/X_train.npy \
    --y_train_path UniMTS_data/pamap2/y_train.npy \
    --X_test_path UniMTS_data/pamap2/X_test.npy \
    --y_test_path UniMTS_data/pamap2/y_test.npy \
    --config_path UniMTS_data/pamap2/pamap2.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --num_class 18 \
    --run_tag pamap2_finetune
