#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=unimts_zeroshot_mhealth
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
cd baselines/language_alignment/UniMTS

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

python evaluate_custom.py \
    --X_path UniMTS_data/mhealth/X_test.npy \
    --y_path UniMTS_data/mhealth/y_test.npy \
    --config_path UniMTS_data/mhealth/mhealth.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint checkpoint/UniMTS.pth \
    --run_tag mhealth_zeroshot \
    --batch_size 64
