#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=unimts_userdata_fromscratch_zeroshot_hhar
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
cd baselines/language_alignment/UniMTS

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

python -u evaluate_custom.py \
    --X_path UniMTS_data/hhar/X_test.npy \
    --y_path UniMTS_data/hhar/y_test.npy \
    --config_path UniMTS_data/hhar/hhar.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint checkpoint/ours_userdata_fromscratch/epoch_90.pth \
    --run_tag hhar_zeroshot_userdata_fromscratch \
    --batch_size 64
