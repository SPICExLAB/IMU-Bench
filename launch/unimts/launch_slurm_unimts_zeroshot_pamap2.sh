#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=unimts_zeroshot_pamap2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
cd baselines/language_alignment/UniMTS

# HF_HUB_DISABLE_IMPLICIT_TOKEN: evaluate_custom.py unconditionally calls
# hf_hub_download() for a generic config.json (unused by our --config_path
# custom-data path) on every run; this environment's cached HF token is
# stale and causes spurious 401s unless implicit token attachment is
# disabled (verified working standalone before this launch script was written).
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

python evaluate_custom.py \
    --X_path UniMTS_data/pamap2/X_test.npy \
    --y_path UniMTS_data/pamap2/y_test.npy \
    --config_path UniMTS_data/pamap2/pamap2.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint checkpoint/UniMTS.pth \
    --run_tag pamap2_zeroshot \
    --batch_size 64
