#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=unimts_alljoint_newjoints_fromscratch_zeroshot_utd_mhad_sanity
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
    --X_path UniMTS_data/utd_mhad/X_test.npy \
    --y_path UniMTS_data/utd_mhad/y_test.npy \
    --config_path UniMTS_data/utd_mhad/utd_mhad.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint checkpoint/ours_alljoint_newjoints_fromscratch/epoch_10.pth \
    --run_tag utd_mhad_zeroshot_alljoint_newjoints_fromscratch_sanity_epoch10 \
    --batch_size 64
