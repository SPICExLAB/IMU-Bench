#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --job-name=unimts_finetune_wisdm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
cd baselines/language_alignment/UniMTS

# mem=180G (up from 32G, which OOM-killed this job on WISDM's 100K-window
# train split): load_custom_data's [N,T,22,6] joint-grid expansion is
# float64 and allocated for train+test simultaneously -- ~13.5GB for the
# WISDM train split alone before resampling, with multiple live copies
# during the function. PAMAP2/UTD-MHAD are small enough not to hit this.
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

python finetune_custom.py \
    --mode full \
    --num_epochs 200 \
    --batch_size 64 \
    --checkpoint checkpoint/UniMTS.pth \
    --X_train_path UniMTS_data/wisdm/X_train.npy \
    --y_train_path UniMTS_data/wisdm/y_train.npy \
    --X_test_path UniMTS_data/wisdm/X_test.npy \
    --y_test_path UniMTS_data/wisdm/y_test.npy \
    --config_path UniMTS_data/wisdm/wisdm.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --num_class 18 \
    --run_tag wisdm_finetune
