#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=train_unimts_userdata_fromscratch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# UniMTS's own ST-GCN architecture, trained from scratch via their own
# unmodified pretrain.py, on the user-provided all-joint HumanML3D dataset
# at /projects/b53017/vbc1591/HumanML3D/UniMTS_data/data (sim/ + aug_texts/,
# ~29,222 clips, essentially complete coverage of the mirrored HumanML3D set
# -- more complete than any all-joint export built earlier this session).

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL/baselines/language_alignment/UniMTS
source /gpfs/home/vbc1591/IMU-SSL/TransfHAR_env/bin/activate

python pretrain.py \
    --data_path /projects/b53017/vbc1591/HumanML3D/UniMTS_data/data \
    --run_tag ours_userdata_fromscratch \
    --num_epochs 100 \
    --batch_size 64 \
    --log 10
