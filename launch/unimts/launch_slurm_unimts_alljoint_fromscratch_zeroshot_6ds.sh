#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=unimts_alljoint_fromscratch_zeroshot_6ds
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Full 6-dataset zero-shot eval of the UniMTS-architecture checkpoint
# trained FROM SCRATCH on our own all-joint-synthesized HumanML3D data
# (checkpoint/ours_alljoint_fromscratch/epoch_90.pth, final saved checkpoint
# of a 100-epoch run, train loss 0.53). Same eval harness/data/joint_list as
# the official UniMTS checkpoint's own 6-dataset eval (launch/launch_slurm_
# unimts_zeroshot_*.sh) for a direct comparison: isolates whether UniMTS's
# architecture + all-joint data richness (this checkpoint) vs. IMUSim
# synthesis fidelity / proprietary pretraining scale (official checkpoint)
# drives its zero-shot advantage over "Ours" (ViT1D + wrist-only HumanML3D).

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
cd baselines/language_alignment/UniMTS

export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

CKPT=checkpoint/ours_alljoint_fromscratch/epoch_90.pth

echo "=== UTD-MHAD ==="
python evaluate_custom.py \
    --X_path UniMTS_data/utd_mhad/X_test.npy \
    --y_path UniMTS_data/utd_mhad/y_test.npy \
    --config_path UniMTS_data/utd_mhad/utd_mhad.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint "$CKPT" \
    --run_tag utd_mhad_zeroshot_alljoint_fromscratch \
    --batch_size 64

echo "=== WISDM ==="
python evaluate_custom.py \
    --X_path UniMTS_data/wisdm/X_test.npy \
    --y_path UniMTS_data/wisdm/y_test.npy \
    --config_path UniMTS_data/wisdm/wisdm.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint "$CKPT" \
    --run_tag wisdm_zeroshot_alljoint_fromscratch \
    --batch_size 64

echo "=== PAMAP2 ==="
python evaluate_custom.py \
    --X_path UniMTS_data/pamap2/X_test.npy \
    --y_path UniMTS_data/pamap2/y_test.npy \
    --config_path UniMTS_data/pamap2/pamap2.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint "$CKPT" \
    --run_tag pamap2_zeroshot_alljoint_fromscratch \
    --batch_size 64

echo "=== WHARF ==="
python evaluate_custom.py \
    --X_path UniMTS_data/wharf/X_test.npy \
    --y_path UniMTS_data/wharf/y_test.npy \
    --config_path UniMTS_data/wharf/wharf.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint "$CKPT" \
    --run_tag wharf_zeroshot_alljoint_fromscratch \
    --batch_size 64

echo "=== HHAR ==="
python evaluate_custom.py \
    --X_path UniMTS_data/hhar/X_test.npy \
    --y_path UniMTS_data/hhar/y_test.npy \
    --config_path UniMTS_data/hhar/hhar.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint "$CKPT" \
    --run_tag hhar_zeroshot_alljoint_fromscratch \
    --batch_size 64

echo "=== MHEALTH ==="
python evaluate_custom.py \
    --X_path UniMTS_data/mhealth/X_test.npy \
    --y_path UniMTS_data/mhealth/y_test.npy \
    --config_path UniMTS_data/mhealth/mhealth.json \
    --joint_list 21 \
    --original_sampling_rate 50 \
    --checkpoint "$CKPT" \
    --run_tag mhealth_zeroshot_alljoint_fromscratch \
    --batch_size 64
