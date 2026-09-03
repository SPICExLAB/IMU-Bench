#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=eval_udalign_verbatim_wristonly
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
cd baselines/language_alignment/UniMTS

CKPT=checkpoint/ours_userdata_fromscratch/epoch_90.pth

echo "=== VERBATIM pretrain.py checkpoint (8054089, exact original recipe; trained on ALL clips incl. val -> upper bound) -- WRIST-ONLY ==="
python -u eval_unimts_userdata_alignment.py --checkpoint "$CKPT" --wrist-only 1

echo "=== same checkpoint -- FULL 22-joint (reference) ==="
python -u eval_unimts_userdata_alignment.py --checkpoint "$CKPT" --wrist-only 0
