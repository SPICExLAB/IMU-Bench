#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=eval_unimts_userdata_alignment
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
cd baselines/language_alignment/UniMTS

echo "=== SELF-TRAINED (ours_userdata_fromscratch epoch_90 -- NOTE: trained on these val clips, upper bound) ==="
python -u eval_unimts_userdata_alignment.py --checkpoint checkpoint/ours_userdata_fromscratch/epoch_90.pth

echo "=== OFFICIAL UniMTS.pth (never saw this data) ==="
python -u eval_unimts_userdata_alignment.py --checkpoint checkpoint/UniMTS.pth
