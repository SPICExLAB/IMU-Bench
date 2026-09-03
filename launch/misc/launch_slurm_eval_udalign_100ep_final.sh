#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=eval_udalign_100ep_final
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
cd baselines/language_alignment/UniMTS

D=checkpoint/unimts_userdata_scratch_jointmask_100ep

echo "=== 100ep FINAL model (last.pth -- original recipe's selection) WRIST-ONLY ==="
python -u eval_unimts_userdata_alignment.py --checkpoint "$D/last.pth" --wrist-only 1
echo "=== 100ep best-val-loss model (best.pth) WRIST-ONLY ==="
python -u eval_unimts_userdata_alignment.py --checkpoint "$D/best.pth" --wrist-only 1
echo "=== 100ep FINAL model FULL-GRID (reference) ==="
python -u eval_unimts_userdata_alignment.py --checkpoint "$D/last.pth" --wrist-only 0
