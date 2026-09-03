#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=eval_openwatch_normwear_aug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

# First-party NormWear-Base on OpenWatch's *augmented* release, so §5a's table
# (which is trained on that release) gets a NormWear row on the same data,
# same subject split and same metric code as every other row in it. §5c's
# existing 52.04% is the no-aug release and belongs to §5's table, not §5a's.
python scripts/eval_openwatch_normwear.py \
    --root /projects/b53017/vbc1591/openwatch_raw/open_watch_augmentations \
    --cache results_openwatch_normwear_aug_embeddings.npz \
    --out-json results_openwatch_normwear_aug.json
