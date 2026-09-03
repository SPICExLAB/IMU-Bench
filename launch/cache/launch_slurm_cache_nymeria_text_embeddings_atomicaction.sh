#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=cache_nymeria_text_emb_atomicaction
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

export HF_HOME=/gpfs/projects/p32139/.cache

python scripts/cache_clip_text_embeddings_nymeria.py \
    --captions-root /projects/b53017/vbc1591/nymeria_captions \
    --caption-source atomic_action \
    --local-files-only --device cuda
