#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --job-name=test_simplify_captions
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

python scripts/generate_simplified_nymeria_captions.py \
    --captions-root /projects/b53017/vbc1591/nymeria_captions \
    --caption-source atomic_action \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --out /gpfs/home/vbc1591/IMU-SSL/test_simplified_map.json \
    --limit 20 \
    --batch-size 20
