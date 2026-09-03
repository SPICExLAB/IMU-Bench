#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=openwatch_normwear_protocol_noaug
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

# OpenWatch §5f fair-comparison protocol (scripts/openwatch_paper_protocol_common.py)
python scripts/train_eval_openwatch_normwear_protocol.py --condition noaug --out-json results_openwatch_normwear_protocol_noaug.json --cache-dir /projects/b53017/vbc1591/openwatch_cache/normwear_protocol
