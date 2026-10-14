#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=relcon_motifdist_5src_native
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Stage 1 of the RelCon native arm: the same cross-attention motif-distance network as
# relcon_motifdist_5src, trained on the same un-normalised input the stage-2 encoder will see.
# Its RevIN is untouched -- that is the normalisation RelCon actually specifies.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

python launch/nativenorm/preflight.py configs/nativenorm/relcon_motifdist_5src_native.yaml

python -u scripts/train_relcon_motifdist.py \
    --config configs/base.yaml \
    --model-config configs/nativenorm/relcon_motifdist_5src_native.yaml \
    --run-name relcon_motifdist_5src_native \
    --runs-root runs \
    --max-steps 20000 \
    --val-every-steps 500 \
    --patience 15
