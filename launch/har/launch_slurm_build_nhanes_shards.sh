#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=150G
#SBATCH --job-name=nhanes_shard
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Sharded NHANES cache build: 10 independent jobs, ~80 GB each instead of one
# 800 GB job. Rationale (2026-08-01): the single-job build could only run on
# the handful of 1TB nodes and sat queued for hours behind them; at 150G a
# shard fits essentially any node, all 10 run concurrently, and the ~4h serial
# read collapses to ~30 min. Also removes the RAM ceiling on how much NHANES
# we can ingest -- raise --num-shards rather than hunting a bigger node.
#
# Each shard takes a deterministic round-robin slice of the SAME participant
# list and train/val split the unsharded path computes, so the union is
# equivalent. merge_nhanes_shards.py stitches them afterwards and refuses to
# run on an incomplete set.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

OUT="/projects/b53017/vbc1591/mae_multidata_cache_stage1_acconly"
python scripts/build_nhanes_shard.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_stage1_acconly_flu.yaml \
    --out-dir "$OUT" \
    --shard-idx "${SLURM_ARRAY_TASK_ID}" --num-shards 10 \
    --split train --num-workers 8

# shard 0 also builds the (small, 400h-capped) val split
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    python scripts/build_nhanes_shard.py \
        --config configs/base.yaml \
        --model-config configs/vit1d_ssl_mae_stage1_acconly_flu.yaml \
        --out-dir "$OUT" \
        --shard-idx 0 --num-shards 1 --split val --num-workers 8
fi
