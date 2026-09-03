#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --job-name=merge_nhanes_shards
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Streams the 10 x 160 GB shards into the single nhanes_{train,val}_{x,mask}.npy
# pair the cache loader expects. Memmap on both sides in 64k-row blocks, so
# 32G is plenty -- peak RAM is one block, not the 1.6 TB array. Refuses to run
# on an incomplete shard set rather than writing a silently-short cache.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
df -h /projects/b53017/vbc1591 | tail -1
python scripts/merge_nhanes_shards.py \
    --out-dir /projects/b53017/vbc1591/mae_multidata_cache_stage1_acconly --split both
df -h /projects/b53017/vbc1591 | tail -1
