#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=simplify_atomicaction_full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

# job 7621827 (batch-size 128) ran 2h24m and got 82048/143054 done before a
# fatal CUDA OOM -- allocator fragmentation grew across batches until an
# allocation could no longer be satisfied even with a non-fatal-retry-sized
# free pool (reserved-but-unallocated memory was large per the error msg).
# expandable_segments avoids that fragmentation, and dropping batch size adds
# headroom. --checkpoint-every makes this resumable: rerunning the same
# --out path skips the 82048 already-mapped captions automatically.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Output MUST be on a shared filesystem (GPFS-backed /projects), not /tmp --
# /tmp turned out to be node-local during the quality-test smoke tests.
python scripts/generate_simplified_nymeria_captions.py \
    --captions-root /projects/b53017/vbc1591/nymeria_captions \
    --caption-source atomic_action \
    --model-id Qwen/Qwen2.5-7B-Instruct \
    --out /projects/b53017/vbc1591/nymeria_captions/atomic_action_simplified_map.json \
    --batch-size 64 \
    --checkpoint-every 2000
