#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --job-name=vit1d_ssl_mae_cleanpool
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# MAE pretraining on the leakage-free 6-source "clean pool" (+NHANES).
# Requires launch_slurm_build_cache_cleanpool.sh to have completed first --
# with no scaling_mode set, this loads straight from the cache (fast start).
#
# --time=24:00:00: at the ~3.85-3.9 steps/sec rate MEASURED on the Section 11
# runs (my earlier ~9 steps/sec estimate was 2.3x too optimistic and TIMEOUT'd
# all three of those jobs at 16h), 300,000 steps needs ~21.4h. 24h gives real
# margin. If it still times out, resume with `--resume latest` -- NOT `best`,
# which desyncs the Trainer's early-stopping bookkeeping restore (it always
# reads latest.pt first; see imu_lm/runtime_consistency/trainer.py:118).

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_ssl_mae_varlen_altattn_cleanpool_flu.yaml \
    --run-name vit1d_ssl_mae_cleanpool
