#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generic Inertia-1 downstream job. Account/partition/time/job-name are passed on the
# sbatch command line by submit_inertia1_downstream.sh. Two modes (env):
#   MODEL_CONFIG=configs/inertia1/<run>.yaml RUN_NAME=<run>   -> harness train run
#   SCORE_CMD="python scripts/...."                            -> arbitrary scoring command
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs analysis/inertia1
# wandb: authenticate once with `wandb login` (never commit an API key)
export HF_HOME=/gpfs/projects/p32139/.cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [ -n "${SCORE_CMD:-}" ]; then
    echo "[score] $SCORE_CMD"
    eval "$SCORE_CMD"
else
     >/dev/null 2>&1 || true
    echo "[train] $RUN_NAME <- $MODEL_CONFIG"
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
        --config configs/base.yaml \
        --model-config "$MODEL_CONFIG" \
        --run-name "$RUN_NAME"
fi
