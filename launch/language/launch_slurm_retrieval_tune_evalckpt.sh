#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=rtune_evalckpt
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Evaluate an arbitrary checkpoint of a tuning-sweep run on val + test (CPU).
#   sbatch --job-name=rtune_evalckpt_<arm>_<ckpt> launch/launch_slurm_retrieval_tune_evalckpt.sh <arm> <ckpt>
# -> analysis/retrieval_tune/{val,test}/<arm>__<ckpt>.json
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
ARM="${1:?}"; CKPT="${2:?}"; RUN="rtune_${ARM}"
for SPLIT in val test; do
    python -u scripts/eval_retrieval_alignment.py --model ours --run "$RUN" --ckpt "$CKPT" \
        --split "$SPLIT" --label "rtune ${ARM} ${CKPT} (${SPLIT})" \
        --out "analysis/retrieval_tune/${SPLIT}/${ARM}__${CKPT}.json"
done
