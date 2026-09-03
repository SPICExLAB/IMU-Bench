#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --job-name=inertia1_pretrain
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Reproduce Inertia-1 pretraining on NHANES (no official weights exist).
#   VARIANT=patchtst | ar_transformer   (env)   -> runs/inertia1_<variant>_small_nhanes
# Recipe: baselines/inertia1/scripts/pretrain_inertia1.py (their configs: small preset,
# 30 s @ 20 Hz triaxial g, batch 1024, AdamW + warmup/cosine, 170k steps as in the paper).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

VARIANT="${VARIANT:-patchtst}"
case "$VARIANT" in
  patchtst)        RUN=inertia1_patchtst_small_nhanes ;;
  ar_transformer)  RUN=inertia1_art_small_nhanes ;;
  *) echo "bad VARIANT=$VARIANT"; exit 1 ;;
esac
CORPUS=/projects/b53017/vbc1591/inertia1_nhanes20hz
N_DONE=$(ls "$CORPUS/participants" | grep -c '\.json$' || true)
echo "corpus participants with metadata: $N_DONE / 14693"
if [ "$N_DONE" -lt 14000 ]; then
  echo "WARNING: corpus build incomplete ($N_DONE participants); training on what exists."
fi

python -u baselines/inertia1/scripts/pretrain_inertia1.py \
    --variant "$VARIANT" --preset small --corpus "$CORPUS" \
    --run-name "$RUN" --runs-root /gpfs/home/vbc1591/IMU-SSL/runs \
    --max-steps "${MAX_STEPS:-170000}" --batch-size 1024 --num-workers 12 --resume
