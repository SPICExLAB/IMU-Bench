#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Inertia-1 "large" preset (d1024/8L/16H/ff4096, ~101M params) on the same NHANES corpus and
# with the same recipe as the small repro (lr 1e-3, batch 1024, 170k steps, warmup 500,
# cosine, grad-clip 1, bf16). VARIANT=patchtst | ar_transformer.
#   -> runs/inertia1_<variant>_large_nhanes ; --resume makes any restart continue from latest.pt
#
# TWIN LOGIC (submit the same script once per partition; account/partition/time/job-name are
# given on the sbatch command line):
#   ROLE=canonical  (b53017; downstream bench cells depend afterok on THIS job): if a twin job
#                   is RUNNING the same run, cancel it and resume from its latest.pt.
#   ROLE=twin       (gengpu): if the canonical job is RUNNING, exit 0 immediately; otherwise
#                   train (with --resume). If the canonical starts later it takes over.
# Whichever partition schedules first does the work; checkpoints are atomic (tmp+rename).
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
VARIANT="${VARIANT:-patchtst}"
ROLE="${ROLE:-canonical}"
case "$VARIANT" in
  patchtst)        RUN=inertia1_patchtst_large_nhanes ;;
  ar_transformer)  RUN=inertia1_art_large_nhanes ;;
  *) echo "bad VARIANT=$VARIANT"; exit 1 ;;
esac
CANON_NAME="inertia1_pretrain_large_${VARIANT}"
TWIN_NAME="${CANON_NAME}_gengpu"
me=${SLURM_JOB_ID:-0}
other_running() {  # $1 = job name -> prints job ids RUNNING with that name (excluding self)
  squeue -u "$USER" -h -n "$1" -t RUNNING -o "%i" | grep -v "^${me}$" || true
}
if [ "$ROLE" = "twin" ]; then
  if [ -n "$(other_running "$CANON_NAME")" ]; then
    echo "[twin] canonical job $(other_running "$CANON_NAME") is RUNNING -> nothing to do"; exit 0
  fi
else
  tw=$(other_running "$TWIN_NAME")
  if [ -n "$tw" ]; then
    echo "[canonical] twin $tw is RUNNING on gengpu -> cancelling it and resuming from its latest.pt"
    scancel $tw; sleep 45
  fi
fi
CORPUS=/projects/b53017/vbc1591/inertia1_nhanes20hz
echo "[$ROLE] job $me on $(hostname) partition ${SLURM_JOB_PARTITION:-?} -> $RUN"
python -u baselines/inertia1/scripts/pretrain_inertia1.py \
    --variant "$VARIANT" --preset large --corpus "$CORPUS" \
    --run-name "$RUN" --runs-root /gpfs/home/vbc1591/IMU-SSL/runs \
    --max-steps "${MAX_STEPS:-170000}" --batch-size "${BATCH:-1024}" --num-workers 12 --resume \
    --lr "${LR:-2e-4}" --warmup-steps "${WARMUP:-2000}"
# LR 2e-4 / warmup 2000 (not the small preset's 1e-3 / 500): at 1e-3 the 101M models diverged
# (PatchTST-L val 0.108 @5k vs 0.023 small; ART-L loss spike to 22 @2.5k) -- first attempt kept in
# runs/*_large_nhanes__lr1e-3_diverged. 2e-4 is the pretraining LR stated in the paper's Appendix A.
