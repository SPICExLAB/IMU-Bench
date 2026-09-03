#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=eval_retrieval_alignment
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Complete bidirectional retrieval eval (R@1/5/10, MRR, MedR; full + fixed
# 64/100 galleries; both caption modes) for the language-alignment rows of
# Experiments_table.md Section 13. CPU is enough (ViT1D-S / ST-GCN encoders on
# ~1.2k-12k items) -- the default partition is `short` (CPU) so this never
# waits on the per-user GPU quota; pass `-p gengpu --gres=gpu:a100:1` to sbatch
# to run on a GPU instead. Rows live in launch/eval_retrieval_alignment_rows.sh.
#
#   sbatch launch/launch_slurm_eval_retrieval_alignment.sh <row_name>   # one row (recommended: fan out)
#   sbatch launch/launch_slurm_eval_retrieval_alignment.sh humanml3d    # all 9 HumanML3D rows, serial
#   sbatch launch/launch_slurm_eval_retrieval_alignment.sh nymeria      # all 5 Nymeria rows, serial
#   sbatch launch/launch_slurm_eval_retrieval_alignment.sh all
#   for r in $(bash -c 'source launch/eval_retrieval_alignment_rows.sh; echo $HUMANML3D_ROWS $NYMERIA_ROWS'); do
#       sbatch --job-name=evalret_$r launch/launch_slurm_eval_retrieval_alignment.sh $r; done
# Then: python scripts/summarize_retrieval_alignment.py analysis/retrieval_eval

set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source launch/eval_retrieval_alignment_rows.sh

WHICH="${1:-all}"
# SPLIT=test (env) evaluates the HumanML3D test split instead of val and writes
# to analysis/retrieval_eval_test/ (used as the held-out confirmation set for
# the 2026-08-15 tuning sweep; Nymeria rows ignore SPLIT).
SPLIT="${SPLIT:-val}"
OUT=analysis/retrieval_eval
[[ "$SPLIT" != "val" ]] && OUT="analysis/retrieval_eval_${SPLIT}"
mkdir -p "$OUT"
COMMON="--gallery-sizes 64,100 --n-subsets 200 --seed 42 --batch-size 128 --split $SPLIT"

case "$WHICH" in
    all)       ROWLIST="$HUMANML3D_ROWS $NYMERIA_ROWS" ;;
    humanml3d) ROWLIST="$HUMANML3D_ROWS" ;;
    nymeria)   ROWLIST="$NYMERIA_ROWS" ;;
    *)         ROWLIST="$WHICH" ;;
esac

FAILED=()
for r in $ROWLIST; do
    # shellcheck disable=SC2086
    run_row "$r" "$OUT" $COMMON || FAILED+=("$r")
done

echo "=================== SUMMARY (all JSONs present so far)"
python scripts/summarize_retrieval_alignment.py "$OUT" || true
if [[ ${#FAILED[@]} -gt 0 ]]; then echo "FAILED rows: ${FAILED[*]}"; exit 1; fi
echo "ALL DONE"
