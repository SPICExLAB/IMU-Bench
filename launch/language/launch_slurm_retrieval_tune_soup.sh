#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=rtune_soup
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Round-2 "soup" arm of the HumanML3D CLIP-alignment tuning sweep (2026-08-15).
#   sbatch --job-name=rtune_soup_<arm> launch/launch_slurm_retrieval_tune_soup.sh <arm>
# 1) trains configs/retrieval_tune/<arm>.yaml (parent recipe + save_step_checkpoints)
#    -> runs/rtune_<arm>/checkpoints/step_*.pt every 500 steps
# 2) evaluates EVERY step checkpoint on the val split (G=100 only, 50 subsets)
#    -> analysis/retrieval_tune/steps/<arm>__step<k>.json  (selection-by-retrieval)
# 3) builds uniform soups over [4000,8000] and [5500,8000] and [3000,8000]
#    -> evaluates each on val + test like every other arm:
#       analysis/retrieval_tune/{val,test}/<arm>__soup<a>_<b>.json
# 4) also evals the run's best.pt (val-loss selection) on val + test
#       analysis/retrieval_tune/{val,test}/<arm>.json
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1

ARM="${1:?usage: $0 <arm>}"
CFG="configs/retrieval_tune/${ARM}.yaml"
[[ -f "$CFG" ]] || { echo "no such config: $CFG"; exit 2; }
RUN="rtune_${ARM}"
mkdir -p analysis/retrieval_tune/val analysis/retrieval_tune/test analysis/retrieval_tune/steps

if [[ ! -f "runs/${RUN}/checkpoints/best.pt" || "${FORCE_RETRAIN:-0}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py --config configs/base.yaml \
        --model-config "$CFG" --run-name "$RUN"
fi

# best.pt on val + test
for SPLIT in val test; do
    python -u scripts/eval_retrieval_alignment.py --model ours --run "$RUN" --ckpt best \
        --split "$SPLIT" --label "rtune ${ARM} best.pt (${SPLIT})" \
        --out "analysis/retrieval_tune/${SPLIT}/${ARM}.json"
done

# per-step val retrieval (cheap protocol: G=100, 50 subsets)
for f in runs/${RUN}/checkpoints/step_*.pt; do
    name=$(basename "$f" .pt); step=$((10#${name#step_}))
    python -u scripts/eval_retrieval_alignment.py --model ours --run "$RUN" --ckpt "$name" \
        --split val --gallery-sizes 100 --n-subsets 50 --label "rtune ${ARM} ${name} (val)" \
        --out "analysis/retrieval_tune/steps/${ARM}__step${step}.json"
done

# soups
# SOUP_RANGES env: space-separated "a-b" pairs (default = 8000-step schedule)
for RANGE in ${SOUP_RANGES:-4000-8000 5500-8000 3000-8000}; do
    A=${RANGE%-*}; B=${RANGE#*-}
    python scripts/soup_checkpoints.py --run "$RUN" --min-step "$A" --max-step "$B" --out "soup_${A}_${B}"
    for SPLIT in val test; do
        python -u scripts/eval_retrieval_alignment.py --model ours --run "$RUN" --ckpt "soup_${A}_${B}" \
            --split "$SPLIT" --label "rtune ${ARM} soup[${A},${B}] (${SPLIT})" \
            --out "analysis/retrieval_tune/${SPLIT}/${ARM}__soup${A}_${B}.json"
    done
done
echo "[rtune-soup] done: $ARM"
