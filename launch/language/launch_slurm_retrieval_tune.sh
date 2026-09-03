#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=rtune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# HumanML3D CLIP-alignment tuning sweep (2026-08-15).
#   sbatch --job-name=rtune_<arm> launch/launch_slurm_retrieval_tune.sh <arm>
# Trains configs/retrieval_tune/<arm>.yaml as run rtune_<arm>, then runs the
# complete retrieval eval (scripts/eval_retrieval_alignment.py) on the val
# split (what Section 13 reports; used for arm selection) AND the never-used
# HumanML3D test split (4,384 clips; held out to confirm the selected recipe).
# JSONs -> analysis/retrieval_tune/{val,test}/<arm>.json
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1

ARM="${1:?usage: $0 <arm>}"
CFG="configs/retrieval_tune/${ARM}.yaml"
[[ -f "$CFG" ]] || { echo "no such config: $CFG"; exit 2; }
RUN="rtune_${ARM}"
mkdir -p analysis/retrieval_tune/val analysis/retrieval_tune/test

if [[ ! -f "runs/${RUN}/checkpoints/best.pt" || "${FORCE_RETRAIN:-0}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
        --config configs/base.yaml \
        --model-config "$CFG" \
        --run-name "$RUN"
else
    echo "[rtune] best.pt exists for $RUN, skipping training (FORCE_RETRAIN=1 to redo)"
fi

for SPLIT in val test; do
    python -u scripts/eval_retrieval_alignment.py --model ours --run "$RUN" --ckpt best \
        --split "$SPLIT" --label "rtune ${ARM} (${SPLIT})" \
        --out "analysis/retrieval_tune/${SPLIT}/${ARM}.json"
done
echo "[rtune] done: $ARM"
