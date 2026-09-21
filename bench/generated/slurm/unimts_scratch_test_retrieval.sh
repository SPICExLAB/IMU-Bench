#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=48G
#SBATCH --job-name=unimts_scratch_test_retrieval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
# UniMTS wrist-only reimplementation, same script / split / galleries / seed as bench.score.language
for split in test val; do
  python scripts/eval_retrieval_alignment.py --model unimts --dataset humanml3d --wrist-only 1 \
    --checkpoint baselines/language_alignment/UniMTS/checkpoint/unimts_userdata_scratch_jointmask_100ep/last.pth \
    --split $split --gallery-sizes 100 --n-subsets 200 --seed 42 \
    --label "UniMTS scratch, joint-mask recipe 100ep last.pth, wrist-only, $split split (bench protocol)" \
    --out analysis/retrieval_eval/humanml3d__unimts_scratch_jointmask_wrist__${split}_bench.json
done
echo "[unimts_scratch_test_retrieval] finished $(date -Is)"
