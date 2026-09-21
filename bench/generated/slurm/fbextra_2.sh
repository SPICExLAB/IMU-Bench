#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=fbextra_2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python -u -m bench.rescore_full_body_extra --cells limubert_5src__probe --max-batches 3     # smoke test first: fails fast
python -u -m bench.rescore_full_body_extra --cells limubert_5src__probe simmtm_5src__finetune relcon_5src__probe mae_5src_noaltattn__finetune control_random__probe
echo "[fbextra_2] finished $(date -Is)"
