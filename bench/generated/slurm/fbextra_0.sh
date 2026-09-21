#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=fbextra_0
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python -u -m bench.rescore_full_body_extra --cells mae_5src__probe --max-batches 3     # smoke test first: fails fast
python -u -m bench.rescore_full_body_extra --cells mae_5src__probe limubert_5src__finetune focal_5src__probe relcon_5src__finetune mae_5src_acconly__probe control_raw__probe
echo "[fbextra_0] finished $(date -Is)"
