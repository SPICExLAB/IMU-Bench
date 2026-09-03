#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=qgpu3006,qgpu3013
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_har_probe_mhealth_relcon_multidata
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# RelCon linear probe on MHEALTH -- fills the missing MHEALTH column cell in the
# HAR comparison table (EXPERIMENTS.md sections 9/10). Same generic
# utd_mhad_probe protocol as the already-reported WISDM/UTD-MHAD/PAMAP2/RecoFit
# cells for this row, dataset swapped via data.train_on_datasets.
# The existing MHEALTH MAE probe (job 7745122/7745124) ran in ~11 min with
# early_stopping patience=20, so 2h is generous margin.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config configs/vit1d_har_probe_mhealth_relcon_multidata_flu.yaml \
    --run-name vit1d_har_probe_mhealth_relcon_multidata
