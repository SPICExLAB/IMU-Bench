#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=inertia1_smoke_downstream
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export WANDB_MODE=disabled
export HF_HOME=/gpfs/projects/p32139/.cache
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -u baselines/inertia1/scripts/smoke_downstream.py --variant "${VARIANT:-patchtst}" ${ONLY:+--only $ONLY}
