#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --job-name=verify_stage2_warmstart
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python analysis/mae_scaling_diagnostics/checks/verify_stage2_warmstart.py
