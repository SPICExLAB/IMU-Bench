#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=genhimem

#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --job-name=check_pretrain_eval_overlap
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python analysis/mae_scaling_diagnostics/checks/check_pretrain_eval_overlap.py
