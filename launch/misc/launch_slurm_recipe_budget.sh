#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --job-name=recipe_budget
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Prints session_fraction x {natural,balanced} epoch/step budgets at
# 5%/25%/100% under the CURRENT production sources block (v3-scale
# ocdetect/capture24/eatingspeed, nhanes excluded) -- analysis/mae_scaling_diagnostics/10_recipe_budget.py.
# CPU-bound raw source build (~2h based on the 01_composition_audit.py
# precedent at the old, smaller ocdetect size -- likely longer now) plus
# cheap index math. No SLURM job for real training is submitted here.

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs
python analysis/mae_scaling_diagnostics/10_recipe_budget.py
