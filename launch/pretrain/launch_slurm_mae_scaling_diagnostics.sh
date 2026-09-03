#!/bin/bash
#SBATCH --account=b1232
#SBATCH --partition=b1232
#SBATCH --gres=gpu:h100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --job-name=mae_scaling_diagnostics
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs the CPU-bound MAE data-scaling diagnostic audits (analysis/mae_scaling_diagnostics/
# 01-04) that build every pretraining source from raw at full resolution once --
# this OOMs under the ~4GB interactive-session memory cgroup, hence a real SLURM
# allocation. Read-only with respect to runs/, configs/, and every canonical
# checkpoint -- all outputs land under analysis/mae_scaling_diagnostics/out/.
#
# 05_val_loss_breakdown.py (needs a GPU + existing best.pt checkpoints) is
# included here too since this job already has gpu:h100:1 allocated; it runs
# last and is comparatively quick (a handful of forward passes per fraction).
#
# Does NOT run 07/08 (checkpoint-sweep / variance-study reruns) -- those launch
# real additional training jobs and need a separate go/no-go, see
# analysis/mae_scaling_diagnostics/README.md.

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL

if [ ! -f TransfHAR_env/bin/activate ]; then
    python -m venv TransfHAR_env
    source TransfHAR_env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m pip install -e .
else
    source TransfHAR_env/bin/activate
fi

mkdir -p logs analysis/mae_scaling_diagnostics/out

echo "=== 01_composition_audit.py ==="
python analysis/mae_scaling_diagnostics/01_composition_audit.py

echo "=== 02_split_audit.py ==="
python analysis/mae_scaling_diagnostics/02_split_audit.py

echo "=== 03_redundancy_audit.py ==="
python analysis/mae_scaling_diagnostics/03_redundancy_audit.py

echo "=== 04_source_weighting_audit.py ==="
python analysis/mae_scaling_diagnostics/04_source_weighting_audit.py

echo "=== 05_val_loss_breakdown.py ==="
python analysis/mae_scaling_diagnostics/05_val_loss_breakdown.py

echo "=== 06_budget_protocols.py ==="
python analysis/mae_scaling_diagnostics/06_budget_protocols.py

echo "All diagnostics complete -- see analysis/mae_scaling_diagnostics/out/"
