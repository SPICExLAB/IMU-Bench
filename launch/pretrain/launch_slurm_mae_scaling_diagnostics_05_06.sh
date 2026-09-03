#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --job-name=mae_scaling_diagnostics_05_06
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Targeted resubmit of only 05_val_loss_breakdown.py + 06_budget_protocols.py --
# 01-04's outputs are already complete and saved under
# analysis/mae_scaling_diagnostics/out/{01_composition,02_split_audit,
# 03_redundancy,04_source_weighting}/ from job 7704913 (cancelled after 04
# finished, since 05 turned out to have a real bug -- see per_batch_breakdown's
# docstring in 05_val_loss_breakdown.py for the full story: it silently
# reimplemented forward_loss by hand and produced systematically wrong
# reconstructed tensors for reasons never fully root-caused; 05 now calls the
# real forward_loss and captures its internal tensors via hooks instead, with
# a hard sanity-check assertion against forward_loss's own reported loss).
# No need to redo the multi-hour raw source rebuild for this rerun.
#
# UPDATE 2026-07-23: the first gengpu attempt (job 7757568) was still stuck on
# fraction 1/6 after 73 minutes and had NO incremental save (all-or-nothing
# json.dump at the very end) -- cancelled before it could burn the full
# 4h budget for nothing. Root cause: 05_val_loss_breakdown.py was rebuilding
# every source's raw val dataset inside the per-fraction loop, i.e. 6x
# redundant raw builds, even though all 6 scaling configs share an identical
# `sources:` block and only the checkpoint differs per fraction. Fixed: val
# batches are now built ONCE up front and reused across all 6 fraction
# evaluations, and results are written after every fraction, not just at the
# end.

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

echo "=== 05_val_loss_breakdown.py ==="
python analysis/mae_scaling_diagnostics/05_val_loss_breakdown.py

echo "=== 06_budget_protocols.py ==="
python analysis/mae_scaling_diagnostics/06_budget_protocols.py

echo "All remaining diagnostics complete -- see analysis/mae_scaling_diagnostics/out/"
