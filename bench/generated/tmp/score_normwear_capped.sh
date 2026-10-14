#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --qos=buyin
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --job-name=score_normwear_capped
#SBATCH --output=bench/generated/tmp/%x_%j.out
#SBATCH --error=bench/generated/tmp/%x_%j.err
# Score NormWear kinematic cells whose run was stopped at the 25,000-step cap. The scorers recompute the metric with a
# GPU forward pass over the Nymeria val sessions, so this cannot run on the login node. Cells are scored from best.pt,
# which for these runs is at a step <= the cap; the note records the capped budget.
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate
NOTE="capped budget: NormWear is ~26x slower per optimiser step than the 21M controlled encoders, so its kinematic runs were stopped early (this cell's own budget is checkpoint_step below); the controlled rows run to early stopping at 29k-219k steps, so this value is an upper bound on the error, not a converged number"
for c in "$@"; do
  set -- $c
  echo "===== $1 / $2"
  python -m bench.score --task "$1" --encoder normwear_base --protocol "$2" --note "$NOTE"
  echo "[exit] $?"
done
echo "[score_normwear_capped] done $(date -Is)"
