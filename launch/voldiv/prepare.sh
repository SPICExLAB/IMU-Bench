#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --job-name=voldiv_prepare
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# Phase 0 of the volume x diversity factorial (EXPERIMENTS.md 11m): everything
# that does NOT need a pretrain. Produces the motion-type definitions, the
# diversity statistics and the index files for all 17 index-file cells.
#   01 embed the full-size 4-source pool with the frozen mae_4src_z encoder (GPU)
#   02 encoder-free rotation-invariant physical features for every window (CPU)
#   03 both motion-type definitions + their agreement/validity diagnostics
#   04 index files + measured diversity per cell
#   05 pretrain configs, launch scripts, bench registry entries
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export PYTHONUNBUFFERED=1
A=analysis/volume_diversity
python -u $A/01_embed_pool.py
python -u $A/02_physical_features.py
python -u $A/03_type_definitions.py
python -u $A/04_build_arms.py
python -u $A/05_gen_configs.py
echo "=== phase 0 complete; review $A/out/{type_definitions,arms}.json, then:"
echo "    python $A/05_gen_configs.py --write-registry && ./launch/voldiv/submit.sh"
