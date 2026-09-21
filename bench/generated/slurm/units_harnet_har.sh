#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --job-name=units_harnet_har
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
# per-window predictions of the HarNet cells (scored 2026-09-10, before sidecars existed); result JSONs are not rewritten
for t in har_hhar har_mhealth; do for c in harnet5__probe__s42 harnet5__finetune__s42; do python -m bench.rescore_units --task $t --cell $c; done; done
echo "[units_harnet_har] finished $(date -Is)"
