#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --job-name=vd_subj_prep
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
# CPU-only: stages 08 (provenance rebuild from the raw loaders) and 09 (numpy index
# selection over precomputed embeddings) never touch the GPU. Running them on gengpu
# only parked them behind this user's own GPU quota (QOSMaxGRESPerUser, ~5 h wait).
# Phase 3 of the volume x diversity study: the WEARER axis, the last data
# factor the scaling curve confounded. 08 rebuilds per-window subject/session
# provenance from the raw loaders (and refuses to write if its positional-join
# probes fail); 09 turns it into a 1/3/10/all subject ladder at fixed N and
# type-matched selection.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export PYTHONUNBUFFERED=1
A=analysis/volume_diversity
python -u $A/08_provenance.py --sources ocdetect
python -u $A/09_subject_arms.py
echo "=== subject ladder built; regenerate configs with:"
echo "    python $A/05_gen_configs.py --write-registry"
