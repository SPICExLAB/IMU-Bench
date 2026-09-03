#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --job-name=openwatch_table6_build
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python scripts/build_openwatch_table6.py --condition both --split test
python scripts/build_openwatch_fair_table.py
