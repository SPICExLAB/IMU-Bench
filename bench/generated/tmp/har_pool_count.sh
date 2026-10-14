#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --job-name=har_pool_count
#SBATCH --output=bench/generated/tmp/%x_%j.out
#SBATCH --error=bench/generated/tmp/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate
python -u analysis/bench_stats/har_pool_per_dataset.py
