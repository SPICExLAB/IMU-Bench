#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --qos=buyin
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=120G
#SBATCH --job-name=write_mt_fb
#SBATCH --output=bench/generated/tmp/%x_%j.out
#SBATCH --error=bench/generated/tmp/%x_%j.err
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate
python bench/generated/tmp/write_multitask_fullbody_cells.py
