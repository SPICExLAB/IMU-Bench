#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=eval_arm_pose_position_error
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

python scripts/eval_arm_pose_position_error.py \
    --val-dir /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist \
    --max-val-files 101 \
    --out-json results_arm_pose_position_error.json
