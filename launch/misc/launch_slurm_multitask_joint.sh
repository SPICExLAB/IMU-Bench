#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --job-name=mt_joint
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Joint multi-task fine-tuning of one shared encoder (scripts/train_multitask_joint.py).
# Pass the run name, then the script's own arguments:
#   sbatch --job-name=mt_body_odom launch/misc/launch_slurm_multitask_joint.sh \
#       --name mt_body_odom --tasks full_body_pose_nymeria odometry_nymeria_local \
#       --ratio 1 1 --steps 40000
#
# Memory: the smoke test (job 7031122, three tasks) peaked at 64.5 G; 40 G was OOM-killed
# (job 7023958). Keep the request at or above 96 G whenever OpenWatch is one of the tasks,
# since its loader preloads every window.
set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

python -u scripts/train_multitask_joint.py "$@"
