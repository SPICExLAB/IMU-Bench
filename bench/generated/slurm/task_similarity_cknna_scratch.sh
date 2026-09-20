#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
#SBATCH --job-name=task_similarity_cknna_scratch
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Representational similarity between encoders trained FROM SCRATCH on each task (the benchmark's
# scratch protocol: random init, the task's own data only, no shared pretraining). Same probe windows,
# anchors and metric as the task-adapted run, so the two matrices are directly comparable.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs analysis/task_similarity_scratch
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
python analysis/compute_cknna_scratch_encoders.py \
  --out-dir analysis/task_similarity_scratch --n-probe 1024 --device cpu \
  --encoders "Arm pose=bench_arm_pose_nymeria__scratch__scratch__s42:best.pt;\
Full-body pose=bench_full_body_pose_nymeria__scratch__scratch__s42:best_mpjpe.pt;\
Odometry=bench_odometry_nymeria_local__scratch__scratch__s42:best.pt;\
Microgesture=bench_microgesture_openwatch__scratch__scratch__s42:best.pt;\
IMU-text=bench_language_humanml3d__scratch__scratch__s42:latest.pt;\
Parkinson's=bench_pd_pads__scratch__scratch__s42:best.pt;\
PRISM=bench_har_prism__scratch__scratch__s42:best.pt;\
HHAR=bench_har_hhar__scratch__scratch__s42:best.pt;\
MHEALTH=bench_har_mhealth__scratch__scratch__s42:best.pt;\
User study=bench_har_userwatch__scratch__scratch__s42:best.pt"
echo "[cknna-scratch] done $(date -Is)"
