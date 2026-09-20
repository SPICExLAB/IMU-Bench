#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=96G
#SBATCH --job-name=task_similarity_cknna
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Representational task similarity: CKNNA / linear CKA between the reference MAE encoder and the
# same encoder after fine-tuning on each IMU-Bench task. One shared set of probe windows from the
# Nymeria held-out sessions goes through every encoder's own preprocessing and window length.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs analysis/task_similarity
export HF_HOME=/gpfs/projects/p32139/.cache HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
python analysis/compute_cknna_scratch_encoders.py \
  --out-dir analysis/task_similarity --n-probe 1024 --device cpu \
  --encoders "MAE (no task)=vit1d_ssl_mae_varlen_altattn_realdata_ocdetect_eatingspeed_motionx_sonic:best.pt;\
Arm pose=bench_arm_pose_nymeria__mae_5src__finetune__s42:best.pt;\
Full-body pose=bench_full_body_pose_nymeria__mae_5src__finetune__s42:best_mpjpe.pt;\
Odometry=bench_odometry_nymeria_local__mae_5src__finetune__s42:best.pt;\
Microgesture=bench_microgesture_openwatch__mae_5src__finetune__s42:best.pt;\
IMU-text=bench_language_humanml3d__mae_5src__finetune__s42:latest.pt;\
Parkinson's=bench_pd_pads__mae_5src__finetune__s42:best.pt;\
PRISM=bench_har_prism__mae_5src__finetune__s42:best.pt"
echo "[cknna] done $(date -Is)"
