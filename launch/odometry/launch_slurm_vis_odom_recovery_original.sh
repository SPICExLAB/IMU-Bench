#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G
#SBATCH --job-name=vis_odom_recovery_original
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
python analysis/mae_scaling_diagnostics/checks/vis_odom_recovery_altattn.py \
    --pkl /projects/b53017/vbc1591/processed_data_test_body_Ty_rwrist/20230607_s0_james_johnson_act3_ifj2gc_cpfbody_rwrist_imu50hz_wristimu50hz_pose50hz.pkl \
    --model-config configs/vit1d_odom_finetune_nymeria_mae_multidata_flu.yaml \
    --run vit1d_odom_finetune_nymeria_mae_multidata --ckpt best \
    --save-fig data_vis/odom_recovery/vit1d_odom_finetune_nymeria_mae_multidata_best_topview.png
