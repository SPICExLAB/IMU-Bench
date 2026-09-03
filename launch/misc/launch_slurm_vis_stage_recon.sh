#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=200G
#SBATCH --job-name=vis_stage_recon
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p data_vis/stage_recon
# stage2 best.pt was truncated by a failed checkpoint write (job 8649608);
# its latest.pt is intact, so plot that instead.
for s in stage1 stage2; do
  CK=best   # stage2 best.pt now exists (step 91,000) -- earlier run used stale latest.pt
  python analysis/mae_scaling_diagnostics/checks/vis_stage_reconstruction.py \
      --stage "$s" --n 4 --ckpt "$CK" --out-dir data_vis/stage_recon || echo "SKIP $s"
done
ls -la data_vis/stage_recon/
