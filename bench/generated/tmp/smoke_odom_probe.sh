#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --job-name=smoke_odom_probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export WANDB_MODE=disabled
rm -rf bench/generated/tmp/runs/smoke_odom_probe
python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/tmp/smoke_odom_probe.yaml --run-name smoke_odom_probe
ls -la bench/generated/tmp/runs/smoke_odom_probe/checkpoints/
python - <<'PY'
import torch
o=torch.load("bench/generated/tmp/runs/smoke_odom_probe/checkpoints/latest.pt",map_location="cpu",weights_only=False)
print("keys",list(o.keys())[:12],"step",o["step"],"has odom_decoder",("odom_decoder" in o))
PY
echo SMOKE_OK
