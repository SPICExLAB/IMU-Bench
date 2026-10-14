#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --job-name=smoke_dino2
#SBATCH --output=bench/generated/tmp/smoke/%x_%j.out
#SBATCH --error=bench/generated/tmp/smoke/%x_%j.err
# DINO smoke rerun after the centre-update fix (tiny encoder, 8 steps + resume to 12).
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
m=dino
rm -rf runs/smoke_$m
echo "===== $m run 1 (8 steps)  $(date +%T)"
python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/tmp/smoke/${m}_cpu.yaml --run-name smoke_$m 2>&1 \
   | grep -E "Traceback|Error|error|step=(4|8) split|split=val|new best|SSL baseline|checkpoint" | grep -v "wandb\|checkpoint not found" | tail -12
echo "[smoke] $m run 1 exit=${PIPESTATUS[0]}  ckpts: $(ls runs/smoke_$m/checkpoints 2>/dev/null | tr '\n' ' ')"
echo "===== $m run 2 (--resume latest -> 12 steps)  $(date +%T)"
python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/tmp/smoke/${m}_cpu_resume.yaml --run-name smoke_$m --resume latest 2>&1 \
   | grep -E "Traceback|Error|error|resume|step=(9|12) split|split=val|new best" | grep -v wandb | tail -8
echo "[smoke] $m run 2 exit=${PIPESTATUS[0]}"
