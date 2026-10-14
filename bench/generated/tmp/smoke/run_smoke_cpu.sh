#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --job-name=smoke_ssl3
#SBATCH --output=bench/generated/tmp/smoke/%x_%j.out
#SBATCH --error=bench/generated/tmp/smoke/%x_%j.err
# CPU smoke test of the three new SSL objectives (tiny encoder, 8 steps, val every 4) + a --resume latest continuation
# to 12 steps that exercises the checkpoint restore of the bundled extra module (DINOState / JEPAState / heads).
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for m in sslwear dino jepa; do
  rm -rf runs/smoke_$m
  echo "===== $m run 1 (8 steps)  $(date +%T)"
  python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/tmp/smoke/${m}_cpu.yaml --run-name smoke_$m 2>&1 \
     | grep -E "Traceback|Error|error|step=(4|8) split|split=val|new best|SSL baseline|checkpoint" | grep -v "wandb\|checkpoint not found" | tail -12
  echo "[smoke] $m run 1 exit=${PIPESTATUS[0]}  ckpts: $(ls runs/smoke_$m/checkpoints 2>/dev/null | tr '\n' ' ')"
  python - "$m" <<'PY'
import sys, yaml
m=sys.argv[1]; p=f"bench/generated/tmp/smoke/{m}_cpu.yaml"; c=yaml.safe_load(open(p)); c["trainer"]["max_steps"]=12
yaml.safe_dump(c, open(f"bench/generated/tmp/smoke/{m}_cpu_resume.yaml","w"), sort_keys=False)
PY
  echo "===== $m run 2 (--resume latest -> 12 steps)  $(date +%T)"
  python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/tmp/smoke/${m}_cpu_resume.yaml --run-name smoke_$m --resume latest 2>&1 \
     | grep -E "Traceback|Error|error|resume|step=(9|12) split|split=val|new best" | grep -v wandb | tail -8
  echo "[smoke] $m run 2 exit=${PIPESTATUS[0]}"
done
echo "[smoke] all done $(date +%T)"
