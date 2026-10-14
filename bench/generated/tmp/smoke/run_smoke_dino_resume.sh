#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --job-name=smoke_dino_resume
#SBATCH --output=bench/generated/tmp/smoke/%x_%j.out
#SBATCH --error=bench/generated/tmp/smoke/%x_%j.err
# True --resume test for DINOState: continue runs/smoke_dino (latest.pt at step 12) to step 16; the restored step
# counter must be 12 (teacher temp / EMA schedule continue) and the centre buffer non-zero.
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
python - <<'PY'
import torch; ck=torch.load("runs/smoke_dino/checkpoints/latest.pt",map_location="cpu",weights_only=False)
d=ck["decoder"]; print("[ckpt] step", int(ck.get("step",-1)), "| DINOState.step", int(d["step"]), "| centre |max|", float(d["center"].abs().max()), "| teacher keys", sum(k.startswith("teacher.") for k in d))
PY
python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/tmp/smoke/dino_cpu_resume16.yaml --run-name smoke_dino --resume latest 2>&1 \
   | grep -E "Traceback|Error|error|resuming|step=(13|16) split|split=val" | grep -v wandb | tail -8
echo "[smoke] dino resume exit=${PIPESTATUS[0]}"
python - <<'PY'
import torch; ck=torch.load("runs/smoke_dino/checkpoints/latest.pt",map_location="cpu",weights_only=False)
print("[ckpt after] step", int(ck.get("step",-1)), "| DINOState.step", int(ck["decoder"]["step"]))
PY
