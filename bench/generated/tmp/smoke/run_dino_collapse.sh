#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=00:40:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=24G
#SBATCH --job-name=dino_collapse_test
#SBATCH --output=bench/generated/tmp/smoke/%x_%j.out
#SBATCH --error=bench/generated/tmp/smoke/%x_%j.err
# Does the BatchNorm head keep DINO's targets informative on real (cached) windows? 150 steps, 4-layer/192-wide
# encoder, K=1024, batch 32, no BN vs BN. Collapse = train loss pinned at log K = 6.93 and proto_eff_k -> 1024.
set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL; source TransfHAR_env/bin/activate
export WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
for bn in 0 1; do
  rm -rf runs/smoke_dino_bn$bn
  echo "===== use_bn=$bn  (log K = 6.931)  $(date +%T)"
  python scripts/run_pretrain.py --config configs/base.yaml --model-config bench/generated/tmp/smoke/dino_collapse_bn$bn.yaml --run-name smoke_dino_bn$bn 2>&1 \
    | grep -E "split=(train|val)" | awk '{for(i=1;i<=NF;i++){if($i~/^step=/)s=$i; if($i~/^split=/)p=$i; if($i~/^loss=/)l=$i; if($i~/^proto_eff_k=/)k=$i}; print s, p, l, k}' | awk 'NR%3==1 || /val/'
done
echo "[collapse-test] done $(date +%T)"
