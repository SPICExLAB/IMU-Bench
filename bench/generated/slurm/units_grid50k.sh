#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --job-name=units_grid50k
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Per-unit sidecars (-> bootstrap CIs) for the 50k-step volume-grid probes that Section 6 reads,
# so "HHAR and Parkinson's peak at 1 %" can be judged against evaluation noise. Inference only.
set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 BENCH_TMP=/gpfs/home/vbc1591/IMU-SSL/bench/generated/tmp
for t in har_hhar har_mhealth har_prism pd_pads microgesture_openwatch; do
  python -m bench.rescore_units --task $t --cell g5_k1000_f0p1__probe --cell g5_k1000_f1__probe \
      --cell g5_k1000_f10__probe --cell g5_k1000_f100__probe || echo "[units] $t: some cells errored"
done
python -m bench.bootstrap --task har_hhar --task har_mhealth --task har_prism --task pd_pads --task microgesture_openwatch \
    --no-pairs --out bench/results/_bootstrap/grid50k.json
echo "[units] done $(date -Is)"
