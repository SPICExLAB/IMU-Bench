#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=nightbeat_hr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Nightbeat-DB: heart rate from an overnight wrist accelerometer.
# Dataset: Moebus et al., IEEE BHI 2024 (https://github.com/eth-siplab/Nightbeat)
# 39 usable participants / 298.4 h (participant 27's wrist file is corrupt in
# the published release itself -- see configs/base_nightbeat.yaml).
#
# Nightbeat-DB is accelerometer-only, so the gyro half of the 6-channel input
# is handled with the pretraining mask token (--gyro-mode mask), exactly as
# capture24/NHANES gyro was handled during the multidata MAE pretrain.
# The `zero` row is the control: same everything, gyro fed as literal zeros.
#
# Prerequisites (run once, CPU-only, ~10 min):
#   bash /projects/b53017/vbc1591/Nightbeat/download_nightbeatdb.sh
#   python scripts/prepare_nightbeat_dataset.py --workers 10

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs runs/nightbeat_hr

BASE=configs/base_nightbeat.yaml
MODEL=configs/vit1d_nightbeat_hr_probe_mae_multidata_flu.yaml
OUT=runs/nightbeat_hr

FAILED=()

run () {   # run <tag> <extra args...>
  local tag="$1"; shift
  echo "=============== ${tag} ($(date +%T)) ==============="
  # Rows are independent, so a failure in one must not abort the sweep --
  # otherwise an 8 h unattended job can lose five good rows to one bad one.
  # `set -e` is suspended for the call and the failure recorded instead.
  set +e
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_nightbeat_hr.py \
      --config "$BASE" --model-config "$MODEL" \
      --folds 5 --seed 42 \
      --out "${OUT}/${tag}.json" "$@"
  local rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "!!! ${tag} FAILED (exit ${rc})"
    FAILED+=("$tag")
  fi
}

# ── Frozen probe: does the pretrained representation linearly carry HR? ──────
run probe_mask_ridge  --mode probe --gyro-mode mask --head ridge
run probe_mask_mlp    --mode probe --gyro-mode mask --head mlp

# ── Control: the same probe with gyro fed as zeros instead of mask tokens ────
run probe_zero_ridge  --mode probe --gyro-mode zero --head ridge

# ── Finetune vs scratch: brackets how much the pretraining is worth here ─────
run finetune_mask     --mode finetune --gyro-mode mask
run scratch_mask      --mode scratch  --gyro-mode mask

# ── Long window: the model-free envelope check improves 5.18 -> 3.57 bpm MAE
#    going 10.24 -> 20.48 s (FFT bin spacing halves), so window length is a real
#    constraint on this task. Only valid under finetune: the MAE pretrain never
#    exceeded 32 time patches, so a FROZEN probe at 20.48 s would index
#    time-positional-embedding rows that were never trained.
run finetune_mask_20s --mode finetune --gyro-mode mask --window-seconds 20.48

echo "=============== summary ($(date +%T)) ==============="
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "ROWS THAT FAILED: ${FAILED[*]}"
else
  echo "all rows completed"
fi
python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("runs/nightbeat_hr/*.json")):
    r = json.load(open(f))
    m, b = r["model"], r["baseline_subject_mean_ORACLE"]
    rows.append((os.path.basename(f)[:-5], m["mae"], m["rmse"], m["cor"],
                 m["mae_participant_mean"], b["mae"]))
print(f"{'run':<22}{'MAE':>8}{'RMSE':>8}{'Cor':>7}{'MAE/subj':>10}{'oracle':>9}")
for n, a, b_, c, d, e in rows:
    print(f"{n:<22}{a:>8.2f}{b_:>8.2f}{c:>7.2f}{d:>10.2f}{e:>9.2f}")
print("\npaper on Nightbeat-DB: Nightbeat 0.88 | Mean-Pred. 3.16 | PWR 8.83 | "
      "Jerks 13.23 | BioInsights 17.47 (MAE bpm)")
PY
