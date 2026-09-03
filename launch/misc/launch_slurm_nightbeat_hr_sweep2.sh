#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=nightbeat_sweep2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Hyperparameter sweep to improve the finetune/mask row.
# Baseline (job 8357554): lr 1e-4, 20 epochs, patience 5, wd 0.05, smooth_l1,
# no warmup, no layer-wise LR decay  ->  test MAE 1.95 / RMSE 3.44 / Cor 0.80
# (per-participant averaged, the paper's convention).
#
# MODEL SELECTION IS ON val MAE, NOT test MAE.
# `val` here is a disjoint set of TRAIN participants held out inside each fold
# (see train_end_to_end); test participants are never touched during training or
# selection. Choosing a config by test MAE would overfit the very folds the
# headline number is reported on, which would quietly invalidate the comparison
# against the paper. Every row prints both; read the val column to pick.
#
# What the sweep is probing, and why:
#  - lr: the single knob the baseline never varied.
#  - epochs/patience: the baseline early-stopped at epochs 7, 8, 13, 17, 18 --
#    and its WORST fold (3: MAE 2.617) stopped earliest, at 7. That pattern
#    suggests premature stopping rather than convergence, so several rows extend
#    the budget and the patience together.
#  - warmup: standard for ViT finetuning; large early updates can wash out
#    pretrained features before the head is usable.
#  - layer-wise LR decay: lets deep blocks adapt while early feature extractors
#    stay near the pretrained solution -- the usual recipe when a frozen probe
#    is weak (6.10) but finetuning is strong (1.95), i.e. the useful features
#    exist but need re-composition rather than replacement.
#  - l1 loss: directly matches the MAE metric being reported.

set -uo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
mkdir -p logs runs/nightbeat_sweep

BASE=configs/base_nightbeat.yaml
MODEL=configs/vit1d_nightbeat_hr_probe_mae_multidata_flu.yaml
OUT=runs/nightbeat_sweep
FAILED=()

run () {   # run <tag> <extra args...>
  local tag="$1"; shift
  echo "=============== ${tag} ($(date +%T)) ==============="
  echo "args: $*"
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_nightbeat_hr.py \
      --config "$BASE" --model-config "$MODEL" \
      --mode finetune --gyro-mode mask --folds 5 --seed 42 \
      --out "${OUT}/${tag}.json" "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then echo "!!! ${tag} FAILED (exit ${rc})"; FAILED+=("$tag"); fi
}

# ── LR grid EXTENSION ────────────────────────────────────────────────────────
# The main sweep (8370559) found val MAE monotonically decreasing across the
# whole LR axis it tested -- 3e-5: 2.648, 1e-4: 2.203, 3e-4: 1.902 -- i.e. the
# optimum lies at or beyond the top of that grid, so 3e-4 cannot be assumed
# optimal. These rows extend past it to find where it turns over.
run lr6e-4   --lr 6.0e-4 --epochs 20 --patience 5
run lr1e-3   --lr 1.0e-3 --epochs 20 --patience 5
run lr2e-3   --lr 2.0e-3 --epochs 20 --patience 5

echo "=============== sweep2 summary ($(date +%T)) ==============="
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; else echo "all rows completed"; fi
python - <<'PYEOF'
import json, glob, os
rows=[]
for f in sorted(glob.glob("runs/nightbeat_sweep/*.json")):
    r=json.load(open(f)); m=r["model"]; h=r["config"]["hparams"]
    rows.append((os.path.basename(f)[:-5], r.get("val_mae_mean"),
                 m["mae_participant_mean"], m["cor_participant_mean"], h["lr"]))
rows.sort(key=lambda r:(r[1] is None, r[1]))
print(f"{'run':<20}{'VAL_MAE*':>10}{'test_MAE':>10}{'test_Cor':>10}{'lr':>9}")
for n,v,a,c,lr in rows:
    vs=f"{v:.3f}" if v is not None else "--"
    print(f"{n:<20}{vs:>10}{a:>10.2f}{c:>10.2f}{lr:>9.1e}")
print("\n* ranked by VAL MAE. Run-to-run noise measured at ~0.11 bpm test MAE,")
print("  so treat test gaps under ~0.15 bpm as ties.")
PYEOF
