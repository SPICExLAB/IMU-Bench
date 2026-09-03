#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:h100:1
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --job-name=nightbeat_sweep
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

# ── LR axis, baseline schedule held fixed ────────────────────────────────────
run lr3e-5            --lr 3.0e-5 --epochs 20 --patience 5
run lr1e-4_baseline   --lr 1.0e-4 --epochs 20 --patience 5
run lr3e-4            --lr 3.0e-4 --epochs 20 --patience 5

# ── Train longer / more patiently at the two best-looking LRs ────────────────
run lr1e-4_long       --lr 1.0e-4 --epochs 40 --patience 10
run lr3e-4_long       --lr 3.0e-4 --epochs 40 --patience 10

# ── Recipe refinements on top of the longer schedule ─────────────────────────
run lr3e-4_warmup     --lr 3.0e-4 --epochs 40 --patience 10 --warmup-frac 0.1
run lr3e-4_llrd       --lr 3.0e-4 --epochs 40 --patience 10 --warmup-frac 0.1 \
                      --layer-lr-decay 0.75
run lr1e-4_l1         --lr 1.0e-4 --epochs 40 --patience 10 --loss l1
run lr3e-4_head512    --lr 3.0e-4 --epochs 40 --patience 10 --warmup-frac 0.1 \
                      --head-hidden 512

echo "=============== sweep summary ($(date +%T)) ==============="
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; else echo "all rows completed"; fi
python - <<'PY'
import json, glob, os
rows = []
for f in sorted(glob.glob("runs/nightbeat_sweep/*.json")):
    r = json.load(open(f)); m = r["model"]; h = r["config"]["hparams"]
    rows.append((os.path.basename(f)[:-5], r.get("val_mae_mean"),
                 m["mae_participant_mean"], m["rmse_participant_mean"],
                 m["cor_participant_mean"], h["lr"], h["epochs"]))
rows.sort(key=lambda r: (r[1] is None, r[1]))   # rank by VAL MAE
print(f"{'run':<20}{'VAL_MAE*':>10}{'test_MAE':>10}{'test_RMSE':>11}{'test_Cor':>10}{'lr':>9}{'ep':>5}")
print("-"*75)
for n, v, a, b, c, lr, ep in rows:
    vs = f"{v:.3f}" if v is not None else "--"
    print(f"{n:<20}{vs:>10}{a:>10.2f}{b:>11.2f}{c:>10.2f}{lr:>9.1e}{ep:>5}")
print("\n* ranked by VAL MAE (held-out train participants) -- the selection")
print("  metric. test_* columns are reported, NOT selected on.")
print("baseline to beat: test MAE 1.95 / RMSE 3.44 / Cor 0.80")
print("paper Nightbeat: 0.88 / 2.24 / 0.81   |   paper Mean-Pred.: 3.16 / 4.07 / 0")
PY
