#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=short
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=pads_paper_baseline
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-35
#
# Re-implementation of the PADS paper's smartwatch-only pipeline (Varghese et al.
# 2024, npj PD) -- 18 configurations x 2 protocols, one per array task.
#   protocol=paper : the paper's nested 5-fold CV over the 390 PD+DD subjects
#   protocol=ours  : train on the 312 train subjects / score the 78 val subjects
#                    of pads_pddd_dataset.parquet (the vit1d_pddd_* split)
# Summarise with scripts/summarize_pads_paper_baseline.py

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

PROTOCOLS=(paper ours)
OPTIONS=(a b)
CHANNELS=(acc rot both)
CLFS=(svm cat nn)

i=${SLURM_ARRAY_TASK_ID}
proto=${PROTOCOLS[$(( i / 18 ))]}; r=$(( i % 18 ))
opt=${OPTIONS[$(( r / 9 ))]};      r=$(( r % 9 ))
ch=${CHANNELS[$(( r / 3 ))]}
clf=${CLFS[$(( r % 3 ))]}

case $clf in
  svm) NJ=8; export OMP_NUM_THREADS=1 ;;
  nn)  NJ=2; export OMP_NUM_THREADS=4 ;;
  cat) NJ=1; export OMP_NUM_THREADS=8; export CATBOOST_THREADS=8 ;;
esac

OUT=analysis/pads_paper_baseline/${proto}_pd_vs_dd_${opt}_${ch}_${clf}.json
echo "task $i -> protocol=$proto option=$opt channels=$ch clf=$clf n_jobs=$NJ"
python scripts/pads_paper_baseline.py --mode pd_vs_dd --protocol $proto \
    --option $opt --channels $ch --classifier $clf --n-jobs $NJ \
    --boss-cache /projects/b53017/vbc1591/pads_boss_cache/${proto} \
    --out $OUT
