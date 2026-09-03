#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=d60dual
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
for t in wisdm utd_mhad pamap2 recofit hhar mhealth samosa ut_watch prism; do
  r=bench_har_${t}__mae_div60_dc__probe__s42
  [ -f runs/$r/checkpoints/best_macro_f1.pt ] && python bench/generated/score_sc10.py $r har_$t
done
echo D60DUAL_DONE
