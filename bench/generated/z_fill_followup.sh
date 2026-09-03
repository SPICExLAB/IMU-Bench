#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=z_fill
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate
for i in $(seq 1 1400); do [ -f runs/mae_4src_z/.done ] && break; sleep 30; done
[ -f runs/mae_4src_z/.done ] || { echo "mae_4src_z did not finish in the wait window"; exit 1; }
echo "[z_fill] pretrain done -> submitting all 30 downstream cells"
python bench/generated/fill_4src.py mae_4src_z
