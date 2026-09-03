#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=odom_baselines_smoke_test
#SBATCH --output=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.out
#SBATCH --error=/gpfs/home/vbc1591/IMU-SSL/logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

CACHE=/gpfs/home/vbc1591/IMU-SSL/baselines/odometry/data_cache_smoke
mkdir -p "$CACHE"
python3 - <<'PYEOF'
import numpy as np, os
cache = "/gpfs/home/vbc1591/IMU-SSL/baselines/odometry/data_cache_smoke"
N, T = 256, 128
rng = np.random.default_rng(0)
for split, n in [("train", N), ("val", 64)]:
    X = rng.standard_normal((n, 6, T)).astype(np.float32)
    POS = np.cumsum(rng.standard_normal((n, T + 1, 3)).astype(np.float32) * 0.01, axis=1)
    POS[:, 0, :] = 0
    np.savez_compressed(os.path.join(cache, f"nymeria_worldframe_{split}.npz"),
                         X=X, POS=POS, window_seconds=2.56, target_hz=50)
print("synthetic cache written")
PYEOF

echo "=== TLIO smoke test ==="
cd baselines/odometry/TLIO
python3 -u train_nymeria_tlio.py --cache-dir "$CACHE" --out-dir /tmp/tlio_smoke_out --epochs 2 --patience 5 --batch-size 32

echo "=== Air-IO smoke test ==="
cd ../Air-IO
python3 -u train_nymeria_airio.py --cache-dir "$CACHE" --out-dir /tmp/airio_smoke_out --epochs 2 --patience 5 --batch-size 32

echo "=== SMOKE TEST PASSED ==="
