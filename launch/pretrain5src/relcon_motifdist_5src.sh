#!/bin/bash
#SBATCH --account=b53017
#SBATCH --partition=b53017
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=relcon_motifdist_5src
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /gpfs/home/vbc1591/IMU-SSL

if [ ! -f TransfHAR_env/bin/activate ]; then
    python -m venv TransfHAR_env
    source TransfHAR_env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    python -m pip install -e .
else
    source TransfHAR_env/bin/activate
fi

export MAE_MULTIDATA_CACHE_DIR="/projects/b53017/vbc1591/mae_multidata_cache"
if [[ ! -d "$MAE_MULTIDATA_CACHE_DIR" ]]; then
    echo "ERROR: cache directory not found: $MAE_MULTIDATA_CACHE_DIR" >&2
    exit 1
fi

BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/pretrain5src/relcon_motifdist_5src.yaml"
CACHE_MODEL_CONFIG="$(mktemp "/tmp/relcon_motifdist_cache_${SLURM_JOB_ID:-manual}_XXXX.yaml")"
trap 'rm -f "$CACHE_MODEL_CONFIG"' EXIT

python - "$MODEL_CONFIG" "$CACHE_MODEL_CONFIG" "$MAE_MULTIDATA_CACHE_DIR" <<'PY'
import os
import sys
import yaml

in_cfg, out_cfg, cache_dir = sys.argv[1], sys.argv[2], sys.argv[3]

with open(in_cfg, "r") as f:
    cfg = yaml.safe_load(f)

mm_cfg = cfg.setdefault("objective", {}).setdefault("mae_multidata_varlen_altattn_flu", {})
mm_cfg["cache_dir"] = cache_dir

with open(out_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

sources = mm_cfg.get("sources", {}) or {}
enabled_sources = [name for name, src_cfg in sources.items() if (src_cfg or {}).get("enabled", True)]
if not enabled_sources:
    raise RuntimeError("No enabled sources found in objective.mae_multidata_varlen_altattn_flu.sources")

missing = []
for source in enabled_sources:
    for split in ("train", "val"):
        for suffix in ("x", "mask"):
            p = os.path.join(cache_dir, f"{source}_{split}_{suffix}.npy")
            if not os.path.isfile(p):
                missing.append(p)

if missing:
    raise FileNotFoundError(
        "Missing cached files for cache-only training:\n" + "\n".join(missing)
    )

print(f"Using MAE multidata cache: {cache_dir}")
print(f"Cache-only sources ({len(enabled_sources)}): {', '.join(enabled_sources)}")
PY

python -u scripts/train_relcon_motifdist.py \
    --config configs/base.yaml \
    --model-config "$CACHE_MODEL_CONFIG" \
    --run-name relcon_motifdist_5src \
    --runs-root runs \
    --max-steps 20000 \
    --val-every-steps 500 \
    --patience 15
