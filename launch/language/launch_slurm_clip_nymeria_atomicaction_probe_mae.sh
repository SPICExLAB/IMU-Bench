#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=vit1d_clip_nymeria_atomicaction_probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

TEXT_CACHE="/projects/b53017/vbc1591/nymeria_captions/clip_text_emb_cache_atomic_action.pt"
if [[ ! -f "$TEXT_CACHE" ]]; then
    echo "ERROR: text embedding cache not found: $TEXT_CACHE" >&2
    echo "Run:  python scripts/cache_clip_text_embeddings_nymeria.py --captions-root /projects/b53017/vbc1591/nymeria_captions --caption-source atomic_action" >&2
    exit 1
fi

BASE_CONFIG="configs/base.yaml"
MODEL_CONFIG="configs/vit1d_clip_nymeria_atomicaction_probe_mae_multidata.yaml"

CUDA_VISIBLE_DEVICES=0 python scripts/run_pretrain.py \
    --config "$BASE_CONFIG" \
    --model-config "$MODEL_CONFIG" \
    --run-name vit1d_clip_nymeria_atomicaction_mae_probe
