#!/bin/bash
#SBATCH --account=p32139
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=relcon_smoke_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /gpfs/home/vbc1591/IMU-SSL
source TransfHAR_env/bin/activate

echo "=== Unit-level check: relcon_ssl.py functions on synthetic data ==="
python3 -u - <<'PY'
import torch
import sys
sys.path.insert(0, '.')
from imu_lm.objectives.relcon_ssl import (
    MotifDistNet, motifdist_forward_loss, relative_contrastive_loss, two_random_augments,
)
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# two_random_augments on a fake [T,C] window
x_tc = np.random.randn(128, 6).astype(np.float32)
aug = two_random_augments(x_tc)
assert aug.shape == x_tc.shape, f"augment shape mismatch: {aug.shape}"
print("two_random_augments OK, shape", aug.shape)

# MotifDistNet forward + loss
net = MotifDistNet(in_channels=6, embed_dim=64, kernel_size=15).to(device)
x = torch.randn(8, 6, 128, device=device)
loss, logs = motifdist_forward_loss((x,), net, {})
loss.backward()
print("motifdist_forward_loss OK, loss=", loss.item())

# calc_distance
net.eval()
q = torch.randn(8, 128, 6, device=device)
k = torch.randn(8, 128, 6, device=device)
dist = net.calc_distance(q, k)
assert dist.shape == (8,), f"distance shape mismatch {dist.shape}"
print("calc_distance OK, shape", dist.shape)

# relative_contrastive_loss with fake embeddings
bs, d = 8, 32
emb = torch.randn(bs, d, requires_grad=True)
emb_cands = torch.stack([torch.cat((emb[b+1:], emb[:b]), dim=0) for b in range(bs)], dim=0)
distances = torch.stack([torch.rand(bs) for _ in range(bs - 1)])
_, sortedinds = torch.sort(distances, dim=0)
loss2 = relative_contrastive_loss(emb, emb_cands, sortedinds, tau=0.1)
loss2.backward()
print("relative_contrastive_loss OK, loss=", loss2.item())
print("=== ALL UNIT CHECKS PASSED ===")
PY

echo
echo "=== Stage 1 smoke test: train_relcon_motifdist.py (tiny step budget) ==="
export MAE_MULTIDATA_CACHE_DIR="/projects/b53017/vbc1591/mae_multidata_cache"
python3 - "configs/relcon_motifdist_multidata.yaml" "/tmp/relcon_smoke_motifdist_cfg.yaml" "$MAE_MULTIDATA_CACHE_DIR" <<'PY'
import sys, yaml
in_cfg, out_cfg, cache_dir = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(in_cfg))
cfg["objective"]["mae_multidata_varlen_altattn_flu"]["cache_dir"] = cache_dir
yaml.safe_dump(cfg, open(out_cfg, "w"), sort_keys=False)
PY

python3 -u scripts/train_relcon_motifdist.py \
    --config configs/base.yaml \
    --model-config /tmp/relcon_smoke_motifdist_cfg.yaml \
    --run-name relcon_motifdist_SMOKETEST \
    --runs-root runs \
    --max-steps 20 \
    --val-every-steps 10 \
    --patience 100

echo
echo "=== Stage 2 smoke test: run_pretrain.py (tiny step budget) ==="
python3 - "configs/vit1d_ssl_relcon_multidata_flu.yaml" "/tmp/relcon_smoke_stage2_cfg.yaml" "$MAE_MULTIDATA_CACHE_DIR" <<'PY'
import sys, yaml
in_cfg, out_cfg, cache_dir = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(in_cfg))
cfg["objective"]["mae_multidata_varlen_altattn_flu"]["cache_dir"] = cache_dir
cfg["objective"]["motifdist_run"] = "relcon_motifdist_SMOKETEST"
cfg["trainer"]["max_steps"] = 20
cfg["trainer"]["val_every_steps"] = 10
cfg["trainer"]["early_stopping"]["patience"] = 100
yaml.safe_dump(cfg, open(out_cfg, "w"), sort_keys=False)
PY

python3 -u scripts/run_pretrain.py \
    --config configs/base.yaml \
    --model-config /tmp/relcon_smoke_stage2_cfg.yaml \
    --run-name vit1d_ssl_relcon_SMOKETEST

echo
echo "=== SMOKE TEST PASSED (both stages) ==="
