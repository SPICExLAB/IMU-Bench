#!/usr/bin/env python
"""Stage 1 of the volume x diversity factorial (EXPERIMENTS.md 11m).

Embeds every train window of the FULL 4-source pool (1,738,094 windows, all
four datasets at full size -- nothing sub-sampled) with the frozen `mae_4src_z`
encoder, in exactly the ConcatDataset order that `scaling_mode: index_file`
rebuilds, so every index file written later addresses the same windows.

The embedding is a READ-OUT only. Subset selection uses the encoder-free
physical clustering from 02/03 -- see the header of diversity_stats.py for why
that distinction matters to the claim.

Writes
  $WORK/emb_f16.npy        [N, 384] float16   frozen CLS embeddings
  $WORK/enc_pca64_f32.npy  [N, 64]  float32
  $WORK/pool_manifest.json  pool geometry + checkpoint provenance
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arms import (BASE_CONFIG, POOL_CKPT, POOL_CONFIG, POOL_N, PCA_ENC, SEED,
                  SOURCE_SIZES, SOURCES, WORK_DIR)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

_AUG_KEYS = ("random_rotation", "gaussian_noise", "jitter", "bias_drift", "sensor_dropout",
             "time_crop", "frequency_mask", "gain_jitter", "axis_bias", "time_warp")


def load_cfg_no_aug():
    from imu_lm.utils.helpers import deep_update, load_yaml
    cfg = deep_update(load_yaml(BASE_CONFIG), load_yaml(POOL_CONFIG))
    for key in _AUG_KEYS:
        blk = cfg.get(key)
        if isinstance(blk, dict):
            blk["enabled"] = False
    # the pool config trains with the balanced sampler; the sequential pass here
    # must see the plain concat order, and index_file arms use natural sampling
    # anyway, so pin it explicitly rather than relying on the sampler being
    # bypassed by our own DataLoader.
    cfg["objective"]["mae_multidata_varlen_altattn_flu"]["sampling_mode"] = "natural"
    return cfg


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    from imu_lm.data.combined_mae_multidata_loader import make_combined_mae_multidata_loaders
    from imu_lm.models.ViT1D.model import ViT1DEncoderAltAttn

    cfg = load_cfg_no_aug()
    loaders = make_combined_mae_multidata_loaders(cfg)
    train_ds = loaders["train_loader"].dataset
    n = len(train_ds)

    # --- the index space this experiment depends on, asserted not assumed ---
    parts = getattr(train_ds, "datasets", None)
    assert parts is not None, "expected a ConcatDataset over the enabled sources"
    sizes = [len(p) for p in parts]
    assert n == POOL_N, f"pool is {n} windows, arms.py says {POOL_N} -- update arms.py"
    assert sizes == [SOURCE_SIZES[s] for s in SOURCES], (
        f"concat order/sizes {sizes} != arms.py {[SOURCE_SIZES[s] for s in SOURCES]} "
        f"for {SOURCES} -- the index files would address the wrong windows")
    print(f"[embed] pool = {n:,} windows ({n * 5.12 / 3600:,.0f} h), order verified", flush=True)

    emb_path = os.path.join(WORK_DIR, "emb_f16.npy")
    if os.path.exists(emb_path) and np.load(emb_path, mmap_mode="r").shape[0] == n:
        print("[embed] embeddings already on disk, skipping forward pass", flush=True)
    else:
        device = torch.device("cuda")
        encoder = ViT1DEncoderAltAttn(cfg).to(device)
        state = torch.load(POOL_CKPT, map_location=device, weights_only=False)
        encoder.load_state_dict(state["model"])
        encoder.eval()
        print(f"[embed] {POOL_CKPT} (step={state.get('step')})", flush=True)

        seq = torch.utils.data.DataLoader(
            train_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True,
            collate_fn=loaders["val_loader"].collate_fn, drop_last=False)
        out = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(n, 384))
        i, t0 = 0, time.time()
        with torch.no_grad():
            for batch in seq:
                cls = encoder(batch[0].to(device, non_blocking=True))
                b = cls.shape[0]
                out[i:i + b] = cls.half().cpu().numpy()
                i += b
                if (i // 512) % 400 == 0:
                    print(f"[embed] {i:,}/{n:,} ({i / max(1e-9, time.time() - t0):.0f} win/s)", flush=True)
        out.flush()
        assert i == n, f"embedded {i} != pool {n}"
        del out

    # --- PCA-64 ---
    pca_path = os.path.join(WORK_DIR, "enc_pca64_f32.npy")
    emb = np.load(emb_path, mmap_mode="r")
    if os.path.exists(pca_path) and np.load(pca_path, mmap_mode="r").shape[0] == n:
        print("[pca] reusing enc_pca64 on disk", flush=True)
        var = None
    else:
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(SEED)
        fit = np.sort(rng.choice(n, size=min(500_000, n), replace=False))
        pca = PCA(n_components=PCA_ENC, random_state=SEED)
        pca.fit(np.asarray(emb[fit], dtype=np.float32))
        var = float(pca.explained_variance_ratio_.sum())
        print(f"[pca] {PCA_ENC}D keeps {100 * var:.1f}% variance", flush=True)
        z = np.lib.format.open_memmap(pca_path, mode="w+", dtype=np.float32, shape=(n, PCA_ENC))
        for s in range(0, n, 200_000):
            e = min(n, s + 200_000)
            z[s:e] = pca.transform(np.asarray(emb[s:e], dtype=np.float32))
        z.flush()

    state_step = -1
    try:
        state_step = int(torch.load(POOL_CKPT, map_location="cpu", weights_only=False).get("step", -1))
    except Exception:
        pass
    with open(os.path.join(WORK_DIR, "pool_manifest.json"), "w") as f:
        json.dump({"n": n, "sources": SOURCES, "sizes": sizes,
                   "cumulative_sizes": list(np.cumsum(sizes).tolist()),
                   "hours": n * 5.12 / 3600.0,
                   "ckpt": POOL_CKPT, "ckpt_step": state_step,
                   "pca_var": var}, f, indent=2)
    print("[embed] done", flush=True)


if __name__ == "__main__":
    main()
